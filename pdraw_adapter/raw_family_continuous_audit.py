"""Raw-optical family 的分级 continuous NCC 审计与候选资产导出。

本运行器只读取冻结 PSF asset，不运行光学传播、不访问真实数据、不训练模型。
所有解析标签都从候选最终 kernel 的 ``mu_left_x-mu_right_x`` 重算；NCC 只用于
诊断和准入，永远不参与 kernel 变换、label 或 PDOFFSET。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from scipy.ndimage import fourier_shift
from scipy.signal import fftconvolve

from .family_bank import (
    _EXPORT_RETARGET_AXES,
    _RAW_ACTIVE_AXES,
    analytic_centroid_labels,
    retarget_centroid_slope,
)
from .multi_profile import (
    _fit_ncc_records,
    _forward_bilinear_translate_psf,
    _bank_centroid_and_covariance,
    _sha256_file,
    continuous_ncc_v2_runtime_kwargs,
    run_profile_continuous_ncc_diagnostic,
    validate_continuous_ncc_v2_declaration,
)


@dataclass(frozen=True)
class LoadedProfile:
    profile_id: str
    path: Path
    sha256: str
    payload: dict[str, Any]


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)
    return _sha256_file(path)


def _validate_data_isolation(value: Any) -> dict[str, bool]:
    required = {
        "real_pdraw_accessed",
        "google_dev_accessed",
        "google_holdout_accessed",
        "dp5k_accessed",
        "stereo_training_run",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"data_isolation 必须精确为 {sorted(required)}")
    result = {key: bool(item) for key, item in value.items()}
    if any(result.values()):
        raise ValueError("raw-family audit 禁止访问真实数据或启动训练")
    return result


def load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "run",
        "source",
        "candidates",
        "screen",
        "validation",
        "gates",
        "selection",
        "export",
        "output",
        "data_isolation",
    }
    if not isinstance(loaded, dict) or int(loaded.get("schema_version", 0)) != 1:
        raise ValueError("audit 配置必须是 schema_version: 1 mapping")
    if set(loaded) != required:
        raise ValueError(f"audit 配置字段必须精确为 {sorted(required)}")
    _validate_data_isolation(loaded["data_isolation"])
    candidates = loaded["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates 必须是非空列表")
    candidate_ids = [str(row.get("id")) for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate id 不得重复")
    for stage_name in ("screen", "validation"):
        stage = loaded[stage_name]
        if not isinstance(stage, dict):
            raise ValueError(f"{stage_name} 必须是 mapping")
        validate_continuous_ncc_v2_declaration(dict(stage["continuous_ncc"]))
    return loaded


def _load_profiles(config: dict[str, Any]) -> tuple[Path, list[LoadedProfile]]:
    source = config["source"]
    root = Path(source["asset_root"]).resolve()
    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.is_file() or _sha256_file(manifest_path) != str(
        source["artifact_manifest_sha256"]
    ):
        raise RuntimeError("raw family manifest 缺失或 SHA256 不匹配")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_hashes = dict(manifest["files"])
    requested = [str(value) for value in source["profile_ids"]]
    profiles: list[LoadedProfile] = []
    for profile_id in requested:
        relative = f"profiles/{profile_id}/psf_bank.pt"
        path = root / relative
        expected_sha = str(file_hashes.get(relative, ""))
        if not path.is_file() or not expected_sha or _sha256_file(path) != expected_sha:
            raise RuntimeError(f"profile 文件缺失或 SHA256 不匹配：{relative}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or str(payload.get("profile_id")) != profile_id:
            raise ValueError(f"profile payload 非法：{path}")
        bank = np.asarray(torch.as_tensor(payload["psf_bank"]), dtype=np.float32)
        labels = np.asarray(
            torch.as_tensor(payload["analytic_disparity_bins_px"]),
            dtype=np.float64,
        )
        recomputed = analytic_centroid_labels(bank)
        if not np.allclose(labels, recomputed, atol=1.0e-6, rtol=0.0):
            raise RuntimeError(f"source kernel/label 不一致：{profile_id}")
        if bool(payload.get("pdoffset_embedded", True)):
            raise RuntimeError(f"source asset 不得嵌入 PDOFFSET：{profile_id}")
        profiles.append(
            LoadedProfile(
                profile_id=profile_id,
                path=path,
                sha256=expected_sha,
                payload=payload,
            )
        )
    return root, profiles


def _normalized(psf: np.ndarray) -> tuple[np.ndarray, float]:
    value = np.asarray(psf, dtype=np.float64)
    energy = float(value.sum())
    if not np.isfinite(value).all() or np.any(value < 0.0) or energy <= 0.0:
        raise ValueError("PSF 必须有限、非负且具有正能量")
    return value / energy, energy


def _centroid(psf: np.ndarray) -> tuple[float, float]:
    stats = _bank_centroid_and_covariance(np.asarray(psf, dtype=np.float64))
    return float(stats["mu_x"]), float(stats["mu_y"])


def _translate_to_centroid(
    psf: np.ndarray,
    *,
    target_x: float,
    target_y: float,
    refinement_iterations: int = 6,
) -> tuple[np.ndarray, float]:
    normalized, energy = _normalized(psf)
    current = normalized.astype(np.float32)
    retained = 1.0
    for _ in range(int(refinement_iterations)):
        mu_x, mu_y = _centroid(current)
        delta_x = float(target_x) - mu_x
        delta_y = float(target_y) - mu_y
        if max(abs(delta_x), abs(delta_y)) <= 1.0e-7:
            break
        current, fraction = _forward_bilinear_translate_psf(
            current,
            shift_x_px=delta_x,
            shift_y_px=delta_y,
            support_padding_px=0,
        )
        retained *= float(fraction)
    return np.asarray(current, dtype=np.float64) * energy, retained


def _fourier_translate_to_centroid(
    psf: np.ndarray,
    *,
    target_x: float,
    target_y: float,
    refinement_iterations: int = 6,
) -> tuple[np.ndarray, float]:
    """用 exact discrete Fourier phase shift 平移，再显式审计非负投影质量。

    Fourier shift 避免 bilinear splat 对极小亚像素位移的约 1/2 apparent response。
    PSF 仍必须非负，因此每次 shift 后把数值 ringing 投影到非负正交锥并归一；返回
    所有迭代中被裁掉的负质量相对正质量最大比例，供资产门禁独立判断。
    """

    normalized, energy = _normalized(psf)
    current = normalized.astype(np.float64)
    clipped_negative_fraction_max = 0.0
    for _ in range(int(refinement_iterations)):
        mu_x, mu_y = _centroid(current)
        delta_x = float(target_x) - mu_x
        delta_y = float(target_y) - mu_y
        if max(abs(delta_x), abs(delta_y)) <= 1.0e-7:
            break
        translated = np.fft.ifftn(
            fourier_shift(np.fft.fftn(current), shift=(delta_y, delta_x))
        ).real
        positive_mass = float(np.maximum(translated, 0.0).sum())
        negative_mass = float(np.maximum(-translated, 0.0).sum())
        clipped_negative_fraction_max = max(
            clipped_negative_fraction_max,
            negative_mass / max(positive_mass, 1.0e-20),
        )
        current = np.maximum(translated, 0.0)
        current, _ = _normalized(current)
    return current * energy, clipped_negative_fraction_max


def _translate_with_mode(
    psf: np.ndarray,
    *,
    target_x: float,
    target_y: float,
    translation_mode: str,
) -> tuple[np.ndarray, float]:
    if translation_mode == "bilinear_splat":
        return _translate_to_centroid(
            psf, target_x=target_x, target_y=target_y
        )
    if translation_mode == "fourier_nonnegative":
        return _fourier_translate_to_centroid(
            psf, target_x=target_x, target_y=target_y
        )
    raise ValueError(
        "translation_mode 必须为 bilinear_splat 或 fourier_nonnegative"
    )


def _smoothstep_weight(coc: np.ndarray, transition: float, power: float) -> np.ndarray:
    ratio = np.clip(np.abs(coc) / float(transition), 0.0, 1.0)
    return np.power(np.square(ratio) * (3.0 - 2.0 * ratio), float(power))


def pair_common_morphology(
    bank: np.ndarray,
    signed_coc_bins_px: np.ndarray,
    *,
    transition_coc_px: float,
    transition_power: float = 1.0,
    alignment: str = "direct_mean",
    translation_mode: str = "bilinear_splat",
    common_shape_mtf_sigma_px: float | None = None,
    centroid_tolerance_px: float = 2.0e-3,
    retained_mass_min: float = 0.98,
    clipped_negative_fraction_max: float = 0.02,
) -> tuple[np.ndarray, dict[str, Any]]:
    """以共同 L/R centered shape 替换 small-CoC morphology，并保留原 centroid。

    ``direct_mean`` 把去 centroid 后的 L/R 直接平均并用于两侧；
    ``mirror_aligned_mean`` 先把 R 关于 x 镜像后平均，再把共同形态镜像回 R。
    每侧输入能量保持不变；side throughput 是独立 metadata，不在此函数内修改。
    """

    source = np.asarray(bank, dtype=np.float32)
    coc = np.asarray(signed_coc_bins_px, dtype=np.float64)
    if coc.ndim == 1:
        coc = np.tile(coc[None, :], (source.shape[0], 1))
    if source.ndim != 7 or source.shape[4] != 2 or coc.shape != source.shape[:2]:
        raise ValueError("pair-common bank/CoC shape 不一致")
    if alignment not in {"direct_mean", "mirror_aligned_mean"}:
        raise ValueError("alignment 必须为 direct_mean 或 mirror_aligned_mean")
    if translation_mode not in {
        "bilinear_splat",
        "fourier_nonnegative",
        "bilinear_center_fourier_place",
    }:
        raise ValueError(
            "translation_mode 必须为 bilinear_splat、fourier_nonnegative 或 "
            "bilinear_center_fourier_place"
        )
    if common_shape_mtf_sigma_px is not None and (
        not math.isfinite(float(common_shape_mtf_sigma_px))
        or float(common_shape_mtf_sigma_px) <= 0.0
    ):
        raise ValueError("common_shape_mtf_sigma_px 必须为有限正数或 null")
    center_translation_mode = (
        "bilinear_splat"
        if translation_mode == "bilinear_center_fourier_place"
        else translation_mode
    )
    place_translation_mode = (
        "fourier_nonnegative"
        if translation_mode == "bilinear_center_fourier_place"
        else translation_mode
    )
    transition = float(transition_coc_px)
    power = float(transition_power)
    if not math.isfinite(transition) or transition <= 0.0:
        raise ValueError("transition_coc_px 必须为有限正数")
    if not math.isfinite(power) or power <= 0.0:
        raise ValueError("transition_power 必须为有限正数")
    output = np.empty_like(source)
    retained_values: list[float] = []
    clipped_negative_values: list[float] = []
    for index in np.ndindex(source.shape[:4]):
        pair = source[index]
        centered: list[np.ndarray] = []
        centroids: list[tuple[float, float]] = []
        energies: list[float] = []
        for side in range(2):
            normalized, energy = _normalized(pair[side])
            mu_x, mu_y = _centroid(normalized)
            zeroed, translation_quality = _translate_with_mode(
                normalized,
                target_x=0.0,
                target_y=0.0,
                translation_mode=center_translation_mode,
            )
            centered.append(zeroed)
            centroids.append((mu_x, mu_y))
            energies.append(energy)
            if center_translation_mode == "bilinear_splat":
                retained_values.append(translation_quality)
            else:
                clipped_negative_values.append(translation_quality)
        if alignment == "direct_mean":
            common = 0.5 * (centered[0] + centered[1])
        else:
            aligned_right = np.flip(centered[1], axis=1).copy()
            common = 0.5 * (centered[0] + aligned_right)
        if common_shape_mtf_sigma_px is not None:
            sigma = float(common_shape_mtf_sigma_px)
            radius = max(1, int(math.ceil(4.0 * sigma)))
            coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
            one_dimensional = np.exp(-0.5 * np.square(coordinates / sigma))
            one_dimensional /= one_dimensional.sum()
            mtf_kernel = np.outer(one_dimensional, one_dimensional)
            mtf_kernel /= mtf_kernel.sum()
            common = fftconvolve(common, mtf_kernel, mode="same")
            common = np.maximum(common, 0.0)
            common, _ = _normalized(common)
        if alignment == "direct_mean":
            side_shapes = (common, common)
        else:
            side_shapes = (common, np.flip(common, axis=1).copy())
        common_outputs: list[np.ndarray] = []
        for side in range(2):
            placed, translation_quality = _translate_with_mode(
                side_shapes[side],
                target_x=centroids[side][0],
                target_y=centroids[side][1],
                translation_mode=place_translation_mode,
            )
            if place_translation_mode == "bilinear_splat":
                retained_values.append(translation_quality)
            else:
                clipped_negative_values.append(translation_quality)
            placed, _ = _normalized(placed)
            common_outputs.append(placed * energies[side])
        aperture_index, coc_index, _, _ = index
        raw_weight = float(
            _smoothstep_weight(
                np.asarray([coc[aperture_index, coc_index]]), transition, power
            )[0]
        )
        for side in range(2):
            raw_normalized, energy = _normalized(pair[side])
            common_normalized, _ = _normalized(common_outputs[side])
            mixed = raw_weight * raw_normalized + (1.0 - raw_weight) * common_normalized
            mixed, _ = _normalized(mixed)
            output[index + (side,)] = np.asarray(mixed * energy, dtype=np.float32)
    source_stats = _bank_centroid_and_covariance(source)
    output_stats = _bank_centroid_and_covariance(output)
    centroid_error = np.maximum(
        np.abs(output_stats["mu_x"] - source_stats["mu_x"]),
        np.abs(output_stats["mu_y"] - source_stats["mu_y"]),
    )
    energy_error = np.abs(output_stats["energy"] - source_stats["energy"])
    centroid_error_max = float(np.max(centroid_error))
    retained_min = float(min(retained_values)) if retained_values else 1.0
    clipped_negative_max = (
        float(max(clipped_negative_values)) if clipped_negative_values else 0.0
    )
    if centroid_error_max > float(centroid_tolerance_px):
        raise RuntimeError(
            f"pair-common centroid 未保持：{centroid_error_max:.9g}px"
        )
    if retained_min < float(retained_mass_min):
        raise RuntimeError(f"pair-common support retained mass 过低：{retained_min:.9g}")
    if clipped_negative_max > float(clipped_negative_fraction_max):
        raise RuntimeError(
            "pair-common Fourier shift 非负投影过强："
            f"{clipped_negative_max:.9g} > {float(clipped_negative_fraction_max):.9g}"
        )
    shape_rows: list[dict[str, Any]] = []
    for index in np.ndindex(output.shape[:4]):
        zero_centered: list[np.ndarray] = []
        zero_centroid_residuals: list[float] = []
        for side in range(2):
            normalized, _ = _normalized(output[index + (side,)])
            mu_x, mu_y = _centroid(normalized)
            centered_side, _ = _translate_with_mode(
                normalized,
                target_x=0.0,
                target_y=0.0,
                translation_mode="bilinear_splat",
            )
            centered_side, _ = _normalized(centered_side)
            residual_x, residual_y = _centroid(centered_side)
            zero_centered.append(centered_side)
            zero_centroid_residuals.append(max(abs(residual_x), abs(residual_y)))
        direct_l1 = float(np.abs(zero_centered[0] - zero_centered[1]).sum())
        mirror_l1 = float(
            np.abs(zero_centered[0] - np.flip(zero_centered[1], axis=1)).sum()
        )
        aperture_index, coc_index, field_y_index, field_x_index = index
        shape_rows.append(
            {
                "aperture_index": aperture_index,
                "coc_index": coc_index,
                "signed_coc_px": float(coc[aperture_index, coc_index]),
                "field_y_index": field_y_index,
                "field_x_index": field_x_index,
                "zero_centered_direct_l1": direct_l1,
                "zero_centered_mirror_aligned_l1": mirror_l1,
                "zero_centroid_residual_abs_max_px": float(
                    max(zero_centroid_residuals)
                ),
                "final_vs_source_centroid_abs_max_px": float(
                    np.max(centroid_error[index])
                ),
                "final_vs_source_energy_abs_max": float(
                    np.max(energy_error[index])
                ),
            }
        )
    return output, {
        "operation": "small_coc_pair_common_morphology_v1",
        "alignment": alignment,
        "translation_mode": translation_mode,
        "center_translation_mode": center_translation_mode,
        "place_translation_mode": place_translation_mode,
        "common_shape_mtf_sigma_px": (
            None
            if common_shape_mtf_sigma_px is None
            else float(common_shape_mtf_sigma_px)
        ),
        "transition_coc_px": transition,
        "transition_power": power,
        "centroid_abs_max_error_px": centroid_error_max,
        "energy_abs_max_error": float(np.max(energy_error)),
        "retained_mass_min": retained_min,
        "fourier_clipped_negative_fraction_max": clipped_negative_max,
        "zero_centered_direct_l1_mean": float(
            np.mean([row["zero_centered_direct_l1"] for row in shape_rows])
        ),
        "zero_centered_direct_l1_max": float(
            np.max([row["zero_centered_direct_l1"] for row in shape_rows])
        ),
        "zero_centered_mirror_aligned_l1_mean": float(
            np.mean(
                [row["zero_centered_mirror_aligned_l1"] for row in shape_rows]
            )
        ),
        "zero_centered_mirror_aligned_l1_max": float(
            np.max(
                [row["zero_centered_mirror_aligned_l1"] for row in shape_rows]
            )
        ),
        "zero_centroid_residual_abs_max_px": float(
            np.max(
                [row["zero_centroid_residual_abs_max_px"] for row in shape_rows]
            )
        ),
        "per_cell_zero_centered_shape_diagnostic": shape_rows,
        "native_optical_centroid_preserved": True,
        "side_throughput_unchanged": True,
        "ncc_used": False,
        "pdoffset_embedded": False,
    }


def impulse_matching_preserving(
    bank: np.ndarray,
    signed_coc_bins_px: np.ndarray,
    *,
    transition_coc_px: float,
    transition_power: float = 1.0,
    centroid_tolerance_px: float = 2.0e-3,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(bank, dtype=np.float32)
    coc = np.asarray(signed_coc_bins_px, dtype=np.float64)
    if coc.ndim == 1:
        coc = np.tile(coc[None, :], (source.shape[0], 1))
    output = np.empty_like(source)
    for index in np.ndindex(source.shape[:4]):
        aperture_index, coc_index, _, _ = index
        raw_weight = float(
            _smoothstep_weight(
                np.asarray([coc[aperture_index, coc_index]]),
                float(transition_coc_px),
                float(transition_power),
            )[0]
        )
        for side in range(2):
            raw, energy = _normalized(source[index + (side,)])
            mu_x, mu_y = _centroid(raw)
            impulse = np.zeros_like(raw)
            center = (raw.shape[0] - 1) / 2.0
            pixel_x = mu_x + center
            pixel_y = mu_y + center
            x0, y0 = int(math.floor(pixel_x)), int(math.floor(pixel_y))
            fx, fy = pixel_x - x0, pixel_y - y0
            for dx, dy, weight in (
                (0, 0, (1.0 - fx) * (1.0 - fy)),
                (1, 0, fx * (1.0 - fy)),
                (0, 1, (1.0 - fx) * fy),
                (1, 1, fx * fy),
            ):
                impulse[min(y0 + dy, raw.shape[0] - 1), min(x0 + dx, raw.shape[1] - 1)] += weight
            mixed, _ = _normalized(raw_weight * raw + (1.0 - raw_weight) * impulse)
            output[index + (side,)] = np.asarray(mixed * energy, dtype=np.float32)
    source_stats = _bank_centroid_and_covariance(source)
    output_stats = _bank_centroid_and_covariance(output)
    error = np.maximum(
        np.abs(output_stats["mu_x"] - source_stats["mu_x"]),
        np.abs(output_stats["mu_y"] - source_stats["mu_y"]),
    )
    error_max = float(np.max(error))
    if error_max > float(centroid_tolerance_px):
        raise RuntimeError(f"impulse matching centroid 未保持：{error_max:.9g}px")
    return output, {
        "operation": "shifted_impulse_matching_preserving_v1",
        "transition_coc_px": float(transition_coc_px),
        "transition_power": float(transition_power),
        "centroid_abs_max_error_px": error_max,
        "ncc_used": False,
        "pdoffset_embedded": False,
    }


def common_sensor_mtf(
    bank: np.ndarray,
    *,
    sigma_px: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """左右施加同一个单位能量、零 centroid Gaussian sensor/MTF。"""

    sigma = float(sigma_px)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_px 必须为有限正数")
    radius = max(1, int(math.ceil(4.0 * sigma)))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    one_dimensional = np.exp(-0.5 * np.square(coordinates / sigma))
    one_dimensional /= one_dimensional.sum()
    kernel = np.outer(one_dimensional, one_dimensional)
    kernel /= kernel.sum()
    source = np.asarray(bank, dtype=np.float32)
    output = np.empty_like(source)
    for index in np.ndindex(source.shape[:-2]):
        normalized, energy = _normalized(source[index])
        filtered = fftconvolve(normalized, kernel, mode="same")
        filtered = np.maximum(filtered, 0.0)
        filtered, _ = _normalized(filtered)
        output[index] = np.asarray(filtered * energy, dtype=np.float32)
    source_stats = _bank_centroid_and_covariance(source)
    output_stats = _bank_centroid_and_covariance(output)
    centroid_drift = np.maximum(
        np.abs(output_stats["mu_x"] - source_stats["mu_x"]),
        np.abs(output_stats["mu_y"] - source_stats["mu_y"]),
    )
    return output, {
        "operation": "pair_common_zero_centroid_gaussian_sensor_mtf_v1",
        "sigma_px": sigma,
        "kernel_size": int(kernel.shape[0]),
        "centroid_abs_max_drift_px": float(np.max(centroid_drift)),
        "same_kernel_left_right": True,
        "ncc_used": False,
        "pdoffset_embedded": False,
    }


def transform_candidate(
    bank: np.ndarray,
    signed_coc_bins_px: np.ndarray,
    candidate: dict[str, Any],
    *,
    centroid_tolerance_px: float,
    retained_mass_min: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    current = np.asarray(bank, dtype=np.float32).copy()
    operations: list[dict[str, Any]] = []
    for operation in candidate["operations"]:
        kind = str(operation["kind"])
        if kind == "native":
            metadata = {
                "operation": "native_raw_optical_v1",
                "ncc_used": False,
                "pdoffset_embedded": False,
            }
        elif kind == "pair_common":
            current, metadata = pair_common_morphology(
                current,
                signed_coc_bins_px,
                transition_coc_px=float(operation["transition_coc_px"]),
                transition_power=float(operation.get("transition_power", 1.0)),
                alignment=str(operation.get("alignment", "direct_mean")),
                translation_mode=str(
                    operation.get("translation_mode", "bilinear_splat")
                ),
                common_shape_mtf_sigma_px=operation.get(
                    "common_shape_mtf_sigma_px"
                ),
                centroid_tolerance_px=centroid_tolerance_px,
                retained_mass_min=retained_mass_min,
                clipped_negative_fraction_max=float(
                    operation.get("clipped_negative_fraction_max", 0.02)
                ),
            )
        elif kind == "matching_impulse":
            current, metadata = impulse_matching_preserving(
                current,
                signed_coc_bins_px,
                transition_coc_px=float(operation["transition_coc_px"]),
                transition_power=float(operation.get("transition_power", 1.0)),
                centroid_tolerance_px=centroid_tolerance_px,
            )
        elif kind == "common_mtf":
            current, metadata = common_sensor_mtf(
                current, sigma_px=float(operation["sigma_px"])
            )
        elif kind == "dcc_retarget":
            current, _, metadata = retarget_centroid_slope(
                current,
                signed_coc_bins_px,
                [float(value) for value in operation["target_slopes_by_aperture"]],
                support_padding_px=int(operation.get("support_padding_px", 0)),
                anchor_mode=str(operation.get("anchor_mode", "preserve_native_coc0")),
                common_anchor_disparity_px=operation.get(
                    "common_anchor_disparity_px"
                ),
            )
        else:
            raise ValueError(f"未知 candidate operation：{kind}")
        operations.append(metadata)
    labels = analytic_centroid_labels(current)
    return current, labels, operations


def _near_coc_indices(coc: np.ndarray, maximum: float) -> np.ndarray:
    value = np.asarray(coc, dtype=np.float64)
    reference = value[0] if value.ndim == 2 else value
    indices = np.flatnonzero(np.abs(reference) <= float(maximum) + 1.0e-12)
    if indices.size < 3:
        raise ValueError("small-CoC subset 至少需要三个 bin")
    if value.ndim == 2 and not np.allclose(value[:, indices], value[0, indices]):
        raise ValueError("当前 audit 要求各 aperture small-CoC bins 相同")
    return indices


def _select_spatial(
    bank: np.ndarray,
    labels: np.ndarray,
    *,
    field_mode: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    if field_mode == "center":
        center_y = bank.shape[2] // 2
        center_x = bank.shape[3] // 2
        return (
            bank[:, :, center_y : center_y + 1, center_x : center_x + 1],
            labels[:, :, center_y : center_y + 1, center_x : center_x + 1],
            1,
        )
    if field_mode == "full_flattened":
        field_count = bank.shape[2] * bank.shape[3]
        return (
            bank.reshape(bank.shape[0], bank.shape[1], 1, field_count, 2, bank.shape[-2], bank.shape[-1]),
            labels.reshape(labels.shape[0], labels.shape[1], 1, field_count),
            field_count,
        )
    raise ValueError("field_mode 必须为 center 或 full_flattened")


def _fit_groups(
    diagnostic: dict[str, Any],
    *,
    profile_ids: Sequence[str],
    fields_per_profile: int,
    aperture_count: int,
    gates: dict[str, Any],
) -> list[dict[str, Any]]:
    grid_width = int(round(math.sqrt(fields_per_profile)))
    if grid_width * grid_width != fields_per_profile:
        raise ValueError("full-field audit 当前要求方形 field grid")

    def fit_cells(
        *,
        aperture_index: int,
        field_start: int,
        field_end: int,
    ) -> dict[str, Any]:
        labels: list[float] = []
        predictions: list[float] = []
        total = 0
        valid = 0
        for seed_row in diagnostic["seed_diagnostics"]:
            for cell in seed_row["cells"]:
                field_index = int(cell["field_x_index"])
                if (
                    int(cell["aperture_index"]) == aperture_index
                    and field_start <= field_index < field_end
                ):
                    total += 1
                    prediction = cell["ncc_disparity_px"]
                    if prediction is not None:
                        valid += 1
                        labels.append(
                            float(cell["analytic_centroid_disparity_px"])
                        )
                        predictions.append(float(prediction))
        fit = _fit_ncc_records(labels, predictions)
        valid_fraction = valid / float(total)
        label_array = np.asarray(labels, dtype=np.float64)
        prediction_array = np.asarray(predictions, dtype=np.float64)
        sign_threshold = float(gates.get("sign_label_abs_min_px", 0.05))
        sign_mask = np.abs(label_array) > sign_threshold
        sign_agreement = (
            float(
                np.mean(
                    np.sign(label_array[sign_mask])
                    == np.sign(prediction_array[sign_mask])
                )
            )
            if np.any(sign_mask)
            else None
        )
        checks = {
            "slope_min": float(fit["slope_ncc_vs_analytic"])
            >= float(gates["slope_min"]),
            "slope_max": float(fit["slope_ncc_vs_analytic"])
            <= float(gates["slope_max"]),
            "r2": float(fit["r2"]) >= float(gates["r2_min"]),
            # 没有超过冻结阈值的 label 时 sign 是 N/A；有 eligible record 时仍
            # 执行绝对门禁。
            "sign": sign_agreement is None
            or sign_agreement >= float(gates["sign_agreement_min"]),
            "valid": valid_fraction >= float(gates["valid_record_fraction_min"]),
        }
        return {
            "fit": fit,
            "sign_agreement_at_frozen_threshold": sign_agreement,
            "sign_label_abs_min_px": sign_threshold,
            "sign_check_applicable": sign_agreement is not None,
            "valid_record_fraction": valid_fraction,
            "checks": checks,
            "pass": bool(all(checks.values())),
        }

    rows: list[dict[str, Any]] = []
    for profile_index, profile_id in enumerate(profile_ids):
        aperture_rows: list[dict[str, Any]] = []
        field_start = profile_index * fields_per_profile
        field_end = field_start + fields_per_profile
        for aperture_index in range(aperture_count):
            aggregate = fit_cells(
                aperture_index=aperture_index,
                field_start=field_start,
                field_end=field_end,
            )
            fields: list[dict[str, Any]] = []
            for local_field_index in range(fields_per_profile):
                fields.append(
                    {
                        "field_index": local_field_index,
                        "field_y_index": local_field_index // grid_width,
                        "field_x_index": local_field_index % grid_width,
                        **fit_cells(
                            aperture_index=aperture_index,
                            field_start=field_start + local_field_index,
                            field_end=field_start + local_field_index + 1,
                        ),
                    }
                )
            aperture_rows.append(
                {
                    "aperture_index": aperture_index,
                    **aggregate,
                    "fields": fields,
                    "all_fields_pass": all(bool(field["pass"]) for field in fields),
                    "pass": bool(aggregate["pass"])
                    and all(bool(field["pass"]) for field in fields),
                }
            )
        slopes = [float(row["fit"]["slope_ncc_vs_analytic"]) for row in aperture_rows]
        rows.append(
            {
                "profile_id": profile_id,
                "per_aperture": aperture_rows,
                "all_apertures_pass": all(bool(row["pass"]) for row in aperture_rows),
                "all_aperture_fields_pass": all(
                    bool(field["pass"])
                    for aperture in aperture_rows
                    for field in aperture["fields"]
                ),
                "mean_abs_slope_error": float(np.mean(np.abs(np.asarray(slopes) - 1.0))),
                "worst_abs_slope_error": float(np.max(np.abs(np.asarray(slopes) - 1.0))),
            }
        )
    return rows


def _run_stage(
    profiles: Sequence[LoadedProfile],
    candidate: dict[str, Any],
    stage: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    transformed: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    operation_rows: dict[str, Any] = {}
    near_max = float(stage["coc_abs_max_px"])
    for profile in profiles:
        source_bank = np.asarray(torch.as_tensor(profile.payload["psf_bank"]), dtype=np.float32)
        coc = np.asarray(
            torch.as_tensor(profile.payload["signed_coc_bins_px"]), dtype=np.float64
        )
        indices = _near_coc_indices(coc, near_max)
        source_bank = source_bank[:, indices]
        if str(stage["field_mode"]) == "center":
            center_y = source_bank.shape[2] // 2
            center_x = source_bank.shape[3] // 2
            source_bank = source_bank[
                :, :, center_y : center_y + 1, center_x : center_x + 1
            ]
        selected_coc = coc[:, indices] if coc.ndim == 2 else coc[indices]
        candidate_bank, candidate_labels, operations = transform_candidate(
            source_bank,
            selected_coc,
            candidate,
            centroid_tolerance_px=float(gates["centroid_abs_max_error_px"]),
            retained_mass_min=float(gates["retained_mass_min"]),
        )
        spatial_bank, spatial_labels, fields = _select_spatial(
            candidate_bank,
            candidate_labels,
            field_mode=str(stage["field_mode"]),
        )
        transformed.append(spatial_bank)
        labels.append(spatial_labels)
        operation_rows[profile.profile_id] = operations
    fields_per_profile = int(fields)
    combined_bank = np.concatenate(transformed, axis=3)
    combined_labels = np.concatenate(labels, axis=3)
    first_coc = np.asarray(
        torch.as_tensor(profiles[0].payload["signed_coc_bins_px"]), dtype=np.float64
    )
    indices = _near_coc_indices(first_coc, near_max)
    selected_coc = first_coc[:, indices] if first_coc.ndim == 2 else first_coc[indices]
    continuous = dict(stage["continuous_ncc"])
    physical, oracle = run_profile_continuous_ncc_diagnostic(
        combined_bank,
        combined_labels,
        signed_coc_bins_px=selected_coc,
        texture_size=int(stage["texture_size"]),
        seeds=[int(value) for value in stage["seeds"]],
        tile_size=int(stage["tile_size"]),
        tiles_per_axis=int(stage["tiles_per_axis"]),
        search_x=int(stage["search_x"]),
        search_y=int(stage["search_y"]),
        **continuous_ncc_v2_runtime_kwargs(continuous),
    )
    profile_rows = _fit_groups(
        physical,
        profile_ids=[profile.profile_id for profile in profiles],
        fields_per_profile=fields_per_profile,
        aperture_count=combined_bank.shape[0],
        gates=gates,
    )
    passed = [row["profile_id"] for row in profile_rows if row["all_apertures_pass"]]
    return {
        "candidate_id": candidate["id"],
        "profile_count": len(profiles),
        "field_mode": stage["field_mode"],
        "fields_per_profile": fields_per_profile,
        "tile_size": int(stage["tile_size"]),
        "profiles_passed": passed,
        "profile_pass_count": len(passed),
        "profiles": profile_rows,
        "operations": operation_rows,
        "physical": physical,
        "oracle": oracle,
        "analytic_label_source": "final_candidate_kernel_centroid",
        "ncc_updates_labels": False,
        "pdoffset_embedded": False,
    }


def _profile_rank(screen: dict[str, Any]) -> list[str]:
    rows = sorted(
        screen["profiles"],
        key=lambda row: (
            not bool(row["all_apertures_pass"]),
            float(row["worst_abs_slope_error"]),
            float(row["mean_abs_slope_error"]),
            str(row["profile_id"]),
        ),
    )
    return [str(row["profile_id"]) for row in rows]


def _candidate_score(screen: dict[str, Any]) -> tuple[int, float, float, str]:
    passing = [row for row in screen["profiles"] if row["all_apertures_pass"]]
    source = passing or screen["profiles"]
    return (
        -len(passing),
        float(np.mean([row["worst_abs_slope_error"] for row in source])),
        float(np.mean([row["mean_abs_slope_error"] for row in source])),
        str(screen["candidate_id"]),
    )


def _export_candidate_asset(
    *,
    config: dict[str, Any],
    candidate: dict[str, Any],
    profiles: Sequence[LoadedProfile],
    validation: dict[str, Any],
) -> dict[str, Any]:
    export_cfg = config["export"]
    output_root = Path(export_cfg["asset_root"]).resolve()
    if output_root.exists():
        raise FileExistsError(f"candidate asset 输出已存在：{output_root}")
    staging = output_root.with_name(f".{output_root.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"candidate asset staging 已存在：{staging}")
    staging.mkdir(parents=True)
    selected_ids = set(validation["profiles_passed_all_validations"])
    profile_rows: list[dict[str, Any]] = []
    for profile in profiles:
        if profile.profile_id not in selected_ids:
            continue
        source_bank = np.asarray(torch.as_tensor(profile.payload["psf_bank"]), dtype=np.float32)
        coc = np.asarray(
            torch.as_tensor(profile.payload["signed_coc_bins_px"]), dtype=np.float64
        )
        transformed, labels, operations = transform_candidate(
            source_bank,
            coc,
            candidate,
            centroid_tolerance_px=float(config["gates"]["centroid_abs_max_error_px"]),
            retained_mass_min=float(config["gates"]["retained_mass_min"]),
        )
        payload = dict(profile.payload)
        payload.update(
            {
                "asset_id": f"{export_cfg['asset_id']}_{profile.profile_id}",
                "centroid_policy": "native_final_kernel",
                "centroid_variant": str(export_cfg["centroid_variant"]),
                "pdoffset_embedded": False,
                "psf_bank": torch.from_numpy(transformed.copy()),
                "analytic_disparity_bins_px": torch.from_numpy(labels.copy()),
                "source_raw_analytic_disparity_bins_px": torch.as_tensor(
                    profile.payload["analytic_disparity_bins_px"]
                ).detach().clone(),
                "candidate_transform": {
                    "candidate_id": candidate["id"],
                    "operations": operations,
                    "analytic_label_source": "final_kernel_centroid_mu_left_x_minus_mu_right_x",
                    "ncc_role": "diagnostic_admission_only_never_label_writeback",
                    "pdoffset_contract": "disp_gt=analytic_centroid_disp+independent_pd_offset_px",
                },
                "profile_axis_contract": {
                    "raw_active_axes": list(_RAW_ACTIVE_AXES),
                    "inactive_source_export_retarget_axes": list(
                        _EXPORT_RETARGET_AXES
                    ),
                    "audit_candidate_operations": candidate["operations"],
                },
            }
        )
        payload["asset_spec"] = {
            **dict(payload.get("asset_spec", {})),
            "kind": "cldefocus_raw_family_continuous_admitted_candidate_v1",
            "analytic_label_source": "final_psf_centroid_mu_left_x_minus_mu_right_x",
            "pd_offset_embedded": False,
            "ncc_role": "diagnostic_admission_only_never_label_writeback",
        }
        path = staging / "profiles" / profile.profile_id / "psf_bank.pt"
        sha = _atomic_torch_save(path, payload)
        profile_rows.append(
            {
                "profile_id": profile.profile_id,
                "relative_path": str(path.relative_to(staging)),
                "path": str(output_root / path.relative_to(staging)),
                "sha256": sha,
                "shape": list(transformed.shape),
                "analytic_label_sha256": hashlib.sha256(
                    np.ascontiguousarray(labels, dtype=np.float64).tobytes()
                ).hexdigest(),
            }
        )
    summary = {
        "schema_version": 1,
        "status": "pass",
        "asset_id": export_cfg["asset_id"],
        "centroid_variant": export_cfg["centroid_variant"],
        "candidate": candidate,
        "profile_count": len(profile_rows),
        "profiles": profile_rows,
        "analytic_label_source": "final_kernel_centroid_mu_left_x_minus_mu_right_x",
        "ncc_updates_labels": False,
        "pdoffset_embedded": False,
        "parent_load_paths": [row["path"] for row in profile_rows],
        **config["data_isolation"],
    }
    _atomic_write_text(
        staging / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    files: dict[str, str] = {}
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files[str(path.relative_to(staging))] = _sha256_file(path)
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "asset_id": export_cfg["asset_id"],
        "candidate_id": candidate["id"],
        "profile_ids": [row["profile_id"] for row in profile_rows],
        "files": files,
    }
    _atomic_write_text(
        staging / "artifact_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, output_root)
    return {
        "asset_root": str(output_root),
        "asset_id": export_cfg["asset_id"],
        "profile_count": len(profile_rows),
        "artifact_manifest_sha256": _sha256_file(
            output_root / "artifact_manifest.json"
        ),
        "summary_sha256": _sha256_file(output_root / "summary.json"),
        "load_paths": [row["path"] for row in profile_rows],
    }


def run(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    source_root, profiles = _load_profiles(config)
    output_root = Path(config["output"]["root"]).resolve()
    if output_root.exists():
        raise FileExistsError(f"audit 输出已存在，拒绝覆盖：{output_root}")
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    screens: list[dict[str, Any]] = []
    for candidate in config["candidates"]:
        candidate_started = time.perf_counter()
        screen = _run_stage(
            profiles,
            candidate,
            config["screen"],
            config["gates"],
        )
        screen["seconds"] = time.perf_counter() - candidate_started
        screens.append(screen)
        print(
            json.dumps(
                {
                    "candidate": candidate["id"],
                    "screen_pass": screen["profile_pass_count"],
                    "screen_total": len(profiles),
                    "seconds": screen["seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    minimum_screen = int(config["selection"]["minimum_screen_profiles"])
    eligible = [row for row in screens if row["profile_pass_count"] >= minimum_screen]
    eligible.sort(key=_candidate_score)
    validations: list[dict[str, Any]] = []
    candidate_by_id = {str(row["id"]): row for row in config["candidates"]}
    maximum_candidates = int(config["selection"]["max_candidates_for_validation"])
    maximum_profiles = int(config["selection"]["max_profiles_for_validation"])
    for screen in eligible[:maximum_candidates]:
        selected_ids = _profile_rank(screen)[:maximum_profiles]
        selected_profiles = [
            profile for profile in profiles if profile.profile_id in set(selected_ids)
        ]
        per_protocol: list[dict[str, Any]] = []
        passed_sets: list[set[str]] = []
        for tile_size in config["validation"]["tile_sizes"]:
            stage = {**config["validation"], "tile_size": int(tile_size)}
            stage.pop("tile_sizes", None)
            result = _run_stage(
                selected_profiles,
                candidate_by_id[screen["candidate_id"]],
                stage,
                config["gates"],
            )
            result["protocol_id"] = (
                f"{stage['field_mode']}_K{int(tile_size)}"
            )
            per_protocol.append(result)
            passed_sets.append(set(result["profiles_passed"]))
        passed_all = sorted(set.intersection(*passed_sets)) if passed_sets else []
        validations.append(
            {
                "candidate_id": screen["candidate_id"],
                "selected_profile_ids": selected_ids,
                "protocols": per_protocol,
                "profiles_passed_all_validations": passed_all,
                "profile_pass_count_all_validations": len(passed_all),
            }
        )
    minimum_validation = int(config["selection"]["minimum_validation_profiles"])
    admitted = [
        row
        for row in validations
        if row["profile_pass_count_all_validations"] >= minimum_validation
    ]
    selected_validation = admitted[0] if admitted else None
    exported = None
    if selected_validation is not None and bool(config["export"]["enabled"]):
        exported = _export_candidate_asset(
            config=config,
            candidate=candidate_by_id[selected_validation["candidate_id"]],
            profiles=profiles,
            validation=selected_validation,
        )
    if selected_validation is not None:
        status = "pass"
    elif eligible and maximum_candidates == 0:
        status = "screen_pass_pending_validation"
    else:
        status = "no_go"
    summary = {
        "schema_version": 1,
        "status": status,
        "run_id": config["run"]["id"],
        "source_asset_root": str(source_root),
        "source_manifest_sha256": config["source"]["artifact_manifest_sha256"],
        "candidate_count": len(config["candidates"]),
        "screen_profile_count": len(profiles),
        "screen_pass_counts": {
            row["candidate_id"]: row["profile_pass_count"] for row in screens
        },
        "validation_candidate_ids": [row["candidate_id"] for row in validations],
        "selected_candidate_id": (
            None if selected_validation is None else selected_validation["candidate_id"]
        ),
        "selected_profiles": (
            []
            if selected_validation is None
            else selected_validation["profiles_passed_all_validations"]
        ),
        "exported_asset": exported,
        "analytic_label_source": "final_candidate_kernel_centroid",
        "ncc_role": "diagnostic_admission_only_never_label_writeback",
        "pdoffset_embedded": False,
        "elapsed_seconds": time.perf_counter() - started,
        **config["data_isolation"],
    }
    _atomic_write_text(
        output_root / "screens.json",
        json.dumps(screens, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(
        output_root / "validations.json",
        json.dumps(validations, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(
        output_root / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(
        output_root / "resolved_config.yaml",
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
    )
    with (output_root / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["stage", "candidate_id", "protocol", "profile_id", "aperture_index", "slope", "r2", "pass"]
        )
        for screen in screens:
            for profile in screen["profiles"]:
                for aperture in profile["per_aperture"]:
                    writer.writerow(
                        [
                            "screen",
                            screen["candidate_id"],
                            f"{screen['field_mode']}_K{screen['tile_size']}",
                            profile["profile_id"],
                            aperture["aperture_index"],
                            aperture["fit"]["slope_ncc_vs_analytic"],
                            aperture["fit"]["r2"],
                            aperture["pass"],
                        ]
                    )
        for validation in validations:
            for protocol in validation["protocols"]:
                for profile in protocol["profiles"]:
                    for aperture in profile["per_aperture"]:
                        writer.writerow(
                            [
                                "validation",
                                validation["candidate_id"],
                                protocol["protocol_id"],
                                profile["profile_id"],
                                aperture["aperture_index"],
                                aperture["fit"]["slope_ncc_vs_analytic"],
                                aperture["fit"]["r2"],
                                aperture["pass"],
                            ]
                        )
    provenance = {
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256_file(config_path),
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": _sha256_file(Path(__file__).resolve()),
        "source_profile_sha256": {
            profile.profile_id: profile.sha256 for profile in profiles
        },
        "python_executable": sys.executable,
        "summary_sha256": _sha256_file(output_root / "summary.json"),
        **config["data_isolation"],
    }
    _atomic_write_text(
        output_root / "provenance.json",
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
    )
    return {
        "output_root": str(output_root),
        "status": status,
        "selected_candidate_id": summary["selected_candidate_id"],
        "selected_profiles": summary["selected_profiles"],
        "exported_asset": exported,
        "summary_sha256": _sha256_file(output_root / "summary.json"),
        "provenance_sha256": _sha256_file(output_root / "provenance.json"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
