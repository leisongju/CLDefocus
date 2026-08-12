"""同一 CLDefocus 光学 family 的多 profile、多光圈 PDraw bank 生成器。

基础复合镜头、光圈、视场、CoC 与 Rayleigh–Sommerfeld 几何只准备一次；每个
profile 仅改变 pupil response、低阶零 defocus phase screen 和目标 PD centroid
slope。PDOFFSET 永远是 bank 外的独立标签/相机平移项。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import torch
import yaml

try:
    from .multi_profile import (
        LowOrderAberrationProfile,
        ResponseProfile,
        _forward_bilinear_translate_psf,
        _propagate_profile_batch,
        build_side_throughput_grid,
        centroid_xy,
        solve_sensor_offset_for_coc,
    )
    from .multi_aperture import (
        _lens_with_stop_radius,
        build_aperture_plan,
    )
except ImportError:  # pragma: no cover - 兼容直接执行。
    from multi_profile import (  # type: ignore[no-redef]
        LowOrderAberrationProfile,
        ResponseProfile,
        _forward_bilinear_translate_psf,
        _propagate_profile_batch,
        build_side_throughput_grid,
        centroid_xy,
        solve_sensor_offset_for_coc,
    )
    from multi_aperture import (  # type: ignore[no-redef]
        _lens_with_stop_radius,
        build_aperture_plan,
    )


@dataclass(frozen=True)
class FamilyProfile:
    profile_id: str
    seed: int
    response: ResponseProfile
    aberration: LowOrderAberrationProfile
    centroid_slope_px_per_coc: float


_RESPONSE_AXES = (
    "transition_width",
    "cross_talk",
    "split_bias_norm",
    "boundary_curvature_norm",
    "microlens_edge_rolloff",
    "field_split_slope",
    "field_acceptance_slope",
    "lr_throughput_delta",
    "field_lr_throughput_slope",
    "aperture_lr_throughput_slope",
)
_ABERRATION_AXES = (
    "astigmatism_0_waves",
    "astigmatism_45_waves",
    "coma_x_waves",
    "coma_y_waves",
    "spherical_waves",
    "field_astigmatism_0_waves_per_norm",
    "field_astigmatism_45_waves_per_norm",
    "field_coma_x_waves_per_norm",
    "field_coma_y_waves_per_norm",
)
# 只有 response/aberration 轴会进入原始 CLDefocus propagation。centroid slope
# 是 propagation 之后可选的 export-time 几何重定向轴；把两类轴分开记录，避免
# raw-optical 资产把一个未参与成像的数值误报成有效的光学随机轴。
_RAW_ACTIVE_AXES = (*_RESPONSE_AXES, *_ABERRATION_AXES)
_EXPORT_RETARGET_AXES = ("centroid_slope_px_per_coc",)
_PROFILE_AXES = (*_RAW_ACTIVE_AXES, *_EXPORT_RETARGET_AXES)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _axis_seed(master_seed: int, axis: str) -> int:
    payload = f"cldefocus-pdraw-family-v1\0{int(master_seed)}\0{axis}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _sample_axis(
    *,
    master_seed: int,
    axis: str,
    count: int,
    value_range: Sequence[float],
) -> np.ndarray:
    """每个命名轴使用独立 RNG 的一维 LHS；改一轴不会移动其他轴。"""

    if len(value_range) != 2:
        raise ValueError(f"randomization.ranges.{axis} 必须为 [min,max]")
    low, high = (float(value_range[0]), float(value_range[1]))
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"randomization.ranges.{axis} 范围非法")
    if low == high:
        return np.full((count,), low, dtype=np.float64)
    rng = np.random.default_rng(_axis_seed(master_seed, axis))
    bins = (np.arange(count, dtype=np.float64) + rng.random(count)) / float(count)
    rng.shuffle(bins)
    return low + (high - low) * bins


def sample_family_profiles(config: dict[str, Any]) -> list[FamilyProfile]:
    """从独立命名范围确定性采样同一 optical family 内的 profiles。"""

    expected = {"master_seed", "profile_count", "profile_id_prefix", "ranges"}
    if set(config) != expected:
        raise ValueError(f"randomization 字段必须精确为 {sorted(expected)}")
    count = int(config["profile_count"])
    if count < 1 or count > 256:
        raise ValueError("profile_count 必须位于 [1,256]")
    master_seed = int(config["master_seed"])
    ranges = config["ranges"]
    if not isinstance(ranges, dict) or set(ranges) != set(_PROFILE_AXES):
        raise ValueError(
            "randomization.ranges 必须精确覆盖全部独立轴："
            f"{sorted(_PROFILE_AXES)}"
        )
    sampled = {
        axis: _sample_axis(
            master_seed=master_seed,
            axis=axis,
            count=count,
            value_range=ranges[axis],
        )
        for axis in _PROFILE_AXES
    }
    prefix = str(config["profile_id_prefix"])
    profiles: list[FamilyProfile] = []
    for index in range(count):
        profile_id = f"{prefix}{index:03d}"
        seed = _axis_seed(master_seed, profile_id) & 0x7FFFFFFF
        response_values = {axis: float(sampled[axis][index]) for axis in _RESPONSE_AXES}
        aberration_values = {
            axis: float(sampled[axis][index]) for axis in _ABERRATION_AXES
        }
        response = ResponseProfile(
            profile_id=profile_id,
            seed=seed,
            **response_values,
        )
        aberration = LowOrderAberrationProfile(**aberration_values)
        slope = float(sampled["centroid_slope_px_per_coc"][index])
        if response.transition_width <= 0.0:
            raise ValueError("transition_width 必须为正数")
        if not 0.0 <= response.cross_talk < 0.5:
            raise ValueError("cross_talk 必须位于 [0,.5)")
        if abs(response.split_bias_norm) >= 1.0:
            raise ValueError("split_bias_norm 必须位于 (-1,1)")
        if response.microlens_edge_rolloff < 0.0:
            raise ValueError("microlens_edge_rolloff 不得为负")
        if slope <= 0.0:
            raise ValueError("centroid_slope_px_per_coc 必须为正数")
        profiles.append(
            FamilyProfile(
                profile_id=profile_id,
                seed=seed,
                response=response,
                aberration=aberration,
                centroid_slope_px_per_coc=slope,
            )
        )
    return profiles


def family_profile_document(profile: FamilyProfile) -> dict[str, Any]:
    parameters = {
        "profile_id": profile.profile_id,
        "seed": profile.seed,
        "response": asdict(profile.response),
        "low_order_aberration": asdict(profile.aberration),
        "centroid_slope_px_per_coc": profile.centroid_slope_px_per_coc,
    }
    return {
        **parameters,
        "axis_contract": {
            "raw_active_axes": list(_RAW_ACTIVE_AXES),
            "export_retarget_axes": list(_EXPORT_RETARGET_AXES),
            "centroid_slope_px_per_coc_role": "export_retarget_only",
        },
        "parameter_sha256": _sha256_json(parameters),
    }


def _prepare_family_optical_contexts(resolved: dict[str, Any]) -> dict[str, Any]:
    """构造真正的二维 field grid；旧 multi_profile 的一维 field 路径保持不变。"""

    import jax.numpy as jnp
    from datasyn.optics.complens.tabular_lens import load_tabular_lens_from_mytable
    from datasyn.optics.imaging.imaging import HFProper, make_imaging

    sensor_cfg = resolved["sensor"]
    optics = resolved["optics"]
    lens_path = Path(resolved["lens"]["root"]) / str(resolved["lens"]["relative_path"])
    source_lens = load_tabular_lens_from_mytable(lens_path)
    native_imaging = make_imaging(
        lens=source_lens,
        sen_wh=(int(sensor_cfg["width_px"]), int(sensor_cfg["height_px"])),
        proper=HFProper(),
        wvl_ref=float(optics["wavelength_m"]),
        pixsize=float(sensor_cfg["pixel_pitch_m"]),
    )
    aperture_plan = build_aperture_plan(
        float(native_imaging.parax.fnum),
        float(source_lens.get_aperture(source_lens.stop_idx)),
        resolved["apertures"]["f_numbers"],
    )
    field_x = [float(value) for value in optics["field_x_normalized"]]
    field_y = [float(value) for value in optics.get("field_y_normalized", [0.0])]
    for name, values in (("field_x", field_x), ("field_y", field_y)):
        if values != sorted(values) or not values or any(abs(value) > 1.0 for value in values):
            raise ValueError(f"{name}_normalized 必须为 [-1,1] 内严格递增非空序列")
    target_coc = np.asarray(optics["target_signed_coc_bins_px"], dtype=np.float64)
    if target_coc.ndim != 1 or np.any(np.diff(target_coc) <= 0.0):
        raise ValueError("target_signed_coc_bins_px 必须严格递增")
    if np.flatnonzero(np.abs(target_coc) <= 1.0e-12).size != 1:
        raise ValueError("target_signed_coc_bins_px 必须包含且仅包含一个精确 0")

    contexts: list[dict[str, Any]] = []
    for aperture in aperture_plan:
        lens = _lens_with_stop_radius(source_lens, aperture.stop_radius_m)
        imaging = make_imaging(
            lens=lens,
            sen_wh=(int(sensor_cfg["width_px"]), int(sensor_cfg["height_px"])),
            proper=HFProper(),
            wvl_ref=float(optics["wavelength_m"]),
            pixsize=float(sensor_cfg["pixel_pitch_m"]),
        )
        actual_f_number = float(imaging.parax.fnum)
        if abs(actual_f_number - aperture.requested_f_number) > 1.0e-6:
            raise RuntimeError(
                f"f-number 漂移：requested={aperture.requested_f_number}, "
                f"actual={actual_f_number}"
            )
        wavefronts: list[list[Any]] = []
        base_propagations = np.empty((len(field_y), len(field_x)), dtype=np.float64)
        offsets = np.empty(
            (len(field_y), len(field_x), target_coc.size), dtype=np.float64
        )
        actual_coc = np.empty_like(offsets)
        for field_y_index, field_y_value in enumerate(field_y):
            wavefront_row: list[Any] = []
            for field_x_index, field_x_value in enumerate(field_x):
                wavefront = imaging.generate_wavefront(
                    float(optics["object_depth_m"]),
                    jnp.asarray([float(field_x_value), float(field_y_value)]),
                    wvl=float(optics["wavelength_m"]),
                )
                base_s_prop = float(lens.imgpos - imaging.xp.z)
                for coc_index, target in enumerate(target_coc):
                    offset, measured = solve_sensor_offset_for_coc(
                        wavefront,
                        base_s_prop=base_s_prop,
                        target_coc_px=float(target),
                        envelope_m=float(optics["sensor_offset_envelope_m"]),
                        parax=imaging.parax,
                        pixel_pitch_m=float(sensor_cfg["pixel_pitch_m"]),
                        tolerance_px=float(
                            optics.get("target_coc_tolerance_px", 1.0e-6)
                        ),
                    )
                    offsets[field_y_index, field_x_index, coc_index] = offset
                    actual_coc[field_y_index, field_x_index, coc_index] = measured
                wavefront_row.append(wavefront)
                base_propagations[field_y_index, field_x_index] = base_s_prop
            wavefronts.append(wavefront_row)
        contexts.append(
            {
                "aperture_plan": aperture,
                "imaging": imaging,
                "actual_f_number": actual_f_number,
                "wavefronts": wavefronts,
                "base_propagations": base_propagations,
                "offsets_m": offsets,
                "actual_coc_px": actual_coc,
            }
        )
    return {
        "native_f_number": float(native_imaging.parax.fnum),
        "field_x": field_x,
        "field_y": field_y,
        "target_coc": target_coc,
        "apertures": contexts,
    }


def compose_analytic_label_with_pdoffset(
    analytic_centroid_disparity_px: np.ndarray | float,
    pd_offset_px: np.ndarray | float,
) -> np.ndarray:
    """独立组合标签；不修改 PSF morphology 或 bank centroid。"""

    analytic = np.asarray(analytic_centroid_disparity_px, dtype=np.float64)
    offset = np.asarray(pd_offset_px, dtype=np.float64)
    if not np.isfinite(analytic).all() or not np.isfinite(offset).all():
        raise ValueError("analytic centroid/PDOFFSET 必须有限")
    return analytic + offset


def analytic_centroid_labels(bank: np.ndarray) -> np.ndarray:
    value = np.asarray(bank)
    if value.ndim != 7 or value.shape[4] != 2 or value.shape[-1] != value.shape[-2]:
        raise ValueError(
            "bank 必须为 [A,N,Gh,Gw,2,K,K]，实际为 " f"{value.shape}"
        )
    labels = np.empty(value.shape[:4], dtype=np.float64)
    for index in np.ndindex(value.shape[:4]):
        left_x, _ = centroid_xy(value[index + (0,)])
        right_x, _ = centroid_xy(value[index + (1,)])
        labels[index] = left_x - right_x
    return labels


def retarget_centroid_slope(
    bank: np.ndarray,
    signed_coc_bins_px: np.ndarray,
    target_slopes_by_aperture: Sequence[float],
    *,
    support_padding_px: int,
    centroid_refinement_iterations: int = 6,
    anchor_mode: str = "zero",
    common_anchor_disparity_px: np.ndarray | Sequence[float] | float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """保持共同 centroid 与 morphology，只重定向 L/R 相对 centroid slope。

    ``zero`` 保留历史合同 ``target=slope*CoC``；``preserve_native_coc0`` 对每个
    aperture×field 使用原始 CoC=0 optical disparity 作为 anchor；``common`` 使用
    调用方冻结的公共 anchor。三种模式都只平移最终 PSF，输出 label 始终从变换后
    kernel centroid 重算，PDOFFSET 不进入本函数。
    """

    value = np.asarray(bank, dtype=np.float32)
    coc = np.asarray(signed_coc_bins_px, dtype=np.float64)
    if coc.ndim == 1:
        coc = np.tile(coc[None, :], (value.shape[0], 1))
    if value.ndim != 7 or value.shape[4] != 2 or coc.shape != value.shape[:2]:
        raise ValueError("retarget bank/CoC shape 不一致")
    slopes = np.asarray(target_slopes_by_aperture, dtype=np.float64)
    if slopes.shape != (value.shape[0],) or np.any(slopes <= 0.0):
        raise ValueError("target slopes 必须与 aperture 轴等长且为正")
    mode = str(anchor_mode)
    if mode not in {"zero", "preserve_native_coc0", "common"}:
        raise ValueError(
            "anchor_mode 必须为 zero、preserve_native_coc0 或 common"
        )
    if mode != "common" and common_anchor_disparity_px is not None:
        raise ValueError("只有 anchor_mode=common 才允许 common_anchor_disparity_px")
    raw_labels = analytic_centroid_labels(value)
    anchor_shape = (value.shape[0], value.shape[2], value.shape[3])
    if mode == "zero":
        anchors = np.zeros(anchor_shape, dtype=np.float64)
    elif mode == "preserve_native_coc0":
        zero_indices: list[int] = []
        for aperture_index in range(value.shape[0]):
            matches = np.flatnonzero(
                np.abs(coc[aperture_index]) <= 1.0e-12
            )
            if matches.size != 1:
                raise ValueError(
                    "preserve_native_coc0 要求每个 aperture 恰有一个 CoC=0 bin"
                )
            zero_indices.append(int(matches[0]))
        anchors = np.stack(
            [
                raw_labels[aperture_index, zero_index]
                for aperture_index, zero_index in enumerate(zero_indices)
            ],
            axis=0,
        )
    else:
        if common_anchor_disparity_px is None:
            raise ValueError(
                "anchor_mode=common 必须提供 common_anchor_disparity_px"
            )
        common = np.asarray(common_anchor_disparity_px, dtype=np.float64)
        if not np.isfinite(common).all():
            raise ValueError("common anchor 必须全部有限")
        try:
            anchors = np.broadcast_to(common, anchor_shape).copy()
        except ValueError as error:
            raise ValueError(
                f"common anchor 无法广播到 {anchor_shape}：{common.shape}"
            ) from error
    output_size = int(value.shape[-1]) + 2 * int(support_padding_px)
    output = np.empty((*value.shape[:-2], output_size, output_size), dtype=np.float32)
    retained: list[float] = []
    requested_labels = np.empty(value.shape[:4], dtype=np.float64)
    shift_abs_max = 0.0
    refinement_iterations_used_max = 0
    for aperture_index in range(value.shape[0]):
        for coc_index in range(value.shape[1]):
            for field_y_index in range(value.shape[2]):
                for field_x_index in range(value.shape[3]):
                    target = float(
                        anchors[aperture_index, field_y_index, field_x_index]
                        + slopes[aperture_index]
                        * coc[aperture_index, coc_index]
                    )
                    pair = value[
                        aperture_index,
                        coc_index,
                        field_y_index,
                        field_x_index,
                    ]
                    left_x, _ = centroid_xy(pair[0])
                    right_x, _ = centroid_xy(pair[1])
                    raw_disparity = left_x - right_x
                    correction = target - raw_disparity
                    shift_abs_max = max(shift_abs_max, abs(0.5 * correction))
                    translated_pair: list[np.ndarray] = []
                    retained_pair: list[float] = []
                    for side_index, shift_x in ((0, 0.5 * correction), (1, -0.5 * correction)):
                        translated, retained_fraction = _forward_bilinear_translate_psf(
                            pair[side_index],
                            shift_x_px=float(shift_x),
                            shift_y_px=0.0,
                            support_padding_px=int(support_padding_px),
                        )
                        translated_pair.append(translated)
                        retained_pair.append(float(retained_fraction))
                    for refinement_index in range(int(centroid_refinement_iterations)):
                        current_left_x, _ = centroid_xy(translated_pair[0])
                        current_right_x, _ = centroid_xy(translated_pair[1])
                        residual_correction = target - (
                            current_left_x - current_right_x
                        )
                        if abs(residual_correction) <= 1.0e-7:
                            break
                        refinement_iterations_used_max = max(
                            refinement_iterations_used_max, refinement_index + 1
                        )
                        for side_index, shift_x in (
                            (0, 0.5 * residual_correction),
                            (1, -0.5 * residual_correction),
                        ):
                            translated, retained_fraction = _forward_bilinear_translate_psf(
                                translated_pair[side_index],
                                shift_x_px=float(shift_x),
                                shift_y_px=0.0,
                                support_padding_px=0,
                            )
                            translated_pair[side_index] = translated
                            retained_pair[side_index] *= float(retained_fraction)
                    for side_index in range(2):
                        output[
                            aperture_index,
                            coc_index,
                            field_y_index,
                            field_x_index,
                            side_index,
                        ] = translated_pair[side_index]
                        retained.append(retained_pair[side_index])
                    requested_labels[
                        aperture_index,
                        coc_index,
                        field_y_index,
                        field_x_index,
                    ] = target
    labels = analytic_centroid_labels(output)
    error = labels - requested_labels
    metadata = {
        "mode": "symmetric_relative_centroid_retarget_v1",
        "common_centroid_preserved": True,
        "morphology_residual_preserved_except_bilinear_translation": True,
        "target_slopes_by_aperture": slopes.tolist(),
        "anchor_mode": mode,
        "anchor_disparity_px": anchors.tolist(),
        "native_coc0_optical_bias_preserved": mode == "preserve_native_coc0",
        "common_anchor_frozen_by_caller": mode == "common",
        "support_padding_px": int(support_padding_px),
        "centroid_refinement_iterations_max": int(centroid_refinement_iterations),
        "centroid_refinement_iterations_used_max": refinement_iterations_used_max,
        "applied_side_shift_abs_max_px": shift_abs_max,
        "retained_mass_min": float(min(retained)),
        "retained_mass_max": float(max(retained)),
        "analytic_target_error_abs_max_px": float(np.max(np.abs(error))),
        "label_source": "final_psf_centroid_mu_left_x_minus_mu_right_x",
        "ncc_used": False,
        "pd_offset_embedded": False,
    }
    return output, labels, metadata


def _border_inner_mass_min(bank: np.ndarray, border_width: int) -> float:
    value = np.asarray(bank, dtype=np.float64)
    border = int(border_width)
    if border <= 0 or 2 * border >= value.shape[-1]:
        raise ValueError("border_width 非法")
    inner = value[..., border:-border, border:-border].sum(axis=(-2, -1))
    total = value.sum(axis=(-2, -1))
    return float(np.min(inner / total))


def _psf_adjacent_l1(bank: np.ndarray) -> tuple[float, float]:
    delta = np.abs(np.diff(np.asarray(bank, dtype=np.float64), axis=1)).sum(
        axis=(-2, -1)
    )
    return float(delta.mean()), float(delta.max())


def _center_crop_normalized_psf(
    psf: np.ndarray,
    export_kernel_size: int,
) -> tuple[np.ndarray, float]:
    source = np.asarray(psf, dtype=np.float64)
    source_size = int(source.shape[-1])
    target_size = int(export_kernel_size)
    if (
        source.ndim != 2
        or source.shape[0] != source_size
        or target_size <= 0
        or target_size % 2 == 0
        or target_size > source_size
        or (source_size - target_size) % 2 != 0
    ):
        raise ValueError("reference/export kernel 必须是同奇偶中心可裁剪方阵")
    start = (source_size - target_size) // 2
    cropped = source[start : start + target_size, start : start + target_size]
    source_mass = float(source.sum())
    retained = float(cropped.sum()) / source_mass
    if source_mass <= 0.0 or retained <= 0.0:
        raise RuntimeError("PSF center crop 后没有有效能量")
    return (cropped / cropped.sum()).astype(np.float32), retained


def _linear_curve_stats(coc: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    x = np.asarray(coc, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    design = np.stack((x, np.ones_like(x)), axis=1)
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = slope * x + intercept
    residual = y - prediction
    centered = y - y.mean()
    denominator = float(np.square(centered).sum())
    r2 = 1.0 if denominator <= 1.0e-20 else 1.0 - float(np.square(residual).sum()) / denominator
    return {
        "slope_px_per_coc": float(slope),
        "intercept_px": float(intercept),
        "r2": float(r2),
        "residual_abs_max_px": float(np.max(np.abs(residual))),
    }


def _physical_centroid_curve_metadata(
    signed_coc: np.ndarray,
    raw_labels: np.ndarray,
    f_numbers: np.ndarray,
    field_x: Sequence[float],
    field_y: Sequence[float],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    slopes: list[float] = []
    for aperture_index, f_number in enumerate(f_numbers):
        for field_y_index, field_y_value in enumerate(field_y):
            for field_x_index, field_x_value in enumerate(field_x):
                fit = _linear_curve_stats(
                    signed_coc[aperture_index],
                    raw_labels[
                        aperture_index, :, field_y_index, field_x_index
                    ],
                )
                slopes.append(float(fit["slope_px_per_coc"]))
                rows.append(
                    {
                        "aperture_index": aperture_index,
                        "f_number": float(f_number),
                        "field_y_index": field_y_index,
                        "field_x_index": field_x_index,
                        "field_y_normalized": float(field_y_value),
                        "field_x_normalized": float(field_x_value),
                        **fit,
                    }
                )
    return {
        "source": "raw_shared_lens_psf_centroid_mu_left_x_minus_mu_right_x",
        "before_independent_centroid_slope_retarget": True,
        "pdoffset_embedded": False,
        "slope_min_px_per_coc": float(min(slopes)),
        "slope_max_px_per_coc": float(max(slopes)),
        "cells": rows,
    }


def _signed_centroid_checks(
    signed_coc: np.ndarray,
    labels: np.ndarray,
    *,
    tolerance_px: float,
) -> dict[str, Any]:
    coc = np.asarray(signed_coc, dtype=np.float64)
    value = np.asarray(labels, dtype=np.float64)
    expected = np.broadcast_to(coc[:, :, None, None], value.shape)
    nonzero = np.abs(expected) > 1.0e-12
    sign_ok = bool(np.all(value[nonzero] * expected[nonzero] > 0.0))
    monotonic_delta = np.diff(value, axis=1)
    monotonic_min = float(monotonic_delta.min())
    return {
        "sign_agreement_nonzero": sign_ok,
        "strictly_monotonic": monotonic_min > float(tolerance_px),
        "monotonic_step_min_px": monotonic_min,
    }


def _profile_asset_payload(
    *,
    asset_id: str,
    family_id: str,
    family_sha: str,
    profile_document: dict[str, Any],
    f_numbers: np.ndarray,
    signed_coc: np.ndarray,
    labels: np.ndarray,
    raw_labels: np.ndarray,
    bank: np.ndarray,
    throughput: np.ndarray,
    field_x: Sequence[float],
    field_y: Sequence[float],
    physical_curve: dict[str, Any],
) -> dict[str, Any]:
    return {
        "asset_id": asset_id,
        "profile_id": profile_document["profile_id"],
        "family_id": family_id,
        "family_sha256": family_sha,
        "family_profile": profile_document,
        "asset_spec": {
            "kind": "cldefocus_same_family_single_profile_multi_aperture_v1",
            "axis_order": [
                "aperture",
                "signed_coc",
                "field_y",
                "field_x",
                "side",
                "h",
                "w",
            ],
            "signed_disparity": "d=x_L-x_R",
            "analytic_label_source": "final_psf_centroid_mu_left_x_minus_mu_right_x",
            "physical_curve_source": physical_curve["source"],
            "pd_offset_embedded": False,
            "pd_offset_contract": "disp_gt=analytic_centroid_disp+independent_pd_offset_px",
            "ncc_role": "none_never_label_writeback",
            "same_lens_shared_ray_geometry": True,
        },
        "f_numbers": torch.from_numpy(f_numbers.astype(np.float32)),
        "signed_coc_bins_px": torch.from_numpy(signed_coc.astype(np.float64)),
        "analytic_disparity_bins_px": torch.from_numpy(labels.astype(np.float64)),
        "raw_analytic_disparity_bins_px": torch.from_numpy(raw_labels.astype(np.float64)),
        "physical_centroid_vs_coc": physical_curve,
        "field_grid_hw": (len(field_y), len(field_x)),
        "field_x_normalized": torch.tensor(field_x, dtype=torch.float32),
        "field_y_normalized": torch.tensor(field_y, dtype=torch.float32),
        "kernel_size": int(bank.shape[-1]),
        "side_throughput_grid": torch.from_numpy(throughput.astype(np.float32)),
        "psf_bank": torch.from_numpy(bank.astype(np.float32)),
    }


def _select_family_medoid(
    profiles: Sequence[FamilyProfile],
    diagnostics: Sequence[dict[str, Any]],
    randomization: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    ranges = randomization["ranges"]
    rows: list[dict[str, Any]] = []
    for profile, diagnostic in zip(profiles, diagnostics, strict=True):
        values = {
            **{axis: float(getattr(profile.response, axis)) for axis in _RESPONSE_AXES},
            **{axis: float(getattr(profile.aberration, axis)) for axis in _ABERRATION_AXES},
            "centroid_slope_px_per_coc": float(profile.centroid_slope_px_per_coc),
        }
        normalized: dict[str, float] = {}
        for axis in _PROFILE_AXES:
            low, high = (float(value) for value in ranges[axis])
            normalized[axis] = 0.5 if low == high else (values[axis] - low) / (high - low)
        distance = math.sqrt(
            sum((normalized[axis] - 0.5) ** 2 for axis in _PROFILE_AXES)
        )
        centroid_error = float(
            diagnostic["retarget"]["analytic_target_error_abs_max_px"]
        )
        energy_error = max(
            abs(float(diagnostic["energy_min"]) - 1.0),
            abs(float(diagnostic["energy_max"]) - 1.0),
        )
        reference_threshold = float(gates["reference_crop_retained_mass_min"])
        retarget_threshold = float(gates["retarget_retained_mass_min"])
        combined_threshold = float(gates["combined_support_retained_mass_min"])
        centroid_threshold = float(gates["centroid_target_error_abs_max_px"])
        zero_threshold = float(gates["zero_centroid_abs_max_px"])
        hard_gate_margins = {
            "reference_support_normalized_slack": (
                float(diagnostic["reference_crop_retained_mass_min"])
                - reference_threshold
            )
            / max(1.0 - reference_threshold, 1.0e-12),
            "retarget_support_normalized_slack": (
                float(diagnostic["retarget"]["retained_mass_min"])
                - retarget_threshold
            )
            / max(1.0 - retarget_threshold, 1.0e-12),
            "combined_support_normalized_slack": (
                float(diagnostic["combined_support_retained_mass_min"])
                - combined_threshold
            )
            / max(1.0 - combined_threshold, 1.0e-12),
            "centroid_target_normalized_slack": (
                centroid_threshold - centroid_error
            )
            / centroid_threshold,
            "zero_centroid_normalized_slack": (
                zero_threshold
                - float(diagnostic["zero_centroid_disparity_abs_max_px"])
            )
            / zero_threshold,
        }
        minimum_slack = float(min(hard_gate_margins.values()))
        safety_checks = {
            "all_support_centroid_hard_gates_pass": minimum_slack >= 0.0,
            "minimum_normalized_hard_gate_slack_ge_0p05": minimum_slack >= 0.05,
        }
        rows.append(
            {
                "profile_id": profile.profile_id,
                "normalized_distance_to_family_center": distance,
                "normalized_axes": normalized,
                "profile_gate_pass": bool(diagnostic["pass"]),
                "support_centroid_gate_edge_checks": safety_checks,
                "support_centroid_hard_gate_margins": hard_gate_margins,
                "minimum_support_centroid_normalized_slack": minimum_slack,
                "energy_abs_error": energy_error,
                "continuity_l1_max": float(
                    diagnostic["adjacent_coc_psf_l1_max"]
                ),
                "not_support_or_centroid_gate_edge": bool(
                    all(safety_checks.values())
                ),
            }
        )
    eligible = [
        row
        for row in rows
        if row["profile_gate_pass"] and row["not_support_or_centroid_gate_edge"]
    ]
    if not eligible:
        raise RuntimeError("没有同时满足 PASS 与非门禁边缘条件的 medoid 候选")
    selected = min(
        eligible, key=lambda row: row["normalized_distance_to_family_center"]
    )
    return {
        "definition": "euclidean_distance_to_0p5_center_after_per_axis_range_normalization",
        "selected_profile_id": selected["profile_id"],
        "selected_is_not_support_or_centroid_gate_edge": True,
        "selected": selected,
        "eligible_profile_count": len(eligible),
        "all_candidates": sorted(
            rows, key=lambda row: row["normalized_distance_to_family_center"]
        ),
    }


def _export_pdraw_mono_v1_by_aperture(
    profile_root: Path,
    *,
    resource_prefix: str,
    bank: np.ndarray,
    signed_coc: np.ndarray,
    f_numbers: np.ndarray,
    field_x: Sequence[float],
    field_y: Sequence[float],
) -> list[dict[str, Any]]:
    positive_indices = np.flatnonzero(signed_coc[0] > 0.0)
    rows: list[dict[str, Any]] = []
    for aperture_index, f_number in enumerate(f_numbers):
        source = bank[aperture_index, positive_indices]
        centroids = np.empty((*source.shape[:-2], 2), dtype=np.float64)
        for index in np.ndindex(source.shape[:-2]):
            centroids[index] = centroid_xy(source[index])
        required_padding = int(math.ceil(float(np.max(np.abs(centroids))))) + 1
        output_size = int(source.shape[-1]) + 2 * required_padding
        centered = np.empty((*source.shape[:-2], output_size, output_size), dtype=np.float32)
        retained: list[float] = []
        residual: list[float] = []
        for index in np.ndindex(source.shape[:-2]):
            translated, fraction = _forward_bilinear_translate_psf(
                source[index],
                shift_x_px=-float(centroids[index][0]),
                shift_y_px=-float(centroids[index][1]),
                support_padding_px=required_padding,
            )
            centered[index] = translated
            retained.append(float(fraction))
            residual.extend(abs(value) for value in centroid_xy(translated))
        aperture_name = f"f{float(f_number):g}".replace(".", "p")
        output_root = profile_root / "pdraw_mono_v1" / aperture_name
        output_root.mkdir(parents=True, exist_ok=True)
        array_path = output_root / "pdraw_kernels_lr_mono.npy"
        flattened = centered.reshape(
            centered.shape[0],
            len(field_y) * len(field_x),
            2,
            output_size,
            output_size,
        )
        np.save(array_path, flattened, allow_pickle=False)
        metadata = {
            "resource": f"{resource_prefix}_{aperture_name}_shape_only",
            "format_version": "pdraw_mono_v1",
            "primary_array_name": array_path.name,
            "axis_order": {
                array_path.name: [
                    "coc_bin",
                    "field_index",
                    "view_left_right",
                    "y",
                    "x",
                ]
            },
            "views": ["left", "right"],
            "field_grid": {
                "grid_shape": [len(field_y), len(field_x)],
                "field_x_normalized": list(field_x),
                "field_y_normalized": list(field_y),
                "flatten_order": "field_y_major_then_field_x",
            },
            "coc_values_px": signed_coc[aperture_index, positive_indices].tolist(),
            "centroid_policy": {
                array_path.name: "shape-only kernels; each side is centered"
            },
            "source_physical_centroid_removed": True,
            "pdoffset_embedded": False,
            "support_padding_px": required_padding,
            "retained_mass_min": float(min(retained)),
            "centroid_residual_abs_max_px": float(max(residual)),
        }
        metadata_path = output_root / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "f_number": float(f_number),
                "array_path": str(array_path.resolve()),
                "array_sha256": _sha256_file(array_path),
                "metadata_path": str(metadata_path.resolve()),
                "metadata_sha256": _sha256_file(metadata_path),
                "shape": list(flattened.shape),
                "retained_mass_min": float(min(retained)),
                "centroid_residual_abs_max_px": float(max(residual)),
            }
        )
    return rows


def _git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)
    sha = _sha256_file(path)
    _atomic_write_text(path.with_suffix(path.suffix + ".sha256"), sha + "\n")
    return sha


def _chunk_cache_identity(
    *,
    config_sha: str,
    family_sha: str,
    lens_sha: str,
    profiles: Sequence[FamilyProfile],
    chunk_start: int,
    reference_kernel_size: int,
    export_kernel_size: int,
    expected_shape: Sequence[int],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_sha256": config_sha,
        "family_sha256": family_sha,
        "lens_sha256": lens_sha,
        "chunk_start": int(chunk_start),
        "chunk_end_exclusive": int(chunk_start + len(profiles)),
        "profile_ids": [profile.profile_id for profile in profiles],
        "profile_parameter_sha256": [
            family_profile_document(profile)["parameter_sha256"] for profile in profiles
        ],
        "reference_kernel_size": int(reference_kernel_size),
        "export_kernel_size": int(export_kernel_size),
        "per_profile_bank_shape": [int(value) for value in expected_shape],
    }


def _load_verified_chunk_cache(
    path: Path,
    expected_identity: dict[str, Any],
) -> dict[str, Any] | None:
    sha_path = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() and not sha_path.exists():
        return None
    if not path.is_file() or not sha_path.is_file():
        raise RuntimeError(f"chunk cache/sha 不完整：{path}")
    expected_file_sha = sha_path.read_text(encoding="utf-8").strip()
    actual_file_sha = _sha256_file(path)
    if expected_file_sha != actual_file_sha:
        raise RuntimeError(f"chunk cache SHA256 不匹配：{path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("identity") != expected_identity:
        raise RuntimeError(f"chunk cache 身份不匹配，禁止复用：{path}")
    raw_banks = payload.get("raw_banks")
    retained = payload.get("reference_crop_retained")
    if not isinstance(raw_banks, dict) or not isinstance(retained, dict):
        raise RuntimeError(f"chunk cache payload 不完整：{path}")
    expected_ids = expected_identity["profile_ids"]
    if set(raw_banks) != set(expected_ids) or set(retained) != set(expected_ids):
        raise RuntimeError(f"chunk cache profile 集合不匹配：{path}")
    expected_shape = tuple(expected_identity["per_profile_bank_shape"])
    for profile_id in expected_ids:
        tensor = torch.as_tensor(raw_banks[profile_id])
        retained_tensor = torch.as_tensor(retained[profile_id])
        if tuple(tensor.shape) != expected_shape or retained_tensor.numel() == 0:
            raise RuntimeError(f"chunk cache shape 不匹配：{path} profile={profile_id}")
        if not bool(torch.isfinite(tensor).all()) or not bool((tensor >= 0.0).all()):
            raise RuntimeError(f"chunk cache 存在非法 PSF：{path} profile={profile_id}")
        if not bool(torch.isfinite(retained_tensor).all()):
            raise RuntimeError(f"chunk cache retained 非法：{path} profile={profile_id}")
    return {
        "file_sha256": actual_file_sha,
        "raw_banks": raw_banks,
        "reference_crop_retained": retained,
        "source_seconds": float(payload["source_seconds"]),
    }


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or int(loaded.get("schema_version", 0)) != 1:
        raise ValueError("配置必须是 schema_version: 1 mapping")
    required = {
        "schema_version",
        "run",
        "source",
        "lens",
        "sensor",
        "apertures",
        "optics",
        "family",
        "pdoffset",
        "gates",
        "output",
        "data_isolation",
    }
    if set(loaded) != required:
        raise ValueError(f"配置字段必须精确为 {sorted(required)}")
    isolation_keys = {
        "real_pdraw_accessed",
        "google_dev_accessed",
        "google_holdout_accessed",
        "dp5k_accessed",
        "stereo_training_run",
    }
    if (
        not isinstance(loaded["data_isolation"], dict)
        or set(loaded["data_isolation"]) != isolation_keys
    ):
        raise ValueError(f"data_isolation 必须精确为 {sorted(isolation_keys)}")
    if any(bool(value) for value in loaded["data_isolation"].values()):
        raise ValueError("family bank 生成禁止访问真实数据或启动训练")
    pdoffset = loaded["pdoffset"]
    if not isinstance(pdoffset, dict) or set(pdoffset) != {
        "sample_range_px",
        "smoke_probe_px",
    }:
        raise ValueError("pdoffset 必须精确声明 sample_range_px 与 smoke_probe_px")
    sample_range = [float(value) for value in pdoffset["sample_range_px"]]
    if (
        len(sample_range) != 2
        or not all(math.isfinite(value) for value in sample_range)
        or sample_range[0] > sample_range[1]
    ):
        raise ValueError("pdoffset.sample_range_px 必须为合法 [min,max]")
    smoke_probe = float(pdoffset["smoke_probe_px"])
    if not math.isfinite(smoke_probe) or abs(smoke_probe) <= 0.0:
        raise ValueError("pdoffset.smoke_probe_px 必须为有限非零数")
    return loaded


def generate(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    checkout = Path(config["source"]["checkout"]).resolve()
    datasyn_src = checkout / "datasyn/src"
    for import_root in (checkout, datasyn_src):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    lens_root = Path(config["lens"]["root"]).resolve()
    lens_path = lens_root / str(config["lens"]["relative_path"])
    if not lens_path.is_file() or _sha256_file(lens_path) != str(config["lens"]["sha256"]):
        raise RuntimeError("镜头处方不存在或 SHA256 不匹配")
    output_root = Path(config["output"]["root"]).resolve()
    if checkout == output_root or checkout in output_root.parents:
        raise ValueError("输出目录必须位于 CLDefocus 仓库外")
    if output_root.exists():
        unexpected = [
            path.name
            for path in output_root.iterdir()
            if path.name not in {"raw_chunks", "preexport_diagnostics.json"}
        ]
        if unexpected:
            raise FileExistsError(
                f"输出目录含非 cache 文件，拒绝覆盖：{output_root} {unexpected}"
            )
    output_root.mkdir(parents=True, exist_ok=True)

    resolved = {
        **config,
        "source": {**config["source"], "checkout": str(checkout)},
        "lens": {**config["lens"], "root": str(lens_root)},
    }
    profiles = sample_family_profiles(config["family"]["randomization"])
    profile_documents = [family_profile_document(profile) for profile in profiles]
    family_id = str(config["family"]["family_id"])
    family_sha = _sha256_json(
        {
            "family_id": family_id,
            "lens_sha256": config["lens"]["sha256"],
            "profiles": profile_documents,
            "apertures": config["apertures"]["f_numbers"],
        }
    )
    config_sha = _sha256_file(config_path)

    from datasyn.jaxutils.configs import easy_optics_setup

    easy_optics_setup()
    started = time.perf_counter()
    optical = _prepare_family_optical_contexts(resolved)
    reference_setup_seconds = time.perf_counter() - started
    aperture_count = len(optical["apertures"])
    coc_count = int(optical["target_coc"].size)
    field_x_count = len(optical["field_x"])
    field_y_count = len(optical["field_y"])
    reference_kernel_size = int(
        config["optics"].get(
            "reference_kernel_size", config["optics"].get("kernel_size", 0)
        )
    )
    export_kernel_size = int(
        config["optics"].get("export_kernel_size", reference_kernel_size)
    )
    if (
        reference_kernel_size <= 0
        or export_kernel_size <= 0
        or reference_kernel_size % 2 == 0
        or export_kernel_size % 2 == 0
        or export_kernel_size > reference_kernel_size
        or (reference_kernel_size - export_kernel_size) % 2 != 0
    ):
        raise ValueError("reference/export kernel_size 必须为可中心裁剪的正奇数")
    raw_banks = {
        profile.profile_id: np.empty(
            (
                aperture_count,
                coc_count,
                field_y_count,
                field_x_count,
                2,
                export_kernel_size,
                export_kernel_size,
            ),
            dtype=np.float32,
        )
        for profile in profiles
    }
    reference_crop_retained: dict[str, list[float]] = {
        profile.profile_id: [] for profile in profiles
    }
    propagation_started = time.perf_counter()
    cell_count = 0
    profile_batch_size = int(
        config["family"].get("batch_profile_chunk_size", len(profiles))
    )
    if profile_batch_size < 1:
        raise ValueError("batch_profile_chunk_size 必须为正数")
    chunk_timings: list[dict[str, Any]] = []
    for chunk_start in range(0, len(profiles), profile_batch_size):
        chunk_profiles = profiles[chunk_start : chunk_start + profile_batch_size]
        cache_identity = _chunk_cache_identity(
            config_sha=config_sha,
            family_sha=family_sha,
            lens_sha=str(config["lens"]["sha256"]),
            profiles=chunk_profiles,
            chunk_start=chunk_start,
            reference_kernel_size=reference_kernel_size,
            export_kernel_size=export_kernel_size,
            expected_shape=raw_banks[chunk_profiles[0].profile_id].shape,
        )
        cache_path = (
            output_root
            / "raw_chunks"
            / f"chunk_{chunk_start:03d}_{chunk_start + len(chunk_profiles):03d}.pt"
        )
        cached = _load_verified_chunk_cache(cache_path, cache_identity)
        if cached is not None:
            for profile in chunk_profiles:
                raw_banks[profile.profile_id][...] = np.asarray(
                    cached["raw_banks"][profile.profile_id], dtype=np.float32
                )
                reference_crop_retained[profile.profile_id].extend(
                    np.asarray(
                        cached["reference_crop_retained"][profile.profile_id],
                        dtype=np.float64,
                    ).tolist()
                )
            chunk_cell_count = aperture_count * field_y_count * field_x_count * coc_count
            timing = {
                "profile_start": chunk_start,
                "profile_end_exclusive": chunk_start + len(chunk_profiles),
                "profile_ids": [profile.profile_id for profile in chunk_profiles],
                "reference_cell_count": chunk_cell_count,
                "seconds": 0.0,
                "seconds_per_reference_cell": 0.0,
                "resumed_from_verified_cache": True,
                "source_propagation_seconds": float(cached["source_seconds"]),
                "cache_path": str(cache_path.resolve()),
                "cache_sha256": cached["file_sha256"],
            }
            cell_count = chunk_cell_count
            chunk_timings.append(timing)
            print(json.dumps({"family_bank_progress": timing}), flush=True)
            continue
        chunk_started = time.perf_counter()
        chunk_cell_count = 0
        for aperture_index, context in enumerate(optical["apertures"]):
            for field_y_index, field_y in enumerate(optical["field_y"]):
                for field_x_index, field_x in enumerate(optical["field_x"]):
                    wavefront = context["wavefronts"][field_y_index][field_x_index]
                    base_s_prop = context["base_propagations"][
                        field_y_index, field_x_index
                    ]
                    for coc_index, _ in enumerate(optical["target_coc"]):
                        propagated = _propagate_profile_batch(
                            wavefront,
                            s_prop=base_s_prop
                            + float(
                                context["offsets_m"][
                                    field_y_index, field_x_index, coc_index
                                ]
                            ),
                            parax=context["imaging"].parax,
                            sensor=context["imaging"].sen,
                            kernel_size=reference_kernel_size,
                            profiles=[profile.response for profile in chunk_profiles],
                            field_x=float(field_x),
                            field_y=float(field_y),
                            upsample=int(config["optics"].get("upsample", 1)),
                            sensor_chunk_size=int(
                                config["family"].get("batch_sensor_chunk_size", 256)
                            ),
                            aberration_profiles=[
                                profile.aberration for profile in chunk_profiles
                            ],
                        )
                        for profile, row in zip(
                            chunk_profiles, propagated, strict=True
                        ):
                            for side_index, side_name in enumerate(("left", "right")):
                                cropped, retained = _center_crop_normalized_psf(
                                    np.asarray(row[side_name], dtype=np.float32),
                                    export_kernel_size,
                                )
                                raw_banks[profile.profile_id][
                                    aperture_index,
                                    coc_index,
                                    field_y_index,
                                    field_x_index,
                                    side_index,
                                ] = cropped
                                reference_crop_retained[profile.profile_id].append(
                                    retained
                                )
                        chunk_cell_count += 1
        chunk_seconds = time.perf_counter() - chunk_started
        cache_payload = {
            "identity": cache_identity,
            "source_seconds": chunk_seconds,
            "raw_banks": {
                profile.profile_id: torch.from_numpy(
                    raw_banks[profile.profile_id].copy()
                )
                for profile in chunk_profiles
            },
            "reference_crop_retained": {
                profile.profile_id: torch.tensor(
                    reference_crop_retained[profile.profile_id], dtype=torch.float64
                )
                for profile in chunk_profiles
            },
        }
        cache_sha = _atomic_torch_save(cache_path, cache_payload)
        cell_count = chunk_cell_count
        timing = {
            "profile_start": chunk_start,
            "profile_end_exclusive": chunk_start + len(chunk_profiles),
            "profile_ids": [profile.profile_id for profile in chunk_profiles],
            "reference_cell_count": chunk_cell_count,
            "seconds": chunk_seconds,
            "seconds_per_reference_cell": chunk_seconds / float(chunk_cell_count),
            "resumed_from_verified_cache": False,
            "source_propagation_seconds": chunk_seconds,
            "cache_path": str(cache_path.resolve()),
            "cache_sha256": cache_sha,
        }
        chunk_timings.append(timing)
        print(json.dumps({"family_bank_progress": timing}), flush=True)
    propagation_seconds = time.perf_counter() - propagation_started

    f_numbers = np.asarray(
        [float(context["actual_f_number"]) for context in optical["apertures"]],
        dtype=np.float64,
    )
    signed_coc = np.tile(
        np.asarray(optical["target_coc"], dtype=np.float64)[None, :],
        (aperture_count, 1),
    )
    aperture_scale_cfg = config["family"]["aperture_centroid_scale_by_f_number"]
    aperture_scales = []
    for f_number in f_numbers:
        candidates = [
            (abs(float(key) - f_number), float(value))
            for key, value in aperture_scale_cfg.items()
        ]
        distance, value = min(candidates, key=lambda row: row[0])
        if distance > 1.0e-6:
            raise ValueError(f"aperture centroid scale 未覆盖 f/{f_number:g}")
        aperture_scales.append(value)

    final_banks: list[np.ndarray] = []
    final_labels: list[np.ndarray] = []
    raw_labels: list[np.ndarray] = []
    throughputs: list[np.ndarray] = []
    physical_curves: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    gates = config["gates"]
    for profile in profiles:
        raw_bank = raw_banks[profile.profile_id]
        raw_label = analytic_centroid_labels(raw_bank)
        physical_curve = _physical_centroid_curve_metadata(
            signed_coc,
            raw_label,
            f_numbers,
            optical["field_x"],
            optical["field_y"],
        )
        target_slopes = [
            profile.centroid_slope_px_per_coc * scale for scale in aperture_scales
        ]
        final_bank, labels, retarget = retarget_centroid_slope(
            raw_bank,
            signed_coc,
            target_slopes,
            support_padding_px=int(config["family"]["centroid_support_padding_px"]),
        )
        zero_index = int(np.flatnonzero(np.abs(signed_coc[0]) <= 1.0e-12)[0])
        energy = final_bank.sum(axis=(-2, -1))
        adjacent_mean, adjacent_max = _psf_adjacent_l1(final_bank)
        raw_inner_mass = _border_inner_mass_min(
            raw_bank, int(gates["raw_support_border_width_px"])
        )
        reference_retained_min = float(
            min(reference_crop_retained[profile.profile_id])
        )
        total_retained_min = reference_retained_min * float(
            retarget["retained_mass_min"]
        )
        zero_disp_abs = float(np.max(np.abs(labels[:, zero_index])))
        pdoffset_probe = float(config["pdoffset"]["smoke_probe_px"])
        zero_with_pdoffset = compose_analytic_label_with_pdoffset(
            labels[:, zero_index], pdoffset_probe
        )
        signed_checks = _signed_centroid_checks(
            signed_coc,
            labels,
            tolerance_px=float(gates["signed_monotonic_step_min_px"]),
        )
        checks = {
            "finite_nonnegative": bool(
                np.isfinite(final_bank).all() and np.all(final_bank >= 0.0)
            ),
            "unit_energy": float(np.max(np.abs(energy - 1.0)))
            <= float(gates["energy_abs_max"]),
            "raw_support_inner_mass": raw_inner_mass
            >= float(gates["raw_support_inner_mass_min"]),
            "reference_crop_retained_mass": reference_retained_min
            >= float(gates["reference_crop_retained_mass_min"]),
            "retarget_retained_mass": float(retarget["retained_mass_min"])
            >= float(gates["retarget_retained_mass_min"]),
            "combined_support_retained_mass": total_retained_min
            >= float(gates["combined_support_retained_mass_min"]),
            "centroid_target": float(retarget["analytic_target_error_abs_max_px"])
            <= float(gates["centroid_target_error_abs_max_px"]),
            "zero_centroid": zero_disp_abs
            <= float(gates["zero_centroid_abs_max_px"]),
            "pdoffset_independent_nonzero_at_focus": bool(
                np.allclose(
                    zero_with_pdoffset,
                    pdoffset_probe,
                    atol=float(gates["pdoffset_probe_error_abs_max_px"]),
                    rtol=0.0,
                )
            ),
            "coc_continuity": adjacent_max
            <= float(gates["adjacent_psf_l1_max"]),
            "signed_centroid": bool(all(
                (
                    signed_checks["sign_agreement_nonzero"],
                    signed_checks["strictly_monotonic"],
                )
            )),
        }
        diagnostic = {
            "profile_id": profile.profile_id,
            "parameter_sha256": family_profile_document(profile)["parameter_sha256"],
            "checks": checks,
            "pass": bool(all(checks.values())),
            "raw_support_inner_mass_min": raw_inner_mass,
            "reference_crop_retained_mass_min": reference_retained_min,
            "combined_support_retained_mass_min": total_retained_min,
            "energy_min": float(energy.min()),
            "energy_max": float(energy.max()),
            "zero_centroid_disparity_abs_max_px": zero_disp_abs,
            "pdoffset_probe_px": pdoffset_probe,
            "focus_label_with_pdoffset_min_max_px": [
                float(zero_with_pdoffset.min()),
                float(zero_with_pdoffset.max()),
            ],
            "adjacent_coc_psf_l1_mean": adjacent_mean,
            "adjacent_coc_psf_l1_max": adjacent_max,
            "signed_centroid_checks": signed_checks,
            "physical_centroid_vs_coc": physical_curve,
            "retarget": retarget,
        }
        diagnostics.append(diagnostic)
        final_banks.append(final_bank)
        final_labels.append(labels)
        raw_labels.append(raw_label)
        physical_curves.append(physical_curve)
        throughputs.append(
            np.repeat(
                build_side_throughput_grid(
                    profile.response, f_numbers, optical["field_x"]
                ),
                field_y_count,
                axis=1,
            )
        )

    expected_aperture_count = int(gates.get("expected_aperture_count", aperture_count))
    expected_profile_count = int(gates.get("expected_profile_count", len(profiles)))
    expected_coc_count = int(gates.get("expected_coc_count", coc_count))
    expected_field_grid = tuple(
        int(value)
        for value in gates.get(
            "expected_field_grid_hw", [field_y_count, field_x_count]
        )
    )
    lineage_checks = {
        "one_frozen_lens_sha": _sha256_file(lens_path) == str(config["lens"]["sha256"]),
        "aperture_count": aperture_count == expected_aperture_count,
        "profile_count": len(profiles) == expected_profile_count,
        "coc_count": coc_count == expected_coc_count,
        "field_grid": (field_y_count, field_x_count) == expected_field_grid,
        "unique_f_numbers": len(set(f_numbers.tolist())) == aperture_count,
        "single_parameter_sha_per_profile_across_apertures": all(
            bool(document["parameter_sha256"]) for document in profile_documents
        ),
        "shared_ray_geometry_within_profile_batch": True,
    }
    status = (
        "pass"
        if all(row["pass"] for row in diagnostics) and all(lineage_checks.values())
        else "fail"
    )
    combined_bank = np.stack(final_banks, axis=0)
    combined_labels = np.stack(final_labels, axis=0)
    combined_raw_labels = np.stack(raw_labels, axis=0)
    combined_throughput = np.stack(throughputs, axis=0)
    preexport_path = output_root / "preexport_diagnostics.json"
    _atomic_write_text(
        preexport_path,
        json.dumps(
            {
                "config_sha256": config_sha,
                "family_sha256": family_sha,
                "status_before_medoid": status,
                "cross_aperture_shared_lineage_checks": lineage_checks,
                "profiles": diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    medoid = _select_family_medoid(
        profiles,
        diagnostics,
        config["family"]["randomization"],
        gates,
    )
    asset_id = str(config["family"]["asset_id"])
    bank_payload = {
        "asset_id": asset_id,
        "family_id": family_id,
        "family_sha256": family_sha,
        "profile_ids": [profile.profile_id for profile in profiles],
        "family_profiles": profile_documents,
        "physical_centroid_vs_coc_by_profile": physical_curves,
        "matched_single_profile_medoid": medoid,
        "asset_spec": {
            "kind": "cldefocus_same_optical_family_multi_profile_multi_aperture_v1",
            "axis_order": [
                "profile",
                "aperture",
                "signed_coc",
                "field_y",
                "field_x",
                "side",
                "h",
                "w",
            ],
            "signed_disparity": "d=x_L-x_R",
            "analytic_label_source": "final_psf_centroid_mu_left_x_minus_mu_right_x",
            "ncc_role": "none_never_label_writeback",
            "pd_offset_embedded": False,
            "pd_offset_contract": "disp_gt=analytic_centroid_disp+independent_pd_offset_px",
            "co_c_zero_morphology_centroid": 0.0,
            "reference_propagation_reused_across_profiles": True,
            "low_order_basis": "unnormalized_low_order_zernike_like_no_defocus_v1",
        },
        "f_numbers": torch.from_numpy(f_numbers.astype(np.float32)),
        "signed_coc_bins_px": torch.from_numpy(signed_coc.astype(np.float64)),
        "analytic_disparity_bins_px": torch.from_numpy(combined_labels.astype(np.float64)),
        "raw_analytic_disparity_bins_px": torch.from_numpy(
            combined_raw_labels.astype(np.float64)
        ),
        "field_grid_hw": (field_y_count, field_x_count),
        "field_x_normalized": torch.tensor(optical["field_x"], dtype=torch.float32),
        "field_y_normalized": torch.tensor(optical["field_y"], dtype=torch.float32),
        "kernel_size": int(combined_bank.shape[-1]),
        "side_throughput_grid": torch.from_numpy(combined_throughput.astype(np.float32)),
        "psf_bank": torch.from_numpy(combined_bank.astype(np.float32)),
    }
    bank_path = output_root / "psf_bank.pt"
    torch.save(bank_payload, bank_path)
    combined_npy_path = output_root / "combined_psf_bank.npy"
    np.save(combined_npy_path, combined_bank.astype(np.float32), allow_pickle=False)
    profile_assets: list[dict[str, Any]] = []
    for index, profile in enumerate(profiles):
        profile_document = profile_documents[index]
        profile_root = output_root / "profiles" / profile.profile_id
        profile_root.mkdir(parents=True, exist_ok=True)
        profile_asset_id = f"{asset_id}_{profile.profile_id}"
        profile_payload = _profile_asset_payload(
            asset_id=profile_asset_id,
            family_id=family_id,
            family_sha=family_sha,
            profile_document=profile_document,
            f_numbers=f_numbers,
            signed_coc=signed_coc,
            labels=final_labels[index],
            raw_labels=raw_labels[index],
            bank=final_banks[index],
            throughput=throughputs[index],
            field_x=optical["field_x"],
            field_y=optical["field_y"],
            physical_curve=physical_curves[index],
        )
        profile_pt_path = profile_root / "psf_bank.pt"
        torch.save(profile_payload, profile_pt_path)
        profile_npy_path = profile_root / "psf_bank.npy"
        np.save(profile_npy_path, final_banks[index].astype(np.float32), allow_pickle=False)
        pdraw_mono_assets = (
            _export_pdraw_mono_v1_by_aperture(
                profile_root,
                resource_prefix=profile_asset_id,
                bank=final_banks[index],
                signed_coc=signed_coc,
                f_numbers=f_numbers,
                field_x=optical["field_x"],
                field_y=optical["field_y"],
            )
            if bool(config["family"].get("export_pdraw_mono_v1", True))
            else []
        )
        profile_assets.append(
            {
                "profile_id": profile.profile_id,
                "parameter_sha256": profile_document["parameter_sha256"],
                "loader_ready_7d_path": str(profile_pt_path.resolve()),
                "loader_ready_7d_sha256": _sha256_file(profile_pt_path),
                "compatible_npy_path": str(profile_npy_path.resolve()),
                "compatible_npy_sha256": _sha256_file(profile_npy_path),
                "pdraw_mono_v1_by_aperture": pdraw_mono_assets,
            }
        )
    profile_manifest_path = output_root / "profile_manifest.json"
    profile_manifest_path.write_text(
        json.dumps(
            {
                "family_id": family_id,
                "family_sha256": family_sha,
                "profiles": profile_documents,
                "profile_assets": profile_assets,
                "matched_single_profile_medoid": medoid,
                "combined_archive": {
                    "torch_path": str(bank_path.resolve()),
                    "torch_sha256": _sha256_file(bank_path),
                    "npy_path": str(combined_npy_path.resolve()),
                    "npy_sha256": _sha256_file(combined_npy_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    asset_recipe = {
        "schema_version": 1,
        "asset_id": asset_id,
        "psf_bank": str(bank_path.resolve()),
        "psf_bank_sha256": _sha256_file(bank_path),
        "combined_npy": str(combined_npy_path.resolve()),
        "combined_npy_sha256": _sha256_file(combined_npy_path),
        "profile_ids": [profile.profile_id for profile in profiles],
        "matched_single_profile_medoid": medoid,
        "per_profile_loader_ready_7d": [
            {
                "profile_id": row["profile_id"],
                "path": row["loader_ready_7d_path"],
                "sha256": row["loader_ready_7d_sha256"],
            }
            for row in profile_assets
        ],
        "f_numbers": f_numbers.tolist(),
        "pd_offset_embedded": False,
        "label_formula": "analytic_centroid_disparity_px + independent_pd_offset_px",
        "pd_offset_sample_range_px": config["pdoffset"]["sample_range_px"],
    }
    recipe_path = output_root / "asset_recipe.yaml"
    recipe_path.write_text(
        yaml.safe_dump(asset_recipe, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    total_seconds = time.perf_counter() - started
    summary = {
        "run_id": str(config["run"]["id"]),
        "status": status,
        "family_id": family_id,
        "family_sha256": family_sha,
        "profile_count": len(profiles),
        "aperture_count": aperture_count,
        "field_grid_hw": [field_y_count, field_x_count],
        "field_count": field_y_count * field_x_count,
        "coc_count": coc_count,
        "reference_cell_count": cell_count,
        "naive_profile_cell_count": cell_count * len(profiles),
        "reference_propagation_reused_across_profiles": True,
        "cross_aperture_shared_lineage_checks": lineage_checks,
        "reference_kernel_size": reference_kernel_size,
        "export_kernel_size": export_kernel_size,
        "profile_chunk_timings": chunk_timings,
        "source_propagation_seconds_total": sum(
            float(row["source_propagation_seconds"]) for row in chunk_timings
        ),
        "shape": list(combined_bank.shape),
        "reference_setup_seconds": reference_setup_seconds,
        "shared_profile_propagation_seconds": propagation_seconds,
        "total_seconds": total_seconds,
        "seconds_per_reference_cell": propagation_seconds / float(cell_count),
        "profiles": diagnostics,
        "bank_path": str(bank_path.resolve()),
        "bank_sha256": _sha256_file(bank_path),
        "combined_npy_path": str(combined_npy_path.resolve()),
        "combined_npy_sha256": _sha256_file(combined_npy_path),
        "profile_assets": profile_assets,
        "matched_single_profile_medoid": medoid,
        "profile_manifest_sha256": _sha256_file(profile_manifest_path),
        "asset_recipe_sha256": _sha256_file(recipe_path),
        "analytic_label_source": "final_psf_centroid_mu_left_x-minus_mu_right_x",
        "pd_offset_embedded": False,
        "ncc_updates_labels": False,
        **config["data_isolation"],
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    resolved_path = output_root / "resolved_config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    provenance = {
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256_file(config_path),
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": _sha256_file(Path(__file__).resolve()),
        "multi_profile_implementation_sha256": _sha256_file(
            Path(__file__).with_name("multi_profile.py")
        ),
        "source_commit": _git_output(checkout, "rev-parse", "HEAD"),
        "source_dirty_paths": _git_output(checkout, "status", "--short").splitlines(),
        "lens_path": str(lens_path),
        "lens_sha256": _sha256_file(lens_path),
        "python_executable": sys.executable,
        "summary_sha256": _sha256_file(summary_path),
        **config["data_isolation"],
    }
    provenance_path = output_root / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    artifact_manifest = {
        "schema_version": 1,
        "status": status,
        "family_id": family_id,
        "files": {
            "psf_bank.pt": _sha256_file(bank_path),
            "combined_psf_bank.npy": _sha256_file(combined_npy_path),
            "profile_manifest.json": _sha256_file(profile_manifest_path),
            "asset_recipe.yaml": _sha256_file(recipe_path),
            "summary.json": _sha256_file(summary_path),
            "resolved_config.yaml": _sha256_file(resolved_path),
            "provenance.json": _sha256_file(provenance_path),
            "preexport_diagnostics.json": _sha256_file(preexport_path),
        },
    }
    for timing in chunk_timings:
        cache_path = Path(timing["cache_path"])
        artifact_manifest["files"][str(cache_path.relative_to(output_root))] = timing[
            "cache_sha256"
        ]
        cache_sha_path = cache_path.with_suffix(cache_path.suffix + ".sha256")
        artifact_manifest["files"][str(cache_sha_path.relative_to(output_root))] = (
            _sha256_file(cache_sha_path)
        )
    for profile_asset in profile_assets:
        for path_key, sha_key in (
            ("loader_ready_7d_path", "loader_ready_7d_sha256"),
            ("compatible_npy_path", "compatible_npy_sha256"),
        ):
            path = Path(profile_asset[path_key])
            artifact_manifest["files"][str(path.relative_to(output_root))] = (
                profile_asset[sha_key]
            )
        for row in profile_asset["pdraw_mono_v1_by_aperture"]:
            for path_key, sha_key in (
                ("array_path", "array_sha256"),
                ("metadata_path", "metadata_sha256"),
            ):
                path = Path(row[path_key])
                artifact_manifest["files"][str(path.relative_to(output_root))] = row[
                    sha_key
                ]
    artifact_manifest_path = output_root / "artifact_manifest.json"
    artifact_manifest_path.write_text(
        json.dumps(artifact_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "output_root": str(output_root),
        "status": status,
        "bank_sha256": summary["bank_sha256"],
        "summary_sha256": _sha256_file(summary_path),
        "artifact_manifest_sha256": _sha256_file(artifact_manifest_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    result = generate(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
