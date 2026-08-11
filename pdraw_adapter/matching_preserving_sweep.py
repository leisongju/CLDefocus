"""冻结 center8 bank 的 matching-preserving near-focus CPU sweep。

该诊断只改变 PSF morphology，不修改解析 centroid label。NCC、Fourier oracle、
cubic/grid-sample baseline 都只用于度量；任何结果都不得回写 label、PDOFFSET 或训练资产。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as torch_functional
import yaml

from dataset.psf_bank_renderer import (
    GeneratedPSFBank,
    compute_psf_stats,
    make_centroid_preserving_morphology_scaled_psf_bank,
    make_matching_preserving_psf_bank,
)
from pdraw_benchmark.ncc_reference import NccProtocol, estimate_bidirectional_ncc

from .multi_profile import (
    _deterministic_texture,
    _fit_ncc_records,
    _git_output,
    _ncc_tiles,
    _sha256_file,
    run_profile_ncc_diagnostic,
    run_shifted_identical_ncc_oracle,
)
from .near_focus_audit import summarize_per_aperture_ncc


@dataclass(frozen=True)
class CandidateSpec:
    """一个显式、可复现的 matching-preserving 变体。"""

    candidate_id: str
    stage: str
    transition_coc_px: float
    transition_power: float
    morphology_scale: float


def build_candidate_specs(
    transition_coc_px: Sequence[float],
    transition_power: Sequence[float],
    morphology_scales: Sequence[float],
    *,
    include_supplement: bool,
) -> list[CandidateSpec]:
    """先生成 6x3 基础网格；必要时再补充非 1 morphology 的笛卡尔积。"""

    transitions = [float(value) for value in transition_coc_px]
    powers = [float(value) for value in transition_power]
    morphologies = [float(value) for value in morphology_scales]
    if not transitions or not powers:
        raise ValueError("transition_coc_px/transition_power 不能为空")
    if any(not math.isfinite(value) or value <= 0.0 for value in transitions):
        raise ValueError("transition_coc_px 必须全部为有限正数")
    if any(not math.isfinite(value) or value <= 0.0 for value in powers):
        raise ValueError("transition_power 必须全部为有限正数")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in morphologies):
        raise ValueError("morphology_scales 必须全部位于 [0,1]")
    if len(set(transitions)) != len(transitions) or len(set(powers)) != len(powers):
        raise ValueError("基础 sweep 轴不允许重复值")
    if len(set(morphologies)) != len(morphologies):
        raise ValueError("morphology_scales 不允许重复值")

    rows: list[CandidateSpec] = []
    index = 0
    for transition in transitions:
        for power in powers:
            rows.append(
                CandidateSpec(
                    candidate_id=f"C{index:03d}",
                    stage="matching_only",
                    transition_coc_px=transition,
                    transition_power=power,
                    morphology_scale=1.0,
                )
            )
            index += 1
    if include_supplement:
        for transition in transitions:
            for power in powers:
                for morphology in morphologies:
                    # morphology=1 已由基础阶段精确覆盖，禁止重复计算。
                    if abs(morphology - 1.0) <= 1.0e-12:
                        continue
                    rows.append(
                        CandidateSpec(
                            candidate_id=f"C{index:03d}",
                            stage="matching_plus_morphology",
                            transition_coc_px=transition,
                            transition_power=power,
                            morphology_scale=morphology,
                        )
                    )
                    index += 1
    identities = {
        (row.transition_coc_px, row.transition_power, row.morphology_scale) for row in rows
    }
    if len(identities) != len(rows):
        raise RuntimeError("候选参数发生重复")
    return rows


def _fit_checks(
    fit: dict[str, Any],
    valid_fraction: float,
    gates: dict[str, Any],
) -> dict[str, bool]:
    checks = {
        "slope_lower": float(fit["slope_ncc_vs_analytic"])
        >= float(gates["slope_min"]),
        "slope_upper": float(fit["slope_ncc_vs_analytic"])
        <= float(gates["slope_max"]),
        "r2": float(fit["r2"]) >= float(gates["r2_min"]),
        "sign": fit["sign_agreement_nonzero"] is not None
        and float(fit["sign_agreement_nonzero"])
        >= float(gates["sign_agreement_min"]),
        "valid_records": float(valid_fraction)
        >= float(gates.get("valid_record_fraction_min", 0.95)),
    }
    # EPE 默认只报告；只有配置显式给出数值时才成为门禁。
    if gates.get("epe_px_max") is not None:
        checks["epe"] = float(fit["epe_px"]) <= float(gates["epe_px_max"])
    return checks


def apply_strict_and_wide_per_aperture_gates(
    summary: dict[str, Any],
    *,
    strict_gates: dict[str, Any],
    wide_gates: dict[str, Any],
) -> dict[str, Any]:
    """逐 aperture 应用绝对门；aggregate 永远只报告。"""

    rows: list[dict[str, Any]] = []
    for source in summary["per_aperture"]:
        strict_checks = _fit_checks(
            source["fit"], float(source["valid_record_fraction"]), strict_gates
        )
        wide_checks = _fit_checks(
            source["fit"], float(source["valid_record_fraction"]), wide_gates
        )
        rows.append(
            {
                **source,
                "strict_gate": {
                    "checks": strict_checks,
                    "pass": bool(all(strict_checks.values())),
                },
                "wide_gate_report_only": {
                    "checks": wide_checks,
                    "pass": bool(all(wide_checks.values())),
                },
            }
        )
    return {
        **summary,
        "per_aperture": rows,
        "strict_all_apertures_pass": bool(rows)
        and all(bool(row["strict_gate"]["pass"]) for row in rows),
        "wide_all_apertures_pass_report_only": bool(rows)
        and all(bool(row["wide_gate_report_only"]["pass"]) for row in rows),
        "pass_rule": "all_apertures_must_pass_absolute_gate",
        "aggregate_is_admission_gate": False,
        "epe_is_admission_gate": strict_gates.get("epe_px_max") is not None,
        "strict_gates": dict(strict_gates),
        "wide_gates_report_only": dict(wide_gates),
    }


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"combined bank 必须保存为 dict：{path}")
    return payload


def _label_grid_sha256(labels: np.ndarray) -> str:
    value = np.ascontiguousarray(labels, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("utf-8"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _make_source_bank(
    payload: dict[str, Any],
    *,
    profile_index: int,
    aperture_index: int,
) -> GeneratedPSFBank:
    psf = payload["psf_bank"][profile_index, aperture_index].to(dtype=torch.float32)
    coc = payload["signed_coc_bins_px"]
    signed_coc = coc if coc.ndim == 1 else coc[aperture_index]
    throughput = payload.get("side_throughput_grid")
    selected_throughput = (
        None
        if throughput is None
        else throughput[profile_index, aperture_index].to(dtype=torch.float32)
    )
    f_number = float(payload["f_numbers"][aperture_index])
    profile_id = str(payload["profile_ids"][profile_index])
    return GeneratedPSFBank(
        psf_bank=psf,
        signed_coc_bins=signed_coc.to(dtype=torch.float32),
        field_grid_hw=tuple(int(value) for value in payload["field_grid_hw"]),
        kernel_size=int(payload["kernel_size"]),
        asset_id=f"{payload.get('asset_id', 'center8')}_{profile_id}_f{f_number:g}",
        side_throughput_grid=selected_throughput,
        profile_id=profile_id,
        requested_aperture_f_number=f_number,
        selected_aperture_f_number=f_number,
    )


def _transform_profile(
    payload: dict[str, Any],
    *,
    profile_index: int,
    candidate: CandidateSpec,
    gates: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    transformed_apertures: list[torch.Tensor] = []
    source_apertures: list[torch.Tensor] = []
    for aperture_index in range(int(payload["f_numbers"].numel())):
        source = _make_source_bank(
            payload,
            profile_index=profile_index,
            aperture_index=aperture_index,
        )
        matching = make_matching_preserving_psf_bank(
            source,
            transition_coc=candidate.transition_coc_px,
            transition_power=candidate.transition_power,
            centroid_tolerance=float(gates["centroid_abs_max_error_px"]),
        )
        transformed = matching
        if candidate.morphology_scale < 1.0 - 1.0e-12:
            transformed = make_centroid_preserving_morphology_scaled_psf_bank(
                matching,
                morphology_scale=candidate.morphology_scale,
                centroid_tolerance=float(gates["centroid_abs_max_error_px"]),
            )
        source_apertures.append(source.psf_bank)
        transformed_apertures.append(transformed.psf_bank)

    source_tensor = torch.stack(source_apertures, dim=0)
    transformed_tensor = torch.stack(transformed_apertures, dim=0)
    source_stats = compute_psf_stats(source_tensor)
    transformed_stats = compute_psf_stats(transformed_tensor)
    centroid_error_x = (transformed_stats["mu_x"] - source_stats["mu_x"]).abs()
    centroid_error_y = (transformed_stats["mu_y"] - source_stats["mu_y"]).abs()
    centroid_error = torch.maximum(centroid_error_x, centroid_error_y)
    transformed_disparity = (
        transformed_stats["mu_x"][..., 0] - transformed_stats["mu_x"][..., 1]
    )
    source_disparity = source_stats["mu_x"][..., 0] - source_stats["mu_x"][..., 1]
    frozen_labels = payload["analytic_disparity_bins_px"][profile_index].to(
        dtype=torch.float32
    )
    signed_coc = payload["signed_coc_bins_px"].to(dtype=torch.float32)
    if signed_coc.ndim == 1:
        signed_coc = signed_coc[None].expand(source_tensor.shape[0], -1)
    coc_grid = signed_coc[:, :, None, None]
    nonzero = coc_grid.abs() > 0.25
    optical_sign = float(
        (
            torch.sign(frozen_labels[nonzero])
            == torch.sign(coc_grid.expand_as(frozen_labels)[nonzero])
        )
        .to(dtype=torch.float32)
        .mean()
        .item()
    )
    energy_error = (transformed_stats["energy"] - 1.0).abs()
    finite_nonnegative = bool(
        torch.isfinite(transformed_tensor).all().item()
        and (transformed_tensor >= 0.0).all().item()
    )
    centroid_error_max = float(centroid_error.max().item())
    source_label_error_max = float((source_disparity - frozen_labels).abs().max().item())
    transformed_label_error_max = float(
        (transformed_disparity - frozen_labels).abs().max().item()
    )
    energy_error_max = float(energy_error.max().item())
    checks = {
        "finite_nonnegative": finite_nonnegative,
        "energy": energy_error_max <= float(gates["energy_abs_tolerance"]),
        "centroid_identity": centroid_error_max
        <= float(gates["centroid_abs_max_error_px"]),
        "source_label_identity": source_label_error_max
        <= float(gates["frozen_label_abs_max_error_px"]),
        "transformed_label_identity": transformed_label_error_max
        <= float(gates["centroid_abs_max_error_px"]),
        "optical_sign": optical_sign >= float(gates["optical_sign_agreement_min"]),
    }
    optical = {
        "checks": checks,
        "pass": bool(all(checks.values())),
        "centroid_abs_max_error_px": centroid_error_max,
        "source_vs_frozen_label_abs_max_error_px": source_label_error_max,
        "transformed_vs_frozen_label_abs_max_error_px": transformed_label_error_max,
        "energy_abs_max_error": energy_error_max,
        "energy_min": float(transformed_stats["energy"].min().item()),
        "energy_max": float(transformed_stats["energy"].max().item()),
        "optical_sign_agreement_nonzero": optical_sign,
        "equivalent_radius_mean_before_px": float(
            source_stats["equivalent_radius"].mean().item()
        ),
        "equivalent_radius_mean_after_px": float(
            transformed_stats["equivalent_radius"].mean().item()
        ),
        "analytic_label_used_by_ncc": "frozen_source_centroid_label_unchanged",
        "label_compensation": False,
    }
    return transformed_tensor.numpy(), optical


def _ncc_kwargs(diagnostic: dict[str, Any]) -> dict[str, Any]:
    return {
        "texture_size": int(diagnostic["texture_size"]),
        "seeds": [int(value) for value in diagnostic["seeds"]],
        "tile_size": int(diagnostic["tile_size"]),
        "tiles_per_axis": int(diagnostic["tiles_per_axis"]),
        "search_x": int(diagnostic["search_x"]),
        "search_y": int(diagnostic["search_y"]),
    }


def _summarize_subsets(
    diagnostic: dict[str, Any],
    *,
    aperture_count: int,
    subsets: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    strict = {
        **dict(gates["strict"]),
        "valid_record_fraction_min": float(gates["valid_record_fraction_min"]),
    }
    wide = {
        **dict(gates["wide_report_only"]),
        "valid_record_fraction_min": float(gates["valid_record_fraction_min"]),
    }
    coc = apply_strict_and_wide_per_aperture_gates(
        summarize_per_aperture_ncc(
            diagnostic,
            aperture_count=aperture_count,
            coc_abs_max_px=float(subsets["coc_abs_max_px"]),
        ),
        strict_gates=strict,
        wide_gates=wide,
    )
    disparity = apply_strict_and_wide_per_aperture_gates(
        summarize_per_aperture_ncc(
            diagnostic,
            aperture_count=aperture_count,
            analytic_disparity_abs_max_px=float(
                subsets["analytic_disparity_abs_max_px"]
            ),
        ),
        strict_gates=strict,
        wide_gates=wide,
    )
    return {
        "abs_coc": coc,
        "abs_analytic_disparity": disparity,
        "both_subsets_required_for_admission": True,
        "strict_pass": bool(
            coc["strict_all_apertures_pass"]
            and disparity["strict_all_apertures_pass"]
        ),
        "wide_pass_report_only": bool(
            coc["wide_all_apertures_pass_report_only"]
            and disparity["wide_all_apertures_pass_report_only"]
        ),
    }


def _shift_pair_grid_sample(
    texture: np.ndarray,
    disparity_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(np.asarray(texture, dtype=np.float32))[None, None]
    height, width = texture.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height),
        torch.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    base = torch.stack((xx, yy), dim=-1)[None]
    outputs: list[np.ndarray] = []
    for shift_px in (0.5 * float(disparity_px), -0.5 * float(disparity_px)):
        grid = base.clone()
        # grid_sample 读取 source 坐标；减号让图像内容向 +x 移动。
        grid[..., 0] -= 2.0 * shift_px / max(width - 1, 1)
        output = torch_functional.grid_sample(
            tensor,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        outputs.append(output[0, 0].numpy())
    return outputs[0], outputs[1]


def _shift_pair_cubic(
    texture: np.ndarray,
    disparity_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import shift

    left = shift(
        texture,
        shift=(0.0, 0.5 * float(disparity_px)),
        order=3,
        mode="wrap",
        prefilter=True,
    ).astype(np.float32)
    right = shift(
        texture,
        shift=(0.0, -0.5 * float(disparity_px)),
        order=3,
        mode="wrap",
        prefilter=True,
    ).astype(np.float32)
    return left, right


def run_shifted_identical_resampling_baseline(
    analytic_labels: np.ndarray,
    *,
    signed_coc_bins_px: np.ndarray,
    kernel_size: int,
    diagnostic: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    """计算非 oracle 的 symmetric resampling baseline。

    ``cubic`` 是 scipy cubic 重采样，``grid_sample_bilinear`` 是 PyTorch
    grid_sample 双线性重采样；二者都不得称为 Fourier exact oracle。
    """

    methods: dict[str, Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]]] = {
        "cubic": _shift_pair_cubic,
        "grid_sample_bilinear": _shift_pair_grid_sample,
    }
    if method not in methods:
        raise ValueError(f"不支持的 resampling baseline：{method}")
    shift_pair = methods[method]
    labels = np.asarray(analytic_labels, dtype=np.float64)
    coc = np.asarray(signed_coc_bins_px, dtype=np.float64)
    if labels.ndim != 4:
        raise ValueError(f"analytic labels 必须为 [A,N,Gh,Gw]：{labels.shape}")
    if coc.ndim == 1:
        coc = np.tile(coc[None, :], (labels.shape[0], 1))
    if coc.shape != labels.shape[:2]:
        raise ValueError(f"signed CoC shape 不一致：{coc.shape} vs {labels.shape[:2]}")
    kwargs = _ncc_kwargs(diagnostic)
    protocol = NccProtocol(
        tile_size=kwargs["tile_size"],
        tile_stride=kwargs["tile_size"],
        margin_x=kwargs["search_x"],
        margin_y=kwargs["search_y"],
        search_x=kwargs["search_x"],
        search_y=kwargs["search_y"],
        min_peak=0.20,
        min_peak_margin=0.002,
        max_lr_error=0.60,
        max_vertical_shift=0.75,
    )
    tiles = _ncc_tiles(
        texture_size=kwargs["texture_size"],
        kernel_size=int(kernel_size),
        tile_size=kwargs["tile_size"],
        tiles_per_axis=kwargs["tiles_per_axis"],
        search_x=kwargs["search_x"],
        search_y=kwargs["search_y"],
    )
    total_cells = int(np.prod(labels.shape))
    seed_rows: list[dict[str, Any]] = []
    aggregate_labels: list[float] = []
    aggregate_predictions: list[float] = []
    for seed in kwargs["seeds"]:
        texture = _deterministic_texture(kwargs["texture_size"], seed)
        pair_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        seed_labels: list[float] = []
        seed_predictions: list[float] = []
        cells: list[dict[str, Any]] = []
        for aperture_index in range(labels.shape[0]):
            for coc_index in range(labels.shape[1]):
                for field_y_index in range(labels.shape[2]):
                    for field_x_index in range(labels.shape[3]):
                        analytic = float(
                            labels[
                                aperture_index,
                                coc_index,
                                field_y_index,
                                field_x_index,
                            ]
                        )
                        key = float(np.round(analytic, 12))
                        if key not in pair_cache:
                            pair_cache[key] = shift_pair(texture, analytic)
                        left, right = pair_cache[key]
                        estimates = [
                            estimate_bidirectional_ncc(left, right, tile, protocol)
                            for tile in tiles
                        ]
                        valid = [row for row in estimates if bool(row["ncc_quality_pass"])]
                        values = [float(row["pseudo_disp_model_px"]) for row in valid]
                        prediction = float(np.median(values)) if values else None
                        cells.append(
                            {
                                "aperture_index": aperture_index,
                                "coc_index": coc_index,
                                "field_y_index": field_y_index,
                                "field_x_index": field_x_index,
                                "signed_coc_px": float(coc[aperture_index, coc_index]),
                                "analytic_centroid_disparity_px": analytic,
                                "ncc_disparity_px": prediction,
                                "valid_tile_count": len(valid),
                                "total_tile_count": len(estimates),
                            }
                        )
                        if prediction is not None:
                            seed_labels.append(analytic)
                            seed_predictions.append(prediction)
                            aggregate_labels.append(analytic)
                            aggregate_predictions.append(prediction)
        seed_rows.append(
            {
                "seed": seed,
                "fit": _fit_ncc_records(seed_labels, seed_predictions),
                "valid_cell_fraction": len(seed_labels) / float(total_cells),
                "cells": cells,
            }
        )
    return {
        "role": f"{method}_shifted_identical_resampling_baseline_report_only",
        "is_fourier_exact_oracle": False,
        "label_compensation_from_baseline": False,
        "baseline_updates_admission": False,
        "protocol": asdict(protocol),
        "texture_size": kwargs["texture_size"],
        "seeds": kwargs["seeds"],
        "aggregate_fit_report_only": _fit_ncc_records(
            aggregate_labels, aggregate_predictions
        ),
        "seed_diagnostics": seed_rows,
    }


def _baseline_summaries(
    labels: np.ndarray,
    signed_coc: np.ndarray,
    *,
    kernel_size: int,
    diagnostic_cfg: dict[str, Any],
    subsets: dict[str, Any],
) -> dict[str, Any]:
    kwargs = _ncc_kwargs(diagnostic_cfg)
    raw = {
        "fourier_exact_oracle": run_shifted_identical_ncc_oracle(
            labels,
            signed_coc_bins_px=signed_coc,
            kernel_size=kernel_size,
            **kwargs,
        ),
        "cubic_resampling": run_shifted_identical_resampling_baseline(
            labels,
            signed_coc_bins_px=signed_coc,
            kernel_size=kernel_size,
            diagnostic=diagnostic_cfg,
            method="cubic",
        ),
        "grid_sample_bilinear": run_shifted_identical_resampling_baseline(
            labels,
            signed_coc_bins_px=signed_coc,
            kernel_size=kernel_size,
            diagnostic=diagnostic_cfg,
            method="grid_sample_bilinear",
        ),
    }
    output: dict[str, Any] = {
        "label_grid_sha256": _label_grid_sha256(labels),
        "all_baselines_report_only": True,
        "baseline_updates_label": False,
        "baseline_updates_admission": False,
        "methods": {},
    }
    for name, diagnostic in raw.items():
        output["methods"][name] = {
            "role": diagnostic["role"],
            "abs_coc": summarize_per_aperture_ncc(
                diagnostic,
                aperture_count=labels.shape[0],
                coc_abs_max_px=float(subsets["coc_abs_max_px"]),
            ),
            "abs_analytic_disparity": summarize_per_aperture_ncc(
                diagnostic,
                aperture_count=labels.shape[0],
                analytic_disparity_abs_max_px=float(
                    subsets["analytic_disparity_abs_max_px"]
                ),
            ),
        }
    return output


def _attach_baseline_comparison(
    subset_summaries: dict[str, Any],
    baselines: dict[str, Any],
) -> None:
    for subset_name in ("abs_coc", "abs_analytic_disparity"):
        physical_rows = subset_summaries[subset_name]["per_aperture"]
        for row in physical_rows:
            aperture_index = int(row["aperture_index"])
            physical_fit = row["fit"]
            comparisons: dict[str, Any] = {}
            for method_name, method in baselines["methods"].items():
                baseline_row = method[subset_name]["per_aperture"][aperture_index]
                baseline_fit = baseline_row["fit"]
                physical_slope = float(physical_fit["slope_ncc_vs_analytic"])
                baseline_slope = float(baseline_fit["slope_ncc_vs_analytic"])
                comparisons[method_name] = {
                    "baseline_fit": baseline_fit,
                    "slope_psf_minus_baseline": physical_slope - baseline_slope,
                    "slope_psf_over_baseline": (
                        None
                        if abs(baseline_slope) <= 1.0e-12
                        else physical_slope / baseline_slope
                    ),
                    "epe_psf_minus_baseline_px": float(physical_fit["epe_px"])
                    - float(baseline_fit["epe_px"]),
                    "role": "report_only_no_label_or_admission_correction",
                }
            row["estimator_baselines_report_only"] = comparisons
    subset_summaries["baseline_label_grid_sha256"] = baselines["label_grid_sha256"]


def _run_profile_candidate(
    payload: dict[str, Any],
    *,
    profile_index: int,
    candidate: CandidateSpec,
    config: dict[str, Any],
    baselines: dict[str, Any],
) -> dict[str, Any]:
    bank, optical = _transform_profile(
        payload,
        profile_index=profile_index,
        candidate=candidate,
        gates=config["gates"],
    )
    labels = payload["analytic_disparity_bins_px"][profile_index].cpu().numpy()
    coc = payload["signed_coc_bins_px"].cpu().numpy()
    ncc = run_profile_ncc_diagnostic(
        bank,
        labels,
        signed_coc_bins_px=coc,
        **_ncc_kwargs(config["diagnostic"]),
    )
    subsets = _summarize_subsets(
        ncc,
        aperture_count=bank.shape[0],
        subsets=config["subsets"],
        gates=config["gates"],
    )
    _attach_baseline_comparison(subsets, baselines)
    strict_pass = bool(optical["pass"] and subsets["strict_pass"])
    wide_pass = bool(optical["pass"] and subsets["wide_pass_report_only"])
    return {
        "profile_id": str(payload["profile_ids"][profile_index]),
        "candidate": asdict(candidate),
        "optical": optical,
        "subsets": subsets,
        "strict_pass": strict_pass,
        "wide_pass_report_only": wide_pass,
        "aggregate_is_admission_gate": False,
        "ncc_updates_label": False,
        "pdoffset_applied": False,
        "ncc_correction_applied": False,
    }


def _candidate_rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    aperture_rows = [
        aperture
        for subset_name in ("abs_coc", "abs_analytic_disparity")
        for aperture in row["subsets"][subset_name]["per_aperture"]
    ]
    strict_count = sum(bool(value["strict_gate"]["pass"]) for value in aperture_rows)
    wide_count = sum(
        bool(value["wide_gate_report_only"]["pass"]) for value in aperture_rows
    )
    maximum_slope_error = max(
        abs(float(value["fit"]["slope_ncc_vs_analytic"]) - 1.0)
        for value in aperture_rows
    )
    mean_epe = float(np.mean([float(value["fit"]["epe_px"]) for value in aperture_rows]))
    candidate = row["candidate"]
    row["ranking_metrics"] = {
        "strict_aperture_subset_pass_count": strict_count,
        "strict_aperture_subset_total": len(aperture_rows),
        "wide_aperture_subset_pass_count_report_only": wide_count,
        "max_abs_slope_error_from_one": maximum_slope_error,
        "mean_epe_px": mean_epe,
    }
    # 绝对门和 slope-to-one 优先；完全相同时偏好更少 morphology 干预。
    return (
        0 if bool(row["optical"]["pass"]) else 1,
        -strict_count,
        -wide_count,
        maximum_slope_error,
        mean_epe,
        -float(candidate["morphology_scale"]),
        float(candidate["transition_coc_px"]),
        float(candidate["transition_power"]),
        str(candidate["candidate_id"]),
    )


def rank_candidates(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """按冻结绝对门排序；oracle/baseline 明确不参与排序。"""

    ranked = sorted(rows, key=_candidate_rank_key)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["rank_uses_oracle_or_resampling_baseline"] = False
    return ranked


def _metric_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for profile_row in rows:
        candidate = profile_row["candidate"]
        optical = profile_row["optical"]
        for subset_name in ("abs_coc", "abs_analytic_disparity"):
            for aperture in profile_row["subsets"][subset_name]["per_aperture"]:
                fit = aperture["fit"]
                baselines = aperture["estimator_baselines_report_only"]
                output.append(
                    {
                        "profile_id": profile_row["profile_id"],
                        "candidate_id": candidate["candidate_id"],
                        "stage": candidate["stage"],
                        "transition_coc_px": candidate["transition_coc_px"],
                        "transition_power": candidate["transition_power"],
                        "morphology_scale": candidate["morphology_scale"],
                        "subset": subset_name,
                        "aperture_index": aperture["aperture_index"],
                        "selected_signed_coc_bins_px": json.dumps(
                            aperture["selected_signed_coc_bins_px"], separators=(",", ":")
                        ),
                        "expected_record_count": aperture["expected_record_count"],
                        "valid_record_fraction": aperture["valid_record_fraction"],
                        "slope": fit["slope_ncc_vs_analytic"],
                        "intercept_px": fit["intercept_px"],
                        "r2": fit["r2"],
                        "epe_px": fit["epe_px"],
                        "rmse_px": fit["rmse_px"],
                        "max_abs_error_px": fit["max_abs_error_px"],
                        "sign_agreement_nonzero": fit["sign_agreement_nonzero"],
                        "strict_pass": aperture["strict_gate"]["pass"],
                        "wide_pass_report_only": aperture[
                            "wide_gate_report_only"
                        ]["pass"],
                        "fourier_oracle_slope": baselines["fourier_exact_oracle"][
                            "baseline_fit"
                        ]["slope_ncc_vs_analytic"],
                        "psf_minus_fourier_slope": baselines[
                            "fourier_exact_oracle"
                        ]["slope_psf_minus_baseline"],
                        "psf_over_fourier_slope": baselines[
                            "fourier_exact_oracle"
                        ]["slope_psf_over_baseline"],
                        "cubic_resampling_slope": baselines["cubic_resampling"][
                            "baseline_fit"
                        ]["slope_ncc_vs_analytic"],
                        "grid_sample_bilinear_slope": baselines[
                            "grid_sample_bilinear"
                        ]["baseline_fit"]["slope_ncc_vs_analytic"],
                        "centroid_abs_max_error_px": optical[
                            "centroid_abs_max_error_px"
                        ],
                        "energy_abs_max_error": optical["energy_abs_max_error"],
                        "optical_sign_agreement_nonzero": optical[
                            "optical_sign_agreement_nonzero"
                        ],
                    }
                )
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"CSV 不能为空：{path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_payload(payload: dict[str, Any], config: dict[str, Any]) -> None:
    required = {
        "profile_ids",
        "f_numbers",
        "signed_coc_bins_px",
        "analytic_disparity_bins_px",
        "field_grid_hw",
        "kernel_size",
        "psf_bank",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"combined bank 缺少键：{missing}")
    expected_profiles = [str(value) for value in config["input"]["profile_ids"]]
    actual_profiles = [str(value) for value in payload["profile_ids"]]
    if actual_profiles != expected_profiles:
        raise RuntimeError(f"profile 顺序不匹配：{actual_profiles} != {expected_profiles}")
    bank = payload["psf_bank"]
    if bank.ndim != 8 or bank.shape[0] != len(actual_profiles):
        raise ValueError(f"combined bank 必须为 [P,A,N,Gh,Gw,2,K,K]：{tuple(bank.shape)}")
    labels = payload["analytic_disparity_bins_px"]
    if tuple(labels.shape) != tuple(bank.shape[:5]):
        raise ValueError(f"label shape 不一致：{tuple(labels.shape)} vs {tuple(bank.shape[:5])}")
    f_numbers = [float(value) for value in payload["f_numbers"]]
    expected_f_numbers = [float(value) for value in config["input"]["f_numbers"]]
    aperture_tolerance = float(config["input"]["aperture_match_tolerance"])
    if len(f_numbers) != len(expected_f_numbers) or not np.allclose(
        f_numbers,
        expected_f_numbers,
        atol=aperture_tolerance,
        rtol=0.0,
    ):
        raise RuntimeError(f"aperture 顺序不匹配：{f_numbers} != {expected_f_numbers}")


def run(config_path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or int(loaded.get("schema_version", 0)) != 1:
        raise ValueError("matching-preserving sweep 配置必须为 schema_version: 1")
    config = dict(loaded)
    project_root = Path(__file__).resolve().parents[3]
    input_cfg = dict(config["input"])
    input_path = Path(input_cfg["combined_bank_path"]).resolve()
    input_cfg["combined_bank_path"] = str(input_path)
    config["input"] = input_cfg
    output_cfg = dict(config["output"])
    output_root = Path(output_cfg["root"]).resolve()
    output_cfg["root"] = str(output_root)
    config["output"] = output_cfg
    if output_root == project_root or project_root in output_root.parents:
        raise ValueError("诊断输出必须位于仓库外部")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_path = output_root / "resolved_config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    actual_sha = _sha256_file(input_path)
    if actual_sha != str(input_cfg["sha256"]):
        raise RuntimeError(f"冻结 bank SHA256 不匹配：{actual_sha} != {input_cfg['sha256']}")
    payload = _torch_load(input_path)
    _validate_payload(payload, config)
    profile_ids = [str(value) for value in payload["profile_ids"]]
    pilot_profile_id = str(config["sweep"]["pilot_profile_id"])
    pilot_index = profile_ids.index(pilot_profile_id)
    labels = payload["analytic_disparity_bins_px"][pilot_index].cpu().numpy()
    coc = payload["signed_coc_bins_px"].cpu().numpy()

    start = time.perf_counter()
    print("[matching-sweep] 计算 P004 Fourier/cubic/grid_sample baseline", flush=True)
    pilot_baselines = _baseline_summaries(
        labels,
        coc,
        kernel_size=int(payload["kernel_size"]),
        diagnostic_cfg=config["diagnostic"],
        subsets=config["subsets"],
    )
    base_specs = build_candidate_specs(
        config["sweep"]["transition_coc_px"],
        config["sweep"]["transition_power"],
        config["sweep"]["supplemental_morphology_scales"],
        include_supplement=False,
    )
    candidate_rows: list[dict[str, Any]] = []
    for spec in base_specs:
        row = _run_profile_candidate(
            payload,
            profile_index=pilot_index,
            candidate=spec,
            config=config,
            baselines=pilot_baselines,
        )
        candidate_rows.append(row)
        slopes = [
            float(value["fit"]["slope_ncc_vs_analytic"])
            for value in row["subsets"]["abs_coc"]["per_aperture"]
        ]
        print(
            f"[matching-sweep] {spec.candidate_id} tc={spec.transition_coc_px:g} "
            f"p={spec.transition_power:g} m=1 strict={row['strict_pass']} "
            f"slope_min={min(slopes):.6f}",
            flush=True,
        )
    base_strict_pass = any(bool(row["strict_pass"]) for row in candidate_rows)
    supplement_ran = bool(
        config["sweep"].get("supplement_if_no_strict_pass", True)
        and not base_strict_pass
    )
    if supplement_ran:
        all_specs = build_candidate_specs(
            config["sweep"]["transition_coc_px"],
            config["sweep"]["transition_power"],
            config["sweep"]["supplemental_morphology_scales"],
            include_supplement=True,
        )
        for spec in all_specs[len(base_specs) :]:
            row = _run_profile_candidate(
                payload,
                profile_index=pilot_index,
                candidate=spec,
                config=config,
                baselines=pilot_baselines,
            )
            candidate_rows.append(row)
            print(
                f"[matching-sweep] {spec.candidate_id} tc={spec.transition_coc_px:g} "
                f"p={spec.transition_power:g} m={spec.morphology_scale:g} "
                f"strict={row['strict_pass']}",
                flush=True,
            )
    ranked = rank_candidates(candidate_rows)
    best = ranked[0]
    best_spec = CandidateSpec(**best["candidate"])

    candidate_json = output_root / "p004_candidate_metrics.json"
    _write_json(
        candidate_json,
        {
            "pilot_profile_id": pilot_profile_id,
            "base_candidate_count": len(base_specs),
            "base_strict_pass": base_strict_pass,
            "supplement_ran": supplement_ran,
            "candidate_count": len(ranked),
            "baseline_cache": pilot_baselines,
            "rank_rule": (
                "absolute optical/strict/wide gates, slope distance to one, EPE, then "
                "least morphology intervention; baseline never participates"
            ),
            "candidates": ranked,
        },
    )
    candidate_csv = output_root / "p004_candidate_metrics.csv"
    _write_csv(candidate_csv, _metric_rows(ranked))

    verification_rows: list[dict[str, Any]] = []
    verification_baselines: dict[str, Any] = {}
    for profile_index, profile_id in enumerate(profile_ids):
        profile_labels = payload["analytic_disparity_bins_px"][profile_index].cpu().numpy()
        label_sha = _label_grid_sha256(profile_labels)
        if label_sha == pilot_baselines["label_grid_sha256"]:
            baselines = pilot_baselines
        else:
            print(f"[matching-sweep] 计算 {profile_id} estimator baselines", flush=True)
            baselines = _baseline_summaries(
                profile_labels,
                coc,
                kernel_size=int(payload["kernel_size"]),
                diagnostic_cfg=config["diagnostic"],
                subsets=config["subsets"],
            )
        verification_baselines[profile_id] = baselines
        row = _run_profile_candidate(
            payload,
            profile_index=profile_index,
            candidate=best_spec,
            config=config,
            baselines=baselines,
        )
        verification_rows.append(row)
        print(
            f"[matching-sweep] verify {profile_id} strict={row['strict_pass']} "
            f"wide={row['wide_pass_report_only']}",
            flush=True,
        )
    all_profiles_strict_pass = bool(verification_rows) and all(
        bool(row["strict_pass"]) for row in verification_rows
    )
    conclusion = "GO" if all_profiles_strict_pass else "NO_GO"
    unique_recommendation = asdict(best_spec) if all_profiles_strict_pass else None

    verification_json = output_root / "center8_verification.json"
    _write_json(
        verification_json,
        {
            "candidate": asdict(best_spec),
            "candidate_selection_used_baseline": False,
            "baselines": verification_baselines,
            "profiles": verification_rows,
            "all_profiles_strict_pass": all_profiles_strict_pass,
        },
    )
    verification_csv = output_root / "center8_verification.csv"
    _write_csv(verification_csv, _metric_rows(verification_rows))

    elapsed = time.perf_counter() - start
    final_metrics = {
        "run_id": str(config["run"]["id"]),
        "status": conclusion,
        "pilot_profile_id": pilot_profile_id,
        "base_candidate_count": len(base_specs),
        "supplement_ran": supplement_ran,
        "evaluated_candidate_count": len(ranked),
        "best_diagnostic_candidate": asdict(best_spec),
        "best_candidate_pilot_strict_pass": bool(best["strict_pass"]),
        "best_candidate_pilot_wide_pass_report_only": bool(
            best["wide_pass_report_only"]
        ),
        "verification_profile_count": len(verification_rows),
        "verification_strict_pass_profile_ids": [
            row["profile_id"] for row in verification_rows if bool(row["strict_pass"])
        ],
        "all_8_profiles_x_5_apertures_x_2_subsets_strict_pass": (
            all_profiles_strict_pass
        ),
        "unique_recommendation": unique_recommendation,
        "best_candidate_is_training_approved": all_profiles_strict_pass,
        "strict_gate_is_absolute_not_oracle_corrected": True,
        "fourier_oracle_role": "report_only_peak_locking_baseline",
        "cubic_and_grid_sample_role": "report_only_resampling_baselines_not_oracles",
        "aggregate_can_override_per_aperture": False,
        "analytic_label_source": "frozen PSF centroid mu_left_x-mu_right_x",
        "label_compensation_from_ncc_or_baseline": False,
        "ncc_correction_applied": False,
        "pdoffset_applied": False,
        "new_psf_bank_written": False,
        "training_run": False,
        "real_pdraw_accessed": False,
        "google_dev_accessed": False,
        "google_holdout_accessed": False,
        "dp5k_accessed": False,
        "input_combined_bank": str(input_path),
        "input_combined_bank_sha256": actual_sha,
        "elapsed_seconds": elapsed,
    }
    final_path = output_root / "final_metrics.json"
    _write_json(final_path, final_metrics)

    source_paths = [
        Path(__file__).resolve(),
        (project_root / "dataset/psf_bank_renderer.py").resolve(),
        Path(__file__).with_name("multi_profile.py").resolve(),
        Path(__file__).with_name("near_focus_audit.py").resolve(),
    ]
    provenance = {
        "project_commit": _git_output(project_root, "rev-parse", "HEAD"),
        "project_dirty_paths": _git_output(
            project_root, "status", "--porcelain"
        ).splitlines(),
        "runtime": "Genfocus",
        "python_executable": sys.executable,
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256_file(config_path),
        "resolved_config_sha256": _sha256_file(resolved_path),
        "source_sha256": {
            str(path): _sha256_file(path) for path in source_paths
        },
        "input_sha256": actual_sha,
        "cpu_only": True,
        "real_data_accessed": False,
        "training_run": False,
    }
    provenance_path = output_root / "provenance.json"
    _write_json(provenance_path, provenance)

    manifest_payloads: dict[str, dict[str, Any]] = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest_payloads[str(path.relative_to(output_root))] = {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
    manifest = {
        "schema_version": 1,
        "run_id": str(config["run"]["id"]),
        "payload_count": len(manifest_payloads),
        "payloads": manifest_payloads,
    }
    manifest_path = output_root / "artifact_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "output_root": str(output_root),
        "status": conclusion,
        "best_diagnostic_candidate": asdict(best_spec),
        "unique_recommendation": unique_recommendation,
        "final_metrics_sha256": _sha256_file(final_path),
        "artifact_manifest_sha256": _sha256_file(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="冻结 center8 bank 的 matching-preserving near-focus CPU sweep"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
