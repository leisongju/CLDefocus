"""对冻结 matching-preserving 候选做 continuous Lanczos-NCC V2 复核。

本运行器只读取冻结 center8 bank 和原 sweep 账本。它不会重新运行 CLDefocus
光学传播、写出新 PSF bank、访问真实 PDraw 数据或启动训练。解析标签始终沿用冻结
PSF centroid ``mu_left_x-mu_right_x``；NCC 和 exact-shift oracle 均不得回写标签。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from dataset.psf_bank_renderer import compute_psf_stats

from .matching_preserving_sweep import (
    CandidateSpec,
    _torch_load,
    _transform_profile,
    _validate_payload,
)
from .multi_profile import (
    _git_output,
    _sha256_file,
    apply_per_aperture_gate,
    compare_aligned_continuous_ncc_to_oracle,
    continuous_ncc_v2_runtime_kwargs,
    run_profile_continuous_ncc_diagnostic,
    summarize_per_aperture_ncc,
    validate_continuous_ncc_v2_declaration,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256_array(value: np.ndarray) -> str:
    import hashlib

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_original_candidates(
    metrics: dict[str, Any],
    expected: Sequence[dict[str, Any]],
) -> dict[str, CandidateSpec]:
    """从冻结原 sweep 账本按 ID 解析候选，并逐字段验证期望映射。"""

    rows = {
        str(row["candidate"]["candidate_id"]): row["candidate"]
        for row in metrics["candidates"]
    }
    output: dict[str, CandidateSpec] = {}
    for declaration in expected:
        candidate_id = str(declaration["candidate_id"])
        if candidate_id not in rows:
            raise RuntimeError(f"原 sweep 账本缺少候选：{candidate_id}")
        source = rows[candidate_id]
        for key in ("stage", "transition_coc_px", "transition_power", "morphology_scale"):
            actual = source[key]
            target = declaration[key]
            equal = (
                str(actual) == str(target)
                if key == "stage"
                else math.isclose(float(actual), float(target), rel_tol=0.0, abs_tol=1.0e-12)
            )
            if not equal:
                raise RuntimeError(
                    f"{candidate_id} 原 sweep 映射漂移：{key}={actual} != {target}"
                )
        output[candidate_id] = CandidateSpec(
            candidate_id=candidate_id,
            stage=str(source["stage"]),
            transition_coc_px=float(source["transition_coc_px"]),
            transition_power=float(source["transition_power"]),
            morphology_scale=float(source["morphology_scale"]),
        )
    return output


def _source_optical_audit(
    bank: np.ndarray,
    frozen_labels: np.ndarray,
    signed_coc_bins_px: np.ndarray,
    gates: dict[str, Any],
) -> dict[str, Any]:
    tensor = torch.from_numpy(np.asarray(bank, dtype=np.float32))
    stats = compute_psf_stats(tensor)
    disparity = stats["mu_x"][..., 0] - stats["mu_x"][..., 1]
    labels = torch.from_numpy(np.asarray(frozen_labels, dtype=np.float32))
    coc = torch.from_numpy(np.asarray(signed_coc_bins_px, dtype=np.float32))
    if coc.ndim == 1:
        coc = coc[None].expand(tensor.shape[0], -1)
    nonzero = coc[:, :, None, None].abs() > 0.25
    sign = float(
        (
            torch.sign(labels[nonzero.expand_as(labels)])
            == torch.sign(coc[:, :, None, None].expand_as(labels)[nonzero.expand_as(labels)])
        )
        .to(torch.float32)
        .mean()
        .item()
    )
    label_error = float((disparity - labels).abs().max().item())
    energy_error = float((stats["energy"] - 1.0).abs().max().item())
    checks = {
        "finite_nonnegative": bool(
            torch.isfinite(tensor).all().item() and (tensor >= 0.0).all().item()
        ),
        "energy": energy_error <= float(gates["energy_abs_tolerance"]),
        "source_label_identity": label_error
        <= float(gates["frozen_label_abs_max_error_px"]),
        "optical_sign": sign >= float(gates["optical_sign_agreement_min"]),
    }
    return {
        "checks": checks,
        "pass": bool(all(checks.values())),
        "source_vs_frozen_label_abs_max_error_px": label_error,
        "transformed_vs_frozen_label_abs_max_error_px": label_error,
        "centroid_abs_max_error_px": 0.0,
        "energy_abs_max_error": energy_error,
        "energy_min": float(stats["energy"].min().item()),
        "energy_max": float(stats["energy"].max().item()),
        "optical_sign_agreement_nonzero": sign,
        "equivalent_radius_mean_before_px": float(
            stats["equivalent_radius"].mean().item()
        ),
        "equivalent_radius_mean_after_px": float(
            stats["equivalent_radius"].mean().item()
        ),
        "analytic_label_used_by_ncc": "frozen_source_psf_centroid_label_unchanged",
        "label_compensation": False,
    }


def audit_continuous_tile_validity(
    diagnostic: dict[str, Any],
    *,
    aperture_count: int,
    coc_abs_max_px: float,
    expected_tile_count: int,
    expected_records_per_aperture: int,
) -> dict[str, Any]:
    """逐 aperture 审计 near-focus cell 是否全部具有精确 9/9 有效 tile。"""

    threshold = float(coc_abs_max_px)
    rows: list[dict[str, Any]] = []
    all_grid_failures: list[dict[str, Any]] = []
    near_focus_failures: list[dict[str, Any]] = []
    for aperture_index in range(int(aperture_count)):
        selected = []
        for seed_row in diagnostic["seed_diagnostics"]:
            seed = int(seed_row["seed"])
            for cell in seed_row["cells"]:
                if int(cell["aperture_index"]) != aperture_index:
                    continue
                identity = {
                    "seed": seed,
                    "aperture_index": aperture_index,
                    "coc_index": int(cell["coc_index"]),
                    "field_x_index": int(cell["field_x_index"]),
                    "signed_coc_px": float(cell["signed_coc_px"]),
                    "valid_tile_count": int(cell["valid_tile_count"]),
                    "total_tile_count": int(cell["total_tile_count"]),
                }
                exact = bool(
                    cell["cell_quality_pass"]
                    and int(cell["valid_tile_count"]) == int(expected_tile_count)
                    and int(cell["total_tile_count"]) == int(expected_tile_count)
                )
                if not exact:
                    all_grid_failures.append(identity)
                if abs(float(cell["signed_coc_px"])) <= threshold + 1.0e-12:
                    selected.append((identity, exact))
                    if not exact:
                        near_focus_failures.append(identity)
        exact_record_count = len(selected) == int(expected_records_per_aperture)
        rows.append(
            {
                "aperture_index": aperture_index,
                "expected_near_focus_record_count": int(expected_records_per_aperture),
                "observed_near_focus_record_count": len(selected),
                "record_count_exact": exact_record_count,
                "near_focus_all_cells_exact_tiles": bool(selected)
                and exact_record_count
                and all(exact for _, exact in selected),
                "minimum_valid_tile_count": min(
                    (row["valid_tile_count"] for row, _ in selected), default=0
                ),
                "maximum_total_tile_count": max(
                    (row["total_tile_count"] for row, _ in selected), default=0
                ),
            }
        )
    return {
        "required_tile_count_per_cell": int(expected_tile_count),
        "required_contract": f"{expected_tile_count}/{expected_tile_count}",
        "near_focus_rows": rows,
        "near_focus_failure_count": len(near_focus_failures),
        "near_focus_failures": near_focus_failures,
        "near_focus_all_apertures_exact": bool(rows)
        and all(bool(row["near_focus_all_cells_exact_tiles"]) for row in rows),
        "full_coc_grid_failure_count_report_only": len(all_grid_failures),
        "full_coc_grid_failures_report_only": all_grid_failures,
    }


def _candidate_bank(
    payload: dict[str, Any],
    *,
    profile_index: int,
    candidate_id: str,
    specs: dict[str, CandidateSpec],
    optical_gates: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    profile_id = str(payload["profile_ids"][profile_index])
    labels = np.asarray(
        payload["analytic_disparity_bins_px"][profile_index].cpu(),
        dtype=np.float64,
    )
    coc = np.asarray(payload["signed_coc_bins_px"].cpu(), dtype=np.float64)
    if candidate_id == "SOURCE_P004":
        if profile_id != "P004":
            raise ValueError("SOURCE_P004 只允许用于原 P004 pilot")
        bank = np.asarray(payload["psf_bank"][profile_index].cpu(), dtype=np.float32)
        optical = _source_optical_audit(bank, labels, coc, optical_gates)
        mapping = {
            "candidate_id": candidate_id,
            "stage": "source_unmodified",
            "transition_coc_px": None,
            "transition_power": None,
            "morphology_scale": None,
            "source_bank_unmodified": True,
        }
        return bank, optical, mapping
    spec = specs[candidate_id]
    bank, optical = _transform_profile(
        payload,
        profile_index=profile_index,
        candidate=spec,
        gates=optical_gates,
    )
    mapping = {
        "candidate_id": spec.candidate_id,
        "stage": spec.stage,
        "transition_coc_px": spec.transition_coc_px,
        "transition_power": spec.transition_power,
        "morphology_scale": spec.morphology_scale,
        "source_bank_unmodified": False,
    }
    return bank, optical, mapping


def _evaluate(
    payload: dict[str, Any],
    *,
    profile_index: int,
    candidate_id: str,
    specs: dict[str, CandidateSpec],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    bank, optical, mapping = _candidate_bank(
        payload,
        profile_index=profile_index,
        candidate_id=candidate_id,
        specs=specs,
        optical_gates=dict(config["optical_gates"]),
    )
    labels = np.asarray(
        payload["analytic_disparity_bins_px"][profile_index].cpu(),
        dtype=np.float64,
    )
    coc = np.asarray(payload["signed_coc_bins_px"].cpu(), dtype=np.float64)
    diagnostic_cfg = dict(config["diagnostic"])
    continuous_cfg = dict(diagnostic_cfg["continuous_ncc"])
    physical, oracle = run_profile_continuous_ncc_diagnostic(
        bank,
        labels,
        signed_coc_bins_px=coc,
        texture_size=int(diagnostic_cfg["texture_size"]),
        seeds=[int(value) for value in diagnostic_cfg["seeds"]],
        tile_size=int(diagnostic_cfg["tile_size"]),
        tiles_per_axis=int(diagnostic_cfg["tiles_per_axis"]),
        search_x=int(diagnostic_cfg["search_x"]),
        search_y=int(diagnostic_cfg["search_y"]),
        **continuous_ncc_v2_runtime_kwargs(continuous_cfg),
    )
    aperture_count = int(bank.shape[0])
    near_focus_cfg = dict(config["near_focus"])
    threshold = float(near_focus_cfg["coc_abs_max_px"])
    summary = summarize_per_aperture_ncc(
        physical,
        aperture_count=aperture_count,
        coc_abs_max_px=threshold,
    )
    gate = apply_per_aperture_gate(summary, dict(config["strict_gates"]))
    oracle_summary = summarize_per_aperture_ncc(
        oracle,
        aperture_count=aperture_count,
        coc_abs_max_px=threshold,
    )
    aligned = compare_aligned_continuous_ncc_to_oracle(
        physical,
        oracle,
        aperture_count=aperture_count,
        coc_abs_max_px=threshold,
    )
    validity = audit_continuous_tile_validity(
        physical,
        aperture_count=aperture_count,
        coc_abs_max_px=threshold,
        expected_tile_count=int(near_focus_cfg["expected_tiles_per_cell"]),
        expected_records_per_aperture=int(
            near_focus_cfg["expected_records_per_aperture"]
        ),
    )
    numerical_pass = bool(optical["pass"] and gate["pass"] and validity["near_focus_all_apertures_exact"])
    nonphysical_impulse = bool(
        candidate_id == "C018" and float(mapping["morphology_scale"]) == 0.0
    )
    admission_eligible = bool(candidate_id == "C017" and not nonphysical_impulse)
    profile_id = str(payload["profile_ids"][profile_index])
    row = {
        "profile_id": profile_id,
        "candidate": mapping,
        "optical": optical,
        "analytic_label_source": "frozen_source_PSF_centroid_mu_left_x_minus_mu_right_x",
        "analytic_label_grid_sha256": _sha256_array(labels),
        "ncc_updates_label": False,
        "oracle_updates_label_or_admission": False,
        "pdoffset_applied": False,
        "continuous_primary_gate": gate,
        "tile_validity": validity,
        "continuous_fourier_oracle_summary_report_only": oracle_summary,
        "aligned_physical_oracle_report_only": aligned,
        "numeric_protocol_identity_sha256": physical[
            "numeric_protocol_identity_sha256"
        ],
        "tile_grid_sha256": physical["tile_grid_sha256"],
        "aligned_label_grid_sha256": aligned["label_grid_sha256"],
        "aligned_valid_mask_sha256": aligned["aligned_valid_mask_sha256"],
        "strict_numerical_pass": numerical_pass,
        "physical_model_class": (
            "centroid_preserving_impulse_degenerate_nonphysical"
            if nonphysical_impulse
            else (
                "matching_preserving_transformed_psf"
                if candidate_id == "C017"
                else "frozen_source_psf_diagnostic_baseline"
            )
        ),
        "admission_eligible": admission_eligible,
        "admission_pass": bool(numerical_pass and admission_eligible),
        "c018_nonphysical_veto": nonphysical_impulse,
    }
    raw = {
        "profile_id": profile_id,
        "candidate": mapping,
        "physical_continuous_ncc": physical,
        "fourier_exact_shift_continuous_oracle_report_only": oracle,
    }
    return row, raw


def _csv_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        for aperture in row["continuous_primary_gate"]["per_aperture"]:
            fit = aperture["fit"]
            validity = row["tile_validity"]["near_focus_rows"][
                int(aperture["aperture_index"])
            ]
            output.append(
                {
                    "profile_id": row["profile_id"],
                    "candidate_id": row["candidate"]["candidate_id"],
                    "aperture_index": aperture["aperture_index"],
                    "selected_signed_coc_bins_px": json.dumps(
                        aperture["selected_signed_coc_bins_px"], separators=(",", ":")
                    ),
                    "expected_record_count": aperture["expected_record_count"],
                    "valid_record_fraction": aperture["valid_record_fraction"],
                    "slope": fit["slope_ncc_vs_analytic"],
                    "r2": fit["r2"],
                    "epe_px": fit["epe_px"],
                    "rmse_px": fit["rmse_px"],
                    "sign_agreement_nonzero": fit["sign_agreement_nonzero"],
                    "all_cells_9_of_9": validity[
                        "near_focus_all_cells_exact_tiles"
                    ],
                    "strict_aperture_pass": aperture["pass"],
                    "profile_strict_numerical_pass": row[
                        "strict_numerical_pass"
                    ],
                    "admission_eligible": row["admission_eligible"],
                    "admission_pass": row["admission_pass"],
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


def _validate_lineage(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    lineage_cfg = dict(config["original_sweep_lineage"])
    provenance_path = Path(lineage_cfg["provenance_path"]).resolve()
    metrics_path = Path(lineage_cfg["candidate_metrics_path"]).resolve()
    actual_provenance_sha = _sha256_file(provenance_path)
    actual_metrics_sha = _sha256_file(metrics_path)
    if actual_provenance_sha != str(lineage_cfg["provenance_sha256"]):
        raise RuntimeError("原 sweep provenance SHA256 不匹配")
    if actual_metrics_sha != str(lineage_cfg["candidate_metrics_sha256"]):
        raise RuntimeError("原 sweep candidate metrics SHA256 不匹配")
    provenance = _load_json(provenance_path)
    original_sources = dict(provenance["source_sha256"])
    checked_sources: dict[str, Any] = {}
    for relative_path in (
        "render/CLDefocus/pdraw_adapter/matching_preserving_sweep.py",
        "dataset/psf_bank_renderer.py",
    ):
        current_path = (project_root / relative_path).resolve()
        matching_original_paths = [
            path for path in original_sources if path.endswith(relative_path)
        ]
        if len(matching_original_paths) != 1:
            raise RuntimeError(f"原 provenance 无法唯一映射源码：{relative_path}")
        original_sha = str(original_sources[matching_original_paths[0]])
        current_sha = _sha256_file(current_path)
        if current_sha != original_sha:
            raise RuntimeError(
                f"候选重建源码相对原 sweep 漂移：{relative_path} {current_sha} != {original_sha}"
            )
        checked_sources[relative_path] = {
            "original_path": matching_original_paths[0],
            "original_sha256": original_sha,
            "current_path": str(current_path),
            "current_sha256": current_sha,
            "exact_match": True,
        }
    return {
        "original_provenance_path": str(provenance_path),
        "original_provenance_sha256": actual_provenance_sha,
        "original_candidate_metrics_path": str(metrics_path),
        "original_candidate_metrics_sha256": actual_metrics_sha,
        "original_source_sha256": original_sources,
        "candidate_reconstruction_source_checks": checked_sources,
        "original_input_sha256": provenance["input_sha256"],
        "candidate_metrics": _load_json(metrics_path),
    }


def run(config_path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or int(loaded.get("schema_version", 0)) != 1:
        raise ValueError("continuous recheck 配置必须为 schema_version: 1")
    config = dict(loaded)
    expected_python = Path(str(config["runtime"]["python_executable"])).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(
            f"必须使用 Genfocus：{sys.executable} != {expected_python}"
        )
    project_root = Path(__file__).resolve().parents[3]
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

    input_cfg = dict(config["input"])
    input_path = Path(input_cfg["combined_bank_path"]).resolve()
    input_sha = _sha256_file(input_path)
    if input_sha != str(input_cfg["sha256"]):
        raise RuntimeError("冻结 combined bank SHA256 不匹配")
    lineage = _validate_lineage(config, project_root)
    if input_sha != str(lineage["original_input_sha256"]):
        raise RuntimeError("combined bank 与原 sweep provenance 的 input SHA 不一致")
    payload = _torch_load(input_path)
    validation_config = {
        "input": {
            "profile_ids": input_cfg["profile_ids"],
            "f_numbers": input_cfg["f_numbers"],
            "aperture_match_tolerance": input_cfg["aperture_match_tolerance"],
        }
    }
    _validate_payload(payload, validation_config)
    continuous_cfg = dict(config["diagnostic"]["continuous_ncc"])
    validate_continuous_ncc_v2_declaration(continuous_cfg)
    specs = _resolve_original_candidates(
        lineage.pop("candidate_metrics"),
        config["pilot"]["transformed_candidates"],
    )
    profile_ids = [str(value) for value in payload["profile_ids"]]
    pilot_profile_id = str(config["pilot"]["profile_id"])
    pilot_index = profile_ids.index(pilot_profile_id)

    start = time.perf_counter()
    pilot_rows: list[dict[str, Any]] = []
    pilot_raw: list[dict[str, Any]] = []
    for candidate_id in config["pilot"]["candidate_ids"]:
        candidate_start = time.perf_counter()
        row, raw = _evaluate(
            payload,
            profile_index=pilot_index,
            candidate_id=str(candidate_id),
            specs=specs,
            config=config,
        )
        row["elapsed_seconds"] = time.perf_counter() - candidate_start
        pilot_rows.append(row)
        pilot_raw.append(raw)
        slopes = [
            float(value["fit"]["slope_ncc_vs_analytic"])
            for value in row["continuous_primary_gate"]["per_aperture"]
        ]
        print(
            f"[continuous-recheck] {pilot_profile_id}/{candidate_id} "
            f"strict={row['strict_numerical_pass']} admission={row['admission_pass']} "
            f"slopes={[round(value, 6) for value in slopes]}",
            flush=True,
        )

    c017_pilot = next(
        row for row in pilot_rows if row["candidate"]["candidate_id"] == "C017"
    )
    full_triggered = bool(
        config["pilot"]["c017_full_center8_if_strict_pass"]
        and c017_pilot["strict_numerical_pass"]
    )
    center8_rows: list[dict[str, Any]] = [c017_pilot]
    center8_raw: list[dict[str, Any]] = [
        raw for raw in pilot_raw if raw["candidate"]["candidate_id"] == "C017"
    ]
    if full_triggered:
        for profile_index, profile_id in enumerate(profile_ids):
            if profile_id == pilot_profile_id:
                continue
            row, raw = _evaluate(
                payload,
                profile_index=profile_index,
                candidate_id="C017",
                specs=specs,
                config=config,
            )
            center8_rows.append(row)
            center8_raw.append(raw)
            print(
                f"[continuous-recheck] center8 {profile_id}/C017 "
                f"strict={row['strict_numerical_pass']}",
                flush=True,
            )
    center8_all_strict = bool(
        full_triggered
        and len(center8_rows) == len(profile_ids)
        and all(bool(row["strict_numerical_pass"]) for row in center8_rows)
    )

    pilot_metrics_path = output_root / "pilot_metrics.json"
    _write_json(
        pilot_metrics_path,
        {
            "profile_id": pilot_profile_id,
            "candidate_count": len(pilot_rows),
            "candidates": pilot_rows,
        },
    )
    pilot_raw_path = output_root / "pilot_continuous_diagnostics.json"
    _write_json(pilot_raw_path, {"candidates": pilot_raw})
    center8_path: Path | None = None
    center8_raw_path: Path | None = None
    if full_triggered:
        center8_path = output_root / "c017_center8_metrics.json"
        _write_json(
            center8_path,
            {
                "trigger": "P004/C017 strict numerical PASS",
                "profile_count": len(center8_rows),
                "all_profiles_strict_numerical_pass": center8_all_strict,
                "profiles": center8_rows,
            },
        )
        center8_raw_path = output_root / "c017_center8_continuous_diagnostics.json"
        _write_json(center8_raw_path, {"profiles": center8_raw})
    csv_path = output_root / "per_aperture_metrics.csv"
    _write_csv(
        csv_path,
        _csv_rows(
            pilot_rows
            + ([row for row in center8_rows if row is not c017_pilot] if full_triggered else [])
        ),
    )

    elapsed = time.perf_counter() - start
    final_metrics = {
        "run_id": str(config["run"]["id"]),
        "status": "GO" if center8_all_strict else "NO_GO",
        "scope": "readonly_matching_preserving_continuous_v2_recheck",
        "combined_bank_path": str(input_path),
        "combined_bank_sha256": input_sha,
        "original_sweep_lineage": lineage,
        "pilot_profile_id": pilot_profile_id,
        "pilot_candidate_ids": [
            str(row["candidate"]["candidate_id"]) for row in pilot_rows
        ],
        "pilot_strict_numerical_pass_ids": [
            str(row["candidate"]["candidate_id"])
            for row in pilot_rows
            if bool(row["strict_numerical_pass"])
        ],
        "pilot_admission_pass_ids": [
            str(row["candidate"]["candidate_id"])
            for row in pilot_rows
            if bool(row["admission_pass"])
        ],
        "c017_pilot_strict_pass": bool(c017_pilot["strict_numerical_pass"]),
        "c017_center8_recheck_triggered": full_triggered,
        "c017_center8_profile_count": len(center8_rows) if full_triggered else 0,
        "c017_center8_all_strict_pass": center8_all_strict,
        "c018_nonphysical_degenerate_veto_unconditional": True,
        "c018_can_be_admitted": False,
        "strict_gate_is_absolute_not_oracle_corrected": True,
        "all_required_cells_must_be_9_of_9": True,
        "aggregate_can_override_per_aperture": False,
        "analytic_label_source": "frozen PSF centroid mu_left_x-mu_right_x",
        "ncc_or_oracle_updates_label": False,
        "pdoffset_applied": False,
        "new_psf_bank_written": False,
        "real_pdraw_accessed": False,
        "google_dev_accessed": False,
        "google_holdout_accessed": False,
        "dp5k_accessed": False,
        "training_run": False,
        "elapsed_seconds": elapsed,
    }
    final_path = output_root / "final_metrics.json"
    _write_json(final_path, final_metrics)

    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("matching_preserving_sweep.py").resolve(),
        Path(__file__).with_name("multi_profile.py").resolve(),
        (project_root / "dataset/psf_bank_renderer.py").resolve(),
        (project_root / "render/cldefocus_pdraw/continuous_ncc.py").resolve(),
        (project_root / "render/cldefocus_pdraw/continuous_ncc_v2.py").resolve(),
    ]
    provenance = {
        "project_commit": _git_output(project_root, "rev-parse", "HEAD"),
        "project_dirty_paths": _git_output(project_root, "status", "--porcelain").splitlines(),
        "runtime": "Genfocus",
        "python_executable": sys.executable,
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256_file(config_path),
        "resolved_config_sha256": _sha256_file(resolved_path),
        "source_sha256": {str(path): _sha256_file(path) for path in source_paths},
        "input_sha256": input_sha,
        "original_sweep_provenance_sha256": lineage[
            "original_provenance_sha256"
        ],
        "cpu_only": True,
        "real_data_accessed": False,
        "training_run": False,
    }
    provenance_path = output_root / "provenance.json"
    _write_json(provenance_path, provenance)

    payloads: dict[str, dict[str, Any]] = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            payloads[str(path.relative_to(output_root))] = {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
    manifest_path = output_root / "artifact_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "run_id": str(config["run"]["id"]),
            "payload_count": len(payloads),
            "payloads": payloads,
        },
    )
    return {
        "output_root": str(output_root),
        "status": final_metrics["status"],
        "c017_pilot_strict_pass": final_metrics["c017_pilot_strict_pass"],
        "c017_center8_recheck_triggered": full_triggered,
        "c017_center8_all_strict_pass": center8_all_strict,
        "final_metrics_sha256": _sha256_file(final_path),
        "artifact_manifest_sha256": _sha256_file(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="冻结 matching-preserving 候选的 continuous V2 复核"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
