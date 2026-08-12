"""执行冻结的 multi-prescription device-atom GPU propagation 与 NCC 准入。

本模块消费 ``multi_prescription_plan.py`` 已经落盘且带 SHA256 的 CPU plan，
为每个 lens atom 依次生成并冻结五类子配置：raw family、raw-optical、K64、
K128 和最终 PairCommon+DCC6 asset。子配置写在外部产物目录，后续 stage 只读取
前一 stage 的冻结 manifest SHA；任何 atom 未完整通过时，聚合
``training_admitted`` 都保持 ``false``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import yaml


_CANDIDATE_ID = "dcc6_pair_common_direct_hybrid_s100_tc16"
_ISOLATION_KEYS = {
    "real_pdraw_accessed",
    "google_dev_accessed",
    "google_holdout_accessed",
    "dp5k_accessed",
    "stereo_training_run",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _git_output(checkout: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args], text=True
    ).strip()


def _validate_isolation(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict) or set(value) != _ISOLATION_KEYS:
        raise ValueError(f"data_isolation 必须精确为 {sorted(_ISOLATION_KEYS)}")
    result = {key: bool(item) for key, item in value.items()}
    if any(result.values()):
        raise ValueError("multi-prescription 正式资产构建禁止访问真实数据或启动训练")
    return result


def load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "run",
        "source_plan",
        "cldefocus",
        "lens",
        "parent",
        "propagation",
        "raw_family_gates",
        "audit",
        "candidate_gates",
        "output",
        "data_isolation",
    }
    if not isinstance(loaded, dict) or int(loaded.get("schema_version", 0)) != 1:
        raise ValueError("执行配置必须是 schema_version: 1 mapping")
    if set(loaded) != required:
        raise ValueError(f"执行配置字段必须精确为 {sorted(required)}")
    if str(loaded["run"].get("mode")) != "gpu_propagate_audit_export":
        raise ValueError("run.mode 必须为 gpu_propagate_audit_export")
    _validate_isolation(loaded["data_isolation"])
    if int(loaded["audit"]["worker_count"]) < 1:
        raise ValueError("audit.worker_count 必须为正数")
    if sorted(int(key) for key in loaded["audit"]["texture_size_by_tile"]) != [64, 128]:
        raise ValueError("texture_size_by_tile 必须精确冻结 K64/K128")
    return loaded


def _verify_implementation(config: dict[str, Any]) -> None:
    checkout = Path(config["cldefocus"]["checkout"]).resolve()
    if _git_output(checkout, "rev-parse", "HEAD") != str(
        config["cldefocus"]["commit"]
    ):
        raise RuntimeError("CLDefocus commit 与执行配置不一致")
    for relative, expected_sha in config["cldefocus"][
        "implementation_sha256"
    ].items():
        path = checkout / str(relative)
        if not path.is_file() or _sha256_file(path) != str(expected_sha):
            raise RuntimeError(f"冻结实现 SHA256 不匹配：{path}")
    parent = config["parent"]
    parent_root = Path(parent["repository"]).resolve()
    for relative, key in (
        ("dataset/psf_bank_renderer.py", "loader_sha256"),
        ("dataset/dp_renderer.py", "renderer_sha256"),
    ):
        path = parent_root / relative
        if not path.is_file() or _sha256_file(path) != str(parent[key]):
            raise RuntimeError(f"冻结 parent 实现 SHA256 不匹配：{path}")


def load_verified_plan(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = config["source_plan"]
    root = Path(source["root"]).resolve()
    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.is_file() or _sha256_file(manifest_path) != str(
        source["artifact_manifest_sha256"]
    ):
        raise RuntimeError("CPU plan artifact_manifest SHA256 不匹配")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "cpu_prepared_pending_gpu_propagation"
        or manifest.get("gpu_propagation_started") is not False
        or manifest.get("training_admitted") is not False
        or manifest.get("pdoffset_embedded") is not False
    ):
        raise RuntimeError("CPU plan 状态合同不匹配")
    plan_path = root / "device_atom_plan.json"
    expected_plan_sha = str(source["device_atom_plan_sha256"])
    if (
        manifest.get("files", {}).get("device_atom_plan.json") != expected_plan_sha
        or not plan_path.is_file()
        or _sha256_file(plan_path) != expected_plan_sha
    ):
        raise RuntimeError("CPU device_atom_plan SHA256 不匹配")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    atoms = list(plan.get("atoms", []))
    if len(atoms) != int(manifest["device_atom_count"]):
        raise RuntimeError("CPU plan atom 数量不一致")
    output_root = Path(config["output"]["root"]).resolve()
    for atom in atoms:
        if atom.get("pdoffset_embedded") is not False:
            raise RuntimeError(f"atom 意外嵌入 PDOFFSET：{atom.get('atom_id')}")
        if atom.get("shared_lineage_across_apertures") is not True:
            raise RuntimeError(f"atom 未声明五光圈共享 lineage：{atom.get('atom_id')}")
        for output in atom["planned_outputs"].values():
            path = Path(output).resolve()
            if output_root not in path.parents:
                raise RuntimeError(f"atom 输出不在冻结根目录：{path}")
    replacements = dict(config["lens"].get("replacements_by_atom", {}))
    active_atoms: list[dict[str, Any]] = []
    for source_atom in atoms:
        source_atom_id = str(source_atom["atom_id"])
        replacement = replacements.get(source_atom_id)
        if replacement is None:
            active_atoms.append(source_atom)
            continue
        required = {
            "replacement_rank",
            "selection_policy",
            "lens_id",
            "lens_relative_path",
            "lens_sha256",
            "replacement_atom_id",
            "negative_evidence_root",
        }
        if not isinstance(replacement, dict) or set(replacement) != required:
            raise ValueError(
                f"replacement 字段必须精确为 {sorted(required)}：{source_atom_id}"
            )
        replacement_id = str(replacement["replacement_atom_id"])
        replacement_root = output_root / replacement_id
        active_atoms.append(
            {
                **source_atom,
                "atom_id": replacement_id,
                "lens_id": str(replacement["lens_id"]),
                "lens_relative_path": str(replacement["lens_relative_path"]),
                "lens_sha256": str(replacement["lens_sha256"]),
                "shared_lineage_id": (
                    f"{manifest['plan_id']}::{str(replacement['lens_sha256'])[:16]}::"
                    "dcc6_head0"
                ),
                "replacement_for_atom_id": source_atom_id,
                "replacement_rank": int(replacement["replacement_rank"]),
                "replacement_selection_policy": str(
                    replacement["selection_policy"]
                ),
                "negative_evidence_root": str(
                    Path(replacement["negative_evidence_root"]).resolve()
                ),
                "planned_outputs": {
                    "raw_family_root": str((replacement_root / "raw_family").resolve()),
                    "raw_optical_root": str((replacement_root / "raw_optical").resolve()),
                    "k64_audit_root": str((replacement_root / "audit_k64").resolve()),
                    "k128_audit_root": str((replacement_root / "audit_k128").resolve()),
                    "final_asset_root": str(
                        (replacement_root / "paircommon_dcc6").resolve()
                    ),
                },
            }
        )
    if len({str(atom["atom_id"]) for atom in active_atoms}) != len(active_atoms):
        raise RuntimeError("replacement 后 atom_id 不唯一")
    return manifest, active_atoms


def _fixed_range(value: float) -> list[float]:
    return [float(value), float(value)]


def _profile_ranges(atom: dict[str, Any]) -> dict[str, list[float]]:
    response = atom["response_profile"]
    aberration = response["low_order_aberration_waves"]
    values = {
        "transition_width": response["transition_width"],
        "cross_talk": response["cross_talk"],
        "split_bias_norm": response["split_bias_norm"],
        "boundary_curvature_norm": response["boundary_curvature_norm"],
        "microlens_edge_rolloff": response["microlens_edge_rolloff"],
        "field_split_slope": response["field_split_slope"],
        "field_acceptance_slope": response["field_acceptance_slope"],
        "lr_throughput_delta": response["lr_throughput_delta"],
        "field_lr_throughput_slope": response["field_lr_throughput_slope"],
        "aperture_lr_throughput_slope": response[
            "aperture_lr_throughput_slope"
        ],
        "astigmatism_0_waves": aberration["astigmatism_0"],
        "astigmatism_45_waves": aberration["astigmatism_45"],
        "coma_x_waves": aberration["coma_x"],
        "coma_y_waves": aberration["coma_y"],
        "spherical_waves": aberration["spherical"],
        "field_astigmatism_0_waves_per_norm": aberration[
            "field_astigmatism_0_per_norm"
        ],
        "field_astigmatism_45_waves_per_norm": aberration[
            "field_astigmatism_45_per_norm"
        ],
        "field_coma_x_waves_per_norm": aberration["field_coma_x_per_norm"],
        "field_coma_y_waves_per_norm": aberration["field_coma_y_per_norm"],
        "centroid_slope_px_per_coc": atom["dcc_px_per_coc"],
    }
    return {key: _fixed_range(float(value)) for key, value in values.items()}


def build_family_config(
    config: dict[str, Any], atom: dict[str, Any]
) -> dict[str, Any]:
    propagation = config["propagation"]
    outputs = atom["planned_outputs"]
    f_numbers = [float(value) for value in atom["f_numbers"]]
    envelope_by_atom = dict(
        propagation.get("sensor_offset_envelope_m_by_atom", {})
    )
    sensor_offset_envelope_m = float(
        envelope_by_atom.get(
            str(atom["atom_id"]), propagation["sensor_offset_envelope_m"]
        )
    )
    if sensor_offset_envelope_m <= 0.0:
        raise ValueError("sensor offset envelope 必须为正数")
    return {
        "schema_version": 1,
        "run": {
            "id": f"{config['run']['id']}__{atom['atom_id']}__raw_family",
            "purpose": "单镜头处方、五光圈共享 lineage 的正式 raw PSF propagation",
        },
        "source": {
            "repository": config["cldefocus"]["repository"],
            "checkout": config["cldefocus"]["checkout"],
            "commit": config["cldefocus"]["commit"],
            "license": config["cldefocus"]["license"],
        },
        "lens": {
            "root": config["lens"]["root"],
            "relative_path": atom["lens_relative_path"],
            "sha256": atom["lens_sha256"],
        },
        "sensor": {
            "width_px": int(propagation["sensor_width_px"]),
            "height_px": int(propagation["sensor_height_px"]),
            "pixel_pitch_m": float(propagation["pixel_pitch_m"]),
        },
        "apertures": {"f_numbers": f_numbers},
        "optics": {
            "wavelength_m": float(propagation["wavelength_m"]),
            "object_depth_m": float(propagation["object_depth_m"]),
            "field_x_normalized": [
                float(value) for value in atom["field_x_normalized"]
            ],
            "field_y_normalized": [
                float(value) for value in atom["field_y_normalized"]
            ],
            "target_signed_coc_bins_px": [
                float(value) for value in atom["signed_coc_bins_px"]
            ],
            "sensor_offset_envelope_m": sensor_offset_envelope_m,
            "target_coc_tolerance_px": float(
                propagation["target_coc_tolerance_px"]
            ),
            "reference_kernel_size": int(propagation["reference_kernel_size"]),
            "export_kernel_size": int(propagation["export_kernel_size"]),
            "upsample": int(propagation["upsample"]),
        },
        "family": {
            "family_id": atom["shared_lineage_id"],
            "asset_id": f"{atom['atom_id']}_raw_family_source",
            "batch_sensor_chunk_size": int(
                propagation["batch_sensor_chunk_size"]
            ),
            "batch_profile_chunk_size": 1,
            "centroid_support_padding_px": 0,
            "export_pdraw_mono_v1": False,
            "aperture_centroid_scale_by_f_number": {
                f"{value:g}": 1.0 for value in f_numbers
            },
            "randomization": {
                "master_seed": int(propagation["master_seed"]),
                "profile_count": 1,
                "profile_id_prefix": str(propagation["profile_id_prefix"]),
                "ranges": _profile_ranges(atom),
            },
        },
        "pdoffset": {
            "sample_range_px": [
                float(value) for value in propagation["pdoffset_sample_range_px"]
            ],
            "smoke_probe_px": float(propagation["pdoffset_smoke_probe_px"]),
        },
        "gates": {
            **dict(config["raw_family_gates"]),
            "expected_profile_count": 1,
            "expected_aperture_count": len(f_numbers),
            "expected_coc_count": len(atom["signed_coc_bins_px"]),
            "expected_field_grid_hw": [
                len(atom["field_y_normalized"]),
                len(atom["field_x_normalized"]),
            ],
        },
        "output": {"root": outputs["raw_family_root"]},
        "data_isolation": dict(config["data_isolation"]),
    }


def build_raw_optical_config(
    config: dict[str, Any],
    atom: dict[str, Any],
    *,
    family_manifest_sha256: str,
    family_bank_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run": {
            "id": f"{config['run']['id']}__{atom['atom_id']}__raw_optical",
            "asset_id": f"{atom['atom_id']}_raw_optical",
        },
        "source": {
            "asset_root": atom["planned_outputs"]["raw_family_root"],
            "artifact_manifest_sha256": family_manifest_sha256,
            "psf_bank_sha256": family_bank_sha256,
            "parent_repository": config["parent"]["repository"],
            "parent_loader_sha256": config["parent"]["loader_sha256"],
            "parent_renderer_sha256": config["parent"]["renderer_sha256"],
        },
        "output": {"root": atom["planned_outputs"]["raw_optical_root"]},
        "gates": {
            "analytic_agreement_abs_max_px": float(
                config["candidate_gates"]["parent_centroid_abs_max_error_px"]
            ),
            "energy_abs_max": float(config["raw_family_gates"]["energy_abs_max"]),
        },
        "data_isolation": dict(config["data_isolation"]),
    }


def _candidate_operations(atom: dict[str, Any]) -> list[dict[str, Any]]:
    pair_common = atom["pair_common"]
    slopes = [float(atom["dcc_px_per_coc"])] * len(atom["f_numbers"])
    return [
        {
            "kind": "dcc_retarget",
            "target_slopes_by_aperture": slopes,
            "support_padding_px": 0,
            "anchor_mode": "preserve_native_coc0",
        },
        {
            "kind": "pair_common",
            "alignment": pair_common["alignment"],
            "translation_mode": "bilinear_center_fourier_place",
            "common_shape_mtf_sigma_px": float(
                pair_common["common_shape_mtf_sigma_px"]
            ),
            "clipped_negative_fraction_max": float(
                pair_common["clipped_negative_fraction_max"]
            ),
            "transition_coc_px": float(pair_common["transition_coc_px"]),
            "transition_power": float(pair_common["transition_power"]),
        },
    ]


def build_audit_config(
    config: dict[str, Any],
    atom: dict[str, Any],
    *,
    raw_manifest_sha256: str,
    tile_size: int,
) -> dict[str, Any]:
    audit = config["audit"]
    texture_size = int(audit["texture_size_by_tile"][str(int(tile_size))])
    continuous = {
        "protocol_version": "continuous_lanczos_v2",
        "interpolation": "separable_lanczos_windowed_sinc_no_lookup_table",
        "lanczos_radius": int(audit["lanczos_radius"]),
        "refinement_radius_px": float(audit["refinement_radius_px"]),
        "coordinate_iterations": int(audit["coordinate_iterations"]),
        "optimizer_xatol_px": float(audit["optimizer_xatol_px"]),
        "min_texture_std": float(audit["min_texture_std"]),
        "min_score": float(audit["min_score"]),
        "max_lr_error_px": float(audit["max_lr_error_px"]),
        "max_vertical_shift_px": float(audit["max_vertical_shift_px"]),
        "peak_exclusion_x": int(audit["peak_exclusion_x"]),
        "peak_exclusion_y": int(audit["peak_exclusion_y"]),
        "final_score_tolerance": float(audit["final_score_tolerance"]),
        "per_cell_valid_tile_fraction_min": 1.0,
        "expected_aperture_count": len(atom["f_numbers"]),
        "worker_count": int(audit["worker_count"]),
        "uses_ground_truth": False,
        "applies_posthoc_label_correction": False,
    }
    stage = {
        "field_mode": "full_flattened",
        "coc_abs_max_px": float(audit["coc_abs_max_px"]),
        "texture_size": texture_size,
        "seeds": [int(value) for value in audit["seeds"]],
        "tile_size": int(tile_size),
        "tiles_per_axis": int(audit["tiles_per_axis"]),
        "search_x": int(audit["search_x"]),
        "search_y": int(audit["search_y"]),
        "continuous_ncc": continuous,
    }
    validation = {
        key: value for key, value in stage.items() if key != "tile_size"
    }
    validation["tile_sizes"] = [int(tile_size)]
    return {
        "schema_version": 1,
        "run": {
            "id": f"{config['run']['id']}__{atom['atom_id']}__K{tile_size}",
            "purpose": f"单 device atom 的 3x3 全场 K{tile_size} continuous NCC 门禁",
        },
        "source": {
            "asset_root": atom["planned_outputs"]["raw_optical_root"],
            "artifact_manifest_sha256": raw_manifest_sha256,
            "profile_ids": [f"{config['propagation']['profile_id_prefix']}000"],
        },
        "candidates": [
            {"id": _CANDIDATE_ID, "operations": _candidate_operations(atom)}
        ],
        "screen": stage,
        "validation": validation,
        "gates": dict(audit["gates"]),
        "selection": {
            "minimum_screen_profiles": 1,
            "max_candidates_for_validation": 0,
            "max_profiles_for_validation": 1,
            "minimum_validation_profiles": 1,
        },
        "export": {
            "enabled": False,
            "asset_id": f"{atom['atom_id']}_no_export_K{tile_size}",
            "centroid_variant": f"{atom['atom_id']}_no_export_K{tile_size}",
            "asset_root": str(
                Path(atom["planned_outputs"]["final_asset_root"]).with_name(
                    f"no_export_K{tile_size}"
                )
            ),
        },
        "output": {
            "root": atom["planned_outputs"][
                "k64_audit_root" if int(tile_size) == 64 else "k128_audit_root"
            ]
        },
        "data_isolation": dict(config["data_isolation"]),
    }


def build_candidate_config(
    config: dict[str, Any],
    atom: dict[str, Any],
    *,
    raw_manifest_sha256: str,
    audit_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if [int(row["tile_size"]) for row in audit_rows] != [64, 128]:
        raise ValueError("candidate audit_rows 必须按 K64/K128 顺序提供")
    return {
        "schema_version": 1,
        "run": {
            "id": f"{config['run']['id']}__{atom['atom_id']}__candidate_export",
            "asset_id": f"{atom['atom_id']}_paircommon_tc16_dcc6_contncc_v1",
            "centroid_variant": "multi_prescription_paircommon_direct_s1_tc16_dcc6_contncc_v1",
        },
        "source": {
            "asset_root": atom["planned_outputs"]["raw_optical_root"],
            "artifact_manifest_sha256": raw_manifest_sha256,
            "profile_ids": [f"{config['propagation']['profile_id_prefix']}000"],
        },
        "audits": [dict(row) for row in audit_rows],
        "candidate": {
            "id": _CANDIDATE_ID,
            "operations": _candidate_operations(atom),
        },
        "parent": {
            "repository": config["parent"]["repository"],
            "loader_sha256": config["parent"]["loader_sha256"],
            "renderer_sha256": config["parent"]["renderer_sha256"],
        },
        "gates": dict(config["candidate_gates"]),
        "output": {"root": atom["planned_outputs"]["final_asset_root"]},
        "data_isolation": dict(config["data_isolation"]),
    }


def _write_frozen_yaml(path: Path, value: dict[str, Any]) -> str:
    rendered = yaml.safe_dump(value, allow_unicode=True, sort_keys=False)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"已存在子配置与本次解析结果不同：{path}")
    else:
        _atomic_write_text(path, rendered)
    return _sha256_file(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON 顶层不是 mapping：{path}")
    return value


def _verified_manifest_stage(root: Path) -> dict[str, Any] | None:
    manifest_path = root / "artifact_manifest.json"
    summary_path = root / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        return None
    manifest = _load_json(manifest_path)
    summary = _load_json(summary_path)
    if manifest.get("status") != "pass" or summary.get("status") != "pass":
        return None
    for relative, expected_sha in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file() or _sha256_file(path) != str(expected_sha):
            raise RuntimeError(f"stage manifest 文件 SHA256 不匹配：{path}")
    return {
        "status": "pass",
        "root": str(root),
        "artifact_manifest_sha256": _sha256_file(manifest_path),
        "summary_sha256": _sha256_file(summary_path),
        "summary": summary,
        "manifest": manifest,
    }


def _verified_audit_stage(
    root: Path, *, tile_size: int, expected_aperture_count: int = 5
) -> dict[str, Any] | None:
    required = [root / name for name in ("summary.json", "provenance.json", "screens.json")]
    if not all(path.is_file() for path in required):
        return None
    summary, provenance = (_load_json(required[index]) for index in (0, 1))
    screens = json.loads(required[2].read_text(encoding="utf-8"))
    matching = [
        row
        for row in screens
        if str(row.get("candidate_id")) == _CANDIDATE_ID
        and int(row.get("tile_size", -1)) == int(tile_size)
    ]
    if len(matching) != 1:
        raise RuntimeError(f"K{tile_size} audit 缺少唯一冻结候选")
    screen = matching[0]
    profiles = list(screen.get("profiles", []))
    if len(profiles) != 1 or not all(
        bool(row.get("all_apertures_pass"))
        and bool(row.get("all_aperture_fields_pass"))
        for row in profiles
    ):
        return None
    fields = [
        field
        for profile in profiles
        for aperture in profile["per_aperture"]
        for field in aperture["fields"]
    ]
    expected_fields = int(expected_aperture_count) * 9
    if len(fields) != expected_fields or not all(
        bool(field.get("pass")) for field in fields
    ):
        return None
    return {
        "status": "pass",
        "root": str(root),
        "tile_size": int(tile_size),
        "summary_sha256": _sha256_file(required[0]),
        "provenance_sha256": _sha256_file(required[1]),
        "screens_sha256": _sha256_file(required[2]),
        "field_count": len(fields),
        "slope_min": min(float(field["fit"]["slope_ncc_vs_analytic"]) for field in fields),
        "slope_max": max(float(field["fit"]["slope_ncc_vs_analytic"]) for field in fields),
        "r2_min": min(float(field["fit"]["r2"]) for field in fields),
        "summary_status": summary.get("status"),
        "config_sha256": provenance.get("config_sha256"),
    }


def inspect_atom(atom: dict[str, Any]) -> dict[str, Any]:
    outputs = atom["planned_outputs"]
    raw_family = _verified_manifest_stage(Path(outputs["raw_family_root"]))
    raw_optical = _verified_manifest_stage(Path(outputs["raw_optical_root"]))
    k64 = _verified_audit_stage(Path(outputs["k64_audit_root"]), tile_size=64)
    k128 = _verified_audit_stage(Path(outputs["k128_audit_root"]), tile_size=128)
    final = _verified_manifest_stage(Path(outputs["final_asset_root"]))
    admitted = all(value is not None for value in (raw_family, raw_optical, k64, k128, final))
    return {
        "atom_id": atom["atom_id"],
        "lens_id": atom["lens_id"],
        "lens_sha256": atom["lens_sha256"],
        "shared_lineage_id": atom["shared_lineage_id"],
        "shared_lineage_across_apertures": atom["shared_lineage_across_apertures"],
        "pdoffset_embedded": False,
        "stages": {
            "raw_family": raw_family,
            "raw_optical": raw_optical,
            "continuous_ncc_k64": k64,
            "continuous_ncc_k128": k128,
            "pair_common_dcc6_parent_smoke": final,
        },
        "training_admitted": admitted,
    }


def _write_aggregate(
    config_path: Path,
    config: dict[str, Any],
    plan_manifest: dict[str, Any],
    atoms: Sequence[dict[str, Any]],
    *,
    last_error: str | None = None,
) -> dict[str, Any]:
    root = Path(config["output"]["root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    atom_rows = [inspect_atom(atom) for atom in atoms]
    admitted = bool(atom_rows) and all(row["training_admitted"] for row in atom_rows)
    status = "pass" if admitted else ("no_go" if last_error else "in_progress_not_admitted")
    summary = {
        "schema_version": 1,
        "status": status,
        "run_id": config["run"]["id"],
        "device_atom_count": len(atom_rows),
        "device_atom_pass_count": sum(row["training_admitted"] for row in atom_rows),
        "training_admitted": admitted,
        "gpu_propagation_started": any(
            Path(atom["planned_outputs"]["raw_family_root"]).exists()
            for atom in atoms
        ),
        "pdoffset_embedded": False,
        "atoms": atom_rows,
        "last_error": last_error,
        **config["data_isolation"],
    }
    resolved_path = root / "resolved_execution_config.yaml"
    _atomic_write_text(
        resolved_path,
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
    )
    summary_path = root / "summary.json"
    _atomic_write_text(
        summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    provenance = {
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256_file(config_path),
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": _sha256_file(Path(__file__).resolve()),
        "source_plan_root": str(Path(config["source_plan"]["root"]).resolve()),
        "source_plan_artifact_manifest_sha256": config["source_plan"][
            "artifact_manifest_sha256"
        ],
        "source_plan_id": plan_manifest["plan_id"],
        "source_commit": _git_output(
            Path(config["cldefocus"]["checkout"]).resolve(), "rev-parse", "HEAD"
        ),
        "source_dirty_paths": _git_output(
            Path(config["cldefocus"]["checkout"]).resolve(), "status", "--short"
        ).splitlines(),
        "python_executable": sys.executable,
        **config["data_isolation"],
    }
    provenance_path = root / "provenance.json"
    _atomic_write_text(
        provenance_path, json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
    generated_configs = {
        str(path.relative_to(root)): _sha256_file(path)
        for path in sorted(root.glob("*/generated_configs/*.yaml"))
    }
    manifest = {
        "schema_version": 1,
        "status": status,
        "run_id": config["run"]["id"],
        "device_atom_count": len(atom_rows),
        "device_atom_pass_count": summary["device_atom_pass_count"],
        "shared_aperture_lineage_required": True,
        "training_admitted": admitted,
        "pdoffset_embedded": False,
        "source_plan_artifact_manifest_sha256": config["source_plan"][
            "artifact_manifest_sha256"
        ],
        "atom_final_manifest_sha256": {
            row["atom_id"]: (
                None
                if row["stages"]["pair_common_dcc6_parent_smoke"] is None
                else row["stages"]["pair_common_dcc6_parent_smoke"][
                    "artifact_manifest_sha256"
                ]
            )
            for row in atom_rows
        },
        "files": {
            "resolved_execution_config.yaml": _sha256_file(resolved_path),
            "summary.json": _sha256_file(summary_path),
            "provenance.json": _sha256_file(provenance_path),
            **generated_configs,
        },
        **config["data_isolation"],
    }
    manifest_path = root / "artifact_manifest.json"
    _atomic_write_text(
        manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return {
        **summary,
        "output_root": str(root),
        "artifact_manifest_sha256": _sha256_file(manifest_path),
        "summary_sha256": _sha256_file(summary_path),
    }


def _stage_config_path(atom: dict[str, Any], name: str) -> Path:
    atom_root = Path(atom["planned_outputs"]["raw_family_root"]).parent
    return atom_root / "generated_configs" / f"{name}.yaml"


def _screen_atom(atom: dict[str, Any]) -> dict[str, Any]:
    atom_root = Path(atom["planned_outputs"]["raw_family_root"]).parent
    screen_root = atom_root / "replacement_screen45"
    return {
        **atom,
        "f_numbers": [1.8],
        "signed_coc_bins_px": [-1.0, -0.5, 0.0, 0.5, 1.0],
        "planned_outputs": {
            "raw_family_root": str((screen_root / "raw_family").resolve()),
            "raw_optical_root": str((screen_root / "raw_optical").resolve()),
            "k64_audit_root": str((screen_root / "audit_k64").resolve()),
            "k128_audit_root": str((screen_root / "unused_audit_k128").resolve()),
            "final_asset_root": str((screen_root / "unused_final").resolve()),
        },
    }


def run_replacement_screen45(
    config_path: Path,
    *,
    atom_id: str,
) -> dict[str, Any]:
    """运行 1 aperture × 5 CoC × 3×3 field 的替补镜头 K64 screen。"""

    config_path = config_path.resolve()
    config = load_config(config_path)
    _verify_implementation(config)
    _, atoms = load_verified_plan(config)
    matching = [atom for atom in atoms if str(atom["atom_id"]) == str(atom_id)]
    if len(matching) != 1:
        raise ValueError(f"replacement screen 要求唯一 atom_id：{atom_id}")
    atom = matching[0]
    if "replacement_for_atom_id" not in atom:
        raise ValueError("45-cell screen 只允许冻结 replacement atom")
    screen_atom = _screen_atom(atom)
    config_root = (
        Path(screen_atom["planned_outputs"]["raw_family_root"]).parent
        / "generated_configs"
    )
    family_config = build_family_config(config, screen_atom)
    family_path = config_root / "01_screen45_raw_family.yaml"
    _write_frozen_yaml(family_path, family_config)

    family_stage = _verified_manifest_stage(
        Path(screen_atom["planned_outputs"]["raw_family_root"])
    )
    if family_stage is None:
        from .family_bank import generate

        result = generate(family_path)
        if result.get("status") != "pass":
            raise RuntimeError("replacement screen raw-family gate 失败")
        family_stage = _verified_manifest_stage(
            Path(screen_atom["planned_outputs"]["raw_family_root"])
        )
    assert family_stage is not None

    raw_config = build_raw_optical_config(
        config,
        screen_atom,
        family_manifest_sha256=family_stage["artifact_manifest_sha256"],
        family_bank_sha256=family_stage["manifest"]["files"]["psf_bank.pt"],
    )
    raw_path = config_root / "02_screen45_raw_optical.yaml"
    _write_frozen_yaml(raw_path, raw_config)
    raw_stage = _verified_manifest_stage(
        Path(screen_atom["planned_outputs"]["raw_optical_root"])
    )
    if raw_stage is None:
        from .raw_optical_export import export_raw_optical

        export_raw_optical(raw_path)
        raw_stage = _verified_manifest_stage(
            Path(screen_atom["planned_outputs"]["raw_optical_root"])
        )
    assert raw_stage is not None

    audit_config = build_audit_config(
        config,
        screen_atom,
        raw_manifest_sha256=raw_stage["artifact_manifest_sha256"],
        tile_size=64,
    )
    audit_path = config_root / "03_screen45_audit_k64.yaml"
    _write_frozen_yaml(audit_path, audit_config)
    audit_stage = _verified_audit_stage(
        Path(screen_atom["planned_outputs"]["k64_audit_root"]),
        tile_size=64,
        expected_aperture_count=1,
    )
    if audit_stage is None:
        from .raw_family_continuous_audit import run as run_audit

        run_audit(audit_path)
        audit_stage = _verified_audit_stage(
            Path(screen_atom["planned_outputs"]["k64_audit_root"]),
            tile_size=64,
            expected_aperture_count=1,
        )
    if audit_stage is None:
        raise RuntimeError("replacement 45-cell K64 screen 失败")
    return {
        "status": "pass",
        "atom_id": atom["atom_id"],
        "replacement_for_atom_id": atom["replacement_for_atom_id"],
        "cell_count": 45,
        "raw_family": {
            key: family_stage[key]
            for key in ("root", "artifact_manifest_sha256", "summary_sha256")
        },
        "raw_optical": {
            key: raw_stage[key]
            for key in ("root", "artifact_manifest_sha256", "summary_sha256")
        },
        "continuous_ncc_k64": audit_stage,
        "training_admitted": False,
        "pdoffset_embedded": False,
        **config["data_isolation"],
    }


def _audit_export_row(stage: dict[str, Any]) -> dict[str, Any]:
    return {
        "tile_size": int(stage["tile_size"]),
        "root": stage["root"],
        "summary_sha256": stage["summary_sha256"],
        "provenance_sha256": stage["provenance_sha256"],
        "screens_sha256": stage["screens_sha256"],
    }


def run(
    config_path: Path,
    *,
    atom_ids: Sequence[str] | None = None,
    prepare_only: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = config_path.resolve()
    config = load_config(config_path)
    _verify_implementation(config)
    plan_manifest, all_atoms = load_verified_plan(config)
    selected_ids = set(atom_ids or [str(atom["atom_id"]) for atom in all_atoms])
    unknown = selected_ids - {str(atom["atom_id"]) for atom in all_atoms}
    if unknown:
        raise ValueError(f"未知 atom_id：{sorted(unknown)}")
    selected_atoms = [atom for atom in all_atoms if atom["atom_id"] in selected_ids]

    try:
        for atom in selected_atoms:
            family_config = build_family_config(config, atom)
            family_config_path = _stage_config_path(atom, "01_raw_family")
            _write_frozen_yaml(family_config_path, family_config)
            if prepare_only:
                continue

            current = inspect_atom(atom)
            family_stage = current["stages"]["raw_family"]
            if family_stage is None:
                from .family_bank import generate

                result = generate(family_config_path)
                if result.get("status") != "pass":
                    raise RuntimeError(
                        f"raw family gate 失败：{atom['atom_id']} status={result.get('status')}"
                    )
                family_stage = inspect_atom(atom)["stages"]["raw_family"]
            assert family_stage is not None

            raw_config = build_raw_optical_config(
                config,
                atom,
                family_manifest_sha256=family_stage["artifact_manifest_sha256"],
                family_bank_sha256=family_stage["manifest"]["files"]["psf_bank.pt"],
            )
            raw_config_path = _stage_config_path(atom, "02_raw_optical")
            _write_frozen_yaml(raw_config_path, raw_config)
            raw_stage = inspect_atom(atom)["stages"]["raw_optical"]
            if raw_stage is None:
                from .raw_optical_export import export_raw_optical

                result = export_raw_optical(raw_config_path)
                if result.get("status") != "pass":
                    raise RuntimeError(f"raw-optical gate 失败：{atom['atom_id']}")
                raw_stage = inspect_atom(atom)["stages"]["raw_optical"]
            assert raw_stage is not None

            audit_stages: list[dict[str, Any]] = []
            for tile_size, key, name in (
                (64, "continuous_ncc_k64", "03_audit_k64"),
                (128, "continuous_ncc_k128", "04_audit_k128"),
            ):
                audit_config = build_audit_config(
                    config,
                    atom,
                    raw_manifest_sha256=raw_stage["artifact_manifest_sha256"],
                    tile_size=tile_size,
                )
                audit_config_path = _stage_config_path(atom, name)
                _write_frozen_yaml(audit_config_path, audit_config)
                audit_stage = inspect_atom(atom)["stages"][key]
                if audit_stage is None:
                    from .raw_family_continuous_audit import run as run_audit

                    run_audit(audit_config_path)
                    audit_stage = inspect_atom(atom)["stages"][key]
                if audit_stage is None:
                    raise RuntimeError(
                        f"K{tile_size} full-field continuous NCC 失败：{atom['atom_id']}"
                    )
                audit_stages.append(audit_stage)

            candidate_config = build_candidate_config(
                config,
                atom,
                raw_manifest_sha256=raw_stage["artifact_manifest_sha256"],
                audit_rows=[_audit_export_row(row) for row in audit_stages],
            )
            candidate_config_path = _stage_config_path(atom, "05_candidate_export")
            _write_frozen_yaml(candidate_config_path, candidate_config)
            final_stage = inspect_atom(atom)["stages"][
                "pair_common_dcc6_parent_smoke"
            ]
            if final_stage is None:
                from .raw_family_candidate_export import export

                result = export(candidate_config_path)
                if result.get("status") != "pass":
                    raise RuntimeError(f"candidate export 失败：{atom['atom_id']}")
                final_stage = inspect_atom(atom)["stages"][
                    "pair_common_dcc6_parent_smoke"
                ]
            if final_stage is None:
                raise RuntimeError(f"parent load/render smoke 失败：{atom['atom_id']}")

            _write_aggregate(config_path, config, plan_manifest, all_atoms)
    except BaseException as error:
        _write_aggregate(
            config_path,
            config,
            plan_manifest,
            all_atoms,
            last_error=f"{type(error).__name__}: {error}",
        )
        raise

    result = _write_aggregate(config_path, config, plan_manifest, all_atoms)
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--atom-id", action="append", default=None)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只校验 plan/实现并冻结 raw-family 子配置，不执行 GPU propagation",
    )
    parser.add_argument(
        "--replacement-screen45-only",
        action="store_true",
        help="只运行一个 replacement atom 的 45-cell K64 screen",
    )
    args = parser.parse_args(argv)
    if args.replacement_screen45_only:
        if args.prepare_only or not args.atom_id or len(args.atom_id) != 1:
            parser.error("replacement screen 要求精确一个 --atom-id 且不得同时 prepare-only")
        result = run_replacement_screen45(args.config, atom_id=args.atom_id[0])
    else:
        result = run(
            args.config,
            atom_ids=args.atom_id,
            prepare_only=bool(args.prepare_only),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
