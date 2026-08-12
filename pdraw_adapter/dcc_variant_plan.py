"""从已冻结 raw-optical atom 准备 DCC5/DCC7 低成本候选分叉。

本模块只在 CPU 上重放确定性的 DCC retarget + PairCommon 变换，并生成后续
K64/K128 continuous NCC 的冻结 YAML。它不运行 NCC、不导出正式 variant asset、
不访问真实数据，也不把 NCC 结果写回 kernel 或解析标签。
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

import numpy as np
import torch
import yaml

from .family_bank import analytic_centroid_labels
from .raw_family_continuous_audit import transform_candidate


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


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "run",
        "sources",
        "variants",
        "pair_common",
        "audit",
        "parent",
        "gates",
        "output",
        "data_isolation",
    }
    if not isinstance(config, dict) or int(config.get("schema_version", 0)) != 1:
        raise ValueError("DCC variant 配置必须是 schema_version: 1 mapping")
    if set(config) != required:
        raise ValueError(f"DCC variant 配置字段必须精确为 {sorted(required)}")
    if str(config["run"].get("mode")) != "cpu_prepare_only":
        raise ValueError("当前 DCC variant planner 只允许 cpu_prepare_only")
    isolation = config["data_isolation"]
    if not isinstance(isolation, dict) or set(isolation) != _ISOLATION_KEYS:
        raise ValueError(f"data_isolation 必须精确为 {sorted(_ISOLATION_KEYS)}")
    if any(bool(value) for value in isolation.values()):
        raise ValueError("DCC variant CPU plan 禁止访问真实数据或启动训练")
    variants = list(config["variants"])
    if [str(row["id"]) for row in variants] != ["dcc5", "dcc7"]:
        raise ValueError("variants 必须按顺序精确冻结 dcc5、dcc7")
    if [float(row["slope_px_per_coc"]) for row in variants] != [0.2, 1.0 / 7.0]:
        raise ValueError("DCC5/DCC7 slope 必须精确为 1/5、1/7")
    if sorted(int(key) for key in config["audit"]["texture_size_by_tile"]) != [64, 128]:
        raise ValueError("audit 必须精确覆盖 K64/K128")
    return config


def candidate_operations(
    *, slope_px_per_coc: float, aperture_count: int, pair_common: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "kind": "dcc_retarget",
            "target_slopes_by_aperture": [float(slope_px_per_coc)]
            * int(aperture_count),
            "support_padding_px": 0,
            "anchor_mode": "preserve_native_coc0",
        },
        {
            "kind": "pair_common",
            "alignment": "direct_mean",
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


def _verified_payload(
    *, root: Path, manifest_sha256: str, profile_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = root / "artifact_manifest.json"
    profile_path = root / "profiles/H000/psf_bank.pt"
    if _sha256_file(manifest_path) != str(manifest_sha256):
        raise RuntimeError(f"asset manifest SHA256 不匹配：{root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass" or manifest.get("pdoffset_embedded", False):
        raise RuntimeError(f"asset 状态/PDOFFSET 合同不匹配：{root}")
    relative = "profiles/H000/psf_bank.pt"
    if (
        _sha256_file(profile_path) != str(profile_sha256)
        or manifest.get("files", {}).get(relative) != str(profile_sha256)
    ):
        raise RuntimeError(f"profile SHA256 不匹配：{profile_path}")
    payload = torch.load(profile_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or bool(payload.get("pdoffset_embedded", True)):
        raise RuntimeError(f"profile payload/PDOFFSET 合同不匹配：{profile_path}")
    bank = np.asarray(torch.as_tensor(payload["psf_bank"]), dtype=np.float32)
    labels = np.asarray(
        torch.as_tensor(payload["analytic_disparity_bins_px"]), dtype=np.float64
    )
    recomputed = analytic_centroid_labels(bank)
    if not np.array_equal(labels, recomputed):
        raise RuntimeError(f"profile label 不是当前 kernel 的精确 centroid：{profile_path}")
    return payload, manifest


def _candidate(source: dict[str, Any], variant: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"{source['atom_id']}__{variant['id']}__paircommon_tc16",
        "operations": candidate_operations(
            slope_px_per_coc=float(variant["slope_px_per_coc"]),
            aperture_count=5,
            pair_common=config["pair_common"],
        ),
    }


def _audit_config(
    config: dict[str, Any],
    source: dict[str, Any],
    variant: dict[str, Any],
    *,
    tile_size: int,
) -> dict[str, Any]:
    audit = config["audit"]
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
        "expected_aperture_count": 5,
        "worker_count": int(audit["worker_count"]),
        "uses_ground_truth": False,
        "applies_posthoc_label_correction": False,
    }
    stage = {
        "field_mode": "full_flattened",
        "coc_abs_max_px": 1.0,
        "texture_size": int(audit["texture_size_by_tile"][str(tile_size)]),
        "seeds": [int(value) for value in audit["seeds"]],
        "tile_size": int(tile_size),
        "tiles_per_axis": 1,
        "search_x": 4,
        "search_y": 2,
        "continuous_ncc": continuous,
    }
    validation = {key: value for key, value in stage.items() if key != "tile_size"}
    validation["tile_sizes"] = [int(tile_size)]
    formal_root = Path(config["output"]["formal_root"]).resolve()
    candidate = _candidate(source, variant, config)
    output_root = formal_root / source["atom_id"] / variant["id"] / f"audit_k{tile_size}"
    return {
        "schema_version": 1,
        "run": {
            "id": f"{config['run']['id']}__{source['atom_id']}__{variant['id']}__k{tile_size}",
            "purpose": "DCC5/DCC7 raw-optical 分叉的冻结 full-field continuous NCC",
        },
        "source": {
            "asset_root": source["raw_optical_root"],
            "artifact_manifest_sha256": source["raw_manifest_sha256"],
            "profile_ids": ["H000"],
        },
        "candidates": [candidate],
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
            "asset_id": f"{candidate['id']}__no_export",
            "centroid_variant": f"{candidate['id']}__no_export",
            "asset_root": str(output_root.with_name("no_export")),
        },
        "output": {"root": str(output_root)},
        "data_isolation": dict(config["data_isolation"]),
    }


def _validate_variant(
    raw_payload: dict[str, Any],
    *,
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw = np.asarray(torch.as_tensor(raw_payload["psf_bank"]), dtype=np.float32)
    coc = np.asarray(
        torch.as_tensor(raw_payload["signed_coc_bins_px"]), dtype=np.float64
    )
    output, labels, operations = transform_candidate(
        raw,
        coc,
        candidate,
        centroid_tolerance_px=float(config["gates"]["centroid_abs_max_error_px"]),
        retained_mass_min=float(config["gates"]["retained_mass_min"]),
    )
    recomputed = analytic_centroid_labels(output)
    if not np.array_equal(labels, recomputed):
        raise RuntimeError("variant label 未严格来自最终 kernel centroid")
    energy_error = float(
        np.max(
            np.abs(
                output.sum(axis=(-2, -1), dtype=np.float64)
                - raw.sum(axis=(-2, -1), dtype=np.float64)
            )
        )
    )
    if energy_error > float(config["gates"]["energy_abs_max"]):
        raise RuntimeError(f"variant energy gate 失败：{energy_error}")
    dcc = operations[0]
    pair_common = operations[1]
    return output, labels, {
        "shape": list(output.shape),
        "centroid_contract_abs_max_px": float(
            dcc["analytic_target_error_abs_max_px"]
        ),
        "retained_mass_min": float(dcc["retained_mass_min"]),
        "pair_common_centroid_abs_max_error_px": float(
            pair_common["centroid_abs_max_error_px"]
        ),
        "fourier_clipped_negative_fraction_max": float(
            pair_common["fourier_clipped_negative_fraction_max"]
        ),
        "energy_abs_max": energy_error,
        "analytic_label_sha256": hashlib.sha256(
            np.ascontiguousarray(labels, dtype=np.float64).tobytes()
        ).hexdigest(),
        "pdoffset_embedded": False,
    }


def prepare(config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = config_path.resolve()
    config = load_config(config_path)
    output_root = Path(config["output"]["plan_root"]).resolve()
    if output_root.exists():
        raise FileExistsError(f"DCC variant CPU plan 已存在：{output_root}")
    output_root.mkdir(parents=True)

    source_rows: list[dict[str, Any]] = []
    generated: dict[str, str] = {}
    for source in config["sources"]:
        raw_payload, _ = _verified_payload(
            root=Path(source["raw_optical_root"]).resolve(),
            manifest_sha256=source["raw_manifest_sha256"],
            profile_sha256=source["raw_profile_sha256"],
        )
        dcc6_payload, _ = _verified_payload(
            root=Path(source["dcc6_reference_root"]).resolve(),
            manifest_sha256=source["dcc6_manifest_sha256"],
            profile_sha256=source["dcc6_profile_sha256"],
        )
        aperture_count = int(torch.as_tensor(raw_payload["f_numbers"]).numel())
        if aperture_count != 5:
            raise RuntimeError(f"source aperture_count 不是 5：{source['atom_id']}")

        reference_candidate = {
            "id": "dcc6_reproduction_check",
            "operations": candidate_operations(
                slope_px_per_coc=1.0 / 6.0,
                aperture_count=aperture_count,
                pair_common=config["pair_common"],
            ),
        }
        reproduced, reproduced_labels, _ = _validate_variant(
            raw_payload, candidate=reference_candidate, config=config
        )
        frozen_dcc6 = np.asarray(
            torch.as_tensor(dcc6_payload["psf_bank"]), dtype=np.float32
        )
        frozen_labels = np.asarray(
            torch.as_tensor(dcc6_payload["analytic_disparity_bins_px"]),
            dtype=np.float64,
        )
        reproduction_kernel_error = float(np.max(np.abs(reproduced - frozen_dcc6)))
        reproduction_label_error = float(
            np.max(np.abs(reproduced_labels - frozen_labels))
        )
        if max(reproduction_kernel_error, reproduction_label_error) > float(
            config["gates"]["dcc6_reproduction_abs_max"]
        ):
            raise RuntimeError(f"DCC6 reference 重放不一致：{source['atom_id']}")

        variants: list[dict[str, Any]] = []
        for variant in config["variants"]:
            candidate = _candidate(source, variant, config)
            _, _, metrics = _validate_variant(
                raw_payload, candidate=candidate, config=config
            )
            audit_paths: dict[str, str] = {}
            for tile_size in (64, 128):
                audit_config = _audit_config(
                    config, source, variant, tile_size=tile_size
                )
                relative = (
                    Path(source["atom_id"])
                    / variant["id"]
                    / f"audit_k{tile_size}.yaml"
                )
                path = output_root / relative
                _atomic_write_text(
                    path,
                    yaml.safe_dump(audit_config, allow_unicode=True, sort_keys=False),
                )
                generated[str(relative)] = _sha256_file(path)
                audit_paths[f"k{tile_size}"] = str(path)
            recipe = {
                "schema_version": 1,
                "status": "pending_formal_k64_k128",
                "source": {
                    "asset_root": source["raw_optical_root"],
                    "artifact_manifest_sha256": source["raw_manifest_sha256"],
                    "profile_ids": ["H000"],
                },
                "candidate": candidate,
                "required_audits": audit_paths,
                "final_output_root": str(
                    Path(config["output"]["formal_root"])
                    / source["atom_id"]
                    / variant["id"]
                    / "paircommon"
                ),
                "parent": dict(config["parent"]),
                "gates": {
                    "centroid_abs_max_error_px": config["gates"][
                        "centroid_abs_max_error_px"
                    ],
                    "parent_centroid_abs_max_error_px": config["gates"][
                        "parent_centroid_abs_max_error_px"
                    ],
                    "retained_mass_min": config["gates"]["retained_mass_min"],
                },
                "analytic_label_source": "final_kernel_centroid_mu_left_x_minus_mu_right_x",
                "ncc_role": "diagnostic_admission_only_never_label_writeback",
                "pdoffset_embedded": False,
                "training_admitted": False,
                **config["data_isolation"],
            }
            recipe_relative = (
                Path(source["atom_id"])
                / variant["id"]
                / "candidate_export_recipe.json"
            )
            recipe_path = output_root / recipe_relative
            _atomic_write_text(
                recipe_path, json.dumps(recipe, ensure_ascii=False, indent=2) + "\n"
            )
            generated[str(recipe_relative)] = _sha256_file(recipe_path)
            variants.append(
                {
                    "variant_id": variant["id"],
                    "slope_px_per_coc": float(variant["slope_px_per_coc"]),
                    "candidate_id": candidate["id"],
                    "static_transform": metrics,
                    "generated_audit_configs": audit_paths,
                    "candidate_export_recipe": str(recipe_path),
                    "formal_status": "pending_k64_k128_not_admitted",
                }
            )
        source_rows.append(
            {
                "atom_id": source["atom_id"],
                "raw_manifest_sha256": source["raw_manifest_sha256"],
                "raw_profile_sha256": source["raw_profile_sha256"],
                "dcc6_reference_manifest_sha256": source["dcc6_manifest_sha256"],
                "dcc6_reference_profile_sha256": source["dcc6_profile_sha256"],
                "dcc6_reproduction_kernel_abs_max": reproduction_kernel_error,
                "dcc6_reproduction_label_abs_max": reproduction_label_error,
                "dcc6_reproduction_bitwise_equal": bool(
                    np.array_equal(reproduced, frozen_dcc6)
                    and np.array_equal(reproduced_labels, frozen_labels)
                ),
                "variants": variants,
            }
        )

    resolved_path = output_root / "resolved_config.yaml"
    _atomic_write_text(
        resolved_path, yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    )
    summary = {
        "schema_version": 1,
        "status": "cpu_prepared_pending_formal_ncc",
        "run_id": config["run"]["id"],
        "source_atom_count": len(source_rows),
        "variant_count_per_atom": len(config["variants"]),
        "formal_candidate_count": len(source_rows) * len(config["variants"]),
        "new_optical_propagation_required": False,
        "dcc6_reference_reproduced_bitwise": all(
            row["dcc6_reproduction_bitwise_equal"] for row in source_rows
        ),
        "formal_k64_k128_run": False,
        "variant_assets_exported": False,
        "training_admitted": False,
        "pdoffset_embedded": False,
        "sources": source_rows,
        "elapsed_seconds": time.perf_counter() - started,
        **config["data_isolation"],
    }
    summary_path = output_root / "summary.json"
    _atomic_write_text(
        summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    checkout = Path(__file__).resolve().parents[1]
    provenance = {
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": _sha256_file(Path(__file__).resolve()),
        "transform_implementation_sha256": _sha256_file(
            Path(__file__).with_name("raw_family_continuous_audit.py")
        ),
        "source_commit": _git_output(checkout, "rev-parse", "HEAD"),
        "source_dirty_paths": _git_output(
            checkout, "status", "--short"
        ).splitlines(),
        "python_executable": sys.executable,
        "execution_device": "cpu_only",
        **config["data_isolation"],
    }
    provenance_path = output_root / "provenance.json"
    _atomic_write_text(
        provenance_path, json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
    files = {
        "resolved_config.yaml": _sha256_file(resolved_path),
        "summary.json": _sha256_file(summary_path),
        "provenance.json": _sha256_file(provenance_path),
        **generated,
    }
    manifest = {
        "schema_version": 1,
        "status": summary["status"],
        "run_id": config["run"]["id"],
        "formal_candidate_count": summary["formal_candidate_count"],
        "new_optical_propagation_required": False,
        "formal_k64_k128_run": False,
        "variant_assets_exported": False,
        "training_admitted": False,
        "pdoffset_embedded": False,
        "files": files,
        **config["data_isolation"],
    }
    manifest_path = output_root / "artifact_manifest.json"
    _atomic_write_text(
        manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return {
        **summary,
        "output_root": str(output_root),
        "artifact_manifest_sha256": _sha256_file(manifest_path),
        "summary_sha256": _sha256_file(summary_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
