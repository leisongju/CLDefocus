"""把已通过 K64/K128 全场 continuous NCC 的 raw-family 候选导出为 parent asset。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from .family_bank import _EXPORT_RETARGET_AXES, _RAW_ACTIVE_AXES, analytic_centroid_labels
from .raw_family_continuous_audit import (
    _atomic_torch_save,
    _atomic_write_text,
    _load_profiles,
    _sha256_file,
    _validate_data_isolation,
    transform_candidate,
)
from .raw_optical_export import validate_with_parent_loader


def load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "run",
        "source",
        "audits",
        "candidate",
        "parent",
        "gates",
        "output",
        "data_isolation",
    }
    if not isinstance(loaded, dict) or int(loaded.get("schema_version", 0)) != 1:
        raise ValueError("candidate export 配置必须为 schema_version: 1 mapping")
    if set(loaded) != required:
        raise ValueError(f"candidate export 配置字段必须精确为 {sorted(required)}")
    _validate_data_isolation(loaded["data_isolation"])
    audits = loaded["audits"]
    if not isinstance(audits, list) or [int(row["tile_size"]) for row in audits] != [64, 128]:
        raise ValueError("audits 必须按顺序冻结 K64 与 K128 两项")
    return loaded


def _load_verified_audit(
    row: dict[str, Any],
    *,
    candidate_id: str,
    expected_profile_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[tuple[str, int, int, int], dict[str, Any]]]:
    root = Path(row["root"]).resolve()
    verified: dict[str, Any] = {}
    for name in ("summary.json", "provenance.json", "screens.json"):
        path = root / name
        expected = str(row[f"{name.removesuffix('.json')}_sha256"])
        actual = _sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"audit {name} SHA256 不匹配：{root}")
        verified[name] = json.loads(path.read_text(encoding="utf-8"))
    screens = verified["screens.json"]
    if not isinstance(screens, list) or not screens:
        raise RuntimeError("audit screens 必须包含冻结候选")
    selected = [
        screen for screen in screens
        if str(screen.get("candidate_id")) == candidate_id
    ]
    if len(selected) != 1:
        raise RuntimeError(
            "audit 必须精确包含一个匹配 export candidate_id 的候选："
            f"candidate_id={candidate_id!r}, matches={len(selected)}"
        )
    screen = selected[0]
    if int(screen["tile_size"]) != int(row["tile_size"]):
        raise RuntimeError("audit tile_size 与 export 配置不一致")
    if str(screen["field_mode"]) != "full_flattened" or int(
        screen["fields_per_profile"]
    ) != 9:
        raise RuntimeError("audit 必须是 3x3 full-field")
    profile_ids = [str(item["profile_id"]) for item in screen["profiles"]]
    requested = list(expected_profile_ids)
    if [profile_id for profile_id in profile_ids if profile_id in requested] != requested:
        raise RuntimeError("audit 未按 source 顺序完整覆盖请求的 profile 子集")
    requested_set = set(requested)
    fields: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for profile in screen["profiles"]:
        if str(profile["profile_id"]) not in requested_set:
            continue
        if not bool(profile["all_apertures_pass"]) or not bool(
            profile["all_aperture_fields_pass"]
        ):
            raise RuntimeError(f"audit profile 未全过：{profile['profile_id']}")
        for aperture in profile["per_aperture"]:
            for field in aperture["fields"]:
                key = (
                    str(profile["profile_id"]),
                    int(aperture["aperture_index"]),
                    int(field["field_y_index"]),
                    int(field["field_x_index"]),
                )
                if key in fields or not bool(field["pass"]):
                    raise RuntimeError(f"audit field 重复或失败：{key}")
                fields[key] = field
    expected_count = len(expected_profile_ids) * 5 * 9
    if len(fields) != expected_count:
        raise RuntimeError(f"audit field 覆盖不完整：{len(fields)}/{expected_count}")
    return {
        "root": str(root),
        "tile_size": int(row["tile_size"]),
        "summary_sha256": str(row["summary_sha256"]),
        "provenance_sha256": str(row["provenance_sha256"]),
        "screens_sha256": str(row["screens_sha256"]),
        "screen_seconds": float(screen["seconds"]),
    }, fields


def _audit_ranges(fields: dict[tuple[str, int, int, int], dict[str, Any]]) -> dict[str, float]:
    slopes = np.asarray(
        [row["fit"]["slope_ncc_vs_analytic"] for row in fields.values()],
        dtype=np.float64,
    )
    r2 = np.asarray([row["fit"]["r2"] for row in fields.values()], dtype=np.float64)
    return {
        "field_count": int(slopes.size),
        "slope_min": float(slopes.min()),
        "slope_max": float(slopes.max()),
        "slope_mean": float(slopes.mean()),
        "r2_min": float(r2.min()),
        "r2_max": float(r2.max()),
    }


def export(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    source_root, profiles = _load_profiles(config)
    profile_ids = [profile.profile_id for profile in profiles]
    candidate = dict(config["candidate"])
    audit_rows: list[dict[str, Any]] = []
    audit_fields: dict[int, dict[tuple[str, int, int, int], dict[str, Any]]] = {}
    for audit in config["audits"]:
        verified, fields = _load_verified_audit(
            audit,
            candidate_id=str(candidate["id"]),
            expected_profile_ids=profile_ids,
        )
        audit_rows.append({**verified, **_audit_ranges(fields)})
        audit_fields[int(audit["tile_size"])] = fields
    keys = sorted(audit_fields[64])
    if keys != sorted(audit_fields[128]):
        raise RuntimeError("K64/K128 field key 不一致")
    slope_deltas = np.asarray(
        [
            float(audit_fields[128][key]["fit"]["slope_ncc_vs_analytic"])
            - float(audit_fields[64][key]["fit"]["slope_ncc_vs_analytic"])
            for key in keys
        ],
        dtype=np.float64,
    )
    r2_deltas = np.asarray(
        [
            float(audit_fields[128][key]["fit"]["r2"])
            - float(audit_fields[64][key]["fit"]["r2"])
            for key in keys
        ],
        dtype=np.float64,
    )
    convergence = {
        "field_count": len(keys),
        "slope_delta_min": float(slope_deltas.min()),
        "slope_delta_max": float(slope_deltas.max()),
        "slope_delta_abs_mean": float(np.mean(np.abs(slope_deltas))),
        "slope_delta_abs_p95": float(np.percentile(np.abs(slope_deltas), 95)),
        "r2_delta_min": float(r2_deltas.min()),
        "r2_delta_max": float(r2_deltas.max()),
        "r2_delta_abs_mean": float(np.mean(np.abs(r2_deltas))),
        "r2_delta_abs_p95": float(np.percentile(np.abs(r2_deltas), 95)),
    }

    output_root = Path(config["output"]["root"]).resolve()
    if output_root.exists():
        raise FileExistsError(f"candidate asset 输出已存在：{output_root}")
    staging = output_root.with_name(f".{output_root.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"candidate asset staging 已存在：{staging}")
    staging.mkdir(parents=True)
    try:
        profile_rows: list[dict[str, Any]] = []
        for profile in profiles:
            source_bank = np.asarray(
                torch.as_tensor(profile.payload["psf_bank"]), dtype=np.float32
            )
            coc = np.asarray(
                torch.as_tensor(profile.payload["signed_coc_bins_px"]),
                dtype=np.float64,
            )
            transformed, labels, operations = transform_candidate(
                source_bank,
                coc,
                candidate,
                centroid_tolerance_px=float(
                    config["gates"]["centroid_abs_max_error_px"]
                ),
                retained_mass_min=float(config["gates"]["retained_mass_min"]),
            )
            recomputed = analytic_centroid_labels(transformed)
            if not np.array_equal(labels, recomputed):
                raise RuntimeError("最终 candidate label 未严格来自最终 kernel centroid")
            source_labels = analytic_centroid_labels(source_bank)
            centroid_delta = float(np.max(np.abs(labels - source_labels)))
            dcc_operations = [
                operation
                for operation in operations
                if operation.get("mode")
                == "symmetric_relative_centroid_retarget_v1"
            ]
            centroid_contract_error = (
                max(
                    float(operation["analytic_target_error_abs_max_px"])
                    for operation in dcc_operations
                )
                if dcc_operations
                else centroid_delta
            )
            energy_delta = float(
                np.max(
                    np.abs(
                        transformed.sum(axis=(-2, -1), dtype=np.float64)
                        - source_bank.sum(axis=(-2, -1), dtype=np.float64)
                    )
                )
            )
            if centroid_contract_error > float(
                config["gates"]["centroid_abs_max_error_px"]
            ):
                raise RuntimeError(f"candidate centroid contract 失败：{profile.profile_id}")
            payload = dict(profile.payload)
            payload.update(
                {
                    "asset_id": f"{config['run']['asset_id']}_{profile.profile_id}",
                    "centroid_policy": "native",
                    "centroid_variant": str(config["run"]["centroid_variant"]),
                    "pdoffset_embedded": False,
                    "psf_bank": torch.from_numpy(transformed.copy()),
                    "analytic_disparity_bins_px": torch.from_numpy(labels.copy()),
                    "raw_analytic_disparity_bins_px": torch.as_tensor(
                        profile.payload["analytic_disparity_bins_px"]
                    ).detach().clone(),
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
                        "inactive_profile_parameters_for_source_raw_export": list(
                            _EXPORT_RETARGET_AXES
                        ),
                        "candidate_operations": candidate["operations"],
                    },
                }
            )
            payload["asset_spec"] = {
                **dict(payload.get("asset_spec", {})),
                "kind": "cldefocus_raw_family_pair_common_continuous_admitted_v1",
                "analytic_label_source": "final_psf_centroid_mu_left_x_minus_mu_right_x",
                "centroid_variant": config["run"]["centroid_variant"],
                "pd_offset_embedded": False,
                "ncc_role": "diagnostic_admission_only_never_label_writeback",
            }
            path = staging / "profiles" / profile.profile_id / "psf_bank.pt"
            sha = _atomic_torch_save(path, payload)
            fourier_quality = [
                float(operation["fourier_clipped_negative_fraction_max"])
                for operation in operations
                if "fourier_clipped_negative_fraction_max" in operation
            ]
            profile_rows.append(
                {
                    "profile_id": profile.profile_id,
                    "relative_path": str(path.relative_to(staging)),
                    "path": str(output_root / path.relative_to(staging)),
                    "sha256": sha,
                    "shape": list(transformed.shape),
                    "source_sha256": profile.sha256,
                    "final_vs_source_centroid_abs_max_px": centroid_delta,
                    "centroid_contract_abs_max_px": centroid_contract_error,
                    "final_vs_source_energy_abs_max": energy_delta,
                    "fourier_clipped_negative_fraction_max": float(
                        max(fourier_quality, default=0.0)
                    ),
                    "analytic_label_sha256": hashlib.sha256(
                        np.ascontiguousarray(labels, dtype=np.float64).tobytes()
                    ).hexdigest(),
                }
            )

        parent_config = {
            "source": {
                "parent_repository": config["parent"]["repository"],
                "parent_loader_sha256": config["parent"]["loader_sha256"],
                "parent_renderer_sha256": config["parent"]["renderer_sha256"],
            }
        }
        parent_validation = validate_with_parent_loader(
            config=parent_config,
            staging_root=staging,
            profile_rows=profile_rows,
            f_numbers=torch.as_tensor(profiles[0].payload["f_numbers"]),
            analytic_tolerance_px=float(
                config["gates"]["parent_centroid_abs_max_error_px"]
            ),
            expected_centroid_policy="native",
            expected_centroid_variant=str(config["run"]["centroid_variant"]),
        )

        with (staging / "k64_k128_field_convergence.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "profile_id",
                    "aperture_index",
                    "field_y_index",
                    "field_x_index",
                    "k64_slope",
                    "k128_slope",
                    "slope_delta",
                    "k64_r2",
                    "k128_r2",
                    "r2_delta",
                ]
            )
            for key, slope_delta, r2_delta in zip(
                keys, slope_deltas.tolist(), r2_deltas.tolist(), strict=True
            ):
                writer.writerow(
                    [
                        *key,
                        audit_fields[64][key]["fit"]["slope_ncc_vs_analytic"],
                        audit_fields[128][key]["fit"]["slope_ncc_vs_analytic"],
                        slope_delta,
                        audit_fields[64][key]["fit"]["r2"],
                        audit_fields[128][key]["fit"]["r2"],
                        r2_delta,
                    ]
                )
        summary = {
            "schema_version": 1,
            "status": "pass",
            "run_id": config["run"]["id"],
            "asset_id": config["run"]["asset_id"],
            "centroid_policy": "native",
            "centroid_variant": config["run"]["centroid_variant"],
            "candidate": candidate,
            "profile_count": len(profile_rows),
            "aperture_count": 5,
            "profile_aperture_parent_validation": "80/80",
            "profiles": profile_rows,
            "audit_lineage": audit_rows,
            "k64_k128_convergence": convergence,
            "parent_validation": parent_validation,
            "analytic_label_source": "final_kernel_centroid_mu_left_x_minus_mu_right_x",
            "ncc_updates_labels": False,
            "pdoffset_embedded": False,
            "source_asset_root": str(source_root),
            **config["data_isolation"],
        }
        _atomic_write_text(
            staging / "summary.json",
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        )
        _atomic_write_text(
            staging / "resolved_config.yaml",
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        )
        report = f"""# CLDefocus pair-common tc16 continuous NCC 准入资产报告

状态：**PASS**。

该资产从冻结 raw-optical family 构造，不重新运行 CLDefocus propagation，不访问真实数据，
不训练模型。候选使用 direct shared morphology、bilinear center、共同 Gaussian MTF
`sigma=1.0 px`、Fourier place、`transition=16 px`。NCC 仅参与只读诊断准入，
解析标签始终从最终 kernel 的 `mu_left_x-mu_right_x` 重算；PDOFFSET 未嵌入。

- asset id：`{config['run']['asset_id']}`
- profile/aperture：`{len(profile_rows)}×5`
- K64 全场：{audit_rows[0]['field_count']}/{audit_rows[0]['field_count']} PASS，slope `{audit_rows[0]['slope_min']:.9f}–{audit_rows[0]['slope_max']:.9f}`，R² min `{audit_rows[0]['r2_min']:.9f}`
- K128 全场：{audit_rows[1]['field_count']}/{audit_rows[1]['field_count']} PASS，slope `{audit_rows[1]['slope_min']:.9f}–{audit_rows[1]['slope_max']:.9f}`，R² min `{audit_rows[1]['r2_min']:.9f}`
- K64→K128 slope 绝对差 mean/P95：`{convergence['slope_delta_abs_mean']:.9f}/{convergence['slope_delta_abs_p95']:.9f}`
- Parent loader profile×aperture：`{parent_validation['combination_count']}/{parent_validation['expected_combination_count']}` PASS
- Parent kernel/analytic centroid 最大误差：`{parent_validation['centroid_agreement_abs_max_px']:.3e} px`
- Candidate centroid 合同最大误差：`{max(row['centroid_contract_abs_max_px'] for row in profile_rows):.3e} px`
- Candidate 最终/源 optical centroid 最大变化：`{max(row['final_vs_source_centroid_abs_max_px'] for row in profile_rows):.3e} px`
- Fourier 非负投影质量最大值：`{max(row['fourier_clipped_negative_fraction_max'] for row in profile_rows):.6f}`

Parent 应按样本光圈直接加载 `profiles/Fxxx/psf_bank.pt`，使用 bank centroid，并保持
`focus_zero_calibrate=false`。PDOFFSET 仍由 parent 独立加入。
"""
        _atomic_write_text(staging / "REPORT.md", report)
        files: dict[str, str] = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "artifact_manifest.json":
                files[str(path.relative_to(staging))] = _sha256_file(path)
        manifest = {
            "schema_version": 1,
            "status": "pass",
            "asset_id": config["run"]["asset_id"],
            "centroid_variant": config["run"]["centroid_variant"],
            "profile_ids": profile_ids,
            "profile_count": len(profile_ids),
            "aperture_count": 5,
            "parent_loader_combination_count": parent_validation["combination_count"],
            "analytic_label_source": "final_kernel_centroid_mu_left_x_minus_mu_right_x",
            "ncc_role": "diagnostic_admission_only_never_label_writeback",
            "pdoffset_embedded": False,
            "files": files,
        }
        _atomic_write_text(
            staging / "artifact_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "output_root": str(output_root),
        "status": "pass",
        "asset_id": config["run"]["asset_id"],
        "profile_count": len(profiles),
        "parent_loader_combinations": parent_validation["combination_count"],
        "artifact_manifest_sha256": _sha256_file(output_root / "artifact_manifest.json"),
        "summary_sha256": _sha256_file(output_root / "summary.json"),
        "report_sha256": _sha256_file(output_root / "REPORT.md"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
