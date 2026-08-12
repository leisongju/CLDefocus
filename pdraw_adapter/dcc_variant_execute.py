"""执行冻结 DCC5/DCC7 K64/K128 CPU 审计、导出与 parent smoke。"""

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
import yaml


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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_output(checkout: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args], text=True
    ).strip()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"schema_version", "run", "source_plan", "output", "data_isolation"}
    if not isinstance(config, dict) or int(config.get("schema_version", 0)) != 1:
        raise ValueError("DCC57 正式执行配置必须是 schema_version: 1 mapping")
    if set(config) != required:
        raise ValueError(f"执行配置字段必须精确为 {sorted(required)}")
    run = config["run"]
    if str(run.get("mode")) != "cpu_formal_audit_export":
        raise ValueError("run.mode 必须为 cpu_formal_audit_export")
    if int(run.get("expected_candidate_count", 0)) != 4:
        raise ValueError("正式执行必须精确包含四个候选")
    if [int(value) for value in run.get("audit_order", [])] != [64, 128]:
        raise ValueError("正式执行顺序必须精确为 K64、K128")
    if bool(run.get("gpu_allowed", True)) or bool(
        run.get("new_optical_propagation_allowed", True)
    ):
        raise ValueError("DCC57 分叉禁止 GPU 和新 optical propagation")
    isolation = config["data_isolation"]
    if not isinstance(isolation, dict) or set(isolation) != _ISOLATION_KEYS:
        raise ValueError(f"data_isolation 必须精确为 {sorted(_ISOLATION_KEYS)}")
    if any(bool(value) for value in isolation.values()):
        raise ValueError("正式执行禁止访问真实数据或启动训练")
    return config


def _require_cpu_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("必须显式以 CUDA_VISIBLE_DEVICES='' 执行，拒绝可见 GPU")


def load_verified_plan(
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = config["source_plan"]
    root = Path(source["root"]).resolve()
    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.is_file() or _sha256_file(manifest_path) != str(
        source["artifact_manifest_sha256"]
    ):
        raise RuntimeError("冻结 CPU plan artifact_manifest SHA256 不匹配")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("status") != "cpu_prepared_pending_formal_ncc"
        or manifest.get("formal_k64_k128_run") is not False
        or manifest.get("variant_assets_exported") is not False
        or manifest.get("training_admitted") is not False
        or manifest.get("pdoffset_embedded") is not False
        or manifest.get("new_optical_propagation_required") is not False
    ):
        raise RuntimeError("冻结 CPU plan 状态合同不匹配")
    for relative, expected in manifest.get("files", {}).items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise RuntimeError(f"CPU plan 文件越界或不存在：{path}")
        if _sha256_file(path) != str(expected):
            raise RuntimeError(f"CPU plan 文件 SHA256 不匹配：{path}")
    summary = _load_json(root / "summary.json")
    variants: list[dict[str, Any]] = []
    formal_root = Path(config["output"]["root"]).resolve()
    for source_row in summary.get("sources", []):
        for variant in source_row.get("variants", []):
            audit_configs = variant["generated_audit_configs"]
            recipe_path = Path(variant["candidate_export_recipe"]).resolve()
            recipe = _load_json(recipe_path)
            if recipe.get("status") != "pending_formal_k64_k128":
                raise RuntimeError(f"candidate recipe 状态合同不匹配：{recipe_path}")
            if str(recipe.get("candidate", {}).get("id")) != str(
                variant["candidate_id"]
            ):
                raise RuntimeError(f"candidate recipe id 不匹配：{recipe_path}")
            output_root = Path(recipe["final_output_root"]).resolve()
            if formal_root not in output_root.parents:
                raise RuntimeError(f"candidate 输出不在冻结 formal root：{output_root}")
            variants.append(
                {
                    "atom_id": str(source_row["atom_id"]),
                    "variant_id": str(variant["variant_id"]),
                    "candidate_id": str(variant["candidate_id"]),
                    "audit_configs": {
                        int(tile): str(Path(path).resolve())
                        for tile, path in ((64, audit_configs["k64"]), (128, audit_configs["k128"]))
                    },
                    "recipe_path": str(recipe_path),
                    "recipe": recipe,
                    "output_root": str(output_root),
                    "static_transform": dict(variant["static_transform"]),
                }
            )
    if len(variants) != int(config["run"]["expected_candidate_count"]):
        raise RuntimeError(f"CPU plan 候选数量不一致：{len(variants)}")
    expected_order = [
        ("MP00", "dcc5"),
        ("MP00", "dcc7"),
        ("MP01", "dcc5"),
        ("MP01", "dcc7"),
    ]
    actual_order = [(row["atom_id"][:4], row["variant_id"]) for row in variants]
    if actual_order != expected_order:
        raise RuntimeError(f"CPU plan 候选顺序不匹配：{actual_order}")
    return manifest, variants


def _audit_output_root(config_path: Path) -> Path:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return Path(loaded["output"]["root"]).resolve()


def inspect_audit(
    root: Path,
    *,
    tile_size: int,
    candidate_id: str,
    config_sha256: str,
) -> dict[str, Any] | None:
    required = [root / name for name in ("summary.json", "provenance.json", "screens.json")]
    if not all(path.is_file() for path in required):
        return None
    summary = _load_json(required[0])
    provenance = _load_json(required[1])
    screens = _load_json(required[2])
    if provenance.get("config_sha256") != config_sha256:
        raise RuntimeError(f"K{tile_size} provenance 配置 SHA256 不匹配：{root}")
    if summary.get("status") != "screen_pass_pending_validation":
        return None
    if summary.get("pdoffset_embedded") is not False or summary.get("exported_asset") is not None:
        raise RuntimeError(f"K{tile_size} audit 状态/PDOFFSET 合同不匹配：{root}")
    if any(bool(summary.get(key)) for key in _ISOLATION_KEYS):
        raise RuntimeError(f"K{tile_size} audit 隔离合同不匹配：{root}")
    matching = [
        row
        for row in screens
        if str(row.get("candidate_id")) == candidate_id
        and int(row.get("tile_size", -1)) == int(tile_size)
    ]
    if len(matching) != 1:
        raise RuntimeError(f"K{tile_size} audit 缺少唯一冻结候选：{candidate_id}")
    screen = matching[0]
    if str(screen.get("field_mode")) != "full_flattened" or int(
        screen.get("fields_per_profile", 0)
    ) != 9:
        raise RuntimeError(f"K{tile_size} audit 不是冻结 3x3 full-field")
    profiles = list(screen.get("profiles", []))
    if len(profiles) != 1 or str(profiles[0].get("profile_id")) != "H000":
        raise RuntimeError(f"K{tile_size} audit profile 覆盖不匹配")
    fields: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    profile = profiles[0]
    if not bool(profile.get("all_apertures_pass")) or not bool(
        profile.get("all_aperture_fields_pass")
    ):
        return None
    apertures = list(profile.get("per_aperture", []))
    if len(apertures) != 5:
        raise RuntimeError(f"K{tile_size} audit 光圈数不等于 5")
    for aperture in apertures:
        if not bool(aperture.get("pass")) or not bool(aperture.get("all_fields_pass")):
            return None
        aperture_fields = list(aperture.get("fields", []))
        if len(aperture_fields) != 9:
            raise RuntimeError(f"K{tile_size} audit 每光圈 field 数不等于 9")
        for field in aperture_fields:
            checks = dict(field.get("checks", {}))
            key = (
                "H000",
                int(aperture["aperture_index"]),
                int(field["field_y_index"]),
                int(field["field_x_index"]),
            )
            if key in fields:
                raise RuntimeError(f"K{tile_size} audit field 重复：{key}")
            if not bool(field.get("pass")) or not checks or not all(
                bool(value) for value in checks.values()
            ):
                return None
            fields[key] = field
    if len(fields) != 45:
        raise RuntimeError(f"K{tile_size} audit field 覆盖不完整：{len(fields)}/45")
    slopes = np.asarray(
        [float(row["fit"]["slope_ncc_vs_analytic"]) for row in fields.values()],
        dtype=np.float64,
    )
    r2 = np.asarray(
        [float(row["fit"]["r2"]) for row in fields.values()], dtype=np.float64
    )
    if not np.all(np.isfinite(slopes)) or not np.all(np.isfinite(r2)):
        raise RuntimeError(f"K{tile_size} audit 指标包含非有限值")
    return {
        "status": "pass",
        "root": str(root),
        "tile_size": int(tile_size),
        "field_count": len(fields),
        "summary_status": summary["status"],
        "summary_sha256": _sha256_file(required[0]),
        "provenance_sha256": _sha256_file(required[1]),
        "screens_sha256": _sha256_file(required[2]),
        "config_sha256": config_sha256,
        "slope_min": float(slopes.min()),
        "slope_max": float(slopes.max()),
        "slope_mean": float(slopes.mean()),
        "r2_min": float(r2.min()),
        "r2_max": float(r2.max()),
        "r2_mean": float(r2.mean()),
        "seconds": float(screen["seconds"]),
        "_fields": fields,
    }


def _public_audit(stage: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in stage.items() if key != "_fields"}


def _convergence(k64: dict[str, Any], k128: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(k64["_fields"])
    if keys != sorted(k128["_fields"]):
        raise RuntimeError("K64/K128 field key 不一致")
    slope_delta = np.asarray(
        [
            float(k128["_fields"][key]["fit"]["slope_ncc_vs_analytic"])
            - float(k64["_fields"][key]["fit"]["slope_ncc_vs_analytic"])
            for key in keys
        ],
        dtype=np.float64,
    )
    r2_delta = np.asarray(
        [
            float(k128["_fields"][key]["fit"]["r2"])
            - float(k64["_fields"][key]["fit"]["r2"])
            for key in keys
        ],
        dtype=np.float64,
    )
    return {
        "field_count": len(keys),
        "slope_delta_min": float(slope_delta.min()),
        "slope_delta_max": float(slope_delta.max()),
        "slope_delta_abs_mean": float(np.mean(np.abs(slope_delta))),
        "slope_delta_abs_p95": float(np.percentile(np.abs(slope_delta), 95)),
        "r2_delta_min": float(r2_delta.min()),
        "r2_delta_max": float(r2_delta.max()),
        "r2_delta_abs_mean": float(np.mean(np.abs(r2_delta))),
        "r2_delta_abs_p95": float(np.percentile(np.abs(r2_delta), 95)),
    }


def build_candidate_config(
    execution_config: dict[str, Any],
    variant: dict[str, Any],
    audits: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if [int(row["tile_size"]) for row in audits] != [64, 128]:
        raise ValueError("candidate export 必须按 K64/K128 顺序提供审计")
    recipe = variant["recipe"]
    return {
        "schema_version": 1,
        "run": {
            "id": f"{execution_config['run']['id']}__{variant['candidate_id']}__export",
            "asset_id": f"{variant['candidate_id']}__contncc_v1",
            "centroid_variant": f"{variant['candidate_id']}__contncc_v1",
        },
        "source": dict(recipe["source"]),
        "audits": [
            {
                key: row[key]
                for key in (
                    "tile_size",
                    "root",
                    "summary_sha256",
                    "provenance_sha256",
                    "screens_sha256",
                )
            }
            for row in audits
        ],
        "candidate": dict(recipe["candidate"]),
        "parent": dict(recipe["parent"]),
        "gates": dict(recipe["gates"]),
        "output": {"root": variant["output_root"]},
        "data_isolation": dict(execution_config["data_isolation"]),
    }


def _write_frozen_yaml(path: Path, value: dict[str, Any]) -> str:
    rendered = yaml.safe_dump(value, allow_unicode=True, sort_keys=False)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"已存在 candidate 配置与本次解析结果不同：{path}")
    else:
        _atomic_write_text(path, rendered)
    return _sha256_file(path)


def inspect_bank(root: Path, *, candidate_id: str) -> dict[str, Any] | None:
    manifest_path = root / "artifact_manifest.json"
    summary_path = root / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        return None
    manifest = _load_json(manifest_path)
    summary = _load_json(summary_path)
    if manifest.get("status") != "pass" or summary.get("status") != "pass":
        return None
    if str(summary.get("candidate", {}).get("id")) != candidate_id:
        raise RuntimeError(f"bank candidate id 不匹配：{root}")
    if summary.get("pdoffset_embedded") is not False or summary.get(
        "ncc_updates_labels"
    ) is not False:
        raise RuntimeError(f"bank label/PDOFFSET 合同不匹配：{root}")
    for relative, expected in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file() or _sha256_file(path) != str(expected):
            raise RuntimeError(f"bank 文件 SHA256 不匹配：{path}")
    parent = summary.get("parent_validation", {})
    if (
        int(parent.get("combination_count", 0)) != 5
        or int(parent.get("expected_combination_count", 0)) != 5
        or int(parent.get("bank_mode_renderer_instance_count", 0)) != 5
        or int(parent.get("bank_mode_render_smoke_count", 0)) != 5
    ):
        return None
    return {
        "status": "pass",
        "root": str(root),
        "artifact_manifest_sha256": _sha256_file(manifest_path),
        "summary_sha256": _sha256_file(summary_path),
        "profile_sha256": manifest["files"]["profiles/H000/psf_bank.pt"],
        "parent_loader_render_smoke": "5/5 PASS",
        "parent_centroid_abs_max_error_px": float(
            parent["centroid_agreement_abs_max_px"]
        ),
    }


def _audit_or_run(
    config_path: Path, *, tile_size: int, candidate_id: str
) -> tuple[dict[str, Any] | None, str | None]:
    config_sha = _sha256_file(config_path)
    root = _audit_output_root(config_path)
    if root.exists():
        try:
            existing = inspect_audit(
                root,
                tile_size=tile_size,
                candidate_id=candidate_id,
                config_sha256=config_sha,
            )
        except BaseException as error:
            return None, f"{type(error).__name__}: {error}"
        if existing is None:
            return None, "已存在 audit 输出不满足冻结门禁，保留 NO-GO 且拒绝覆盖"
        return existing, None
    try:
        from .raw_family_continuous_audit import run as run_audit

        run_audit(config_path)
        result = inspect_audit(
            root,
            tile_size=tile_size,
            candidate_id=candidate_id,
            config_sha256=config_sha,
        )
        if result is None:
            return None, "formal audit 未满足冻结 45/45 field 门禁"
        return result, None
    except BaseException as error:
        return None, f"{type(error).__name__}: {error}"


def _write_aggregate(
    config_path: Path,
    config: dict[str, Any],
    plan_manifest: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    root = Path(config["output"]["root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    admitted_count = sum(bool(row["training_admitted"]) for row in rows)
    admitted = len(rows) == int(config["run"]["expected_candidate_count"]) and admitted_count == len(rows)
    status = "pass" if admitted else "no_go"
    summary = {
        "schema_version": 1,
        "status": status,
        "run_id": config["run"]["id"],
        "candidate_count": len(rows),
        "candidate_pass_count": admitted_count,
        "training_admitted": admitted,
        "formal_k64_k128_run": True,
        "variant_assets_exported": admitted_count,
        "new_optical_propagation_run": False,
        "gpu_used": False,
        "pdoffset_embedded": False,
        "candidates": list(rows),
        "elapsed_seconds": float(elapsed_seconds),
        **config["data_isolation"],
    }
    resolved = root / "resolved_execution_config.yaml"
    _atomic_write_text(resolved, yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    summary_path = root / "summary.json"
    _atomic_write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    checkout = Path(__file__).resolve().parents[1]
    provenance = {
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256_file(config_path),
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": _sha256_file(Path(__file__).resolve()),
        "source_plan_root": str(Path(config["source_plan"]["root"]).resolve()),
        "source_plan_artifact_manifest_sha256": config["source_plan"]["artifact_manifest_sha256"],
        "source_plan_manifest_status": plan_manifest["status"],
        "python_executable": sys.executable,
        "source_commit": _git_output(checkout, "rev-parse", "HEAD"),
        "source_dirty_paths": _git_output(checkout, "status", "--short").splitlines(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **config["data_isolation"],
    }
    provenance_path = root / "provenance.json"
    _atomic_write_text(provenance_path, json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    generated = {
        str(path.relative_to(root)): _sha256_file(path)
        for path in sorted(root.glob("*/*/generated_configs/*.yaml"))
    }
    manifest = {
        "schema_version": 1,
        "status": status,
        "run_id": config["run"]["id"],
        "candidate_count": len(rows),
        "candidate_pass_count": admitted_count,
        "training_admitted": admitted,
        "new_optical_propagation_run": False,
        "gpu_used": False,
        "pdoffset_embedded": False,
        "source_plan_artifact_manifest_sha256": config["source_plan"]["artifact_manifest_sha256"],
        "bank_manifest_sha256": {
            row["candidate_id"]: None if row.get("bank") is None else row["bank"]["artifact_manifest_sha256"]
            for row in rows
        },
        "files": {
            "resolved_execution_config.yaml": _sha256_file(resolved),
            "summary.json": _sha256_file(summary_path),
            "provenance.json": _sha256_file(provenance_path),
            **generated,
        },
        **config["data_isolation"],
    }
    manifest_path = root / "artifact_manifest.json"
    _atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return {
        **summary,
        "output_root": str(root),
        "artifact_manifest_sha256": _sha256_file(manifest_path),
        "summary_sha256": _sha256_file(summary_path),
    }


def run(config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = config_path.resolve()
    config = load_config(config_path)
    _require_cpu_environment()
    plan_manifest, variants = load_verified_plan(config)
    rows: list[dict[str, Any]] = []
    for variant in variants:
        print(json.dumps({"candidate": variant["candidate_id"], "stage": "start"}, ensure_ascii=False), flush=True)
        audits: list[dict[str, Any]] = []
        errors: list[str] = []
        for tile_size in config["run"]["audit_order"]:
            audit_path = Path(variant["audit_configs"][int(tile_size)])
            stage, error = _audit_or_run(
                audit_path,
                tile_size=int(tile_size),
                candidate_id=variant["candidate_id"],
            )
            if stage is not None:
                audits.append(stage)
                print(
                    json.dumps(
                        {
                            "candidate": variant["candidate_id"],
                            "stage": f"K{tile_size}",
                            "status": "pass",
                            "slope": [stage["slope_min"], stage["slope_max"]],
                            "r2_min": stage["r2_min"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            else:
                errors.append(f"K{tile_size}: {error}")
                print(json.dumps({"candidate": variant["candidate_id"], "stage": f"K{tile_size}", "status": "no_go", "reason": error}, ensure_ascii=False), flush=True)
        convergence = None
        bank = None
        if len(audits) == 2:
            try:
                convergence = _convergence(audits[0], audits[1])
                export_config = build_candidate_config(config, variant, audits)
                export_path = (
                    Path(variant["output_root"]).parent
                    / "generated_configs"
                    / "candidate_export.yaml"
                )
                export_config_sha = _write_frozen_yaml(export_path, export_config)
                bank_root = Path(variant["output_root"])
                bank = inspect_bank(bank_root, candidate_id=variant["candidate_id"])
                if bank is None and not bank_root.exists():
                    from .raw_family_candidate_export import export

                    export(export_path)
                    bank = inspect_bank(bank_root, candidate_id=variant["candidate_id"])
                if bank is None:
                    raise RuntimeError("candidate bank/parent load-render smoke 未通过")
                bank["export_config_path"] = str(export_path)
                bank["export_config_sha256"] = export_config_sha
            except BaseException as error:
                errors.append(f"export: {type(error).__name__}: {error}")
                bank = None
        row = {
            "atom_id": variant["atom_id"],
            "variant_id": variant["variant_id"],
            "candidate_id": variant["candidate_id"],
            "status": "pass" if bank is not None and not errors else "no_go",
            "training_admitted": bank is not None and not errors,
            "audits": [_public_audit(stage) for stage in audits],
            "k64_k128_convergence": convergence,
            "bank": bank,
            "static_transform": variant["static_transform"],
            "errors": errors,
            "new_optical_propagation_run": False,
            "gpu_used": False,
            "pdoffset_embedded": False,
        }
        rows.append(row)
        print(json.dumps({"candidate": variant["candidate_id"], "stage": "complete", "status": row["status"]}, ensure_ascii=False), flush=True)
    return _write_aggregate(
        config_path,
        config,
        plan_manifest,
        rows,
        elapsed_seconds=time.perf_counter() - started,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
