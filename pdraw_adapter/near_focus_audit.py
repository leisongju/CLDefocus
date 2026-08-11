"""冻结 CLDefocus bank 的 near-focus、逐 aperture NCC 只读审计。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .multi_profile import (
    _bank_centroid_and_covariance,
    _centroid_labels_from_stats,
    _git_output,
    _sha256_file,
    apply_per_aperture_gate,
    run_profile_ncc_diagnostic,
    summarize_per_aperture_ncc,
)


def _torch_load(path: Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"冻结 bank 必须保存为 dict：{path}")
    return payload


def run(config_path: Path) -> dict[str, Any]:
    loaded_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded_config, dict) or int(loaded_config.get("schema_version", 0)) != 1:
        raise ValueError("near-focus audit 配置必须为 schema_version: 1")
    project_root = Path(__file__).resolve().parents[3]
    resolved = dict(loaded_config)
    input_cfg = dict(resolved["input"])
    input_path = Path(input_cfg["combined_bank_path"]).resolve()
    input_cfg["combined_bank_path"] = str(input_path)
    resolved["input"] = input_cfg
    output_cfg = dict(resolved["output"])
    output_root = Path(output_cfg["root"]).resolve()
    output_cfg["root"] = str(output_root)
    resolved["output"] = output_cfg
    if output_root == project_root or project_root in output_root.parents:
        raise ValueError("审计输出必须位于仓库外部")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_path = output_root / "resolved_config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    expected_sha = str(input_cfg["sha256"])
    actual_sha = _sha256_file(input_path)
    if actual_sha != expected_sha:
        raise RuntimeError(f"冻结 bank SHA256 不匹配：{actual_sha} != {expected_sha}")
    payload = _torch_load(input_path)
    profile_ids = [str(value) for value in payload["profile_ids"]]
    expected_profile_ids = [str(value) for value in input_cfg["profile_ids"]]
    if profile_ids != expected_profile_ids:
        raise RuntimeError(f"profile 顺序不匹配：{profile_ids} != {expected_profile_ids}")
    bank = np.asarray(payload["psf_bank"].detach().cpu(), dtype=np.float32)
    labels = np.asarray(
        payload["analytic_disparity_bins_px"].detach().cpu(),
        dtype=np.float64,
    )
    coc = np.asarray(payload["signed_coc_bins_px"].detach().cpu(), dtype=np.float64)
    if bank.ndim != 8 or bank.shape[0] != len(profile_ids):
        raise ValueError(f"combined bank 必须为 [P,A,N,Gh,Gw,2,K,K]：{bank.shape}")
    if labels.shape != bank.shape[:5]:
        raise ValueError(f"analytic label shape 不一致：{labels.shape} vs {bank.shape[:5]}")

    diagnostic_cfg = resolved["diagnostic"]
    gates = dict(resolved["gates"])
    profile_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for profile_index, profile_id in enumerate(profile_ids):
        profile_bank = bank[profile_index]
        recomputed = _centroid_labels_from_stats(
            _bank_centroid_and_covariance(profile_bank)
        )
        label_recompute_abs_max = float(
            np.max(np.abs(recomputed - labels[profile_index]))
        )
        label_check = label_recompute_abs_max <= float(
            diagnostic_cfg.get("label_recompute_abs_max_px", 2.0e-6)
        )
        diagnostic = run_profile_ncc_diagnostic(
            profile_bank,
            labels[profile_index],
            signed_coc_bins_px=coc,
            texture_size=int(diagnostic_cfg["texture_size"]),
            seeds=[int(value) for value in diagnostic_cfg["seeds"]],
            tile_size=int(diagnostic_cfg["tile_size"]),
            tiles_per_axis=int(diagnostic_cfg["tiles_per_axis"]),
            search_x=int(diagnostic_cfg["search_x"]),
            search_y=int(diagnostic_cfg["search_y"]),
        )
        coc_subset = apply_per_aperture_gate(
            summarize_per_aperture_ncc(
                diagnostic,
                aperture_count=profile_bank.shape[0],
                coc_abs_max_px=float(resolved["subsets"]["coc_abs_max_px"]),
            ),
            gates,
        )
        disparity_subset = apply_per_aperture_gate(
            summarize_per_aperture_ncc(
                diagnostic,
                aperture_count=profile_bank.shape[0],
                analytic_disparity_abs_max_px=float(
                    resolved["subsets"]["analytic_disparity_abs_max_px"]
                ),
            ),
            gates,
        )
        profile_pass = bool(label_check and coc_subset["pass"])
        profile_rows.append(
            {
                "profile_id": profile_id,
                "label_recompute_abs_max_px": label_recompute_abs_max,
                "label_recompute_pass": label_check,
                "coc_subset": coc_subset,
                "analytic_disparity_subset": disparity_subset,
                "primary_profile_pass": profile_pass,
                "primary_pass_rule": "label_identity_and_all_apertures_abs_coc_le_1",
            }
        )
        print(
            f"[near-focus] {profile_id} "
            f"coc_all_apertures={'PASS' if coc_subset['pass'] else 'FAIL'} "
            f"disp_all_apertures={'PASS' if disparity_subset['pass'] else 'FAIL'}",
            flush=True,
        )
    elapsed = time.perf_counter() - start
    passed_profiles = [
        row["profile_id"] for row in profile_rows if bool(row["primary_profile_pass"])
    ]
    conclusion = "GO" if passed_profiles else "NO_GO"
    final_metrics = {
        "run_id": str(resolved["run"]["id"]),
        "status": conclusion,
        "scope": "readonly_frozen_center8_near_focus_ncc",
        "primary_subset": "abs(signed_coc_px)<=1.0",
        "secondary_subset": "abs(analytic_disparity_px)<=0.25",
        "aggregate_can_override_per_aperture": False,
        "profile_count": len(profile_rows),
        "passed_profile_ids": passed_profiles,
        "all_profiles_fail_primary": not passed_profiles,
        "profiles": profile_rows,
        "elapsed_seconds": elapsed,
        "input_combined_bank": str(input_path),
        "input_combined_bank_sha256": actual_sha,
        "real_pdraw_accessed": False,
        "google_dev_accessed": False,
        "google_holdout_accessed": False,
        "dp5k_accessed": False,
        "stereo_training_run": False,
        "field_raytrace_run": False,
        "new_psf_bank_written": False,
    }
    final_path = output_root / "final_metrics.json"
    final_path.write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "project_commit": _git_output(project_root, "rev-parse", "HEAD"),
        "project_dirty_paths": _git_output(project_root, "status", "--porcelain").splitlines(),
        "runtime": "Genfocus",
        "python_executable": sys.executable,
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256_file(config_path),
        "resolved_config_sha256": _sha256_file(resolved_path),
        "source_path": str(Path(__file__).resolve()),
        "source_sha256": _sha256_file(Path(__file__).resolve()),
        "input_sha256": actual_sha,
        "real_data_accessed": False,
        "training_run": False,
        "field_raytrace_run": False,
    }
    provenance_path = output_root / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    artifact_manifest: dict[str, dict[str, Any]] = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifact_manifest[str(path.relative_to(output_root))] = {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
    artifact_path = output_root / "artifact_manifest.json"
    artifact_path.write_text(
        json.dumps(artifact_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "output_root": str(output_root),
        "status": conclusion,
        "passed_profile_ids": passed_profiles,
        "final_metrics_sha256": _sha256_file(final_path),
        "artifact_manifest_sha256": _sha256_file(artifact_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="只读审计冻结 center8 bank 的 near-focus NCC")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
