"""从 family raw_chunks 缓存导出 parent 可直接消费的 raw-optical PSF asset。

本模块刻意不导入 CLDefocus 光学传播代码。输入必须是 ``family_bank.py`` 已经
生成并带 SHA256 sidecar 的 raw chunk cache；因此该导出不会重新运行传播，也不会
改写原 retargeted family asset。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from .family_bank import (
    _EXPORT_RETARGET_AXES,
    _RAW_ACTIVE_AXES,
    analytic_centroid_labels,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"schema_version", "run", "source", "output", "gates", "data_isolation"}
    if not isinstance(config, dict) or int(config.get("schema_version", 0)) != 1:
        raise ValueError("raw-optical 导出配置必须为 schema_version: 1 mapping")
    if set(config) != required:
        raise ValueError(f"raw-optical 导出配置字段必须精确为 {sorted(required)}")
    isolation_keys = {
        "real_pdraw_accessed",
        "google_dev_accessed",
        "google_holdout_accessed",
        "dp5k_accessed",
        "stereo_training_run",
    }
    if not isinstance(config["data_isolation"], dict) or set(config["data_isolation"]) != isolation_keys:
        raise ValueError(f"data_isolation 必须精确为 {sorted(isolation_keys)}")
    if any(bool(value) for value in config["data_isolation"].values()):
        raise ValueError("raw-optical cache 导出禁止访问真实数据或启动训练")
    return config


def _load_verified_source(config: dict[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    source = config["source"]
    source_root = Path(source["asset_root"]).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source family asset 不存在：{source_root}")
    expected_manifest_sha = str(source["artifact_manifest_sha256"])
    manifest_path = source_root / "artifact_manifest.json"
    if _sha256_file(manifest_path) != expected_manifest_sha:
        raise RuntimeError("source artifact_manifest SHA256 不匹配")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank_path = source_root / "psf_bank.pt"
    expected_bank_sha = str(source["psf_bank_sha256"])
    if _sha256_file(bank_path) != expected_bank_sha:
        raise RuntimeError("source combined psf_bank SHA256 不匹配")
    payload = torch.load(bank_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(manifest, dict):
        raise RuntimeError("source family asset payload/manifest 非法")
    return source_root, payload, manifest


def _load_raw_chunks(
    source_root: Path,
    source_payload: dict[str, Any],
    source_manifest: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    profile_ids = [str(value) for value in source_payload["profile_ids"]]
    profile_documents = source_payload["family_profiles"]
    parameter_sha = {
        str(row["profile_id"]): str(row["parameter_sha256"])
        for row in profile_documents
    }
    files = source_manifest.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("source manifest 缺少 files")
    raw_banks: dict[str, torch.Tensor] = {}
    chunk_rows: list[dict[str, Any]] = []
    for cache_path in sorted((source_root / "raw_chunks").glob("chunk_*.pt")):
        relative = str(cache_path.relative_to(source_root))
        expected_sha = files.get(relative)
        if not isinstance(expected_sha, str) or _sha256_file(cache_path) != expected_sha:
            raise RuntimeError(f"raw chunk manifest SHA256 不匹配：{cache_path}")
        sidecar = cache_path.with_suffix(cache_path.suffix + ".sha256")
        if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != expected_sha:
            raise RuntimeError(f"raw chunk sidecar SHA256 不匹配：{cache_path}")
        chunk = torch.load(cache_path, map_location="cpu", weights_only=True)
        identity = chunk.get("identity") if isinstance(chunk, dict) else None
        banks = chunk.get("raw_banks") if isinstance(chunk, dict) else None
        if not isinstance(identity, dict) or not isinstance(banks, dict):
            raise RuntimeError(f"raw chunk schema 非法：{cache_path}")
        chunk_ids = [str(value) for value in identity.get("profile_ids", [])]
        if set(chunk_ids) != set(banks):
            raise RuntimeError(f"raw chunk profile 集合不一致：{cache_path}")
        identity_parameter_sha = identity.get("profile_parameter_sha256", [])
        for profile_id, digest in zip(chunk_ids, identity_parameter_sha, strict=True):
            if parameter_sha.get(profile_id) != str(digest):
                raise RuntimeError(f"raw chunk profile lineage 不匹配：{profile_id}")
            bank = torch.as_tensor(banks[profile_id]).detach().float().contiguous()
            if tuple(bank.shape) != tuple(identity["per_profile_bank_shape"]):
                raise RuntimeError(f"raw chunk bank shape 不匹配：{profile_id}")
            if not bool(torch.isfinite(bank).all()) or bool((bank < 0.0).any()):
                raise RuntimeError(f"raw chunk PSF 存在非法值：{profile_id}")
            if profile_id in raw_banks:
                raise RuntimeError(f"raw chunk profile 重复：{profile_id}")
            raw_banks[profile_id] = bank
        chunk_rows.append(
            {
                "path": str(cache_path.resolve()),
                "sha256": expected_sha,
                "identity": identity,
            }
        )
    if set(raw_banks) != set(profile_ids):
        raise RuntimeError("raw chunks 未完整覆盖 source profile 集合")
    return raw_banks, chunk_rows


def build_raw_profile_payload(
    *,
    source_payload: dict[str, Any],
    profile_index: int,
    raw_bank: torch.Tensor,
    analytic_tolerance_px: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    """构造一个 raw-optical profile payload，并执行核/标签一致性门禁。"""

    profile_id = str(source_payload["profile_ids"][profile_index])
    expected_shape = (
        int(torch.as_tensor(source_payload["f_numbers"]).numel()),
        int(torch.as_tensor(source_payload["signed_coc_bins_px"]).shape[-1]),
        *tuple(int(value) for value in source_payload["field_grid_hw"]),
        2,
        int(source_payload["kernel_size"]),
        int(source_payload["kernel_size"]),
    )
    if tuple(raw_bank.shape) != expected_shape:
        raise ValueError(f"raw bank shape 不匹配：{tuple(raw_bank.shape)} != {expected_shape}")
    raw_np = raw_bank.detach().cpu().numpy().astype(np.float32, copy=False)
    analytic = analytic_centroid_labels(raw_np)
    source_raw = torch.as_tensor(source_payload["raw_analytic_disparity_bins_px"])[
        profile_index
    ].detach().cpu().numpy().astype(np.float64, copy=False)
    agreement = float(np.max(np.abs(analytic - source_raw)))
    if agreement > float(analytic_tolerance_px):
        raise RuntimeError(
            "raw kernel 重算 centroid 与 source raw_analytic 不一致："
            f"profile={profile_id}, max_error={agreement:.9g}px"
        )
    energy = raw_np.sum(axis=(-2, -1), dtype=np.float64)
    if not np.isfinite(energy).all() or float(energy.min()) <= 0.0:
        raise RuntimeError(f"raw kernel energy 非法：{profile_id}")
    profile_document = source_payload["family_profiles"][profile_index]
    throughput = torch.as_tensor(source_payload["side_throughput_grid"])[profile_index]
    asset_id = f"{source_payload['asset_id']}_{profile_id}_raw_optical_v1"
    payload = {
        "asset_id": asset_id,
        "profile_id": profile_id,
        "family_id": source_payload["family_id"],
        "family_sha256": source_payload["family_sha256"],
        "family_profile": profile_document,
        "profile_axis_contract": {
            "raw_active_axes": list(_RAW_ACTIVE_AXES),
            "inactive_export_retarget_axes": list(_EXPORT_RETARGET_AXES),
            "inactive_profile_parameters_for_raw_export": list(
                _EXPORT_RETARGET_AXES
            ),
            "centroid_slope_px_per_coc_active": False,
        },
        "centroid_policy": "native",
        "centroid_variant": "raw_optical_v1",
        "pdoffset_embedded": False,
        "asset_spec": {
            "kind": "cldefocus_same_family_raw_optical_single_profile_multi_aperture_v1",
            "axis_order": [
                "aperture", "signed_coc", "field_y", "field_x", "side", "h", "w"
            ],
            "signed_disparity": "d=x_L-x_R",
            "analytic_label_source": "raw_psf_centroid_mu_left_x_minus_mu_right_x",
            "centroid_variant": "raw_optical_v1",
            "pd_offset_embedded": False,
            "pd_offset_contract": "disp_gt=analytic_centroid_disp+independent_pd_offset_px",
            "ncc_role": "none_never_label_writeback",
            "same_lens_shared_ray_geometry": True,
            "source_family_asset_id": source_payload["asset_id"],
            "raw_active_axes": list(_RAW_ACTIVE_AXES),
            "inactive_export_retarget_axes": list(_EXPORT_RETARGET_AXES),
        },
        "f_numbers": torch.as_tensor(source_payload["f_numbers"]).detach().clone(),
        "signed_coc_bins_px": torch.as_tensor(
            source_payload["signed_coc_bins_px"]
        ).detach().clone(),
        "analytic_disparity_bins_px": torch.from_numpy(analytic.copy()),
        "raw_analytic_disparity_bins_px": torch.from_numpy(analytic.copy()),
        "source_raw_analytic_disparity_bins_px": torch.from_numpy(source_raw.copy()),
        "physical_centroid_vs_coc": source_payload["physical_centroid_vs_coc_by_profile"][
            profile_index
        ],
        "field_grid_hw": tuple(int(value) for value in source_payload["field_grid_hw"]),
        "field_x_normalized": torch.as_tensor(
            source_payload["field_x_normalized"]
        ).detach().clone(),
        "field_y_normalized": torch.as_tensor(
            source_payload["field_y_normalized"]
        ).detach().clone(),
        "kernel_size": int(source_payload["kernel_size"]),
        "side_throughput_grid": throughput.detach().clone(),
        "psf_bank": raw_bank.detach().clone(),
    }
    stats = {
        "analytic_agreement_abs_max_px": agreement,
        "energy_abs_max": float(np.max(np.abs(energy - 1.0))),
        "energy_min": float(energy.min()),
        "energy_max": float(energy.max()),
    }
    return payload, stats


def _slope_stats(coc: torch.Tensor, disparity: torch.Tensor) -> tuple[float, float]:
    slopes: list[float] = []
    for aperture_index in range(int(coc.shape[0])):
        x = coc[aperture_index].double()
        denominator = (x - x.mean()).square().sum()
        rows = disparity[aperture_index].double().permute(1, 2, 0).reshape(-1, x.numel())
        for row in rows:
            slopes.append(float(((x - x.mean()) * (row - row.mean())).sum() / denominator))
    return min(slopes), max(slopes)


def validate_with_parent_loader(
    *,
    config: dict[str, Any],
    staging_root: Path,
    profile_rows: Sequence[dict[str, Any]],
    f_numbers: torch.Tensor,
    analytic_tolerance_px: float,
    expected_centroid_policy: str = "native",
    expected_centroid_variant: str = "raw_optical_v1",
) -> dict[str, Any]:
    """用冻结 parent loader 全量读取 profile×aperture，并做 bank-mode smoke。"""

    parent_root = Path(config["source"]["parent_repository"]).resolve()
    loader_path = parent_root / "dataset" / "psf_bank_renderer.py"
    renderer_path = parent_root / "dataset" / "dp_renderer.py"
    for path, key in (
        (loader_path, "parent_loader_sha256"),
        (renderer_path, "parent_renderer_sha256"),
    ):
        if _sha256_file(path) != str(config["source"][key]):
            raise RuntimeError(f"parent 实现 SHA256 不匹配：{path}")
    if str(parent_root) not in os.sys.path:
        os.sys.path.insert(0, str(parent_root))
    from dataset.dp_renderer import DPRenderer, DPRendererConfig
    from dataset.psf_bank_renderer import compute_psf_stats, load_generated_psf_bank

    combinations = 0
    centroid_error_max = 0.0
    render_smokes = 0
    for profile_row in profile_rows:
        relative_path = Path(str(profile_row["relative_path"]))
        profile_path = staging_root / relative_path
        payload = torch.load(profile_path, map_location="cpu", weights_only=True)
        if payload.get("centroid_policy") != expected_centroid_policy or payload.get(
            "centroid_variant"
        ) != expected_centroid_variant:
            raise RuntimeError(f"顶层 centroid 契约不匹配：{profile_path}")
        for aperture_index, f_number_value in enumerate(f_numbers.tolist()):
            f_number = float(f_number_value)
            bank = load_generated_psf_bank(
                psf_bank_path=profile_path,
                aperture_f_number=f_number,
                aperture_match_tolerance=1.0e-4,
                device="cpu",
                dtype=torch.float32,
            )
            if (
                bank.centroid_policy != expected_centroid_policy
                or bank.profile_id != profile_row["profile_id"]
            ):
                raise RuntimeError("parent loader 未保留 centroid/profile 契约")
            stats = compute_psf_stats(bank.psf_bank)
            actual = stats["mu_x"][..., 0] - stats["mu_x"][..., 1]
            expected = torch.as_tensor(payload["analytic_disparity_bins_px"])[
                aperture_index
            ].to(dtype=torch.float32)
            error = float((actual - expected).abs().max())
            centroid_error_max = max(centroid_error_max, error)
            if error > analytic_tolerance_px:
                raise RuntimeError(
                    "parent loader kernel centroid 与 asset analytic label 不一致："
                    f"profile={profile_row['profile_id']}, f/{f_number:g}, error={error:.9g}"
                )
            renderer = DPRenderer(
                DPRendererConfig(
                    block_size=32,
                    overlap=16,
                    method="spatial",
                    linear_light=False,
                    physical_centroid_mode="bank",
                    focus_zero_calibrate=False,
                    centroid_slope_target=None,
                ),
                psf_bank=bank,
            )
            if renderer.config.physical_centroid_mode != "bank":
                raise RuntimeError("parent DPRenderer 未进入 bank centroid 模式")
            combinations += 1

            # 每个 aperture 在第一个 profile 上做一次真实 render branch smoke；其余
            # 组合已完整 load、重算 centroid 并实例化 bank-mode renderer。
            if profile_row["profile_id"] == profile_rows[0]["profile_id"]:
                rgb = torch.zeros(1, 1, 32, 32)
                rgb[..., 16, 16] = 1.0
                coc = torch.zeros(1, 1, 32, 32)
                output = renderer.render(
                    rgb,
                    coc=coc,
                    pd_offset=torch.zeros_like(coc),
                    mode="physical_psf_fast",
                )
                if output["render_meta"].get("physical_centroid_model") != "bank":
                    raise RuntimeError("parent render smoke 未执行 bank centroid 分支")
                if any(
                    str(key).startswith("physical_centroid_delta")
                    for key in output["render_meta"]
                ):
                    raise RuntimeError("bank centroid 分支意外执行 dense centroid retarget")
                render_smokes += 1
    expected_combinations = len(profile_rows) * int(f_numbers.numel())
    if combinations != expected_combinations:
        raise RuntimeError("parent loader 未覆盖全部 profile×aperture 组合")
    return {
        "status": "pass",
        "parent_repository": str(parent_root),
        "parent_loader_sha256": _sha256_file(loader_path),
        "parent_renderer_sha256": _sha256_file(renderer_path),
        "combination_count": combinations,
        "expected_combination_count": expected_combinations,
        "bank_mode_renderer_instance_count": combinations,
        "bank_mode_render_smoke_count": render_smokes,
        "centroid_agreement_abs_max_px": centroid_error_max,
        "dense_centroid_retarget_used": False,
    }


def export_raw_optical(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    source_root, source_payload, source_manifest = _load_verified_source(config)
    raw_banks, chunk_rows = _load_raw_chunks(source_root, source_payload, source_manifest)
    output_root = Path(config["output"]["root"]).resolve()
    if output_root.exists():
        raise FileExistsError(f"raw-optical 输出已存在，拒绝覆盖：{output_root}")
    staging = output_root.with_name(f".{output_root.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"raw-optical staging 已存在：{staging}")
    staging.mkdir(parents=True)

    tolerance = float(config["gates"]["analytic_agreement_abs_max_px"])
    energy_tolerance = float(config["gates"]["energy_abs_max"])
    profile_rows: list[dict[str, Any]] = []
    all_labels: list[torch.Tensor] = []
    for profile_index, profile_id_value in enumerate(source_payload["profile_ids"]):
        profile_id = str(profile_id_value)
        payload, stats = build_raw_profile_payload(
            source_payload=source_payload,
            profile_index=profile_index,
            raw_bank=raw_banks[profile_id],
            analytic_tolerance_px=tolerance,
        )
        if stats["energy_abs_max"] > energy_tolerance:
            raise RuntimeError(
                f"raw kernel energy gate 失败：profile={profile_id}, "
                f"error={stats['energy_abs_max']:.9g}"
            )
        profile_path = staging / "profiles" / profile_id / "psf_bank.pt"
        sha = _atomic_torch_save(profile_path, payload)
        profile_rows.append(
            {
                "profile_id": profile_id,
                "parameter_sha256": payload["family_profile"]["parameter_sha256"],
                "path": str((output_root / "profiles" / profile_id / "psf_bank.pt")),
                "relative_path": str(profile_path.relative_to(staging)),
                "sha256": sha,
                "shape": list(payload["psf_bank"].shape),
                **stats,
            }
        )
        all_labels.append(payload["analytic_disparity_bins_px"])

    parent_validation = validate_with_parent_loader(
        config=config,
        staging_root=staging,
        profile_rows=profile_rows,
        f_numbers=torch.as_tensor(source_payload["f_numbers"]),
        analytic_tolerance_px=tolerance,
    )

    labels = torch.stack(all_labels)
    coc = torch.as_tensor(source_payload["signed_coc_bins_px"])
    zero_indices = coc.abs().argmin(dim=1)
    zero_values = torch.stack(
        [labels[:, aperture, int(index)] for aperture, index in enumerate(zero_indices)]
    )
    field_ranges = labels.amax(dim=(-2, -1)) - labels.amin(dim=(-2, -1))
    slope_min, slope_max = _slope_stats(
        coc,
        labels.permute(1, 2, 0, 3, 4).reshape(
            coc.shape[0], coc.shape[1], -1, labels.shape[-1]
        ),
    )
    summary = {
        "schema_version": 1,
        "status": "pass",
        "run_id": config["run"]["id"],
        "asset_id": config["run"]["asset_id"],
        "centroid_policy": "native",
        "centroid_variant": "raw_optical_v1",
        "pdoffset_embedded": False,
        "profile_axis_contract": {
            "raw_active_axes": list(_RAW_ACTIVE_AXES),
            "inactive_export_retarget_axes": list(_EXPORT_RETARGET_AXES),
            "inactive_profile_parameters_for_raw_export": list(
                _EXPORT_RETARGET_AXES
            ),
            "centroid_slope_px_per_coc_active": False,
        },
        "source_asset_root": str(source_root),
        "source_asset_manifest_sha256": config["source"]["artifact_manifest_sha256"],
        "source_psf_bank_sha256": config["source"]["psf_bank_sha256"],
        "source_chunk_count": len(chunk_rows),
        "source_propagation_reused": True,
        "new_optical_propagation_run": False,
        "profile_count": len(profile_rows),
        "aperture_count": int(torch.as_tensor(source_payload["f_numbers"]).numel()),
        "shape_per_profile": profile_rows[0]["shape"],
        "combined_shape": [len(profile_rows), *profile_rows[0]["shape"]],
        "raw_coc_zero_disparity_abs_max_px": float(zero_values.abs().max()),
        "raw_field_disparity_range_max_px": float(field_ranges.max()),
        "raw_centroid_slope_min_px_per_coc": slope_min,
        "raw_centroid_slope_max_px_per_coc": slope_max,
        "analytic_agreement_abs_max_px": max(
            float(row["analytic_agreement_abs_max_px"]) for row in profile_rows
        ),
        "energy_abs_max": max(float(row["energy_abs_max"]) for row in profile_rows),
        "profiles": profile_rows,
        "source_chunks": chunk_rows,
        "parent_validation": parent_validation,
        **config["data_isolation"],
    }
    _atomic_write_text(
        staging / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    resolved = {
        **config,
        "source": {**config["source"], "asset_root": str(source_root)},
        "output": {**config["output"], "root": str(output_root)},
    }
    _atomic_write_text(
        staging / "resolved_config.yaml",
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
    )
    report = f"""# CLDefocus raw-optical family asset 导出报告

状态：**PASS**。

本资产从原 family asset 已通过 SHA256 门禁的 `raw_chunks` 缓存重打包，未运行任何
新的 CLDefocus 光学传播，也未修改原 retargeted asset。每个 profile 的
`psf_bank.pt` 都是 parent loader 可直接读取的 7D multi-aperture bank。

- profile 数：{summary['profile_count']}
- 光圈数：{summary['aperture_count']}
- 单 profile 形状：`{summary['shape_per_profile']}`
- `CoC=0` raw disparity 最大绝对值：{summary['raw_coc_zero_disparity_abs_max_px']:.9f} px
- raw field disparity range 最大值：{summary['raw_field_disparity_range_max_px']:.9f} px
- raw centroid slope 范围：{slope_min:.9f}–{slope_max:.9f} px/CoC
- raw kernel 与冻结 raw analytic label 最大误差：{summary['analytic_agreement_abs_max_px']:.3e} px
- PSF energy 最大误差：{summary['energy_abs_max']:.3e}
- Parent loader/bank-mode 全量组合：{parent_validation['combination_count']}/{parent_validation['expected_combination_count']} PASS
- Parent kernel/analytic centroid 最大误差：{parent_validation['centroid_agreement_abs_max_px']:.3e} px
- Parent 实际 bank render smoke：{parent_validation['bank_mode_render_smoke_count']} 个光圈，未执行 dense centroid retarget

Parent 使用时应设置 `psf_physical_centroid_mode: bank`、
`psf_centroid_slope_target: null`、`psf_focus_zero_calibrate: false`，并按样本光圈
选择 `profiles/Fxxx/psf_bank.pt`。PDOFFSET 仍由 bank 外独立加入。
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
        "centroid_variant": "raw_optical_v1",
        "raw_active_axes": list(_RAW_ACTIVE_AXES),
        "inactive_export_retarget_axes": list(_EXPORT_RETARGET_AXES),
        "inactive_profile_parameters_for_raw_export": list(
            _EXPORT_RETARGET_AXES
        ),
        "files": files,
    }
    _atomic_write_text(
        staging / "artifact_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, output_root)
    return {
        "output_root": str(output_root),
        "status": "pass",
        "artifact_manifest_sha256": _sha256_file(output_root / "artifact_manifest.json"),
        "summary_sha256": _sha256_file(output_root / "summary.json"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    result = export_raw_optical(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
