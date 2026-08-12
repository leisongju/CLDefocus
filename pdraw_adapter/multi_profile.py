"""CLDefocus 多光圈、空间变化 Dual-Pixel response-profile bank。

该生成器只使用冻结复合镜头处方和确定性合成纹理。每个 profile 的唯一训练标签来自
归一化左右 PSF 的解析质心差 ``mu_left_x-mu_right_x``；tile NCC 只做独立准入诊断，
绝不回写、缩放或平移 label。PDOFFSET 不进入 bank，左右 throughput 作为归一化 PSF
之外的显式 ``side_throughput_grid`` 交给 DPRenderer 应用。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import yaml

try:
    from .multi_aperture import (
        _deterministic_texture,
        _git_output,
        _lens_with_stop_radius,
        _linear_fit,
        _normalize_psf_jax,
        _sha256_file,
        _sha256_json,
        _signed_coc_px_at_propagation,
        build_aperture_plan,
        centroid_xy,
    )
except ImportError:  # 直接执行本文件时没有 package context。
    from multi_aperture import (  # type: ignore[no-redef]
        _deterministic_texture,
        _git_output,
        _lens_with_stop_radius,
        _linear_fit,
        _normalize_psf_jax,
        _sha256_file,
        _sha256_json,
        _signed_coc_px_at_propagation,
        build_aperture_plan,
        centroid_xy,
    )


@dataclass(frozen=True)
class ResponseProfile:
    """同一镜头上的受约束 DP sensor-response 先验，而非独立镜头处方。"""

    profile_id: str
    seed: int
    transition_width: float
    cross_talk: float
    split_bias_norm: float
    boundary_curvature_norm: float
    microlens_edge_rolloff: float
    field_split_slope: float
    field_acceptance_slope: float
    lr_throughput_delta: float
    field_lr_throughput_slope: float
    aperture_lr_throughput_slope: float


@dataclass(frozen=True)
class LowOrderAberrationProfile:
    """同一基础镜头 family 上叠加的低阶 pupil phase，系数单位为 waves。

    基函数刻意不含 defocus：signed CoC 仍只由 sensor propagation 决定。field 项
    在归一化水平视场上作线性变化，使像差随机化与 pupil response、PD centroid
    slope、aperture 四个轴彼此独立。
    """

    astigmatism_0_waves: float = 0.0
    astigmatism_45_waves: float = 0.0
    coma_x_waves: float = 0.0
    coma_y_waves: float = 0.0
    spherical_waves: float = 0.0
    field_astigmatism_0_waves_per_norm: float = 0.0
    field_astigmatism_45_waves_per_norm: float = 0.0
    field_coma_x_waves_per_norm: float = 0.0
    field_coma_y_waves_per_norm: float = 0.0


def low_order_aberration_waves(
    pupil_x: Any,
    pupil_y: Any,
    valid_mask: Any,
    profile: LowOrderAberrationProfile,
    *,
    field_x: float,
    field_y: float = 0.0,
) -> tuple[Any, dict[str, Any]]:
    """在归一化椭圆 pupil 上生成零 piston 的低阶 phase screen（waves）。"""

    import jax.numpy as jnp

    x = jnp.asarray(pupil_x)
    y = jnp.asarray(pupil_y)
    valid = jnp.asarray(valid_mask, dtype=bool) & jnp.isfinite(x) & jnp.isfinite(y)
    inf = jnp.asarray(jnp.inf, dtype=x.dtype)
    x_min = jnp.min(jnp.where(valid, x, inf))
    x_max = jnp.max(jnp.where(valid, x, -inf))
    y_min = jnp.min(jnp.where(valid, y, inf))
    y_max = jnp.max(jnp.where(valid, y, -inf))
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    radius_x = jnp.maximum(0.5 * (x_max - x_min), 1.0e-20)
    radius_y = jnp.maximum(0.5 * (y_max - y_min), 1.0e-20)
    ux = (x - center_x) / radius_x
    uy = (y - center_y) / radius_y
    radius_squared = ux**2 + uy**2
    field_value = float(field_x)
    field_y_value = float(field_y)
    astigmatism_0 = (
        float(profile.astigmatism_0_waves)
        + field_value * float(profile.field_astigmatism_0_waves_per_norm)
    )
    coma_x = (
        float(profile.coma_x_waves)
        + field_value * float(profile.field_coma_x_waves_per_norm)
    )
    astigmatism_45 = (
        float(profile.astigmatism_45_waves)
        + field_y_value * float(profile.field_astigmatism_45_waves_per_norm)
    )
    coma_y = (
        float(profile.coma_y_waves)
        + field_y_value * float(profile.field_coma_y_waves_per_norm)
    )
    screen = (
        astigmatism_0 * (ux**2 - uy**2)
        + astigmatism_45 * (2.0 * ux * uy)
        + coma_x * ((3.0 * radius_squared - 2.0) * ux)
        + coma_y * ((3.0 * radius_squared - 2.0) * uy)
        + float(profile.spherical_waves)
        * (6.0 * radius_squared**2 - 6.0 * radius_squared + 1.0)
    )
    valid_count = jnp.maximum(jnp.count_nonzero(valid), 1)
    piston = jnp.where(valid, screen, 0.0).sum() / valid_count
    screen = jnp.where(valid, screen - piston, 0.0)
    return screen, {
        "basis": "unnormalized_low_order_zernike_like_no_defocus_v1",
        "coefficient_unit": "waves",
        "effective_astigmatism_0_waves": astigmatism_0,
        "effective_astigmatism_45_waves": astigmatism_45,
        "effective_coma_x_waves": coma_x,
        "effective_coma_y_waves": coma_y,
        "piston_removed_waves": piston,
    }


def _profile_seed(master_seed: int, profile_id: str, values: Sequence[float]) -> int:
    encoded = "|".join(
        ["cldefocus-v7", str(int(master_seed)), str(profile_id)]
        + [f"{float(value):.17g}" for value in values]
    )
    return int.from_bytes(hashlib.sha256(encoded.encode("utf-8")).digest()[:4], "big") & 0x7FFFFFFF


def build_profile_catalog(
    *,
    count: int = 32,
    master_seed: int = 20260812,
) -> list[ResponseProfile]:
    """生成冻结 P000 anchor + 31 行 LHS response-profile 目录。"""

    from scipy.stats import qmc

    if int(count) != 32:
        raise ValueError("冻结 V7 目录固定为 32 个 profile")
    primary = qmc.LatinHypercube(
        d=3,
        scramble=True,
        strength=1,
        optimization=None,
        seed=int(master_seed),
    ).random(n=31, workers=1)
    extra = qmc.LatinHypercube(
        d=7,
        scramble=True,
        strength=1,
        optimization=None,
        seed=int(master_seed) + 1,
    ).random(n=31, workers=1)

    rows: list[ResponseProfile] = []
    anchor_values = (0.74, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rows.append(
        ResponseProfile(
            profile_id="P000",
            seed=_profile_seed(master_seed, "P000", anchor_values),
            transition_width=anchor_values[0],
            cross_talk=anchor_values[1],
            split_bias_norm=anchor_values[2],
            boundary_curvature_norm=anchor_values[3],
            microlens_edge_rolloff=anchor_values[4],
            field_split_slope=anchor_values[5],
            field_acceptance_slope=anchor_values[6],
            lr_throughput_delta=anchor_values[7],
            field_lr_throughput_slope=anchor_values[8],
            aperture_lr_throughput_slope=anchor_values[9],
        )
    )
    for index in range(31):
        profile_id = f"P{index + 1:03d}"
        transition = 0.55 + 0.40 * float(primary[index, 0])
        cross_talk = 0.05 * float(primary[index, 1])
        split_bias = -0.025 + 0.05 * float(primary[index, 2])
        # 额外维度只覆盖温和、可解释的 sensor response variation。
        curvature = -0.015 + 0.030 * float(extra[index, 0])
        edge_rolloff = 0.18 * float(extra[index, 1])
        field_split = -0.020 + 0.040 * float(extra[index, 2])
        field_acceptance = -0.08 + 0.16 * float(extra[index, 3])
        lr_delta = -0.025 + 0.050 * float(extra[index, 4])
        field_lr = -0.015 + 0.030 * float(extra[index, 5])
        aperture_lr = -0.012 + 0.024 * float(extra[index, 6])
        values = (
            transition,
            cross_talk,
            split_bias,
            curvature,
            edge_rolloff,
            field_split,
            field_acceptance,
            lr_delta,
            field_lr,
            aperture_lr,
        )
        rows.append(
            ResponseProfile(
                profile_id=profile_id,
                seed=_profile_seed(master_seed, profile_id, values),
                transition_width=transition,
                cross_talk=cross_talk,
                split_bias_norm=split_bias,
                boundary_curvature_norm=curvature,
                microlens_edge_rolloff=edge_rolloff,
                field_split_slope=field_split,
                field_acceptance_slope=field_acceptance,
                lr_throughput_delta=lr_delta,
                field_lr_throughput_slope=field_lr,
                aperture_lr_throughput_slope=aperture_lr,
            )
        )
    return rows


def build_explicit_profile_catalog(
    rows: Sequence[dict[str, Any]],
) -> list[ResponseProfile]:
    """从版本化配置构造显式 profile，要求完整且无歧义的 dataclass 字段。

    旧 ``build_profile_catalog`` 不调用本函数，其 32 行 LHS 数值、顺序和 seed 因而保持
    完全不变。
    """

    parameter_names = (
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
    expected_keys = {"profile_id", "seed", *parameter_names}
    output: list[ResponseProfile] = []
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    for index, source in enumerate(rows):
        if not isinstance(source, dict):
            raise ValueError(f"explicit profile 第 {index} 行必须为 mapping")
        missing = expected_keys - set(source)
        unexpected = set(source) - expected_keys
        if missing or unexpected:
            raise ValueError(
                f"explicit profile 第 {index} 行字段不完整："
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        profile_id = str(source.get("profile_id", "")).strip()
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", profile_id) is None:
            raise ValueError(f"explicit profile ID 不安全：{profile_id!r}")
        if profile_id in seen_ids:
            raise ValueError(f"explicit profile ID 必须唯一：{profile_id!r}")
        seen_ids.add(profile_id)
        seed_value = source["seed"]
        if isinstance(seed_value, bool) or not isinstance(seed_value, int):
            raise ValueError(f"explicit profile {profile_id} seed 必须为 int")
        seed = int(seed_value)
        if not 0 <= seed <= 0x7FFFFFFF or seed in seen_seeds:
            raise ValueError(f"explicit profile seed 必须唯一且位于 [0,2^31-1]：{seed}")
        seen_seeds.add(seed)
        raw_values = [source[name] for name in parameter_names]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_values):
            raise ValueError(f"explicit profile {profile_id} 参数必须为 numeric 且不得为 bool")
        values = tuple(float(value) for value in raw_values)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"explicit profile {profile_id} 参数必须全部有限")
        transition_width, cross_talk = values[:2]
        if transition_width <= 0.0:
            raise ValueError(f"explicit profile {profile_id} transition_width 必须为正数")
        if not 0.0 <= cross_talk < 0.5:
            raise ValueError(f"explicit profile {profile_id} cross_talk 必须位于 [0,.5)")
        if abs(values[2]) >= 1.0:
            raise ValueError(f"explicit profile {profile_id} split_bias_norm 必须位于 (-1,1)")
        if values[4] < 0.0:
            raise ValueError(f"explicit profile {profile_id} microlens_edge_rolloff 不得为负")
        output.append(
            ResponseProfile(
                profile_id=profile_id,
                seed=seed,
                transition_width=values[0],
                cross_talk=values[1],
                split_bias_norm=values[2],
                boundary_curvature_norm=values[3],
                microlens_edge_rolloff=values[4],
                field_split_slope=values[5],
                field_acceptance_slope=values[6],
                lr_throughput_delta=values[7],
                field_lr_throughput_slope=values[8],
                aperture_lr_throughput_slope=values[9],
            )
        )
    if not output:
        raise ValueError("explicit_profiles 不得为空")
    return output


def resolve_profile_catalog(profile_cfg: dict[str, Any]) -> tuple[list[ResponseProfile], str]:
    """选择旧冻结 LHS 或显式 profile 路径。"""

    explicit = profile_cfg.get("explicit_profiles")
    if explicit is None:
        return (
            build_profile_catalog(
                count=int(profile_cfg["catalog_size"]),
                master_seed=int(profile_cfg["master_seed"]),
            ),
            "frozen_lhs_v7",
        )
    if not isinstance(explicit, list):
        raise ValueError("explicit_profiles 必须为 YAML list")
    ambiguous = {"catalog_size", "master_seed"} & set(profile_cfg)
    if ambiguous:
        raise ValueError(f"explicit_profiles 不得同时声明生成式字段：{sorted(ambiguous)}")
    return (
        build_explicit_profile_catalog(explicit),
        "explicit_response_profiles_v1",
    )


def deterministic_profile_chunks(
    profiles: Sequence[ResponseProfile],
    chunk_size: int | None,
) -> list[list[ResponseProfile]]:
    """按原顺序确定性分块，降低 shared-RS profile 轴显存。"""

    values = list(profiles)
    if not values:
        return []
    size = len(values) if chunk_size is None else int(chunk_size)
    if size < 1:
        raise ValueError("batch_profile_chunk_size 必须为正数或 null")
    return [values[start : start + size] for start in range(0, len(values), size)]


def resolve_aperture_transition_power_law(
    profile_cfg: dict[str, Any],
    catalog: Sequence[ResponseProfile],
) -> dict[str, Any]:
    """解析可选的 f/# 条件化 projected-pupil transition width。

    该项改变左右 pupil response 和最终 PSF，不接触解析 label 或 NCC 结果。未声明时
    scale 恒为 1，从而保持旧单 profile/shared-RS 路径不变。
    """

    profile_ids = [profile.profile_id for profile in catalog]
    source = profile_cfg.get("aperture_transition_power_law")
    if source is None:
        return {
            "enabled": False,
            "mode": "disabled_identity",
            "reference_f_number": None,
            "exponent_by_profile_id": {profile_id: 0.0 for profile_id in profile_ids},
            "changes_psf_response": False,
            "changes_labels_or_ncc_admission": False,
        }
    if not isinstance(source, dict):
        raise ValueError("aperture_transition_power_law 必须为 mapping")
    expected = {"mode", "reference_f_number", "exponent_by_profile_id"}
    missing = expected - set(source)
    unexpected = set(source) - expected
    if missing or unexpected:
        raise ValueError(
            "aperture_transition_power_law 字段不完整："
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    mode = str(source["mode"])
    if mode != "normalized_pupil_power_law_v1":
        raise ValueError(f"未知 aperture transition law：{mode}")
    reference = source["reference_f_number"]
    if isinstance(reference, bool) or not isinstance(reference, (int, float)):
        raise ValueError("reference_f_number 必须为 numeric")
    reference_f_number = float(reference)
    if not math.isfinite(reference_f_number) or reference_f_number <= 0.0:
        raise ValueError("reference_f_number 必须为有限正数")
    raw_exponents = source["exponent_by_profile_id"]
    if not isinstance(raw_exponents, dict):
        raise ValueError("exponent_by_profile_id 必须为 mapping")
    if set(raw_exponents) != set(profile_ids):
        raise ValueError(
            "exponent_by_profile_id 必须与 catalog ID 精确一致："
            f"missing={sorted(set(profile_ids) - set(raw_exponents))}, "
            f"unexpected={sorted(set(raw_exponents) - set(profile_ids))}"
        )
    exponents: dict[str, float] = {}
    for profile_id in profile_ids:
        value = raw_exponents[profile_id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"profile {profile_id} aperture transition exponent 必须为 numeric")
        exponent = float(value)
        if not math.isfinite(exponent) or abs(exponent) > 8.0:
            raise ValueError(
                f"profile {profile_id} aperture transition exponent 必须有限且 |gamma|<=8"
            )
        exponents[profile_id] = exponent
    return {
        "enabled": True,
        "mode": mode,
        "reference_f_number": reference_f_number,
        "exponent_by_profile_id": exponents,
        "changes_psf_response": True,
        "changes_labels_or_ncc_admission": False,
    }


def aperture_transition_scale(
    profile: ResponseProfile,
    f_number: float,
    transition_law: dict[str, Any],
) -> float:
    """返回 ``(f_ref/f)^gamma``；disabled law 精确返回 1。"""

    if not bool(transition_law["enabled"]):
        return 1.0
    reference = float(transition_law["reference_f_number"])
    f_value = float(f_number)
    if not math.isfinite(f_value) or f_value <= 0.0:
        raise ValueError("f_number 必须为有限正数")
    exponent = float(transition_law["exponent_by_profile_id"][profile.profile_id])
    scale = (reference / f_value) ** exponent
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"profile {profile.profile_id} aperture transition scale 非法")
    return float(scale)


def resolve_aperture_cross_talk_by_f_number(
    profile_cfg: dict[str, Any],
    catalog: Sequence[ResponseProfile],
    f_numbers: Sequence[float],
) -> dict[str, Any]:
    """解析可选的 profile×f-number 精确 cross-talk 表。

    未声明时严格返回 profile 自带 ``cross_talk``。启用时每个 profile 必须为配置
    中每档 f-number 提供且仅提供一个值；该表只进入 pupil power mixing，不接触
    centroid label、NCC 估计或准入阈值。
    """

    profile_ids = [profile.profile_id for profile in catalog]
    aperture_values = [float(value) for value in f_numbers]
    source = profile_cfg.get("aperture_cross_talk_by_f_number")
    if source is None:
        return {
            "enabled": False,
            "mode": "disabled_profile_default",
            "match_tolerance": 0.0,
            "values_by_profile_id": {
                profile.profile_id: {
                    f"{value:g}": float(profile.cross_talk)
                    for value in aperture_values
                }
                for profile in catalog
            },
            "changes_psf_response": False,
            "changes_labels_or_ncc_admission": False,
        }
    if not isinstance(source, dict):
        raise ValueError("aperture_cross_talk_by_f_number 必须为 mapping 或 null")
    expected = {"mode", "match_tolerance", "values_by_profile_id"}
    if set(source) != expected:
        raise ValueError(
            "aperture_cross_talk_by_f_number 字段必须精确为 "
            f"{sorted(expected)}"
        )
    if str(source["mode"]) != "exact_f_number_map_v1":
        raise ValueError("aperture_cross_talk_by_f_number.mode 不受支持")
    tolerance = source["match_tolerance"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError("aperture cross-talk match_tolerance 必须为 numeric")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("aperture cross-talk match_tolerance 必须为有限非负数")
    raw_profiles = source["values_by_profile_id"]
    if not isinstance(raw_profiles, dict) or set(raw_profiles) != set(profile_ids):
        raise ValueError(
            "aperture cross-talk values_by_profile_id 必须与 catalog ID 精确一致"
        )
    resolved_values: dict[str, dict[str, float]] = {}
    for profile_id in profile_ids:
        row = raw_profiles[profile_id]
        if not isinstance(row, dict):
            raise ValueError(f"profile {profile_id} cross-talk 表必须为 mapping")
        parsed: list[tuple[float, float]] = []
        for raw_f_number, raw_value in row.items():
            if isinstance(raw_f_number, bool):
                raise ValueError("aperture cross-talk f-number key 不得为 bool")
            try:
                f_number = float(raw_f_number)
            except (TypeError, ValueError) as error:
                raise ValueError("aperture cross-talk f-number key 必须可解析为数值") from error
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError("aperture cross-talk value 必须为 numeric")
            value = float(raw_value)
            if not math.isfinite(f_number) or f_number <= 0.0:
                raise ValueError("aperture cross-talk f-number 必须为有限正数")
            if not math.isfinite(value) or not 0.0 <= value < 0.5:
                raise ValueError("aperture cross-talk value 必须位于 [0,0.5)")
            parsed.append((f_number, value))
        if len(parsed) != len(aperture_values):
            raise ValueError(
                f"profile {profile_id} cross-talk 表必须覆盖全部 f-number"
            )
        used: set[int] = set()
        canonical: dict[str, float] = {}
        for configured_f, value in parsed:
            distances = [abs(configured_f - target) for target in aperture_values]
            index = int(np.argmin(distances))
            if distances[index] > tolerance or index in used:
                raise ValueError(
                    f"profile {profile_id} cross-talk f-number 无法唯一匹配冻结光圈"
                )
            used.add(index)
            canonical[f"{aperture_values[index]:g}"] = value
        if len(used) != len(aperture_values):
            raise ValueError(
                f"profile {profile_id} cross-talk 表未覆盖全部冻结光圈"
            )
        resolved_values[profile_id] = canonical
    return {
        "enabled": True,
        "mode": "exact_f_number_map_v1",
        "match_tolerance": tolerance,
        "values_by_profile_id": resolved_values,
        "changes_psf_response": True,
        "changes_labels_or_ncc_admission": False,
    }


def aperture_cross_talk(
    profile: ResponseProfile,
    f_number: float,
    law: dict[str, Any],
) -> float:
    """返回当前 profile/f-number 的 pupil cross-talk。"""

    if not bool(law["enabled"]):
        return float(profile.cross_talk)
    values = law["values_by_profile_id"][profile.profile_id]
    target = float(f_number)
    candidates = [(abs(float(key) - target), float(value)) for key, value in values.items()]
    distance, value = min(candidates, key=lambda row: row[0])
    if distance > float(law["match_tolerance"]):
        raise ValueError(
            f"profile {profile.profile_id} 没有匹配 f/{target:g} 的 cross-talk"
        )
    return value


def build_profile_catalog_document(
    catalog: Sequence[ResponseProfile],
    *,
    catalog_source: str,
    profile_cfg: dict[str, Any],
) -> dict[str, Any]:
    """构造可落盘 catalog；冻结 LHS 分支保持 V7 字节级 JSON 语义。"""

    catalog_payload = [
        {
            **asdict(profile),
            "design_center_logical_left_fraction": circular_pupil_positive_fraction(
                profile.split_bias_norm
            ),
            "catalog_status": "candidate",
        }
        for profile in catalog
    ]
    if catalog_source == "frozen_lhs_v7":
        return {
            "schema_version": 1,
            "kind": "same_lens_constrained_dp_response_profiles",
            "not_independent_lenses": True,
            "master_seed": int(profile_cfg["master_seed"]),
            "lhs_algorithm": {
                "implementation": "scipy.stats.qmc.LatinHypercube",
                "scipy_version": "1.15.1",
                "scramble": True,
                "strength": 1,
                "optimization": None,
                "primary_seed": int(profile_cfg["master_seed"]),
                "extra_seed": int(profile_cfg["master_seed"]) + 1,
            },
            "profiles": catalog_payload,
        }
    if catalog_source != "explicit_response_profiles_v1":
        raise ValueError(f"未知 profile catalog_source：{catalog_source}")
    document = {
        "schema_version": 1,
        "kind": "same_lens_explicit_dp_response_profiles",
        "not_independent_lenses": True,
        "catalog_source": catalog_source,
        "explicit_order_preserved": True,
        "profiles": catalog_payload,
    }
    if profile_cfg.get("aperture_transition_power_law") is not None:
        document["aperture_transition_power_law"] = profile_cfg[
            "aperture_transition_power_law"
        ]
    return document


def circular_pupil_positive_fraction(split_bias_norm: float) -> float:
    """圆瞳中 ``x > split_bias_norm * radius`` 的面积比例。"""

    u = float(np.clip(split_bias_norm, -1.0, 1.0))
    return float((math.acos(u) - u * math.sqrt(max(0.0, 1.0 - u * u))) / math.pi)


def build_side_throughput_grid(
    profile: ResponseProfile,
    f_numbers: Sequence[float],
    field_x: Sequence[float],
    *,
    reference_f_number: float = 2.8,
) -> np.ndarray:
    """构造 `[A,1,Gw,2]` L/R 增益，并逐 cell 归一为左右均值 1。"""

    output = np.empty((len(f_numbers), 1, len(field_x), 2), dtype=np.float32)
    for aperture_index, f_number in enumerate(f_numbers):
        aperture_term = profile.aperture_lr_throughput_slope * math.log(
            float(f_number) / float(reference_f_number)
        )
        for field_index, field in enumerate(field_x):
            delta = (
                profile.lr_throughput_delta
                + aperture_term
                + profile.field_lr_throughput_slope * float(field)
            )
            left = 1.0 + delta
            right = 1.0 - delta
            pair_mean = 0.5 * (left + right)
            output[aperture_index, 0, field_index] = (left / pair_mean, right / pair_mean)
    if not np.isfinite(output).all() or float(output.min()) <= 0.0:
        raise ValueError(f"profile {profile.profile_id} 产生非法 throughput")
    return output


@dataclass(frozen=True)
class FieldOriginCalibrationResult:
    """离轴 field 坐标原点校准结果。

    ``raw_centroid_xy_px`` 与 ``calibrated_centroid_xy_px`` 的 shape 都是
    ``[A,N,Gh,Gw,2,2]``，最后两轴依次为 ``side=[L,R]`` 和 ``xy=[x,y]``。
    """

    psf_bank: np.ndarray
    analytic_labels_px: np.ndarray
    raw_analytic_labels_px: np.ndarray
    raw_centroid_xy_px: np.ndarray
    calibrated_centroid_xy_px: np.ndarray
    metadata: dict[str, Any]


def _bank_centroid_and_covariance(bank: np.ndarray) -> dict[str, np.ndarray]:
    """计算任意 leading shape PSF 的质心与中心二阶矩。"""

    value = np.asarray(bank, dtype=np.float64)
    if value.ndim < 2 or value.shape[-1] != value.shape[-2]:
        raise ValueError(f"PSF 最后两轴必须为方阵，实际为 {value.shape}")
    kernel_size = int(value.shape[-1])
    coordinates = np.arange(kernel_size, dtype=np.float64) - (kernel_size - 1) / 2.0
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    energy = value.sum(axis=(-2, -1))
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(energy <= 0.0):
        raise ValueError("PSF 必须有限、非负且具有正能量")
    mu_x = (value * xx).sum(axis=(-2, -1)) / energy
    mu_y = (value * yy).sum(axis=(-2, -1)) / energy
    centered_x = xx - mu_x[..., None, None]
    centered_y = yy - mu_y[..., None, None]
    covariance_xx = (value * np.square(centered_x)).sum(axis=(-2, -1)) / energy
    covariance_xy = (value * centered_x * centered_y).sum(axis=(-2, -1)) / energy
    covariance_yy = (value * np.square(centered_y)).sum(axis=(-2, -1)) / energy
    return {
        "energy": energy,
        "mu_x": mu_x,
        "mu_y": mu_y,
        "covariance_xx": covariance_xx,
        "covariance_xy": covariance_xy,
        "covariance_yy": covariance_yy,
    }


def _centroid_xy_array(stats: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack((stats["mu_x"], stats["mu_y"]), axis=-1)


def _centroid_labels_from_stats(stats: dict[str, np.ndarray]) -> np.ndarray:
    mu_x = np.asarray(stats["mu_x"], dtype=np.float64)
    if mu_x.ndim != 5 or mu_x.shape[-1] != 2:
        raise ValueError(f"bank centroid shape 必须为 [A,N,Gh,Gw,2]，实际为 {mu_x.shape}")
    return mu_x[..., 0] - mu_x[..., 1]


def _forward_bilinear_translate_psf(
    psf: np.ndarray,
    *,
    shift_x_px: float,
    shift_y_px: float,
    support_padding_px: int,
) -> tuple[np.ndarray, float]:
    """把单个 PSF 以前向双线性 splat 平移到带对称零 padding 的 support。

    padding 严格覆盖平移量时，离散 splat 同时守恒总质量和一阶矩。函数仍返回
    renormalize 前的 retained-mass fraction，供资产门禁独立检查 support 截断。
    """

    source = np.asarray(psf, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError(f"psf 必须为二维方阵，实际为 {source.shape}")
    if not np.isfinite(source).all() or np.any(source < 0.0):
        raise ValueError("psf 必须有限且非负")
    source_mass = float(source.sum())
    if source_mass <= 0.0:
        raise ValueError("psf 必须具有正能量")
    padding = int(support_padding_px)
    if padding < 0:
        raise ValueError("support_padding_px 不得为负数")
    output_size = int(source.shape[0]) + 2 * padding
    source_y, source_x = np.indices(source.shape, dtype=np.float64)
    target_x = source_x + float(padding) + float(shift_x_px)
    target_y = source_y + float(padding) + float(shift_y_px)
    x0 = np.floor(target_x).astype(np.int64)
    y0 = np.floor(target_y).astype(np.int64)
    fx = target_x - x0
    fy = target_y - y0
    output = np.zeros((output_size, output_size), dtype=np.float64)
    for delta_y, weight_y in ((0, 1.0 - fy), (1, fy)):
        for delta_x, weight_x in ((0, 1.0 - fx), (1, fx)):
            target_ix = x0 + delta_x
            target_iy = y0 + delta_y
            valid = (
                (target_ix >= 0)
                & (target_ix < output_size)
                & (target_iy >= 0)
                & (target_iy < output_size)
            )
            np.add.at(
                output,
                (target_iy[valid], target_ix[valid]),
                source[valid] * weight_x[valid] * weight_y[valid],
            )
    retained_mass = float(output.sum())
    if retained_mass <= 0.0:
        raise RuntimeError("field-origin 平移后 PSF support 内没有剩余能量")
    output /= retained_mass
    return output.astype(np.float32), retained_mass / source_mass


def calibrate_field_origin(
    bank: np.ndarray,
    signed_coc_bins_px: np.ndarray,
    *,
    centroid_residual_max_px: float = 0.01,
    curve_identity_abs_max_px: float = 2.0e-5,
    requested_support_padding_px: int = 2,
    retained_mass_min: float = 0.999999,
    centered_covariance_abs_delta_max_px2: float = 0.25,
) -> FieldOriginCalibrationResult:
    """按 aperture×field×side 去除 CoC=0 静态坐标原点。

    对每个 cell 先分解
    ``common=(mu_L0+mu_R0)/2``、``relative=(mu_L0-mu_R0)/2``，再把固定
    ``s_L=-(common+relative)=-mu_L0`` 与
    ``s_R=-(common-relative)=-mu_R0`` 应用于该 cell 的全部 CoC。NCC 不参与
    求解；PDOFFSET 不进入本函数。输出 label 只从校准后 PSF 的一阶矩重算。
    """

    value = np.asarray(bank, dtype=np.float32)
    coc = np.asarray(signed_coc_bins_px, dtype=np.float64)
    if value.ndim != 7 or value.shape[4] != 2 or value.shape[-1] != value.shape[-2]:
        raise ValueError(f"field-origin bank 必须为 [A,N,Gh,Gw,2,K,K]，实际为 {value.shape}")
    if coc.ndim == 1:
        coc = np.tile(coc[None, :], (value.shape[0], 1))
    if coc.shape != value.shape[:2]:
        raise ValueError(f"signed CoC shape 不一致：{coc.shape} vs {value.shape[:2]}")
    tolerance = float(centroid_residual_max_px)
    identity_tolerance = float(curve_identity_abs_max_px)
    covariance_tolerance = float(centered_covariance_abs_delta_max_px2)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("centroid_residual_max_px 必须为有限正数")
    if not math.isfinite(identity_tolerance) or identity_tolerance <= 0.0:
        raise ValueError("curve_identity_abs_max_px 必须为有限正数")
    if not 0.0 < float(retained_mass_min) <= 1.0:
        raise ValueError("retained_mass_min 必须位于 (0,1]")

    zero_indices: list[int] = []
    for aperture_index in range(value.shape[0]):
        exact = np.flatnonzero(np.abs(coc[aperture_index]) <= 1.0e-12)
        if exact.size != 1:
            raise ValueError("每个 aperture 的 signed CoC 必须包含且仅包含一个精确 0")
        zero_indices.append(int(exact[0]))

    raw_stats = _bank_centroid_and_covariance(value)
    raw_centroids = _centroid_xy_array(raw_stats)
    raw_labels = _centroid_labels_from_stats(raw_stats)
    raw_zero = np.stack(
        [raw_centroids[a, zero_indices[a]] for a in range(value.shape[0])],
        axis=0,
    )
    common = 0.5 * (raw_zero[..., 0, :] + raw_zero[..., 1, :])
    relative = 0.5 * (raw_zero[..., 0, :] - raw_zero[..., 1, :])
    shifts = -raw_zero
    largest_shift = float(np.max(np.abs(shifts)))
    minimum_padding = int(math.ceil(largest_shift)) + 1
    padding = max(int(requested_support_padding_px), minimum_padding)
    output_size = int(value.shape[-1]) + 2 * padding
    calibrated = np.empty((*value.shape[:-2], output_size, output_size), dtype=np.float32)
    retained = np.empty(value.shape[:-2], dtype=np.float64)
    for aperture_index in range(value.shape[0]):
        for coc_index in range(value.shape[1]):
            for field_y_index in range(value.shape[2]):
                for field_x_index in range(value.shape[3]):
                    for side_index in range(2):
                        shift_x, shift_y = shifts[
                            aperture_index,
                            field_y_index,
                            field_x_index,
                            side_index,
                        ]
                        translated, retained_fraction = _forward_bilinear_translate_psf(
                            value[
                                aperture_index,
                                coc_index,
                                field_y_index,
                                field_x_index,
                                side_index,
                            ],
                            shift_x_px=float(shift_x),
                            shift_y_px=float(shift_y),
                            support_padding_px=padding,
                        )
                        calibrated[
                            aperture_index,
                            coc_index,
                            field_y_index,
                            field_x_index,
                            side_index,
                        ] = translated
                        retained[
                            aperture_index,
                            coc_index,
                            field_y_index,
                            field_x_index,
                            side_index,
                        ] = retained_fraction

    calibrated_stats = _bank_centroid_and_covariance(calibrated)
    calibrated_centroids = _centroid_xy_array(calibrated_stats)
    calibrated_labels = _centroid_labels_from_stats(calibrated_stats)
    calibrated_zero = np.stack(
        [calibrated_centroids[a, zero_indices[a]] for a in range(value.shape[0])],
        axis=0,
    )
    expected_centroids = raw_centroids - raw_zero[:, None, ...]
    side_curve_identity_error = calibrated_centroids - expected_centroids
    raw_zero_disparity = raw_zero[..., 0, 0] - raw_zero[..., 1, 0]
    expected_labels = raw_labels - raw_zero_disparity[:, None, ...]
    disparity_curve_identity_error = calibrated_labels - expected_labels
    covariance_names = ("covariance_xx", "covariance_xy", "covariance_yy")
    covariance_delta = np.stack(
        [
            calibrated_stats[name] - raw_stats[name]
            for name in covariance_names
        ],
        axis=-1,
    )
    energy_error = np.abs(np.asarray(calibrated_stats["energy"]) - 1.0)
    checks = {
        "finite_nonnegative": bool(np.isfinite(calibrated).all() and np.all(calibrated >= 0.0)),
        "unit_energy": float(np.max(energy_error)) <= 2.0e-6,
        "retained_mass": float(np.min(retained)) >= float(retained_mass_min),
        "focus_side_centroid_residual": float(np.max(np.abs(calibrated_zero))) <= tolerance,
        "side_curve_identity": float(np.max(np.abs(side_curve_identity_error)))
        <= identity_tolerance,
        "disparity_curve_identity": float(np.max(np.abs(disparity_curve_identity_error)))
        <= identity_tolerance,
        "centered_covariance_translation_bound": float(np.max(np.abs(covariance_delta)))
        <= covariance_tolerance,
    }
    per_aperture_field: list[dict[str, Any]] = []
    for aperture_index in range(value.shape[0]):
        for field_y_index in range(value.shape[2]):
            for field_x_index in range(value.shape[3]):
                left_zero = raw_zero[aperture_index, field_y_index, field_x_index, 0]
                right_zero = raw_zero[aperture_index, field_y_index, field_x_index, 1]
                cell_identity = side_curve_identity_error[
                    aperture_index, :, field_y_index, field_x_index
                ]
                cell_disp_identity = disparity_curve_identity_error[
                    aperture_index, :, field_y_index, field_x_index
                ]
                per_aperture_field.append(
                    {
                        "aperture_index": aperture_index,
                        "field_y_index": field_y_index,
                        "field_x_index": field_x_index,
                        "raw_mu_left_zero_xy_px": left_zero.tolist(),
                        "raw_mu_right_zero_xy_px": right_zero.tolist(),
                        "raw_common_zero_xy_px": common[
                            aperture_index, field_y_index, field_x_index
                        ].tolist(),
                        "raw_relative_half_zero_xy_px": relative[
                            aperture_index, field_y_index, field_x_index
                        ].tolist(),
                        "applied_shift_left_xy_px": shifts[
                            aperture_index, field_y_index, field_x_index, 0
                        ].tolist(),
                        "applied_shift_right_xy_px": shifts[
                            aperture_index, field_y_index, field_x_index, 1
                        ].tolist(),
                        "calibrated_mu_left_zero_xy_px": calibrated_zero[
                            aperture_index, field_y_index, field_x_index, 0
                        ].tolist(),
                        "calibrated_mu_right_zero_xy_px": calibrated_zero[
                            aperture_index, field_y_index, field_x_index, 1
                        ].tolist(),
                        "raw_zero_disparity_px": float(
                            raw_zero_disparity[aperture_index, field_y_index, field_x_index]
                        ),
                        "calibrated_zero_disparity_px": float(
                            calibrated_labels[
                                aperture_index,
                                zero_indices[aperture_index],
                                field_y_index,
                                field_x_index,
                            ]
                        ),
                        "side_curve_identity_abs_max_px": float(
                            np.max(np.abs(cell_identity))
                        ),
                        "disparity_curve_identity_abs_max_px": float(
                            np.max(np.abs(cell_disp_identity))
                        ),
                        "retained_mass_min": float(
                            np.min(
                                retained[
                                    aperture_index, :, field_y_index, field_x_index
                                ]
                            )
                        ),
                    }
                )
    metadata = {
        "schema_version": 1,
        "mode": "per_aperture_field_side_focus_centroid_translation_v1",
        "role": "field_coordinate_origin_calibration_only",
        "is_ncc_correction": False,
        "ncc_used_to_solve": False,
        "is_pdoffset": False,
        "pd_offset_embedded": False,
        "label_source": "calibrated PSF centroid mu_left_x-mu_right_x",
        "formula": {
            "common": "c0=(mu_L0+mu_R0)/2",
            "relative_half": "r0=(mu_L0-mu_R0)/2",
            "left_shift": "s_L=-(c0+r0)=-mu_L0",
            "right_shift": "s_R=-(c0-r0)=-mu_R0",
            "side_curve": "mu'_side(c)=mu_side(c)-mu_side(0)",
            "disparity_curve": "d'(c)=d_raw(c)-d_raw(0)",
        },
        "axis_order": {
            "raw_zero_centroid_xy_px": ["aperture", "field_y", "field_x", "side", "xy"],
            "common_relative_xy_px": ["aperture", "field_y", "field_x", "xy"],
        },
        "raw_kernel_size": int(value.shape[-1]),
        "calibrated_kernel_size": output_size,
        "requested_support_padding_px": int(requested_support_padding_px),
        "minimum_support_padding_px": minimum_padding,
        "effective_support_padding_px": padding,
        "translation_operator": "forward_bilinear_splat_then_unit_energy_renormalize",
        "raw_zero_centroid_xy_px": raw_zero.tolist(),
        "raw_common_zero_xy_px": common.tolist(),
        "raw_relative_half_zero_xy_px": relative.tolist(),
        "applied_shift_xy_px": shifts.tolist(),
        "calibrated_zero_centroid_xy_px": calibrated_zero.tolist(),
        "raw_zero_disparity_px": raw_zero_disparity.tolist(),
        "raw_analytic_disparity_bins_px": raw_labels.tolist(),
        "calibrated_analytic_disparity_bins_px": calibrated_labels.tolist(),
        "checks": checks,
        "pass": bool(all(checks.values())),
        "focus_side_centroid_residual_abs_max_px": float(np.max(np.abs(calibrated_zero))),
        "side_curve_identity_abs_max_px": float(np.max(np.abs(side_curve_identity_error))),
        "disparity_curve_identity_abs_max_px": float(
            np.max(np.abs(disparity_curve_identity_error))
        ),
        "retained_mass_min": float(np.min(retained)),
        "retained_mass_max": float(np.max(retained)),
        "unit_energy_abs_max": float(np.max(energy_error)),
        "centered_covariance_components": list(covariance_names),
        "centered_covariance_abs_delta_max_px2": float(np.max(np.abs(covariance_delta))),
        "per_aperture_field": per_aperture_field,
    }
    return FieldOriginCalibrationResult(
        psf_bank=calibrated,
        analytic_labels_px=calibrated_labels,
        raw_analytic_labels_px=raw_labels,
        raw_centroid_xy_px=raw_centroids,
        calibrated_centroid_xy_px=calibrated_centroids,
        metadata=metadata,
    )


def apply_field_origin_to_propagation_records(
    propagation_records: Sequence[dict[str, Any]],
    calibration: FieldOriginCalibrationResult,
) -> list[dict[str, Any]]:
    """把 propagation audit 中的 raw 一阶矩改写为显式 raw/calibrated 双记录。"""

    shifts = np.asarray(calibration.metadata["applied_shift_xy_px"], dtype=np.float64)
    output: list[dict[str, Any]] = []
    for source in propagation_records:
        row = dict(source)
        aperture_index = int(row["aperture_index"])
        coc_index = int(row["coc_index"])
        field_y_index = int(row.get("field_y_index", 0))
        field_x_index = int(row["field_x_index"])
        raw = calibration.raw_centroid_xy_px[
            aperture_index, coc_index, field_y_index, field_x_index
        ]
        calibrated = calibration.calibrated_centroid_xy_px[
            aperture_index, coc_index, field_y_index, field_x_index
        ]
        row.update(
            {
                "field_origin_calibrated": True,
                "field_origin_calibration_role": "coordinate_origin_only_not_ncc_not_pdoffset",
                "raw_analytic_centroid_disparity_px": float(raw[0, 0] - raw[1, 0]),
                "raw_mu_left_x_px": float(raw[0, 0]),
                "raw_mu_left_y_px": float(raw[0, 1]),
                "raw_mu_right_x_px": float(raw[1, 0]),
                "raw_mu_right_y_px": float(raw[1, 1]),
                "analytic_centroid_disparity_px": float(calibrated[0, 0] - calibrated[1, 0]),
                "mu_left_x_px": float(calibrated[0, 0]),
                "mu_left_y_px": float(calibrated[0, 1]),
                "mu_right_x_px": float(calibrated[1, 0]),
                "mu_right_y_px": float(calibrated[1, 1]),
                "field_origin_shift_left_xy_px": shifts[
                    aperture_index, field_y_index, field_x_index, 0
                ].tolist(),
                "field_origin_shift_right_xy_px": shifts[
                    aperture_index, field_y_index, field_x_index, 1
                ].tolist(),
            }
        )
        output.append(row)
    return output


def _profile_power_weights(
    pupil_x: Any,
    pupil_y: Any,
    valid_mask: Any,
    *,
    split_bias_norm: float,
    boundary_curvature_norm: float,
    transition_width: float,
    cross_talk: float,
    edge_rolloff: float,
) -> tuple[Any, Any, dict[str, Any]]:
    """在出口瞳坐标上构造连续、互补且带温和径向接受的 L/R 功率。"""

    import jax.numpy as jnp

    x = jnp.asarray(pupil_x)
    y = jnp.asarray(pupil_y)
    valid = jnp.asarray(valid_mask, dtype=bool) & jnp.isfinite(x) & jnp.isfinite(y)
    inf = jnp.asarray(jnp.inf, dtype=x.dtype)
    x_min = jnp.min(jnp.where(valid, x, inf))
    x_max = jnp.max(jnp.where(valid, x, -inf))
    y_min = jnp.min(jnp.where(valid, y, inf))
    y_max = jnp.max(jnp.where(valid, y, -inf))
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    radius_x = jnp.maximum(0.5 * (x_max - x_min), 1.0e-20)
    radius_y = jnp.maximum(0.5 * (y_max - y_min), 1.0e-20)
    ux = (x - center_x) / radius_x
    uy = (y - center_y) / radius_y
    boundary_u = float(split_bias_norm) + float(boundary_curvature_norm) * uy**2
    signed_u = ux - boundary_u
    width = float(transition_width)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("transition_width 必须为有限正数")
    amount = float(cross_talk)
    if not 0.0 <= amount < 0.5:
        raise ValueError("cross_talk 必须位于 [0,0.5)")
    rolloff = float(edge_rolloff)
    if not math.isfinite(rolloff) or rolloff < 0.0:
        raise ValueError("edge_rolloff 必须为有限非负数")

    logical_left = 0.5 * (jnp.tanh(0.5 * signed_u / width) + 1.0)
    logical_right = 1.0 - logical_left
    if amount > 0.0:
        logical_left, logical_right = (
            (1.0 - amount) * logical_left + amount * logical_right,
            (1.0 - amount) * logical_right + amount * logical_left,
        )
    radius_norm = jnp.sqrt(ux**2 + uy**2)
    acceptance = jnp.exp(-rolloff * radius_norm**4)
    left = jnp.where(valid, logical_left * acceptance, 0.0)
    right = jnp.where(valid, logical_right * acceptance, 0.0)
    return left, right, {
        "pupil_center_x_m": center_x,
        "pupil_center_y_m": center_y,
        "pupil_radius_x_m": radius_x,
        "pupil_radius_y_m": radius_y,
        "split_x_center_m": center_x + float(split_bias_norm) * radius_x,
    }


def _propagate_profile_pair(
    wf: Any,
    *,
    s_prop: float,
    parax: Any,
    sensor: Any,
    kernel_size: int,
    pixel_pitch_m: float,
    profile: ResponseProfile,
    field_x: float,
    field_y: float = 0.0,
    upsample: int,
    aperture_transition_scale_value: float = 1.0,
    aberration_profile: LowOrderAberrationProfile | None = None,
) -> dict[str, Any]:
    """将同一完整镜头出口波前按一个 response profile 分成左右通道传播。"""

    import jax.numpy as jnp
    import datasyn.jaxutils.nputils as nputils
    import datasyn.mathutils.vecop as vecop
    import datasyn.optics.safeop as safeop
    from datasyn.mathutils.complex import sqabs_complex
    from datasyn.optics.imaging.defocus import downsample_integer
    from datasyn.optics.imaging.imaging import wh_physics_to_image
    from datasyn.optics.imaging.pupil_function.pf import opd_to_phase_factor
    from datasyn.optics.imaging.rs import rayleigh_sommerfeld
    from datasyn.optics.ray import project_to_z

    if kernel_size <= 0 or kernel_size % 2 == 0 or upsample <= 0:
        raise ValueError("kernel_size 必须是正奇数，upsample 必须为正数")
    z_prop = parax.xp.z + float(s_prop)
    viewport_xy_world = project_to_z(z_prop, wf.chief).ray.o[0:2]
    viewport_xy_pix = sensor.quantize(viewport_xy_world).index
    half = kernel_size // 2
    viewport = sensor.slice(
        (viewport_xy_pix[0] - half, viewport_xy_pix[1] - half),
        (kernel_size, kernel_size),
    )
    viewport_fine = viewport.upsample(upsample)
    eval_points = nputils.flatten(viewport_fine.grid_points(), 0, 2)
    eval_points = vecop.xy2xyz(eval_points, z_prop)

    local_split = profile.split_bias_norm + profile.field_split_slope * float(field_x)
    local_width = profile.transition_width * float(aperture_transition_scale_value) * (
        1.0 + profile.field_acceptance_slope * float(field_x)
    )
    left_power, right_power, pupil_meta = _profile_power_weights(
        wf.wf.pts[..., 0],
        wf.wf.pts[..., 1],
        wf.wf.mask,
        split_bias_norm=local_split,
        boundary_curvature_norm=profile.boundary_curvature_norm,
        transition_width=local_width,
        cross_talk=profile.cross_talk,
        edge_rolloff=profile.microlens_edge_rolloff,
    )
    phase = opd_to_phase_factor(wf.wf.wvl, wf.wf.opd)
    aberration_meta: dict[str, Any] = {
        "basis": "none",
        "coefficient_unit": "waves",
        "effective_astigmatism_0_waves": 0.0,
        "effective_coma_x_waves": 0.0,
        "piston_removed_waves": 0.0,
    }
    if aberration_profile is not None:
        screen_waves, aberration_meta = low_order_aberration_waves(
            wf.wf.pts[..., 0],
            wf.wf.pts[..., 1],
            wf.wf.mask,
            aberration_profile,
            field_x=float(field_x),
            field_y=float(field_y),
        )
        phase = phase * jnp.exp(2j * jnp.pi * screen_waves)
    base_amplitude = jnp.asarray(wf.wf.amp)
    normal = safeop.normdir(wf.xp_sphere.c[None] - wf.wf.pts).v

    def propagate(power: Any) -> Any:
        wave = base_amplitude * jnp.sqrt(jnp.maximum(power, 0.0)) * phase
        field = rayleigh_sommerfeld(
            wf.wf.pts,
            wave,
            eval_points,
            wf.wf.wvl,
            ns=normal,
        )
        field = nputils.unflatten(field, 0, viewport_fine.shape)
        intensity = downsample_integer(sqabs_complex(field), upsample)
        intensity = wh_physics_to_image(intensity, 0, 1)
        return _normalize_psf_jax(intensity)

    left = propagate(left_power)
    right = propagate(right_power)
    valid_count = jnp.maximum(jnp.count_nonzero(jnp.asarray(wf.wf.mask, dtype=bool)), 1)
    return {
        "left": left,
        "right": right,
        "left_effective_pupil_power": left_power.sum(),
        "right_effective_pupil_power": right_power.sum(),
        "pupil_overlap_fraction": (
            jnp.count_nonzero((left_power > 1.0e-10) & (right_power > 1.0e-10))
            / valid_count
        ),
        "local_split_bias_norm": local_split,
        "local_transition_width": local_width,
        "low_order_aberration": aberration_meta,
        **pupil_meta,
    }


def _propagate_profile_batch(
    wf: Any,
    *,
    s_prop: float,
    parax: Any,
    sensor: Any,
    kernel_size: int,
    profiles: Sequence[ResponseProfile],
    field_x: float,
    field_y: float = 0.0,
    upsample: int,
    sensor_chunk_size: int = 256,
    aperture_transition_scales: Sequence[float] | None = None,
    aberration_profiles: Sequence[LowOrderAberrationProfile] | None = None,
) -> list[dict[str, Any]]:
    """共享同一 RS 几何核，一次传播多个 profile 的左右复振幅。"""

    import jax.numpy as jnp
    import datasyn.jaxutils.nputils as nputils
    import datasyn.mathutils.vecop as vecop
    import datasyn.optics.safeop as safeop
    from datasyn.jaxutils import wrappings
    from datasyn.mathutils.complex import sqabs_complex
    from datasyn.optics.imaging.defocus import downsample_integer
    from datasyn.optics.imaging.imaging import wh_physics_to_image
    from datasyn.optics.imaging.pupil_function.pf import opd_to_phase_factor
    from datasyn.optics.ray import project_to_z

    if not profiles:
        raise ValueError("profiles 不能为空")
    if int(sensor_chunk_size) < 1:
        raise ValueError("sensor_chunk_size 必须为正数")
    transition_scales = (
        [1.0] * len(profiles)
        if aperture_transition_scales is None
        else [float(value) for value in aperture_transition_scales]
    )
    if len(transition_scales) != len(profiles) or any(
        not math.isfinite(value) or value <= 0.0 for value in transition_scales
    ):
        raise ValueError("aperture_transition_scales 必须与 profiles 等长且全部为有限正数")
    if aberration_profiles is not None and len(aberration_profiles) != len(profiles):
        raise ValueError("aberration_profiles 必须为 null 或与 profiles 等长")
    z_prop = parax.xp.z + float(s_prop)
    viewport_xy_world = project_to_z(z_prop, wf.chief).ray.o[0:2]
    viewport_xy_pix = sensor.quantize(viewport_xy_world).index
    half = int(kernel_size) // 2
    viewport = sensor.slice(
        (viewport_xy_pix[0] - half, viewport_xy_pix[1] - half),
        (int(kernel_size), int(kernel_size)),
    )
    viewport_fine = viewport.upsample(int(upsample))
    eval_points = nputils.flatten(viewport_fine.grid_points(), 0, 2)
    eval_points = vecop.xy2xyz(eval_points, z_prop)

    powers: list[Any] = []
    aberration_phases: list[Any] = []
    metas: list[dict[str, Any]] = []
    resolved_aberrations = (
        [None] * len(profiles)
        if aberration_profiles is None
        else list(aberration_profiles)
    )
    for profile, transition_scale, aberration_profile in zip(
        profiles,
        transition_scales,
        resolved_aberrations,
        strict=True,
    ):
        local_split = profile.split_bias_norm + profile.field_split_slope * float(field_x)
        local_width = profile.transition_width * transition_scale * (
            1.0 + profile.field_acceptance_slope * float(field_x)
        )
        left, right, pupil_meta = _profile_power_weights(
            wf.wf.pts[..., 0],
            wf.wf.pts[..., 1],
            wf.wf.mask,
            split_bias_norm=local_split,
            boundary_curvature_norm=profile.boundary_curvature_norm,
            transition_width=local_width,
            cross_talk=profile.cross_talk,
            edge_rolloff=profile.microlens_edge_rolloff,
        )
        powers.extend((left, right))
        aberration_meta: dict[str, Any] = {
            "basis": "none",
            "coefficient_unit": "waves",
            "effective_astigmatism_0_waves": 0.0,
            "effective_coma_x_waves": 0.0,
            "piston_removed_waves": 0.0,
        }
        if aberration_profile is None:
            profile_phase = jnp.ones_like(left, dtype=jnp.complex64)
        else:
            screen_waves, aberration_meta = low_order_aberration_waves(
                wf.wf.pts[..., 0],
                wf.wf.pts[..., 1],
                wf.wf.mask,
                aberration_profile,
                field_x=float(field_x),
                field_y=float(field_y),
            )
            profile_phase = jnp.exp(2j * jnp.pi * screen_waves)
        aberration_phases.extend((profile_phase, profile_phase))
        metas.append(
            {
                "left_effective_pupil_power": left.sum(),
                "right_effective_pupil_power": right.sum(),
                "pupil_overlap_fraction": (
                    jnp.count_nonzero((left > 1.0e-10) & (right > 1.0e-10))
                    / jnp.maximum(jnp.count_nonzero(jnp.asarray(wf.wf.mask, dtype=bool)), 1)
                ),
                "local_split_bias_norm": local_split,
                "local_transition_width": local_width,
                "low_order_aberration": aberration_meta,
                **pupil_meta,
            }
        )
    power_stack = jnp.stack(powers, axis=0)
    phase = opd_to_phase_factor(wf.wf.wvl, wf.wf.opd)
    base_amplitude = jnp.asarray(wf.wf.amp)
    if aberration_profiles is None:
        # 保留旧路径的运算顺序，默认配置输出不因新接口产生数值漂移。
        waves = (
            base_amplitude[None]
            * jnp.sqrt(jnp.maximum(power_stack, 0.0))
            * phase[None]
        )
    else:
        aberration_phase_stack = jnp.stack(aberration_phases, axis=0)
        waves = (
            base_amplitude[None]
            * jnp.sqrt(jnp.maximum(power_stack, 0.0))
            * phase[None]
            * aberration_phase_stack
        )
    normal = safeop.normdir(wf.xp_sphere.c[None] - wf.wf.pts).v
    pts = wf.wf.pts
    wavelength = wf.wf.wvl
    jk = 2j * jnp.pi / wavelength
    ray_count = pts.shape[0]

    def propagate_sensor_chunk(sensor_points: Any) -> Any:
        delta = sensor_points[:, None] - pts[None]
        distance = safeop.norm(delta).v
        greens = jnp.exp(jk * distance) / distance
        cosine = (delta * normal[None]).sum(axis=2) / distance
        greens = cosine * greens
        scale = 1.0 / (1j * wavelength * ray_count)
        return scale * jnp.einsum("cn,bn->bc", waves, greens)

    field = wrappings.chunk_vmapped(
        propagate_sensor_chunk,
        chunk_size=int(sensor_chunk_size),
    )(eval_points)
    field = nputils.unflatten(field, 0, viewport_fine.shape)
    intensity = downsample_integer(sqabs_complex(field), int(upsample))
    intensity = wh_physics_to_image(intensity, 0, 1)
    intensity = jnp.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    intensity = jnp.maximum(intensity, 0.0)
    energy = intensity.sum(axis=(0, 1), keepdims=True)
    intensity = intensity / jnp.maximum(energy, 1.0e-20)
    kernels = jnp.transpose(intensity, (2, 0, 1))
    outputs: list[dict[str, Any]] = []
    for index, meta in enumerate(metas):
        outputs.append(
            {
                "left": kernels[2 * index],
                "right": kernels[2 * index + 1],
                **meta,
            }
        )
    return outputs


def solve_sensor_offset_for_coc(
    wf: Any,
    *,
    base_s_prop: float,
    target_coc_px: float,
    envelope_m: float,
    parax: Any,
    pixel_pitch_m: float,
    tolerance_px: float = 1.0e-6,
) -> tuple[float, float]:
    """在固定 envelope 内二分求出指定 signed CoC 的传感器 offset。"""

    target = float(target_coc_px)
    envelope = float(envelope_m)
    if not math.isfinite(envelope) or envelope <= 0.0:
        raise ValueError("sensor_offset_envelope_m 必须为有限正数")

    def value(offset: float) -> float:
        return _signed_coc_px_at_propagation(
            wf,
            s_prop=float(base_s_prop) + float(offset),
            parax=parax,
            pixel_pitch_m=float(pixel_pitch_m),
        )

    low_offset, high_offset = -envelope, envelope
    low_value, high_value = value(low_offset), value(high_offset)
    if (low_value - target) * (high_value - target) > 0.0:
        raise RuntimeError(
            "sensor offset envelope 未包围 target CoC："
            f"target={target:g}, low={low_value:g}, high={high_value:g}, envelope={envelope:g}"
        )
    for _ in range(72):
        mid_offset = 0.5 * (low_offset + high_offset)
        mid_value = value(mid_offset)
        if abs(mid_value - target) <= float(tolerance_px):
            return mid_offset, mid_value
        if (low_value - target) * (mid_value - target) <= 0.0:
            high_offset, high_value = mid_offset, mid_value
        else:
            low_offset, low_value = mid_offset, mid_value
    offset = 0.5 * (low_offset + high_offset)
    return offset, value(offset)


def _ncc_tiles(
    *,
    texture_size: int,
    kernel_size: int,
    tile_size: int,
    tiles_per_axis: int,
    search_x: int,
    search_y: int,
    lanczos_radius: int | None = None,
) -> list[dict[str, int]]:
    optical_margin = kernel_size // 2 + max(int(search_x), int(search_y)) + 8
    continuous_margin = (
        0
        if lanczos_radius is None
        else int(lanczos_radius) + max(int(search_x), int(search_y))
    )
    safe_margin = max(optical_margin, continuous_margin)
    low = safe_margin
    high = int(texture_size) - safe_margin - int(tile_size)
    if high < low:
        raise ValueError("texture_size 无法容纳 kernel、tile 与 NCC 搜索边界")
    positions = np.linspace(low, high, num=int(tiles_per_axis), dtype=int)
    return [
        {"x": int(x), "y": int(y), "width": int(tile_size), "height": int(tile_size)}
        for y in positions
        for x in positions
    ]


def _fit_ncc_records(labels: Sequence[float], predictions: Sequence[float]) -> dict[str, Any]:
    reference = np.asarray(labels, dtype=np.float64)
    prediction = np.asarray(predictions, dtype=np.float64)
    if reference.size < 2 or float(reference.var()) <= 1.0e-12:
        raise RuntimeError("NCC 有效记录不足，无法拟合")
    error = prediction - reference
    nonzero = np.abs(reference) > 0.05
    return {
        **_linear_fit(reference, prediction),
        "record_count": int(reference.size),
        "epe_px": float(np.mean(np.abs(error))),
        "rmse_px": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_px": float(np.max(np.abs(error))),
        "sign_agreement_nonzero": (
            float(np.mean(np.sign(reference[nonzero]) == np.sign(prediction[nonzero])))
            if np.any(nonzero)
            else None
        ),
    }


def run_profile_ncc_diagnostic(
    bank: np.ndarray,
    analytic_labels: np.ndarray,
    *,
    signed_coc_bins_px: np.ndarray | None = None,
    small_coc_abs_max_px: float | None = None,
    texture_size: int,
    seeds: Sequence[int],
    tile_size: int,
    tiles_per_axis: int,
    search_x: int,
    search_y: int,
) -> dict[str, Any]:
    """对 `[A,N,1,Gw,2,K,K]` profile 做逐 field 独立 tile NCC。"""

    from scipy.signal import fftconvolve
    from pdraw_benchmark.ncc_reference import NccProtocol, estimate_bidirectional_ncc

    value = np.asarray(bank, dtype=np.float32)
    labels = np.asarray(analytic_labels, dtype=np.float64)
    if value.ndim != 7 or value.shape[2] != 1 or value.shape[4] != 2:
        raise ValueError(f"profile bank 必须为 [A,N,1,Gw,2,K,K]，实际为 {value.shape}")
    if labels.shape != value.shape[:4]:
        raise ValueError(f"analytic_labels shape 不一致：{labels.shape} vs {value.shape[:4]}")
    coc: np.ndarray | None = None
    if signed_coc_bins_px is not None:
        coc = np.asarray(signed_coc_bins_px, dtype=np.float64)
        if coc.ndim == 1:
            coc = np.tile(coc[None, :], (value.shape[0], 1))
        if coc.shape != value.shape[:2]:
            raise ValueError(f"signed CoC shape 不一致：{coc.shape} vs {value.shape[:2]}")
    if small_coc_abs_max_px is not None:
        if coc is None:
            raise ValueError("small_coc_abs_max_px 要求同时提供 signed_coc_bins_px")
        if not math.isfinite(float(small_coc_abs_max_px)) or float(small_coc_abs_max_px) <= 0.0:
            raise ValueError("small_coc_abs_max_px 必须为有限正数")
    if int(search_x) <= math.ceil(float(np.max(np.abs(labels)))):
        raise ValueError("NCC search_x 必须严格覆盖解析视差范围")

    protocol = NccProtocol(
        tile_size=int(tile_size),
        tile_stride=int(tile_size),
        margin_x=int(search_x),
        margin_y=int(search_y),
        search_x=int(search_x),
        search_y=int(search_y),
        min_peak=0.20,
        min_peak_margin=0.002,
        max_lr_error=0.60,
        max_vertical_shift=0.75,
    )
    tiles = _ncc_tiles(
        texture_size=int(texture_size),
        kernel_size=int(value.shape[-1]),
        tile_size=int(tile_size),
        tiles_per_axis=int(tiles_per_axis),
        search_x=int(search_x),
        search_y=int(search_y),
    )
    all_labels: list[float] = []
    all_predictions: list[float] = []
    seed_rows: list[dict[str, Any]] = []
    field_small_labels: list[list[float]] = [[] for _ in range(value.shape[3])]
    field_small_predictions: list[list[float]] = [[] for _ in range(value.shape[3])]
    field_small_expected = [0 for _ in range(value.shape[3])]
    total_cells = int(np.prod(value.shape[:4]))
    for seed in seeds:
        texture = _deterministic_texture(int(texture_size), int(seed))
        seed_labels: list[float] = []
        seed_predictions: list[float] = []
        cell_rows: list[dict[str, Any]] = []
        for aperture_index in range(value.shape[0]):
            for coc_index in range(value.shape[1]):
                for field_x_index in range(value.shape[3]):
                    pair = value[aperture_index, coc_index, 0, field_x_index]
                    left = fftconvolve(texture, pair[0], mode="same").astype(np.float32)
                    right = fftconvolve(texture, pair[1], mode="same").astype(np.float32)
                    estimates = [
                        estimate_bidirectional_ncc(left, right, tile, protocol) for tile in tiles
                    ]
                    valid = [row for row in estimates if bool(row["ncc_quality_pass"])]
                    predictions = [float(row["pseudo_disp_model_px"]) for row in valid]
                    analytic = float(labels[aperture_index, coc_index, 0, field_x_index])
                    prediction = float(np.median(predictions)) if predictions else None
                    tile_rows = []
                    for tile_index, (tile, estimate) in enumerate(zip(tiles, estimates, strict=True)):
                        tile_rows.append(
                            {
                                "tile_index": tile_index,
                                "x": tile["x"],
                                "y": tile["y"],
                                "ncc_disparity_px": float(estimate["pseudo_disp_model_px"]),
                                "quality_pass": bool(estimate["ncc_quality_pass"]),
                                "quality_reasons": list(estimate["quality_reasons"]),
                                "lr_error_px": float(estimate["lr_error"]),
                                "left_to_right": estimate["left_to_right"],
                                "right_to_left": estimate["right_to_left"],
                            }
                        )
                    cell = {
                        "aperture_index": aperture_index,
                        "coc_index": coc_index,
                        "field_x_index": field_x_index,
                        "signed_coc_px": (
                            None if coc is None else float(coc[aperture_index, coc_index])
                        ),
                        "analytic_centroid_disparity_px": analytic,
                        "ncc_disparity_px": prediction,
                        "valid_tile_count": len(valid),
                        "total_tile_count": len(estimates),
                        "tiles": tile_rows,
                    }
                    if prediction is not None:
                        cell["ncc_minus_analytic_px"] = prediction - analytic
                        seed_labels.append(analytic)
                        seed_predictions.append(prediction)
                        all_labels.append(analytic)
                        all_predictions.append(prediction)
                        if (
                            coc is not None
                            and small_coc_abs_max_px is not None
                            and abs(float(coc[aperture_index, coc_index]))
                            <= float(small_coc_abs_max_px) + 1.0e-12
                        ):
                            field_small_labels[field_x_index].append(analytic)
                            field_small_predictions[field_x_index].append(prediction)
                    cell_rows.append(cell)
        seed_rows.append(
            {
                "seed": int(seed),
                "fit": _fit_ncc_records(seed_labels, seed_predictions),
                "valid_cell_fraction": len(seed_labels) / float(total_cells),
                "cells": cell_rows,
            }
        )
    per_field_small_coc: list[dict[str, Any]] = []
    if coc is not None and small_coc_abs_max_px is not None:
        for field_x_index in range(value.shape[3]):
            expected_per_seed = int(
                sum(
                    np.count_nonzero(
                        np.abs(coc[aperture_index])
                        <= float(small_coc_abs_max_px) + 1.0e-12
                    )
                    for aperture_index in range(value.shape[0])
                )
            )
            field_small_expected[field_x_index] = expected_per_seed * len(seeds)
            per_field_small_coc.append(
                {
                    "field_x_index": field_x_index,
                    "small_coc_abs_max_px": float(small_coc_abs_max_px),
                    "fit": _fit_ncc_records(
                        field_small_labels[field_x_index],
                        field_small_predictions[field_x_index],
                    ),
                    "valid_cell_fraction": len(field_small_labels[field_x_index])
                    / float(field_small_expected[field_x_index]),
                }
            )
    result = {
        "role": "independent_tile_diagnostic_only",
        "texture_kind": "deterministic_synthetic",
        "analytic_label_source": "PSF centroid mu_left_x-mu_right_x",
        "label_compensation_from_ncc": False,
        "protocol": asdict(protocol),
        "texture_size": int(texture_size),
        "seeds": [int(value) for value in seeds],
        "aggregate_fit": _fit_ncc_records(all_labels, all_predictions),
        "aggregate_valid_cell_fraction": len(all_labels) / float(total_cells * len(seeds)),
        "seed_diagnostics": seed_rows,
    }
    if per_field_small_coc:
        result["per_field_small_coc"] = per_field_small_coc
    return result


def validate_continuous_ncc_v2_declaration(config: dict[str, Any]) -> None:
    """拒绝 YAML 的协议声明与实际 continuous V2 实现漂移。"""

    required = {
        "protocol_version",
        "interpolation",
        "lanczos_radius",
        "refinement_radius_px",
        "coordinate_iterations",
        "optimizer_xatol_px",
        "min_texture_std",
        "min_score",
        "max_lr_error_px",
        "max_vertical_shift_px",
        "peak_exclusion_x",
        "peak_exclusion_y",
        "final_score_tolerance",
        "per_cell_valid_tile_fraction_min",
        "expected_aperture_count",
        "worker_count",
        "uses_ground_truth",
        "applies_posthoc_label_correction",
    }
    if set(config) != required:
        raise ValueError(
            "continuous_ncc V2 配置字段必须精确冻结："
            f"missing={sorted(required - set(config))}, "
            f"unexpected={sorted(set(config) - required)}"
        )
    if str(config["protocol_version"]) != "continuous_lanczos_v2":
        raise ValueError("continuous_ncc protocol_version 必须为 continuous_lanczos_v2")
    if str(config["interpolation"]) != (
        "separable_lanczos_windowed_sinc_no_lookup_table"
    ):
        raise ValueError("continuous_ncc interpolation 声明与 V2 实现不一致")
    if config["uses_ground_truth"] is not False:
        raise ValueError("continuous_ncc uses_ground_truth 必须严格为 false")
    if config["applies_posthoc_label_correction"] is not False:
        raise ValueError(
            "continuous_ncc applies_posthoc_label_correction 必须严格为 false"
        )


def continuous_ncc_v2_runtime_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """把冻结 YAML 声明逐字段转成 runner 参数，便于严格转发回归。"""

    validate_continuous_ncc_v2_declaration(config)
    return {
        "lanczos_radius": int(config["lanczos_radius"]),
        "refinement_radius_px": float(config["refinement_radius_px"]),
        "coordinate_iterations": int(config["coordinate_iterations"]),
        "optimizer_xatol_px": float(config["optimizer_xatol_px"]),
        "min_texture_std": float(config["min_texture_std"]),
        "min_score": float(config["min_score"]),
        "max_lr_error_px": float(config["max_lr_error_px"]),
        "max_vertical_shift_px": float(config["max_vertical_shift_px"]),
        "peak_exclusion_x": int(config["peak_exclusion_x"]),
        "peak_exclusion_y": int(config["peak_exclusion_y"]),
        "final_score_tolerance": float(config["final_score_tolerance"]),
        "per_cell_valid_tile_fraction_min": float(
            config["per_cell_valid_tile_fraction_min"]
        ),
        "expected_aperture_count": int(config["expected_aperture_count"]),
        "worker_count": int(config["worker_count"]),
    }


def run_profile_continuous_ncc_diagnostic(
    bank: np.ndarray,
    analytic_labels: np.ndarray,
    *,
    signed_coc_bins_px: np.ndarray,
    texture_size: int,
    seeds: Sequence[int],
    tile_size: int,
    tiles_per_axis: int,
    search_x: int,
    search_y: int,
    lanczos_radius: int,
    refinement_radius_px: float,
    coordinate_iterations: int,
    optimizer_xatol_px: float,
    min_texture_std: float,
    min_score: float,
    max_lr_error_px: float,
    max_vertical_shift_px: float,
    peak_exclusion_x: int = 2,
    peak_exclusion_y: int = 1,
    final_score_tolerance: float = 1.0e-8,
    per_cell_valid_tile_fraction_min: float = 1.0,
    expected_aperture_count: int | None = None,
    worker_count: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """同时评估 physical PSF 与同 label Fourier exact-shift continuous NCC。

    continuous estimator 的函数签名不含 label；``analytic_labels`` 只在估计结束后用于
    拟合，或用于构造独立的 shifted-identical oracle。两者都不回写 label/admission。
    """

    from concurrent.futures import ThreadPoolExecutor

    import cv2
    from scipy.ndimage import fourier_shift
    from scipy.signal import fftconvolve
    from threadpoolctl import threadpool_limits

    from render.cldefocus_pdraw.continuous_ncc_v2 import (
        ContinuousNccProtocolV2,
        estimate_bidirectional_continuous_ncc_v2,
    )

    value = np.asarray(bank, dtype=np.float32)
    labels = np.asarray(analytic_labels, dtype=np.float64)
    coc = np.asarray(signed_coc_bins_px, dtype=np.float64)
    if value.ndim != 7 or value.shape[2] != 1 or value.shape[4] != 2:
        raise ValueError(f"continuous profile bank shape 非法：{value.shape}")
    if labels.shape != value.shape[:4]:
        raise ValueError(f"continuous analytic label shape 不一致：{labels.shape}")
    if coc.ndim == 1:
        coc = np.tile(coc[None, :], (value.shape[0], 1))
    if coc.shape != value.shape[:2]:
        raise ValueError(f"continuous signed CoC shape 不一致：{coc.shape}")
    if not np.isfinite(value).all() or not np.isfinite(labels).all() or not np.isfinite(coc).all():
        raise ValueError("continuous bank/label/CoC 必须全部有限")
    seed_values = [int(value) for value in seeds]
    if not seed_values or len(seed_values) != len(set(seed_values)):
        raise ValueError("continuous seeds 必须为非空唯一序列")
    if expected_aperture_count is not None and value.shape[0] != int(
        expected_aperture_count
    ):
        raise ValueError(
            f"continuous aperture 数量必须为 {expected_aperture_count}：{value.shape[0]}"
        )
    if int(search_x) <= math.ceil(float(np.max(np.abs(labels)))):
        raise ValueError("continuous search_x 必须严格覆盖解析视差范围")
    cell_tile_fraction_min = float(per_cell_valid_tile_fraction_min)
    if not math.isfinite(cell_tile_fraction_min) or not 0.0 < cell_tile_fraction_min <= 1.0:
        raise ValueError("per_cell_valid_tile_fraction_min 必须位于 (0,1]")
    workers = int(worker_count)
    if workers < 1 or workers > 32:
        raise ValueError("continuous NCC worker_count 必须位于 [1,32]")
    protocol = ContinuousNccProtocolV2(
        search_x=int(search_x),
        search_y=int(search_y),
        lanczos_radius=int(lanczos_radius),
        refinement_radius_px=float(refinement_radius_px),
        coordinate_iterations=int(coordinate_iterations),
        optimizer_xatol_px=float(optimizer_xatol_px),
        min_texture_std=float(min_texture_std),
        min_score=float(min_score),
        max_lr_error_px=float(max_lr_error_px),
        max_vertical_shift_px=float(max_vertical_shift_px),
        peak_exclusion_x=int(peak_exclusion_x),
        peak_exclusion_y=int(peak_exclusion_y),
        final_score_tolerance=float(final_score_tolerance),
    )
    protocol.validate()
    tiles = _ncc_tiles(
        texture_size=int(texture_size),
        kernel_size=int(value.shape[-1]),
        tile_size=int(tile_size),
        tiles_per_axis=int(tiles_per_axis),
        search_x=int(search_x),
        search_y=int(search_y),
        lanczos_radius=int(lanczos_radius),
    )

    physical_seed_rows: list[dict[str, Any]] = []
    oracle_seed_rows: list[dict[str, Any]] = []
    physical_all_labels: list[float] = []
    physical_all_predictions: list[float] = []
    oracle_all_labels: list[float] = []
    oracle_all_predictions: list[float] = []
    total_cells = int(np.prod(value.shape[:4]))

    def summarize_pair(
        left: np.ndarray,
        right: np.ndarray,
        executor: ThreadPoolExecutor | None,
    ) -> dict[str, Any]:
        def estimate(tile: dict[str, int]) -> dict[str, Any]:
            return estimate_bidirectional_continuous_ncc_v2(left, right, tile, protocol)

        estimates = (
            [estimate(tile) for tile in tiles]
            if executor is None
            else list(executor.map(estimate, tiles))
        )
        valid = [row for row in estimates if bool(row["quality_pass"])]
        valid_tile_fraction = len(valid) / float(len(estimates))
        cell_quality_pass = valid_tile_fraction >= cell_tile_fraction_min
        predictions = (
            [float(row["diagnostic_disp_model_px"]) for row in valid]
            if cell_quality_pass
            else []
        )
        scores = [
            0.5
            * (
                float(row["left_to_right"]["score"])
                + float(row["right_to_left"]["score"])
            )
            for row in valid
        ]
        lr_errors = [float(row["lr_error_px"]) for row in valid]
        peak_margins = [
            min(
                float(row["left_to_right"]["integer_peak_margin"]),
                float(row["right_to_left"]["integer_peak_margin"]),
            )
            for row in estimates
        ]
        final_score_improvements = [
            min(
                float(row["left_to_right"]["final_minus_integer_score"]),
                float(row["right_to_left"]["final_minus_integer_score"]),
            )
            for row in estimates
        ]
        return {
            "prediction_px": float(np.median(predictions)) if predictions else None,
            "valid_tile_count": len(valid),
            "total_tile_count": len(estimates),
            "valid_tile_fraction": valid_tile_fraction,
            "per_cell_valid_tile_fraction_min": cell_tile_fraction_min,
            "cell_quality_pass": cell_quality_pass,
            "score_median": float(np.median(scores)) if scores else None,
            "lr_error_median_px": (
                float(np.median(lr_errors)) if lr_errors else None
            ),
            "integer_peak_margin_report_only_min": float(np.min(peak_margins)),
            "integer_peak_margin_report_only_median": float(np.median(peak_margins)),
            "integer_peak_margin_report_only_max": float(np.max(peak_margins)),
            "final_minus_integer_score_min": float(
                np.min(final_score_improvements)
            ),
            "quality_failure_reasons": sorted(
                {
                    str(reason)
                    for row in estimates
                    if not bool(row["quality_pass"])
                    for reason in row["quality_reasons"]
                }
            ),
        }

    previous_cv2_threads = int(cv2.getNumThreads())
    cv2.setNumThreads(1)
    thread_limits = threadpool_limits(limits=1)
    executor_context = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for seed in seed_values:
            texture = np.asarray(
                _deterministic_texture(int(texture_size), int(seed)),
                dtype=np.float32,
            )
            spectrum = np.fft.fftn(np.asarray(texture, dtype=np.float64))
            shifted_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
            oracle_estimate_cache: dict[float, dict[str, Any]] = {}
            physical_cells: list[dict[str, Any]] = []
            oracle_cells: list[dict[str, Any]] = []
            physical_seed_labels: list[float] = []
            physical_seed_predictions: list[float] = []
            oracle_seed_labels: list[float] = []
            oracle_seed_predictions: list[float] = []
            for aperture_index in range(value.shape[0]):
                for coc_index in range(value.shape[1]):
                    for field_x_index in range(value.shape[3]):
                        analytic = float(
                            labels[aperture_index, coc_index, 0, field_x_index]
                        )
                        pair = value[
                            aperture_index,
                            coc_index,
                            0,
                            field_x_index,
                        ]
                        physical = summarize_pair(
                            fftconvolve(texture, pair[0], mode="same").astype(np.float32),
                            fftconvolve(texture, pair[1], mode="same").astype(np.float32),
                            executor_context,
                        )
                        cache_key = analytic
                        if cache_key not in shifted_cache:
                            left = np.fft.ifftn(
                                fourier_shift(spectrum, shift=(0.0, 0.5 * analytic))
                            ).real.astype(np.float32)
                            right = np.fft.ifftn(
                                fourier_shift(spectrum, shift=(0.0, -0.5 * analytic))
                            ).real.astype(np.float32)
                            shifted_cache[cache_key] = (left, right)
                        if cache_key not in oracle_estimate_cache:
                            oracle_estimate_cache[cache_key] = summarize_pair(
                                *shifted_cache[cache_key],
                                executor_context,
                            )
                        oracle = dict(oracle_estimate_cache[cache_key])
                        common = {
                            "aperture_index": aperture_index,
                            "coc_index": coc_index,
                            "field_x_index": field_x_index,
                            "signed_coc_px": float(coc[aperture_index, coc_index]),
                            "analytic_centroid_disparity_px": analytic,
                        }
                        physical_prediction = physical.pop("prediction_px")
                        oracle_prediction = oracle.pop("prediction_px")
                        physical_cell = {
                            **common,
                            "ncc_disparity_px": physical_prediction,
                            **physical,
                        }
                        oracle_cell = {
                            **common,
                            "ncc_disparity_px": oracle_prediction,
                            "oracle_ncc_disparity_px": oracle_prediction,
                            **oracle,
                        }
                        if physical_prediction is not None:
                            physical_cell["ncc_minus_analytic_px"] = (
                                float(physical_prediction) - analytic
                            )
                            physical_seed_labels.append(analytic)
                            physical_seed_predictions.append(float(physical_prediction))
                            physical_all_labels.append(analytic)
                            physical_all_predictions.append(float(physical_prediction))
                        if oracle_prediction is not None:
                            oracle_cell["oracle_minus_analytic_px"] = (
                                float(oracle_prediction) - analytic
                            )
                            oracle_seed_labels.append(analytic)
                            oracle_seed_predictions.append(float(oracle_prediction))
                            oracle_all_labels.append(analytic)
                            oracle_all_predictions.append(float(oracle_prediction))
                        physical_cells.append(physical_cell)
                        oracle_cells.append(oracle_cell)
            physical_seed_rows.append(
                {
                    "seed": int(seed),
                    "fit": _fit_ncc_records(
                        physical_seed_labels,
                        physical_seed_predictions,
                    ),
                    "valid_cell_fraction": len(physical_seed_labels) / float(total_cells),
                    "cells": physical_cells,
                }
            )
            oracle_seed_rows.append(
                {
                    "seed": int(seed),
                    "fit": _fit_ncc_records(oracle_seed_labels, oracle_seed_predictions),
                    "valid_cell_fraction": len(oracle_seed_labels) / float(total_cells),
                    "cells": oracle_cells,
                }
            )
    finally:
        if executor_context is not None:
            executor_context.shutdown(wait=True)
        thread_limits.restore_original_limits()
        cv2.setNumThreads(previous_cv2_threads)

    numeric_protocol_identity = {
        "protocol": asdict(protocol),
        "texture_size": int(texture_size),
        "texture_algorithm": "cldefocus_deterministic_texture_v1",
        "seeds": seed_values,
        "tile_grid": tiles,
        "cell_reducer": "median_of_quality_valid_tiles",
        "per_cell_valid_tile_fraction_min": cell_tile_fraction_min,
    }
    common_metadata = {
        "estimator": "continuous_lanczos_bounded_coordinate_ncc_v2",
        "estimator_uses_ground_truth": False,
        "label_compensation_from_ncc": False,
        "protocol": asdict(protocol),
        "texture_size": int(texture_size),
        "texture_algorithm": "cldefocus_deterministic_texture_v1",
        "seeds": seed_values,
        "worker_count": workers,
        "worker_map_preserves_input_order": True,
        "worker_inner_cv2_and_blas_threads": 1,
        "integer_peak_margin_role": "report_only_no_threshold_no_veto",
        "per_cell_valid_tile_fraction_min": cell_tile_fraction_min,
        "cell_reducer": "median_of_quality_valid_tiles",
        "tile_grid": tiles,
        "tile_grid_sha256": _sha256_json(tiles),
        "numeric_protocol_identity_sha256": _sha256_json(
            numeric_protocol_identity
        ),
    }
    physical_result = {
        **common_metadata,
        "role": "primary_gt_free_continuous_lanczos_physical_psf_diagnostic",
        "analytic_label_source": "PSF centroid mu_left_x-mu_right_x",
        "aggregate_fit": _fit_ncc_records(
            physical_all_labels,
            physical_all_predictions,
        ),
        "aggregate_valid_cell_fraction": len(physical_all_labels)
        / float(total_cells * len(seeds)),
        "seed_diagnostics": physical_seed_rows,
    }
    oracle_result = {
        **common_metadata,
        "role": "continuous_lanczos_fourier_shifted_identical_baseline_only",
        "analytic_label_source": "candidate PSF centroid grid copied read-only",
        "oracle_generation_uses_label_only_to_construct_shifted_pair": True,
        "oracle_estimator_uses_label": False,
        "oracle_updates_admission": False,
        "shift_operator": "scipy.ndimage.fourier_shift_on_numpy_fftn",
        "shift_convention": "left=+d/2; right=-d/2; d=x_L-x_R",
        "cubic_or_grid_sample_resampling_baseline_included": False,
        "aggregate_fit_report_only": _fit_ncc_records(
            oracle_all_labels,
            oracle_all_predictions,
        ),
        "seed_diagnostics": oracle_seed_rows,
    }
    return physical_result, oracle_result


def run_shifted_identical_ncc_oracle(
    analytic_labels: np.ndarray,
    *,
    signed_coc_bins_px: np.ndarray,
    kernel_size: int,
    texture_size: int,
    seeds: Sequence[int],
    tile_size: int,
    tiles_per_axis: int,
    search_x: int,
    search_y: int,
) -> dict[str, Any]:
    """用 Fourier-shift identical pair 量化 NCC 亚像素 peak-locking。

    oracle 只复用候选的 analytic label grid 和冻结 NCC protocol，不读取或修正 PSF。
    它永远不参与 label、profile 参数或训练资产的回写。
    """

    from scipy.ndimage import fourier_shift
    from pdraw_benchmark.ncc_reference import NccProtocol, estimate_bidirectional_ncc

    labels = np.asarray(analytic_labels, dtype=np.float64)
    if labels.ndim != 4:
        raise ValueError(f"oracle analytic labels 必须为 [A,N,Gh,Gw]：{labels.shape}")
    coc = np.asarray(signed_coc_bins_px, dtype=np.float64)
    if coc.ndim == 1:
        coc = np.tile(coc[None, :], (labels.shape[0], 1))
    if coc.shape != labels.shape[:2]:
        raise ValueError(f"oracle signed CoC shape 不一致：{coc.shape} vs {labels.shape[:2]}")
    if int(search_x) <= math.ceil(float(np.max(np.abs(labels)))):
        raise ValueError("oracle NCC search_x 必须严格覆盖解析视差范围")
    protocol = NccProtocol(
        tile_size=int(tile_size),
        tile_stride=int(tile_size),
        margin_x=int(search_x),
        margin_y=int(search_y),
        search_x=int(search_x),
        search_y=int(search_y),
        min_peak=0.20,
        min_peak_margin=0.002,
        max_lr_error=0.60,
        max_vertical_shift=0.75,
    )
    tiles = _ncc_tiles(
        texture_size=int(texture_size),
        kernel_size=int(kernel_size),
        tile_size=int(tile_size),
        tiles_per_axis=int(tiles_per_axis),
        search_x=int(search_x),
        search_y=int(search_y),
    )
    seed_rows: list[dict[str, Any]] = []
    all_labels: list[float] = []
    all_predictions: list[float] = []
    total_cells = int(np.prod(labels.shape))
    for seed in seeds:
        texture = np.asarray(
            _deterministic_texture(int(texture_size), int(seed)),
            dtype=np.float64,
        )
        spectrum = np.fft.fftn(texture)
        shifted_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
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
                        cache_key = float(np.round(analytic, 12))
                        if cache_key not in shifted_cache:
                            if abs(analytic) <= 1.0e-15:
                                left = texture.astype(np.float32)
                                right = left
                            else:
                                left = np.fft.ifftn(
                                    fourier_shift(spectrum, shift=(0.0, 0.5 * analytic))
                                ).real.astype(np.float32)
                                right = np.fft.ifftn(
                                    fourier_shift(spectrum, shift=(0.0, -0.5 * analytic))
                                ).real.astype(np.float32)
                            shifted_cache[cache_key] = (left, right)
                        left, right = shifted_cache[cache_key]
                        estimates = [
                            estimate_bidirectional_ncc(left, right, tile, protocol)
                            for tile in tiles
                        ]
                        valid = [row for row in estimates if bool(row["ncc_quality_pass"])]
                        predictions = [float(row["pseudo_disp_model_px"]) for row in valid]
                        prediction = float(np.median(predictions)) if predictions else None
                        cell = {
                            "aperture_index": aperture_index,
                            "coc_index": coc_index,
                            "field_y_index": field_y_index,
                            "field_x_index": field_x_index,
                            "signed_coc_px": float(coc[aperture_index, coc_index]),
                            "analytic_centroid_disparity_px": analytic,
                            "oracle_ncc_disparity_px": prediction,
                            "ncc_disparity_px": prediction,
                            "valid_tile_count": len(valid),
                            "total_tile_count": len(estimates),
                        }
                        if prediction is not None:
                            cell["oracle_minus_analytic_px"] = prediction - analytic
                            seed_labels.append(analytic)
                            seed_predictions.append(prediction)
                            all_labels.append(analytic)
                            all_predictions.append(prediction)
                        cells.append(cell)
        seed_rows.append(
            {
                "seed": int(seed),
                "fit": _fit_ncc_records(seed_labels, seed_predictions),
                "valid_cell_fraction": len(seed_labels) / float(total_cells),
                "cells": cells,
            }
        )
    return {
        "role": "fourier_shifted_identical_peak_locking_baseline_only",
        "shift_operator": "scipy.ndimage.fourier_shift_on_numpy_fftn",
        "shift_convention": "left=+d/2; right=-d/2; d=x_L-x_R",
        "boundary_condition": "periodic_fourier_shift",
        "cubic_or_grid_sample_resampling_baseline_included": False,
        "cubic_or_grid_sample_is_not_fourier_oracle": True,
        "analytic_label_source": "candidate PSF centroid grid copied read-only",
        "label_compensation_from_oracle": False,
        "oracle_updates_admission": False,
        "protocol": asdict(protocol),
        "texture_size": int(texture_size),
        "seeds": [int(value) for value in seeds],
        "aggregate_fit_report_only": _fit_ncc_records(all_labels, all_predictions),
        "seed_diagnostics": seed_rows,
    }


def compare_per_aperture_ncc_to_oracle(
    physical: dict[str, Any],
    oracle: dict[str, Any],
) -> list[dict[str, Any]]:
    """逐 aperture 报告 physical 相对 peak-locking oracle 的差与比；只作解释。"""

    physical_rows = physical["per_aperture"]
    oracle_rows = oracle["per_aperture"]
    if len(physical_rows) != len(oracle_rows):
        raise ValueError("physical/oracle aperture 数量不一致")
    output: list[dict[str, Any]] = []
    for physical_row, oracle_row in zip(physical_rows, oracle_rows, strict=True):
        if int(physical_row["aperture_index"]) != int(oracle_row["aperture_index"]):
            raise ValueError("physical/oracle aperture 顺序不一致")
        physical_fit = physical_row["fit"]
        oracle_fit = oracle_row["fit"]
        physical_slope = float(physical_fit["slope_ncc_vs_analytic"])
        oracle_slope = float(oracle_fit["slope_ncc_vs_analytic"])
        output.append(
            {
                "aperture_index": int(physical_row["aperture_index"]),
                "physical_fit": physical_fit,
                "oracle_fit": oracle_fit,
                "slope_physical_minus_oracle": physical_slope - oracle_slope,
                "slope_physical_over_oracle": (
                    None if abs(oracle_slope) <= 1.0e-12 else physical_slope / oracle_slope
                ),
                "epe_physical_minus_oracle_px": float(physical_fit["epe_px"])
                - float(oracle_fit["epe_px"]),
                "role": "report_only_no_label_or_admission_correction",
            }
        )
    return output


def compare_aligned_continuous_ncc_to_oracle(
    physical: dict[str, Any],
    oracle: dict[str, Any],
    *,
    aperture_count: int,
    coc_abs_max_px: float,
) -> dict[str, Any]:
    """在相同 seed/cell/有效 mask 上重拟合 physical 与 exact-shift excess。"""

    identity_fields = (
        "protocol",
        "texture_size",
        "seeds",
        "tile_grid_sha256",
        "numeric_protocol_identity_sha256",
    )
    if any(physical[field] != oracle[field] for field in identity_fields):
        raise ValueError("continuous physical/oracle numeric protocol identity 不一致")
    physical_seeds = {int(row["seed"]): row for row in physical["seed_diagnostics"]}
    oracle_seeds = {int(row["seed"]): row for row in oracle["seed_diagnostics"]}
    if set(physical_seeds) != set(oracle_seeds):
        raise ValueError("continuous physical/oracle seed 集合不一致")
    threshold = float(coc_abs_max_px)
    rows: list[dict[str, Any]] = []
    all_grid_records: list[dict[str, Any]] = []
    all_valid_mask_records: list[dict[str, Any]] = []
    for aperture_index in range(int(aperture_count)):
        labels: list[float] = []
        physical_predictions: list[float] = []
        oracle_predictions: list[float] = []
        expected = 0
        for seed in sorted(physical_seeds):
            def key(cell: dict[str, Any]) -> tuple[int, int, int]:
                return (
                    int(cell["aperture_index"]),
                    int(cell["coc_index"]),
                    int(cell["field_x_index"]),
                )

            physical_cells = {key(cell): cell for cell in physical_seeds[seed]["cells"]}
            oracle_cells = {key(cell): cell for cell in oracle_seeds[seed]["cells"]}
            if set(physical_cells) != set(oracle_cells):
                raise ValueError("continuous physical/oracle cell grid 不一致")
            for cell_key in sorted(physical_cells):
                if cell_key[0] != aperture_index:
                    continue
                physical_cell = physical_cells[cell_key]
                oracle_cell = oracle_cells[cell_key]
                signed_coc = float(physical_cell["signed_coc_px"])
                if abs(signed_coc) > threshold + 1.0e-12:
                    continue
                expected += 1
                analytic = float(physical_cell["analytic_centroid_disparity_px"])
                oracle_analytic = float(
                    oracle_cell["analytic_centroid_disparity_px"]
                )
                if analytic != oracle_analytic or signed_coc != float(
                    oracle_cell["signed_coc_px"]
                ):
                    raise ValueError("continuous physical/oracle label grid 非逐值一致")
                physical_prediction = physical_cell["ncc_disparity_px"]
                oracle_prediction = oracle_cell["ncc_disparity_px"]
                record_id = {
                    "seed": seed,
                    "aperture_index": aperture_index,
                    "coc_index": cell_key[1],
                    "field_x_index": cell_key[2],
                }
                all_grid_records.append(
                    {
                        **record_id,
                        "signed_coc_px": signed_coc,
                        "analytic_centroid_disparity_px": analytic,
                    }
                )
                aligned_valid = (
                    physical_prediction is not None and oracle_prediction is not None
                )
                all_valid_mask_records.append(
                    {**record_id, "aligned_valid": aligned_valid}
                )
                if not aligned_valid:
                    continue
                labels.append(analytic)
                physical_predictions.append(float(physical_prediction))
                oracle_predictions.append(float(oracle_prediction))
        physical_fit = _fit_ncc_records(labels, physical_predictions)
        oracle_fit = _fit_ncc_records(labels, oracle_predictions)
        physical_slope = float(physical_fit["slope_ncc_vs_analytic"])
        oracle_slope = float(oracle_fit["slope_ncc_vs_analytic"])
        rows.append(
            {
                "aperture_index": aperture_index,
                "expected_record_count": expected,
                "aligned_valid_record_count": len(labels),
                "aligned_valid_record_fraction": len(labels) / float(expected),
                "physical_fit_on_aligned_mask": physical_fit,
                "oracle_fit_on_aligned_mask": oracle_fit,
                "slope_physical_minus_oracle": physical_slope - oracle_slope,
                "slope_physical_over_oracle": (
                    None
                    if abs(oracle_slope) <= 1.0e-12
                    else physical_slope / oracle_slope
                ),
                "physical_minus_oracle_prediction_epe_px": float(
                    np.mean(
                        np.abs(
                            np.asarray(physical_predictions, dtype=np.float64)
                            - np.asarray(oracle_predictions, dtype=np.float64)
                        )
                    )
                ),
                "role": "report_only_no_label_or_admission_correction",
            }
        )
    return {
        "selector": "abs_signed_coc_le",
        "threshold_px": threshold,
        "join_key": ["seed", "aperture_index", "coc_index", "field_x_index"],
        "same_valid_mask_required": True,
        "label_grid_sha256": _sha256_json(all_grid_records),
        "aligned_valid_mask_sha256": _sha256_json(all_valid_mask_records),
        "protocol_sha256": _sha256_json(physical["protocol"]),
        "tile_grid_sha256": physical["tile_grid_sha256"],
        "numeric_protocol_identity_sha256": physical[
            "numeric_protocol_identity_sha256"
        ],
        "per_aperture": rows,
        "role": "report_only_no_label_or_admission_correction",
    }


def profile_optical_diagnostics(
    bank: np.ndarray,
    signed_coc_bins_px: np.ndarray,
    analytic_labels: np.ndarray,
    propagation_records: Sequence[dict[str, Any]],
    *,
    gates: dict[str, Any],
) -> dict[str, Any]:
    """计算不依赖 NCC 的能量、零残差、符号、D/CoC 和 pupil-power 门禁。"""

    value = np.asarray(bank, dtype=np.float64)
    coc = np.asarray(signed_coc_bins_px, dtype=np.float64)
    labels = np.asarray(analytic_labels, dtype=np.float64)
    energy = value.sum(axis=(-2, -1))
    if coc.ndim != 2 or labels.shape != value.shape[:4]:
        raise ValueError("optical diagnostic 的 CoC/label shape 不一致")
    zero_indices = [int(np.argmin(np.abs(coc[a]))) for a in range(coc.shape[0])]
    zero_coc_abs = float(max(abs(float(coc[a, zero_indices[a]])) for a in range(coc.shape[0])))
    zero_disp_abs = float(
        max(
            np.max(np.abs(labels[a, zero_indices[a]]))
            for a in range(labels.shape[0])
        )
    )
    slopes: list[dict[str, Any]] = []
    sign_reference: list[float] = []
    sign_prediction: list[float] = []
    monotonic_min = float("inf")
    for aperture_index in range(value.shape[0]):
        for field_index in range(value.shape[3]):
            cur_coc = coc[aperture_index]
            cur_disp = labels[aperture_index, :, 0, field_index]
            mask = np.abs(cur_coc) > 0.25
            slope = float(np.dot(cur_coc[mask], cur_disp[mask]) / np.dot(cur_coc[mask], cur_coc[mask]))
            slopes.append(
                {
                    "aperture_index": aperture_index,
                    "field_x_index": field_index,
                    "d_to_coc_slope": slope,
                    "label_span_px": float(np.ptp(cur_disp)),
                    "zero_disparity_abs_px": float(
                        abs(cur_disp[zero_indices[aperture_index]])
                    ),
                }
            )
            sign_reference.extend(cur_coc[mask].tolist())
            sign_prediction.extend(cur_disp[mask].tolist())
            monotonic_min = min(monotonic_min, float(np.min(np.diff(cur_disp))))
    sign_reference_array = np.asarray(sign_reference)
    sign_prediction_array = np.asarray(sign_prediction)
    sign_agreement = float(
        np.mean(np.sign(sign_reference_array) == np.sign(sign_prediction_array))
    )
    power_ratios = np.asarray(
        [
            float(record["left_effective_pupil_power"])
            / max(float(record["right_effective_pupil_power"]), 1.0e-20)
            for record in propagation_records
        ],
        dtype=np.float64,
    )
    slope_values = np.asarray([row["d_to_coc_slope"] for row in slopes], dtype=np.float64)
    checks = {
        "finite": bool(np.isfinite(value).all()),
        "energy": bool(
            np.allclose(
                energy,
                1.0,
                atol=float(gates.get("energy_abs_tolerance", 2.0e-5)),
                rtol=0.0,
            )
        ),
        "zero_coc": zero_coc_abs <= float(gates.get("exact_zero_coc_max_px", 1.0e-5)),
        "zero_disparity": zero_disp_abs
        <= float(gates.get("exact_zero_disparity_max_px", 0.02)),
        "signed_disparity": sign_agreement >= float(gates.get("sign_agreement_min", 1.0)),
        "d_to_coc_lower": float(slope_values.min())
        >= float(gates.get("d_to_coc_ratio_min", 0.06)),
        "d_to_coc_upper": float(slope_values.max())
        <= float(gates.get("d_to_coc_ratio_max", 0.30)),
        "monotonic": monotonic_min
        >= -float(gates.get("monotonic_drop_max_px", 0.03)),
        "pupil_power_ratio_lower": float(power_ratios.min())
        >= float(gates.get("pupil_power_ratio_min", 0.85)),
        "pupil_power_ratio_upper": float(power_ratios.max())
        <= float(gates.get("pupil_power_ratio_max", 1.18)),
    }
    return {
        "checks": checks,
        "pass": bool(all(checks.values())),
        "energy_min": float(energy.min()),
        "energy_max": float(energy.max()),
        "zero_coc_abs_max_px": zero_coc_abs,
        "zero_disparity_abs_max_px": zero_disp_abs,
        "sign_agreement_nonzero": sign_agreement,
        "d_to_coc_slope_min": float(slope_values.min()),
        "d_to_coc_slope_max": float(slope_values.max()),
        "d_to_coc_slope_mean": float(slope_values.mean()),
        "monotonic_min_delta_px": monotonic_min,
        "pupil_power_ratio_min": float(power_ratios.min()),
        "pupil_power_ratio_max": float(power_ratios.max()),
        "per_aperture_field": slopes,
    }


def ncc_admission(ncc: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    fit = ncc["aggregate_fit"]
    checks = {
        "slope_lower": float(fit["slope_ncc_vs_analytic"])
        >= float(gates.get("slope_min", 0.9)),
        "slope_upper": float(fit["slope_ncc_vs_analytic"])
        <= float(gates.get("slope_max", 1.1)),
        "offset": abs(float(fit["intercept_px"]))
        < float(gates.get("offset_abs_max_px", 0.05)),
        "r2": float(fit["r2"]) >= float(gates.get("r2_min", 0.98)),
        "epe": float(fit["epe_px"]) <= float(gates.get("epe_px_max", 0.10)),
        "sign": fit["sign_agreement_nonzero"] is not None
        and float(fit["sign_agreement_nonzero"])
        >= float(gates.get("sign_agreement_min", 1.0)),
        "valid_cells": float(ncc["aggregate_valid_cell_fraction"])
        >= float(gates.get("valid_cell_fraction_min", 0.95)),
    }
    per_field_rows: list[dict[str, Any]] = []
    for source in ncc.get("per_field_small_coc", []):
        field_fit = source["fit"]
        field_checks = {
            "slope_lower": float(field_fit["slope_ncc_vs_analytic"])
            >= float(gates.get("per_field_slope_min", gates.get("slope_min", 0.9))),
            "slope_upper": float(field_fit["slope_ncc_vs_analytic"])
            <= float(gates.get("per_field_slope_max", gates.get("slope_max", 1.1))),
            "r2": float(field_fit["r2"])
            >= float(gates.get("per_field_r2_min", gates.get("r2_min", 0.99))),
            "sign": field_fit["sign_agreement_nonzero"] is not None
            and float(field_fit["sign_agreement_nonzero"])
            >= float(
                gates.get(
                    "per_field_sign_agreement_min",
                    gates.get("sign_agreement_min", 1.0),
                )
            ),
            "valid_cells": float(source["valid_cell_fraction"])
            >= float(
                gates.get(
                    "per_field_valid_cell_fraction_min",
                    gates.get("valid_cell_fraction_min", 0.95),
                )
            ),
        }
        per_field_rows.append(
            {
                **source,
                "checks": field_checks,
                "pass": bool(all(field_checks.values())),
            }
        )
    if bool(gates.get("require_per_field_small_coc", False)):
        checks["per_field_small_coc_present"] = bool(per_field_rows)
        checks["per_field_small_coc"] = bool(per_field_rows) and all(
            bool(row["pass"]) for row in per_field_rows
        )
    return {
        "checks": checks,
        "pass": bool(all(checks.values())),
        "per_field_small_coc": per_field_rows,
    }


def summarize_per_aperture_ncc(
    diagnostic: dict[str, Any],
    *,
    aperture_count: int,
    coc_abs_max_px: float | None = None,
    analytic_disparity_abs_max_px: float | None = None,
) -> dict[str, Any]:
    """从 NCC cell 逐 aperture 拟合一个 near-focus 子集。"""

    selectors = int(coc_abs_max_px is not None) + int(
        analytic_disparity_abs_max_px is not None
    )
    if selectors != 1:
        raise ValueError("必须且只能声明一个 near-focus selector")
    if coc_abs_max_px is not None:
        threshold = float(coc_abs_max_px)
        selector_name = "abs_signed_coc_le"
    else:
        threshold = float(analytic_disparity_abs_max_px)
        selector_name = "abs_analytic_disparity_le"
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("near-focus threshold 必须为有限正数")

    rows: list[dict[str, Any]] = []
    aggregate_labels: list[float] = []
    aggregate_predictions: list[float] = []
    for aperture_index in range(int(aperture_count)):
        labels: list[float] = []
        predictions: list[float] = []
        expected = 0
        selected_coc_values: set[float] = set()
        for seed in diagnostic["seed_diagnostics"]:
            for cell in seed["cells"]:
                if int(cell["aperture_index"]) != aperture_index:
                    continue
                signed_coc = float(cell["signed_coc_px"])
                analytic = float(cell["analytic_centroid_disparity_px"])
                selected = (
                    abs(signed_coc) <= threshold + 1.0e-12
                    if coc_abs_max_px is not None
                    else abs(analytic) <= threshold + 1.0e-12
                )
                if not selected:
                    continue
                expected += 1
                selected_coc_values.add(signed_coc)
                prediction = cell["ncc_disparity_px"]
                if prediction is None:
                    continue
                labels.append(analytic)
                predictions.append(float(prediction))
        fit = _fit_ncc_records(labels, predictions)
        aggregate_labels.extend(labels)
        aggregate_predictions.extend(predictions)
        rows.append(
            {
                "aperture_index": aperture_index,
                "selected_signed_coc_bins_px": sorted(selected_coc_values),
                "expected_record_count": expected,
                "valid_record_fraction": len(labels) / float(expected),
                "fit": fit,
            }
        )
    return {
        "selector": selector_name,
        "threshold_px": threshold,
        "per_aperture": rows,
        "aggregate_fit_report_only": _fit_ncc_records(
            aggregate_labels,
            aggregate_predictions,
        ),
        "aggregate_is_admission_gate": False,
    }


def apply_per_aperture_gate(
    summary: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    """严格逐 aperture 判定；aggregate 只报告，不能覆盖任一失败。"""

    output_rows: list[dict[str, Any]] = []
    for source in summary["per_aperture"]:
        fit = source["fit"]
        checks = {
            "slope_lower": float(fit["slope_ncc_vs_analytic"])
            >= float(gates.get("slope_min", 0.9)),
            "slope_upper": float(fit["slope_ncc_vs_analytic"])
            <= float(gates.get("slope_max", 1.1)),
            "r2": float(fit["r2"]) >= float(gates.get("r2_min", 0.99)),
            "sign": fit["sign_agreement_nonzero"] is not None
            and float(fit["sign_agreement_nonzero"])
            >= float(gates.get("sign_agreement_min", 1.0)),
        }
        if "valid_record_fraction_min" in gates:
            checks["valid_records"] = float(source["valid_record_fraction"]) >= float(
                gates["valid_record_fraction_min"]
            )
        output_rows.append(
            {
                **source,
                "checks": checks,
                "pass": bool(all(checks.values())),
            }
        )
    return {
        **summary,
        "per_aperture": output_rows,
        "pass": bool(output_rows) and all(bool(row["pass"]) for row in output_rows),
        "pass_rule": "all_apertures_must_pass; aggregate_never_overrides",
        "gates": {
            "slope_min": float(gates.get("slope_min", 0.9)),
            "slope_max": float(gates.get("slope_max", 1.1)),
            "r2_min": float(gates.get("r2_min", 0.99)),
            "sign_agreement_min": float(gates.get("sign_agreement_min", 1.0)),
            **(
                {
                    "valid_record_fraction_min": float(
                        gates["valid_record_fraction_min"]
                    )
                }
                if "valid_record_fraction_min" in gates
                else {}
            ),
        },
    }


def profile_psf_moment_diagnostics(
    bank: np.ndarray,
    signed_coc_bins_px: np.ndarray,
) -> dict[str, Any]:
    """记录逐 PSF 一阶/中心二阶矩，供 response 搜索解释 morphology。"""

    value = np.asarray(bank, dtype=np.float32)
    coc = np.asarray(signed_coc_bins_px, dtype=np.float64)
    if value.ndim != 7 or value.shape[4] != 2:
        raise ValueError(f"PSF moment bank shape 非法：{value.shape}")
    if coc.ndim == 1:
        coc = np.tile(coc[None, :], (value.shape[0], 1))
    if coc.shape != value.shape[:2]:
        raise ValueError(f"PSF moment CoC shape 不一致：{coc.shape} vs {value.shape[:2]}")
    stats = _bank_centroid_and_covariance(value)
    rows: list[dict[str, Any]] = []
    trace_values: list[float] = []
    for aperture_index in range(value.shape[0]):
        for coc_index in range(value.shape[1]):
            for field_y_index in range(value.shape[2]):
                for field_x_index in range(value.shape[3]):
                    for side_index, side_name in enumerate(("left", "right")):
                        covariance_xx = float(
                            stats["covariance_xx"][
                                aperture_index,
                                coc_index,
                                field_y_index,
                                field_x_index,
                                side_index,
                            ]
                        )
                        covariance_xy = float(
                            stats["covariance_xy"][
                                aperture_index,
                                coc_index,
                                field_y_index,
                                field_x_index,
                                side_index,
                            ]
                        )
                        covariance_yy = float(
                            stats["covariance_yy"][
                                aperture_index,
                                coc_index,
                                field_y_index,
                                field_x_index,
                                side_index,
                            ]
                        )
                        trace = covariance_xx + covariance_yy
                        trace_values.append(trace)
                        rows.append(
                            {
                                "aperture_index": aperture_index,
                                "coc_index": coc_index,
                                "signed_coc_px": float(coc[aperture_index, coc_index]),
                                "field_y_index": field_y_index,
                                "field_x_index": field_x_index,
                                "side_index": side_index,
                                "side": side_name,
                                "energy": float(
                                    stats["energy"][
                                        aperture_index,
                                        coc_index,
                                        field_y_index,
                                        field_x_index,
                                        side_index,
                                    ]
                                ),
                                "mu_x_px": float(
                                    stats["mu_x"][
                                        aperture_index,
                                        coc_index,
                                        field_y_index,
                                        field_x_index,
                                        side_index,
                                    ]
                                ),
                                "mu_y_px": float(
                                    stats["mu_y"][
                                        aperture_index,
                                        coc_index,
                                        field_y_index,
                                        field_x_index,
                                        side_index,
                                    ]
                                ),
                                "covariance_xx_px2": covariance_xx,
                                "covariance_xy_px2": covariance_xy,
                                "covariance_yy_px2": covariance_yy,
                                "centered_second_moment_trace_px2": trace,
                            }
                        )
    return {
        "axis_contract": "per aperture/coc/field/side; centered covariance in pixel coordinates",
        "centered_second_moment_trace_min_px2": float(min(trace_values)),
        "centered_second_moment_trace_max_px2": float(max(trace_values)),
        "rows": rows,
    }


def _deep_merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并版本化 YAML；list/scalar 由 overlay 整体替换。"""

    output = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _deep_merge_config(dict(output[key]), value)
        else:
            output[key] = value
    return output


def _resolve_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    project_root = Path(__file__).resolve().parents[3]
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or int(loaded.get("schema_version", 0)) != 1:
        raise ValueError("配置必须是 schema_version: 1 的 YAML mapping")
    extends = loaded.get("extends")
    if extends is not None:
        base_path = Path(str(extends))
        if not base_path.is_absolute():
            base_path = (project_root / base_path).resolve()
        if project_root not in base_path.parents or not base_path.is_file():
            raise ValueError(f"extends 必须指向仓库内存在的版本化 YAML：{base_path}")
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        if not isinstance(base, dict) or int(base.get("schema_version", 0)) != 1:
            raise ValueError("base 配置必须是 schema_version: 1 的 YAML mapping")
        if base.get("extends") is not None:
            raise ValueError("只允许一层 extends，避免配置谱系歧义")
        actual_base_sha256 = _sha256_file(base_path)
        expected_base_sha256 = loaded.get("extends_sha256")
        if (
            expected_base_sha256 is not None
            and str(expected_base_sha256) != actual_base_sha256
        ):
            raise RuntimeError(
                "extends base SHA256 不匹配："
                f"expected={expected_base_sha256}, actual={actual_base_sha256}"
            )
        overlay = {
            key: value
            for key, value in loaded.items()
            if key not in {"extends", "extends_sha256"}
        }
        loaded = _deep_merge_config(base, overlay)
        loaded["config_inheritance"] = {
            "base_path": str(base_path),
            "base_sha256": actual_base_sha256,
            "expected_base_sha256": (
                None if expected_base_sha256 is None else str(expected_base_sha256)
            ),
            "base_sha256_pin_enforced": expected_base_sha256 is not None,
            "overlay_path": str(config_path.resolve()),
            "overlay_sha256": _sha256_file(config_path),
        }
    resolved = dict(loaded)
    source = dict(resolved["source"])
    source["checkout"] = str((project_root / source.get("checkout", "render/CLDefocus")).resolve())
    resolved["source"] = source
    lens = dict(resolved["lens"])
    lens["root"] = str(Path(lens["root"]).resolve())
    resolved["lens"] = lens
    output = dict(resolved["output"])
    output["root"] = str(Path(output["root"]).resolve())
    resolved["output"] = output
    return resolved, project_root


def _prepare_optical_contexts(resolved: dict[str, Any]) -> dict[str, Any]:
    """一次构造 5 档真实 stop、3 个视场和共同 target-CoC 采样位置。"""

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
    if field_x != sorted(field_x) or not field_x or any(abs(value) > 1.0 for value in field_x):
        raise ValueError("field_x_normalized 必须为 [-1,1] 内严格递增非空序列")
    target_coc = np.asarray(optics["target_signed_coc_bins_px"], dtype=np.float64)
    if target_coc.ndim != 1 or np.any(np.diff(target_coc) <= 0.0):
        raise ValueError("target_signed_coc_bins_px 必须严格递增")
    if float(np.min(np.abs(target_coc))) > 1.0e-12:
        raise ValueError("target_signed_coc_bins_px 必须包含精确 0")

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
                f"f-number 漂移：requested={aperture.requested_f_number}, actual={actual_f_number}"
            )
        wavefronts: list[Any] = []
        offsets = np.empty((len(field_x), target_coc.size), dtype=np.float64)
        actual_coc = np.empty_like(offsets)
        base_propagations: list[float] = []
        for field_index, field in enumerate(field_x):
            wavefront = imaging.generate_wavefront(
                float(optics["object_depth_m"]),
                jnp.asarray([float(field), 0.0]),
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
                    tolerance_px=float(optics.get("target_coc_tolerance_px", 1.0e-6)),
                )
                offsets[field_index, coc_index] = offset
                actual_coc[field_index, coc_index] = measured
            wavefronts.append(wavefront)
            base_propagations.append(base_s_prop)
        contexts.append(
            {
                "aperture_plan": aperture,
                "lens": lens,
                "imaging": imaging,
                "actual_f_number": actual_f_number,
                "exit_pupil_radius_m": float(imaging.parax.xp.r),
                "entrance_pupil_radius_m": float(imaging.parax.ep.r),
                "wavefronts": wavefronts,
                "base_propagations": base_propagations,
                "offsets_m": offsets,
                "actual_coc_px": actual_coc,
            }
        )
    return {
        "native_f_number": float(native_imaging.parax.fnum),
        "native_stop_radius_m": float(source_lens.get_aperture(source_lens.stop_idx)),
        "field_x": field_x,
        "target_coc": target_coc,
        "apertures": contexts,
    }


def _render_profile(
    profile: ResponseProfile,
    optical: dict[str, Any],
    resolved: dict[str, Any],
    transition_law: dict[str, Any],
    cross_talk_law: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    optics_cfg = resolved["optics"]
    sensor_cfg = resolved["sensor"]
    aperture_count = len(optical["apertures"])
    coc_count = int(optical["target_coc"].size)
    field_count = len(optical["field_x"])
    kernel_size = int(optics_cfg["kernel_size"])
    bank = np.empty(
        (aperture_count, coc_count, 1, field_count, 2, kernel_size, kernel_size),
        dtype=np.float32,
    )
    labels = np.empty((aperture_count, coc_count, 1, field_count), dtype=np.float64)
    records: list[dict[str, Any]] = []
    for aperture_index, context in enumerate(optical["apertures"]):
        local_cross_talk = aperture_cross_talk(
            profile, float(context["actual_f_number"]), cross_talk_law
        )
        aperture_profile = replace(profile, cross_talk=local_cross_talk)
        transition_scale = aperture_transition_scale(
            profile,
            float(context["actual_f_number"]),
            transition_law,
        )
        transition_exponent = float(
            transition_law["exponent_by_profile_id"][profile.profile_id]
        )
        for field_index, field_x in enumerate(optical["field_x"]):
            wavefront = context["wavefronts"][field_index]
            base_s_prop = context["base_propagations"][field_index]
            for coc_index, target_coc in enumerate(optical["target_coc"]):
                offset = float(context["offsets_m"][field_index, coc_index])
                propagated = _propagate_profile_pair(
                    wavefront,
                    s_prop=base_s_prop + offset,
                    parax=context["imaging"].parax,
                    sensor=context["imaging"].sen,
                    kernel_size=kernel_size,
                    pixel_pitch_m=float(sensor_cfg["pixel_pitch_m"]),
                    profile=aperture_profile,
                    field_x=float(field_x),
                    upsample=int(optics_cfg.get("upsample", 1)),
                    aperture_transition_scale_value=transition_scale,
                )
                left = np.asarray(propagated["left"], dtype=np.float32)
                right = np.asarray(propagated["right"], dtype=np.float32)
                bank[aperture_index, coc_index, 0, field_index, 0] = left
                bank[aperture_index, coc_index, 0, field_index, 1] = right
                mu_left_x, mu_left_y = centroid_xy(left)
                mu_right_x, mu_right_y = centroid_xy(right)
                disparity = mu_left_x - mu_right_x
                labels[aperture_index, coc_index, 0, field_index] = disparity
                records.append(
                    {
                        "aperture_index": aperture_index,
                        "f_number": float(context["actual_f_number"]),
                        "field_x_index": field_index,
                        "field_x_normalized": float(field_x),
                        "coc_index": coc_index,
                        "target_signed_coc_px": float(target_coc),
                        "actual_signed_coc_px": float(
                            context["actual_coc_px"][field_index, coc_index]
                        ),
                        "sensor_offset_m": offset,
                        "analytic_centroid_disparity_px": disparity,
                        "mu_left_x_px": mu_left_x,
                        "mu_left_y_px": mu_left_y,
                        "mu_right_x_px": mu_right_x,
                        "mu_right_y_px": mu_right_y,
                        "left_effective_pupil_power": float(
                            propagated["left_effective_pupil_power"]
                        ),
                        "right_effective_pupil_power": float(
                            propagated["right_effective_pupil_power"]
                        ),
                        "pupil_overlap_fraction": float(propagated["pupil_overlap_fraction"]),
                        "aperture_transition_power_exponent": transition_exponent,
                        "aperture_transition_scale": transition_scale,
                        "aperture_cross_talk": local_cross_talk,
                        "pupil_center_x_m": float(propagated["pupil_center_x_m"]),
                        "pupil_radius_x_m": float(propagated["pupil_radius_x_m"]),
                        "split_x_center_m": float(propagated["split_x_center_m"]),
                        "local_split_bias_norm": float(propagated["local_split_bias_norm"]),
                        "local_transition_width": float(propagated["local_transition_width"]),
                    }
                )
    return bank, labels, records


def _render_profile_batch(
    profiles: Sequence[ResponseProfile],
    optical: dict[str, Any],
    resolved: dict[str, Any],
    transition_law: dict[str, Any],
    cross_talk_law: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]]:
    """用共享 RS 几何批量渲染一组 profile，输出与逐 profile 路径同形。"""

    if not profiles:
        return {}
    optics_cfg = resolved["optics"]
    profiles_cfg = resolved["profiles"]
    aperture_count = len(optical["apertures"])
    coc_count = int(optical["target_coc"].size)
    field_count = len(optical["field_x"])
    kernel_size = int(optics_cfg["kernel_size"])
    banks = {
        profile.profile_id: np.empty(
            (aperture_count, coc_count, 1, field_count, 2, kernel_size, kernel_size),
            dtype=np.float32,
        )
        for profile in profiles
    }
    labels = {
        profile.profile_id: np.empty(
            (aperture_count, coc_count, 1, field_count),
            dtype=np.float64,
        )
        for profile in profiles
    }
    records: dict[str, list[dict[str, Any]]] = {
        profile.profile_id: [] for profile in profiles
    }
    for aperture_index, context in enumerate(optical["apertures"]):
        aperture_cross_talks = [
            aperture_cross_talk(
                profile, float(context["actual_f_number"]), cross_talk_law
            )
            for profile in profiles
        ]
        aperture_profiles = [
            replace(profile, cross_talk=value)
            for profile, value in zip(profiles, aperture_cross_talks, strict=True)
        ]
        transition_scales = [
            aperture_transition_scale(
                profile,
                float(context["actual_f_number"]),
                transition_law,
            )
            for profile in profiles
        ]
        for field_index, field_x in enumerate(optical["field_x"]):
            wavefront = context["wavefronts"][field_index]
            base_s_prop = context["base_propagations"][field_index]
            for coc_index, target_coc in enumerate(optical["target_coc"]):
                offset = float(context["offsets_m"][field_index, coc_index])
                propagated_rows = _propagate_profile_batch(
                    wavefront,
                    s_prop=base_s_prop + offset,
                    parax=context["imaging"].parax,
                    sensor=context["imaging"].sen,
                    kernel_size=kernel_size,
                    profiles=aperture_profiles,
                    field_x=float(field_x),
                    upsample=int(optics_cfg.get("upsample", 1)),
                    sensor_chunk_size=int(
                        profiles_cfg.get("batch_sensor_chunk_size", 256)
                    ),
                    aperture_transition_scales=transition_scales,
                )
                for profile_index, (profile, propagated) in enumerate(
                    zip(profiles, propagated_rows, strict=True)
                ):
                    left = np.asarray(propagated["left"], dtype=np.float32)
                    right = np.asarray(propagated["right"], dtype=np.float32)
                    bank = banks[profile.profile_id]
                    bank[aperture_index, coc_index, 0, field_index, 0] = left
                    bank[aperture_index, coc_index, 0, field_index, 1] = right
                    mu_left_x, mu_left_y = centroid_xy(left)
                    mu_right_x, mu_right_y = centroid_xy(right)
                    disparity = mu_left_x - mu_right_x
                    labels[profile.profile_id][aperture_index, coc_index, 0, field_index] = disparity
                    records[profile.profile_id].append(
                        {
                            "aperture_index": aperture_index,
                            "f_number": float(context["actual_f_number"]),
                            "field_x_index": field_index,
                            "field_x_normalized": float(field_x),
                            "coc_index": coc_index,
                            "target_signed_coc_px": float(target_coc),
                            "actual_signed_coc_px": float(
                                context["actual_coc_px"][field_index, coc_index]
                            ),
                            "sensor_offset_m": offset,
                            "analytic_centroid_disparity_px": disparity,
                            "mu_left_x_px": mu_left_x,
                            "mu_left_y_px": mu_left_y,
                            "mu_right_x_px": mu_right_x,
                            "mu_right_y_px": mu_right_y,
                            "left_effective_pupil_power": float(
                                propagated["left_effective_pupil_power"]
                            ),
                            "right_effective_pupil_power": float(
                                propagated["right_effective_pupil_power"]
                            ),
                            "pupil_overlap_fraction": float(
                                propagated["pupil_overlap_fraction"]
                            ),
                            "aperture_transition_power_exponent": float(
                                transition_law["exponent_by_profile_id"][profile.profile_id]
                            ),
                            "aperture_transition_scale": float(
                                transition_scales[profile_index]
                            ),
                            "aperture_cross_talk": float(
                                aperture_cross_talks[profile_index]
                            ),
                            "pupil_center_x_m": float(propagated["pupil_center_x_m"]),
                            "pupil_radius_x_m": float(propagated["pupil_radius_x_m"]),
                            "split_x_center_m": float(propagated["split_x_center_m"]),
                            "local_split_bias_norm": float(
                                propagated["local_split_bias_norm"]
                            ),
                            "local_transition_width": float(
                                propagated["local_transition_width"]
                            ),
                        }
                    )
    return {
        profile.profile_id: (
            banks[profile.profile_id],
            labels[profile.profile_id],
            records[profile.profile_id],
        )
        for profile in profiles
    }


def _write_preview(path: Path, banks: Sequence[np.ndarray], profile_ids: Sequence[str]) -> None:
    from PIL import Image, ImageDraw

    if not banks:
        return
    kernel_size = int(banks[0].shape[-1])
    coc_count = int(banks[0].shape[1])
    aperture_count = int(banks[0].shape[0])
    pad = 2
    rows = len(banks) * aperture_count
    canvas = Image.new(
        "L",
        (coc_count * 2 * (kernel_size + pad), rows * (kernel_size + pad)),
        color=0,
    )
    center_field = banks[0].shape[3] // 2
    for profile_index, bank in enumerate(banks):
        for aperture_index in range(aperture_count):
            row = profile_index * aperture_count + aperture_index
            for coc_index in range(coc_count):
                for side in range(2):
                    psf = bank[aperture_index, coc_index, 0, center_field, side]
                    visible = np.power(
                        np.clip(psf / max(float(psf.max()), 1.0e-20), 0.0, 1.0),
                        0.25,
                    )
                    cell = Image.fromarray((visible * 255.0 + 0.5).astype(np.uint8), mode="L")
                    x = (2 * coc_index + side) * (kernel_size + pad)
                    y = row * (kernel_size + pad)
                    canvas.paste(cell, (x, y))
    ImageDraw.Draw(canvas)
    canvas.save(path)


def _training_recipe(
    accepted: Sequence[dict[str, Any]],
    f_numbers: Sequence[float],
    coc_abs_max: float,
) -> dict[str, Any]:
    assets: list[str] = []
    paths: list[str] = []
    apertures: list[float] = []
    for row in accepted:
        for aperture in f_numbers:
            aperture_id = str(float(aperture)).replace(".", "p")
            assets.append(f"{row['profile_id']}_f{aperture_id}")
            paths.append(str(row["bank_path"]))
            apertures.append(float(aperture))
    weight = 1.0 / float(len(assets))
    return {
        "synthesis": {
            "name": "dp_renderer",
            "dp_mode": "physical_psf_source_field",
            "dp_validation_mode": "physical_psf_source_field",
            "dp_linear_light": True,
            "psf_assets": assets,
            "psf_asset_weights": [weight] * len(assets),
            "psf_banks": paths,
            "psf_apertures": apertures,
            "psf_aperture_conditioned": True,
            "psf_aperture_match_tolerance": 1.0e-4,
            "psf_apply_side_throughput": True,
            "psf_coc_abs_maxes": [float(coc_abs_max)] * len(assets),
            "psf_focus_zero_calibrate": False,
            "psf_centroid_slope_target": None,
            "psf_physical_centroid_mode": "bank",
            "ncc_anchor_enable": False,
        },
        "contract": {
            "label_source": "analytic PSF centroid only",
            "pd_offset_embedded": False,
            "field_origin_calibration": "asset_precalibrated_when_declared",
            "field_origin_calibration_is_ncc_correction": False,
            "field_origin_calibration_is_pdoffset": False,
            "side_throughput_applied_by": "DPRenderer",
            "field_coordinate_runtime": "normalized current input extent",
        },
    }


def generate(config_path: Path) -> dict[str, Any]:
    """生成冻结 profile 目录，并仅把完整通过门禁的候选落为外部资产。"""

    resolved, project_root = _resolve_config(config_path)
    profile_cfg = resolved["profiles"]
    catalog, catalog_source = resolve_profile_catalog(profile_cfg)
    transition_law = resolve_aperture_transition_power_law(profile_cfg, catalog)
    cross_talk_law = resolve_aperture_cross_talk_by_f_number(
        profile_cfg,
        catalog,
        resolved["apertures"]["f_numbers"],
    )
    early_diagnostic_cfg = dict(resolved["diagnostic"])
    early_ncc_gates = dict(early_diagnostic_cfg["ncc_gates"])
    if str(early_ncc_gates.get("admission_mode")) == (
        "per_aperture_continuous_near_focus_v1"
    ):
        validate_continuous_ncc_v2_declaration(
            dict(early_diagnostic_cfg["continuous_ncc"])
        )
        expected_apertures = int(
            early_diagnostic_cfg["continuous_ncc"]["expected_aperture_count"]
        )
        if len(resolved["apertures"]["f_numbers"]) != expected_apertures:
            raise ValueError(
                "continuous V2 expected_aperture_count 与 f_numbers 数量不一致"
            )
    catalog_by_id = {row.profile_id: row for row in catalog}
    pilot_ids = [str(value) for value in profile_cfg.get("pilot_profile_ids", [])]
    if len(pilot_ids) != len(set(pilot_ids)) or any(value not in catalog_by_id for value in pilot_ids):
        raise ValueError("pilot_profile_ids 必须唯一且来自已解析 profile catalog")
    candidate_order = pilot_ids + [row.profile_id for row in catalog if row.profile_id not in pilot_ids]
    candidate_limit = min(int(profile_cfg.get("max_candidates_to_render", len(catalog))), len(catalog))
    accepted_target = int(profile_cfg["accepted_profiles_target"])
    if accepted_target < 1 or candidate_limit < accepted_target:
        raise ValueError("accepted_profiles_target/max_candidates_to_render 非法")
    for profile in catalog:
        for f_number in resolved["apertures"]["f_numbers"]:
            transition_scale = aperture_transition_scale(
                profile,
                float(f_number),
                transition_law,
            )
            local_cross_talk = aperture_cross_talk(
                profile,
                float(f_number),
                cross_talk_law,
            )
            if not math.isfinite(local_cross_talk) or not 0.0 <= local_cross_talk < 0.5:
                raise ValueError(
                    f"profile {profile.profile_id} 在 f/{f_number} 的 cross-talk 非法"
                )
            for field_x in resolved["optics"]["field_x_normalized"]:
                local_width = profile.transition_width * transition_scale * (
                    1.0 + profile.field_acceptance_slope * float(field_x)
                )
                local_split = (
                    profile.split_bias_norm + profile.field_split_slope * float(field_x)
                )
                if not math.isfinite(local_width) or local_width <= 0.0:
                    raise ValueError(
                        f"profile {profile.profile_id} 在 f/{f_number}, field={field_x} "
                        "的 width 非法"
                    )
                if not math.isfinite(local_split) or abs(local_split) >= 1.0:
                    raise ValueError(
                        f"profile {profile.profile_id} 在 field={field_x} 的 split 非法"
                    )
        build_side_throughput_grid(
            profile,
            resolved["apertures"]["f_numbers"],
            resolved["optics"]["field_x_normalized"],
        )
    source_root = Path(resolved["source"]["checkout"])
    datasyn_src = source_root / "datasyn" / "src"
    for import_root in (project_root, datasyn_src):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    from datasyn.jaxutils.configs import easy_optics_setup

    easy_optics_setup()
    import torch

    output_root = Path(resolved["output"]["root"])
    if output_root == project_root or project_root in output_root.parents:
        raise ValueError(f"输出目录必须在仓库外部：{output_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{output_root}")
    source_commit = _git_output(source_root, "rev-parse", "HEAD")
    if source_commit != str(resolved["source"]["commit"]):
        raise RuntimeError(
            f"CLDefocus commit 不匹配：expected={resolved['source']['commit']}, actual={source_commit}"
        )
    lens_path = Path(resolved["lens"]["root"]) / str(resolved["lens"]["relative_path"])
    if not lens_path.is_file() or _sha256_file(lens_path) != str(resolved["lens"]["sha256"]):
        raise RuntimeError("镜头处方不存在或 SHA256 不匹配")
    output_root.mkdir(parents=True, exist_ok=True)
    profiles_root = output_root / "profiles"
    profiles_root.mkdir()
    candidate_diagnostics_root = output_root / "candidate_diagnostics"
    candidate_diagnostics_root.mkdir()

    catalog_document = build_profile_catalog_document(
        catalog,
        catalog_source=catalog_source,
        profile_cfg=profile_cfg,
    )
    (output_root / "profile_catalog.json").write_text(
        json.dumps(
            catalog_document,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    resolved_path = output_root / "resolved_config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    setup_start = time.perf_counter()
    optical = _prepare_optical_contexts(resolved)
    setup_seconds = time.perf_counter() - setup_start
    f_numbers = [float(row["actual_f_number"]) for row in optical["apertures"]]
    signed_coc_bins = np.tile(
        optical["target_coc"][None, :],
        (len(f_numbers), 1),
    )

    accepted: list[dict[str, Any]] = []
    accepted_banks: list[np.ndarray] = []
    accepted_labels: list[np.ndarray] = []
    accepted_raw_labels: list[np.ndarray] = []
    accepted_calibrations: list[dict[str, Any] | None] = []
    accepted_throughput: list[np.ndarray] = []
    statuses: list[dict[str, Any]] = []
    ncc_profiles: dict[str, Any] = {}
    oracle_profiles: dict[str, Any] = {}
    continuous_ncc_profiles: dict[str, Any] = {}
    continuous_oracle_profiles: dict[str, Any] = {}
    rendered_pair_count = 0
    generation_start = time.perf_counter()
    diagnostic_cfg = resolved["diagnostic"]
    optical_gates = dict(diagnostic_cfg["optical_gates"])
    ncc_gates = dict(diagnostic_cfg["ncc_gates"])
    calibration_cfg = dict(resolved.get("field_origin_calibration", {}))
    calibration_enabled = bool(calibration_cfg.get("enabled", False))
    version_tag = str(
        resolved.get("run", {}).get(
            "log_tag",
            "V8" if calibration_enabled else "V7",
        )
    )

    batch_results: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]] = {}
    batch_shared_rs = bool(profile_cfg.get("batch_shared_rs", False))
    batch_render_seconds = 0.0
    if batch_shared_rs:
        batch_profiles = [
            catalog_by_id[profile_id] for profile_id in candidate_order[:candidate_limit]
        ]
        batch_chunks = deterministic_profile_chunks(
            batch_profiles,
            profile_cfg.get("batch_profile_chunk_size"),
        )
        print(
            f"[{version_tag}] shared-RS batch render profiles={len(batch_profiles)} "
            f"chunks={len(batch_chunks)} fields={len(optical['field_x'])} "
            f"upsample={resolved['optics'].get('upsample', 1)} "
            f"sensor_chunk={profile_cfg.get('batch_sensor_chunk_size', 256)}",
            flush=True,
        )
        batch_start = time.perf_counter()
        for chunk_index, profile_chunk in enumerate(batch_chunks):
            print(
                f"[{version_tag}] shared-RS chunk={chunk_index + 1}/{len(batch_chunks)} "
                f"profiles={[row.profile_id for row in profile_chunk]}",
                flush=True,
            )
            chunk_results = _render_profile_batch(
                profile_chunk,
                optical,
                resolved,
                transition_law,
                cross_talk_law,
            )
            overlap = set(batch_results) & set(chunk_results)
            if overlap:
                raise RuntimeError(f"shared-RS profile 分块出现重复结果：{sorted(overlap)}")
            batch_results.update(chunk_results)
        batch_render_seconds = time.perf_counter() - batch_start
        if batch_results:
            one_bank = next(iter(batch_results.values()))[0]
            rendered_pair_count = int(np.prod(one_bank.shape[:4])) * len(batch_results)
        print(
            f"[{version_tag}] shared-RS batch completed seconds={batch_render_seconds:.3f}",
            flush=True,
        )

    for candidate_index, profile_id in enumerate(candidate_order[:candidate_limit]):
        if len(accepted) >= accepted_target and not bool(
            profile_cfg.get("evaluate_all_candidates", False)
        ):
            break
        profile = catalog_by_id[profile_id]
        print(
            f"[{version_tag}] render {profile_id} candidate={candidate_index + 1}/{candidate_limit} "
            f"accepted={len(accepted)}/{accepted_target}",
            flush=True,
        )
        candidate_start = time.perf_counter()
        status: dict[str, Any] = {
            "profile_id": profile_id,
            "candidate_index": candidate_index,
            "parameters": asdict(profile),
        }
        try:
            if batch_shared_rs:
                bank, labels, propagation_records = batch_results[profile_id]
            else:
                bank, labels, propagation_records = _render_profile(
                    profile,
                    optical,
                    resolved,
                    transition_law,
                    cross_talk_law,
                )
            raw_render_labels = np.asarray(labels, dtype=np.float64).copy()
            calibration: FieldOriginCalibrationResult | None = None
            calibration_metadata: dict[str, Any] | None = None
            if calibration_enabled:
                calibration = calibrate_field_origin(
                    bank,
                    signed_coc_bins,
                    centroid_residual_max_px=float(
                        calibration_cfg.get("centroid_residual_max_px", 0.01)
                    ),
                    curve_identity_abs_max_px=float(
                        calibration_cfg.get("curve_identity_abs_max_px", 2.0e-5)
                    ),
                    requested_support_padding_px=int(
                        calibration_cfg.get("support_padding_px", 2)
                    ),
                    retained_mass_min=float(
                        calibration_cfg.get("retained_mass_min", 0.999999)
                    ),
                    centered_covariance_abs_delta_max_px2=float(
                        calibration_cfg.get(
                            "centered_covariance_abs_delta_max_px2",
                            0.25,
                        )
                    ),
                )
                raw_label_recompute_abs_max = float(
                    np.max(
                        np.abs(
                            calibration.raw_analytic_labels_px
                            - raw_render_labels
                        )
                    )
                )
                calibration.metadata["raw_render_label_recompute_abs_max_px"] = (
                    raw_label_recompute_abs_max
                )
                calibration.metadata["checks"]["raw_render_label_recompute"] = (
                    raw_label_recompute_abs_max
                    <= float(calibration_cfg.get("raw_label_recompute_abs_max_px", 2.0e-6))
                )
                calibration.metadata["pass"] = bool(
                    all(calibration.metadata["checks"].values())
                )
                bank = calibration.psf_bank
                labels = calibration.analytic_labels_px
                propagation_records = apply_field_origin_to_propagation_records(
                    propagation_records,
                    calibration,
                )
                calibration_metadata = calibration.metadata
            pair_count = int(np.prod(bank.shape[:4]))
            if not batch_shared_rs:
                rendered_pair_count += pair_count
            optical_diag = profile_optical_diagnostics(
                bank,
                signed_coc_bins,
                labels,
                propagation_records,
                gates=optical_gates,
            )
            psf_moments = profile_psf_moment_diagnostics(bank, signed_coc_bins)
            ncc_admission_mode = str(ncc_gates.get("admission_mode", "aggregate_v7"))
            parabolic_compatibility_error: dict[str, str] | None = None
            try:
                ncc = run_profile_ncc_diagnostic(
                    bank,
                    labels,
                    signed_coc_bins_px=signed_coc_bins,
                    small_coc_abs_max_px=(
                        None
                        if diagnostic_cfg.get("small_coc_abs_max_px") is None
                        else float(diagnostic_cfg["small_coc_abs_max_px"])
                    ),
                    texture_size=int(diagnostic_cfg["texture_size"]),
                    seeds=[int(value) for value in diagnostic_cfg["seeds"]],
                    tile_size=int(diagnostic_cfg["tile_size"]),
                    tiles_per_axis=int(diagnostic_cfg["tiles_per_axis"]),
                    search_x=int(diagnostic_cfg["search_x"]),
                    search_y=int(diagnostic_cfg["search_y"]),
                )
            except Exception as error:
                if ncc_admission_mode != "per_aperture_continuous_near_focus_v1":
                    raise
                parabolic_compatibility_error = {
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                ncc = {
                    "role": "current_parabolic_compatibility_report_failed_no_veto",
                    "label_compensation_from_ncc": False,
                    "compatibility_error": parabolic_compatibility_error,
                }
            oracle: dict[str, Any] | None = None
            continuous_ncc: dict[str, Any] | None = None
            continuous_oracle: dict[str, Any] | None = None
            admission_aggregate_fit = ncc.get("aggregate_fit")
            if ncc_admission_mode == "per_aperture_continuous_near_focus_v1":
                coc_abs_max = float(ncc_gates.get("coc_abs_max_px", 1.0))
                continuous_cfg = dict(diagnostic_cfg["continuous_ncc"])
                continuous_runtime_kwargs = continuous_ncc_v2_runtime_kwargs(
                    continuous_cfg
                )
                continuous_ncc, continuous_oracle = run_profile_continuous_ncc_diagnostic(
                    bank,
                    labels,
                    signed_coc_bins_px=signed_coc_bins,
                    texture_size=int(diagnostic_cfg["texture_size"]),
                    seeds=[int(value) for value in diagnostic_cfg["seeds"]],
                    tile_size=int(diagnostic_cfg["tile_size"]),
                    tiles_per_axis=int(diagnostic_cfg["tiles_per_axis"]),
                    search_x=int(diagnostic_cfg["search_x"]),
                    search_y=int(diagnostic_cfg["search_y"]),
                    **continuous_runtime_kwargs,
                )
                primary_gate = apply_per_aperture_gate(
                    summarize_per_aperture_ncc(
                        continuous_ncc,
                        aperture_count=bank.shape[0],
                        coc_abs_max_px=coc_abs_max,
                    ),
                    dict(ncc_gates["primary"]),
                )
                continuous_legacy_wide_gate = apply_per_aperture_gate(
                    summarize_per_aperture_ncc(
                        continuous_ncc,
                        aperture_count=bank.shape[0],
                        coc_abs_max_px=coc_abs_max,
                    ),
                    dict(ncc_gates["legacy_wide"]),
                )
                parabolic_strict_report: dict[str, Any] | None = None
                parabolic_legacy_report: dict[str, Any] | None = None
                continuous_oracle_summary = summarize_per_aperture_ncc(
                    continuous_oracle,
                    aperture_count=bank.shape[0],
                    coc_abs_max_px=coc_abs_max,
                )
                continuous_oracle["per_aperture_summary"] = continuous_oracle_summary
                continuous_oracle["physical_comparison"] = (
                    compare_aligned_continuous_ncc_to_oracle(
                        continuous_ncc,
                        continuous_oracle,
                        aperture_count=bank.shape[0],
                        coc_abs_max_px=coc_abs_max,
                    )
                )
                # 旧 parabolic estimator 与其 Fourier oracle 仅作兼容解释；任何异常不 veto。
                if parabolic_compatibility_error is None:
                    try:
                        parabolic_strict_report = apply_per_aperture_gate(
                            summarize_per_aperture_ncc(
                                ncc,
                                aperture_count=bank.shape[0],
                                coc_abs_max_px=coc_abs_max,
                            ),
                            dict(ncc_gates["primary"]),
                        )
                        parabolic_legacy_report = apply_per_aperture_gate(
                            summarize_per_aperture_ncc(
                                ncc,
                                aperture_count=bank.shape[0],
                                coc_abs_max_px=coc_abs_max,
                            ),
                            dict(ncc_gates["legacy_wide"]),
                        )
                        oracle = run_shifted_identical_ncc_oracle(
                            labels,
                            signed_coc_bins_px=signed_coc_bins,
                            kernel_size=int(bank.shape[-1]),
                            texture_size=int(diagnostic_cfg["texture_size"]),
                            seeds=[int(value) for value in diagnostic_cfg["seeds"]],
                            tile_size=int(diagnostic_cfg["tile_size"]),
                            tiles_per_axis=int(diagnostic_cfg["tiles_per_axis"]),
                            search_x=int(diagnostic_cfg["search_x"]),
                            search_y=int(diagnostic_cfg["search_y"]),
                        )
                        oracle_summary = summarize_per_aperture_ncc(
                            oracle,
                            aperture_count=bank.shape[0],
                            coc_abs_max_px=coc_abs_max,
                        )
                        oracle["per_aperture_summary"] = oracle_summary
                        oracle["physical_comparison"] = (
                            compare_per_aperture_ncc_to_oracle(
                                parabolic_strict_report,
                                oracle_summary,
                            )
                        )
                    except Exception as error:
                        parabolic_compatibility_error = {
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                        oracle = None
                ncc_gate = {
                    "mode": ncc_admission_mode,
                    "primary_estimator": "gt_free_continuous_lanczos_v2",
                    "pass": bool(primary_gate["pass"]),
                    "primary": primary_gate,
                    "continuous_legacy_wide_report_only": continuous_legacy_wide_gate,
                    "current_parabolic_strict_report_only": parabolic_strict_report,
                    "current_parabolic_legacy_wide_report_only": parabolic_legacy_report,
                    "current_parabolic_compatibility_error_no_veto": (
                        parabolic_compatibility_error
                    ),
                    "aggregate_is_admission_gate": False,
                    "oracle_is_admission_correction": False,
                    "ncc_or_oracle_updates_analytic_label": False,
                }
                continuous_ncc["admission"] = ncc_gate
                continuous_ncc_profiles[profile_id] = continuous_ncc
                continuous_oracle_profiles[profile_id] = continuous_oracle
                if oracle is not None:
                    oracle_profiles[profile_id] = oracle
                admission_aggregate_fit = continuous_ncc["aggregate_fit"]
            elif ncc_admission_mode == "per_aperture_near_focus_v1":
                coc_abs_max = float(ncc_gates.get("coc_abs_max_px", 1.0))
                primary_gate = apply_per_aperture_gate(
                    summarize_per_aperture_ncc(
                        ncc,
                        aperture_count=bank.shape[0],
                        coc_abs_max_px=coc_abs_max,
                    ),
                    dict(ncc_gates["primary"]),
                )
                legacy_wide_gate = apply_per_aperture_gate(
                    summarize_per_aperture_ncc(
                        ncc,
                        aperture_count=bank.shape[0],
                        coc_abs_max_px=coc_abs_max,
                    ),
                    dict(ncc_gates["legacy_wide"]),
                )
                oracle = run_shifted_identical_ncc_oracle(
                    labels,
                    signed_coc_bins_px=signed_coc_bins,
                    kernel_size=int(bank.shape[-1]),
                    texture_size=int(diagnostic_cfg["texture_size"]),
                    seeds=[int(value) for value in diagnostic_cfg["seeds"]],
                    tile_size=int(diagnostic_cfg["tile_size"]),
                    tiles_per_axis=int(diagnostic_cfg["tiles_per_axis"]),
                    search_x=int(diagnostic_cfg["search_x"]),
                    search_y=int(diagnostic_cfg["search_y"]),
                )
                oracle_summary = summarize_per_aperture_ncc(
                    oracle,
                    aperture_count=bank.shape[0],
                    coc_abs_max_px=coc_abs_max,
                )
                oracle["per_aperture_summary"] = oracle_summary
                oracle["physical_comparison"] = compare_per_aperture_ncc_to_oracle(
                    primary_gate,
                    oracle_summary,
                )
                ncc_gate = {
                    "mode": ncc_admission_mode,
                    "pass": bool(primary_gate["pass"]),
                    "primary": primary_gate,
                    "legacy_wide_report_only": legacy_wide_gate,
                    "aggregate_is_admission_gate": False,
                    "oracle_is_admission_correction": False,
                }
                oracle_profiles[profile_id] = oracle
            else:
                ncc_gate = ncc_admission(ncc, ncc_gates)
                ncc_gate["mode"] = ncc_admission_mode
            ncc["admission"] = ncc_gate
            ncc_profiles[profile_id] = ncc
            admitted = bool(
                optical_diag["pass"]
                and ncc_gate["pass"]
                and (
                    calibration_metadata is None
                    or bool(calibration_metadata["pass"])
                )
            )
            status.update(
                {
                    "status": "accepted" if admitted else "rejected_by_gate",
                    "optical_diagnostics": optical_diag,
                    "ncc_summary": {**admission_aggregate_fit, **ncc_gate},
                    "continuous_ncc_diagnostic": continuous_ncc,
                    "continuous_shifted_identical_oracle": continuous_oracle,
                    "field_origin_calibration": calibration_metadata,
                    "psf_moment_diagnostics": psf_moments,
                    "ncc_shifted_identical_oracle": oracle,
                    "propagation_records": propagation_records,
                }
            )
            (candidate_diagnostics_root / f"{profile_id}.json").write_text(
                json.dumps(
                    {
                        "profile_id": profile_id,
                        "parameters": asdict(profile),
                        "optical_diagnostics": optical_diag,
                        "ncc_diagnostic": ncc,
                        "continuous_ncc_diagnostic": continuous_ncc,
                        "continuous_shifted_identical_oracle": continuous_oracle,
                        "ncc_shifted_identical_oracle": oracle,
                        "propagation_records": propagation_records,
                        "psf_moment_diagnostics": psf_moments,
                        "field_origin_calibration": calibration_metadata,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if admitted:
                profile_dir = profiles_root / profile_id
                profile_dir.mkdir()
                throughput = build_side_throughput_grid(
                    profile,
                    f_numbers,
                    optical["field_x"],
                )
                bank_path = profile_dir / "psf_bank.pt"
                asset_id = f"{resolved['run']['id']}_{profile_id}"
                torch.save(
                    {
                        "asset_id": asset_id,
                        "profile_id": profile_id,
                        "profile_parameters": asdict(profile),
                        "asset_spec": {
                            "kind": "cldefocus_multi_aperture_dp_response_profile",
                            "axis_order": [
                                "aperture",
                                "signed_coc",
                                "field_y",
                                "field_x",
                                "pdraw_side_left_right",
                                "kernel_y",
                                "kernel_x",
                            ],
                            "signed_disparity": "d=x_L-x_R; near positive; far negative",
                            "analytic_label_source": "PSF centroid mu_left_x-mu_right_x",
                            "ncc_role": "diagnostic_only_no_label_compensation",
                            "field_origin_calibrated": calibration_metadata is not None,
                            "field_origin_role": "coordinate_origin_only_not_ncc_not_pdoffset",
                            "pd_offset_embedded": False,
                            "pd_offset_px": 0.0,
                            "side_throughput_normalization": "pair_mean_one",
                            "field_coordinate_runtime": "normalized current input extent",
                        },
                        "f_numbers": torch.tensor(f_numbers, dtype=torch.float32),
                        "signed_coc_bins_px": torch.from_numpy(signed_coc_bins.copy()),
                        "analytic_disparity_bins_px": torch.from_numpy(labels.copy()),
                        "raw_analytic_disparity_bins_px": torch.from_numpy(
                            (
                                calibration.raw_analytic_labels_px
                                if calibration is not None
                                else raw_render_labels
                            ).copy()
                        ),
                        "field_origin_calibration": calibration_metadata,
                        "field_grid_hw": (1, len(optical["field_x"])),
                        "field_x_normalized": torch.tensor(optical["field_x"]),
                        "kernel_size": int(bank.shape[-1]),
                        "side_throughput_grid": torch.from_numpy(throughput),
                        "psf_bank": torch.from_numpy(bank),
                    },
                    bank_path,
                )
                bank_record = {
                    "profile_id": profile_id,
                    "bank_path": str(bank_path.resolve()),
                    "bank_relative_path": str(bank_path.relative_to(output_root)),
                    "bank_sha256": _sha256_file(bank_path),
                    "bank_bytes": bank_path.stat().st_size,
                    "shape": list(bank.shape),
                    "parameters": asdict(profile),
                    "optical_diagnostics": optical_diag,
                    "ncc_summary": admission_aggregate_fit,
                    "near_focus_ncc_admission": ncc_gate,
                    "psf_second_moment_trace_min_px2": float(
                        psf_moments["centered_second_moment_trace_min_px2"]
                    ),
                    "psf_second_moment_trace_max_px2": float(
                        psf_moments["centered_second_moment_trace_max_px2"]
                    ),
                    "field_origin_calibration_summary": (
                        None
                        if calibration_metadata is None
                        else {
                            "pass": bool(calibration_metadata["pass"]),
                            "focus_side_centroid_residual_abs_max_px": float(
                                calibration_metadata[
                                    "focus_side_centroid_residual_abs_max_px"
                                ]
                            ),
                            "side_curve_identity_abs_max_px": float(
                                calibration_metadata["side_curve_identity_abs_max_px"]
                            ),
                            "disparity_curve_identity_abs_max_px": float(
                                calibration_metadata[
                                    "disparity_curve_identity_abs_max_px"
                                ]
                            ),
                            "retained_mass_min": float(
                                calibration_metadata["retained_mass_min"]
                            ),
                            "centered_covariance_abs_delta_max_px2": float(
                                calibration_metadata[
                                    "centered_covariance_abs_delta_max_px2"
                                ]
                            ),
                        }
                    ),
                    "side_throughput_min": float(throughput.min()),
                    "side_throughput_max": float(throughput.max()),
                    "pd_offset_embedded": False,
                }
                accepted.append(bank_record)
                accepted_banks.append(bank)
                accepted_labels.append(labels)
                accepted_raw_labels.append(
                    calibration.raw_analytic_labels_px
                    if calibration is not None
                    else raw_render_labels
                )
                accepted_calibrations.append(calibration_metadata)
                accepted_throughput.append(throughput)
                status["bank"] = bank_record
            if ncc_admission_mode in {
                "per_aperture_near_focus_v1",
                "per_aperture_continuous_near_focus_v1",
            }:
                primary_rows = ncc_gate["primary"]["per_aperture"]
                slopes = [float(row["fit"]["slope_ncc_vs_analytic"]) for row in primary_rows]
                r2_values = [float(row["fit"]["r2"]) for row in primary_rows]
                print(
                    f"[{version_tag}] {profile_id} {status['status']} "
                    f"per_ap_slope={min(slopes):.6f}..{max(slopes):.6f} "
                    f"per_ap_r2={min(r2_values):.6f}..{max(r2_values):.6f}",
                    flush=True,
                )
            else:
                print(
                    f"[{version_tag}] {profile_id} {status['status']} "
                    f"slope={ncc['aggregate_fit']['slope_ncc_vs_analytic']:.6f} "
                    f"offset={ncc['aggregate_fit']['intercept_px']:.6f} "
                    f"r2={ncc['aggregate_fit']['r2']:.6f}",
                    flush=True,
                )
        except Exception as error:  # 单个 profile 自动拒收，继续寻找 8 个通过项。
            status.update(
                {
                    "status": "rejected_by_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            (candidate_diagnostics_root / f"{profile_id}.json").write_text(
                json.dumps(status, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[{version_tag}] {profile_id} rejected_by_error: {error}", flush=True)
        status["elapsed_seconds"] = time.perf_counter() - candidate_start
        statuses.append(status)

    generation_seconds = time.perf_counter() - generation_start
    pass_count = len(accepted)
    admission_pass = pass_count >= accepted_target
    combined_path: Path | None = None
    if accepted_banks:
        combined_path = output_root / "psf_bank_profiles.pt"
        torch.save(
            {
                "asset_id": str(resolved["run"]["id"]),
                "profile_ids": [row["profile_id"] for row in accepted],
                "profile_parameters": [row["parameters"] for row in accepted],
                "f_numbers": torch.tensor(f_numbers, dtype=torch.float32),
                "signed_coc_bins_px": torch.from_numpy(signed_coc_bins.copy()),
                "analytic_disparity_bins_px": torch.from_numpy(
                    np.stack(accepted_labels, axis=0)
                ),
                "raw_analytic_disparity_bins_px": torch.from_numpy(
                    np.stack(accepted_raw_labels, axis=0)
                ),
                "field_origin_calibration": accepted_calibrations,
                "field_grid_hw": (1, len(optical["field_x"])),
                "field_x_normalized": torch.tensor(optical["field_x"]),
                "kernel_size": int(accepted_banks[0].shape[-1]),
                "side_throughput_grid": torch.from_numpy(
                    np.stack(accepted_throughput, axis=0)
                ),
                "pd_offset_embedded": False,
                "pd_offset_px": 0.0,
                "psf_bank": torch.from_numpy(np.stack(accepted_banks, axis=0)),
            },
            combined_path,
        )

    profiles_manifest_path = output_root / "profiles_manifest.json"
    profiles_manifest = {
        "schema_version": 1,
        "run_id": str(resolved["run"]["id"]),
        "status": "pass" if admission_pass else "insufficient_accepted_profiles",
        "catalog_size": len(catalog),
        "catalog_source": catalog_source,
        "explicit_profile_order_preserved": catalog_source == "explicit_response_profiles_v1",
        "catalog_sha256": _sha256_file(output_root / "profile_catalog.json"),
        "accepted_target": accepted_target,
        "accepted_count": pass_count,
        "rendered_candidate_count": len(statuses),
        "profile_axis_archive": None if combined_path is None else str(combined_path.resolve()),
        "profile_axis_archive_is_directly_selectable_by_loader": True,
        "recommended_training_layout": "per-profile 7D bank expanded by aperture",
        "f_numbers": f_numbers,
        "signed_coc_bins_px": signed_coc_bins.tolist(),
        "field_grid_hw": [1, len(optical["field_x"])],
        "field_x_normalized": optical["field_x"],
        "field_coordinate_runtime": "normalized current input extent",
        "side_order": ["left", "right"],
        "side_throughput_normalization": "pair_mean_one",
        "pd_offset_embedded": False,
        "pd_offset_px": 0.0,
        "field_origin_calibration_enabled": calibration_enabled,
        "field_origin_calibration_role": "coordinate_origin_only_not_ncc_not_pdoffset",
        "label_source": "analytic PSF centroid mu_left_x-mu_right_x",
        "ncc_updates_labels": False,
        "ncc_admission_mode": str(ncc_gates.get("admission_mode", "aggregate_v7")),
        "ncc_primary_estimator": (
            "gt_free_continuous_lanczos_v2"
            if str(ncc_gates.get("admission_mode"))
            == "per_aperture_continuous_near_focus_v1"
            else "current_parabolic_or_aggregate_legacy"
        ),
        "aperture_transition_response": transition_law,
        "aperture_cross_talk_response": cross_talk_law,
        "shifted_identical_oracle_updates_labels": False,
        "shifted_identical_oracle_updates_admission": False,
        "accepted_profiles": accepted,
        "candidate_status": statuses,
    }
    profiles_manifest_path.write_text(
        json.dumps(profiles_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ncc_path = output_root / "ncc_diagnostics.json"
    ncc_path.write_text(
        json.dumps(
            {
                "role": "independent_tile_diagnostic_only",
                "label_compensation_from_ncc": False,
                "profiles": ncc_profiles,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    oracle_path = output_root / "ncc_shifted_identical_oracle.json"
    oracle_path.write_text(
        json.dumps(
            {
                "role": "fourier_shifted_identical_peak_locking_baseline_only",
                "label_compensation_from_oracle": False,
                "oracle_updates_admission": False,
                "profiles": oracle_profiles,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if continuous_ncc_profiles:
        continuous_ncc_path = output_root / "continuous_ncc_diagnostics.json"
        continuous_ncc_path.write_text(
            json.dumps(
                {
                    "role": "primary_gt_free_continuous_lanczos_physical_psf_diagnostic",
                    "label_compensation_from_ncc": False,
                    "profiles": continuous_ncc_profiles,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        continuous_oracle_path = output_root / "continuous_shifted_identical_oracle.json"
        continuous_oracle_path.write_text(
            json.dumps(
                {
                    "role": "continuous_lanczos_fourier_shifted_identical_baseline_only",
                    "label_compensation_from_oracle": False,
                    "oracle_updates_admission": False,
                    "profiles": continuous_oracle_profiles,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    recipe = _training_recipe(
        accepted,
        f_numbers,
        float(np.max(np.abs(optical["target_coc"]))),
    ) if accepted else {"status": "no_accepted_profiles"}
    recipe_path = output_root / "dprenderer_training_assets.yaml"
    recipe_path.write_text(
        yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    preview_path = output_root / "psf_preview_center_field.png"
    _write_preview(preview_path, accepted_banks, [row["profile_id"] for row in accepted])

    measured_seconds_per_pair = (
        generation_seconds / float(rendered_pair_count) if rendered_pair_count else None
    )
    measured_seconds_per_candidate = (
        generation_seconds / float(len(statuses)) if statuses else None
    )
    summary = {
        "run_id": str(resolved["run"]["id"]),
        "status": "pass" if admission_pass else "fail",
        "accepted_count": pass_count,
        "accepted_target": accepted_target,
        "accepted_profile_ids": [row["profile_id"] for row in accepted],
        "rendered_candidate_count": len(statuses),
        "rejected_count": len(statuses) - pass_count,
        "shape_per_profile": list(accepted_banks[0].shape) if accepted_banks else None,
        "combined_shape": (
            [len(accepted_banks), *list(accepted_banks[0].shape)] if accepted_banks else None
        ),
        "setup_seconds": setup_seconds,
        "generation_and_ncc_seconds": generation_seconds,
        "shared_rs_batch": batch_shared_rs,
        "shared_rs_profile_chunk_size": profile_cfg.get("batch_profile_chunk_size"),
        "shared_rs_sensor_chunk_size": profile_cfg.get("batch_sensor_chunk_size", 256),
        "shared_rs_batch_render_seconds": batch_render_seconds,
        "rendered_psf_pair_count": rendered_pair_count,
        "measured_seconds_per_pair_including_ncc": measured_seconds_per_pair,
        "measured_seconds_per_candidate": measured_seconds_per_candidate,
        "projected_32_profile_seconds": (
            None if measured_seconds_per_candidate is None else measured_seconds_per_candidate * 32.0
        ),
        "combined_bank_bytes": None if combined_path is None else combined_path.stat().st_size,
        "per_profile_bank_bytes": [row["bank_bytes"] for row in accepted],
        "ncc_label_compensation": False,
        "ncc_admission_mode": str(ncc_gates.get("admission_mode", "aggregate_v7")),
        "ncc_primary_estimator": (
            "gt_free_continuous_lanczos_v2"
            if str(ncc_gates.get("admission_mode"))
            == "per_aperture_continuous_near_focus_v1"
            else "current_parabolic_or_aggregate_legacy"
        ),
        "current_parabolic_is_compatibility_report_only": str(
            ncc_gates.get("admission_mode")
        )
        == "per_aperture_continuous_near_focus_v1",
        "aperture_transition_response": transition_law,
        "aperture_cross_talk_response": cross_talk_law,
        "shifted_identical_oracle_label_compensation": False,
        "shifted_identical_oracle_admission_correction": False,
        "field_origin_calibration_enabled": calibration_enabled,
        "field_origin_calibration_is_ncc_correction": False,
        "field_origin_calibration_is_pdoffset": False,
        "pd_offset_embedded": False,
        "real_pdraw_accessed": False,
        "google_dev_accessed": False,
        "google_holdout_accessed": False,
        "dp5k_accessed": False,
        "stereo_training_run": False,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("multi_aperture.py").resolve(),
        (project_root / "dataset/psf_bank_renderer.py").resolve(),
        (project_root / "dataset/dp_renderer.py").resolve(),
    ]
    if str(ncc_gates.get("admission_mode")) == "per_aperture_continuous_near_focus_v1":
        source_paths.append(
            (project_root / "render/cldefocus_pdraw/continuous_ncc.py").resolve()
        )
        source_paths.append(
            (project_root / "render/cldefocus_pdraw/continuous_ncc_v2.py").resolve()
        )
    provenance = {
        "source_repository": str(resolved["source"]["repository"]),
        "source_commit": source_commit,
        "project_commit": _git_output(project_root, "rev-parse", "HEAD"),
        "project_dirty_paths": _git_output(project_root, "status", "--porcelain").splitlines(),
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256_file(config_path),
        "resolved_config_sha256": _sha256_file(resolved_path),
        "implementation_source_sha256": {
            str(path.relative_to(project_root)): _sha256_file(path) for path in source_paths
        },
        "lens_path": str(lens_path),
        "lens_sha256": _sha256_file(lens_path),
        "runtime": "Genfocus",
        "python_executable": sys.executable,
        "real_data_accessed": False,
        "stereo_training_run": False,
    }
    provenance_path = output_root / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    artifact_entries = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifact_entries[str(path.relative_to(output_root))] = {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
    artifact_manifest_path = output_root / "artifact_manifest.json"
    artifact_manifest_path.write_text(
        json.dumps(artifact_entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "output_root": str(output_root),
        "status": summary["status"],
        "accepted_profile_ids": summary["accepted_profile_ids"],
        "summary_sha256": _sha256_file(summary_path),
        "profiles_manifest_sha256": _sha256_file(profiles_manifest_path),
        "artifact_manifest_sha256": _sha256_file(artifact_manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 CLDefocus 多 response-profile PDraw bank")
    parser.add_argument("--config", type=Path, required=True, help="版本化 YAML 配置")
    args = parser.parse_args()
    result = generate(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
