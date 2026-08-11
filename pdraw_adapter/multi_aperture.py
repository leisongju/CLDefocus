"""同一 CLDefocus 复合镜头的多光圈左右半瞳 PSF bank 导出。

训练标签只由左右 PSF 的一阶矩 ``mu_left_x - mu_right_x`` 给出。NCC 只在
独立的合成纹理上验证这一解析标签，不参与标签回写、标定或补偿。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import yaml


@dataclass(frozen=True)
class AperturePlan:
    """一档由同一镜头原生 aperture stop 收缩得到的光圈。"""

    index: int
    requested_f_number: float
    stop_scale: float
    stop_radius_m: float


def _finite_positive(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} 必须是有限正数，实际为 {value!r}")
    return number


def build_aperture_plan(
    native_f_number: float,
    native_stop_radius_m: float,
    requested_f_numbers: Sequence[float],
    *,
    opening_tolerance: float = 1.0e-6,
) -> list[AperturePlan]:
    """把目标 f-number 转成物理 stop 半径，不允许超出处方原生最大开口。"""

    native_f = _finite_positive(native_f_number, name="native_f_number")
    native_radius = _finite_positive(native_stop_radius_m, name="native_stop_radius_m")
    if not requested_f_numbers:
        raise ValueError("apertures.f_numbers 不能为空")

    plan: list[AperturePlan] = []
    previous = -math.inf
    for index, requested in enumerate(requested_f_numbers):
        target = _finite_positive(requested, name=f"apertures.f_numbers[{index}]")
        if target <= previous:
            raise ValueError("apertures.f_numbers 必须严格递增")
        scale = native_f / target
        if scale > 1.0 + float(opening_tolerance):
            raise ValueError(
                f"目标 f/{target:g} 需要把 stop 放大到原生的 {scale:.6f} 倍，"
                f"超出原处方 f/{native_f:.6f} 的有效范围"
            )
        scale = min(scale, 1.0)
        plan.append(
            AperturePlan(
                index=index,
                requested_f_number=target,
                stop_scale=scale,
                stop_radius_m=native_radius * scale,
            )
        )
        previous = target
    return plan


def logical_subaperture_masks(
    pupil_x: Any,
    valid_mask: Any,
    *,
    split_x: float = 0.0,
) -> tuple[Any, Any]:
    """返回逻辑左右图的出口瞳掩膜，并包含微透镜的水平成像反向。"""

    import jax.numpy as jnp

    x = jnp.asarray(pupil_x)
    valid = jnp.asarray(valid_mask, dtype=bool) & jnp.isfinite(x)
    physical_left = valid & (x < float(split_x))
    physical_right = valid & ~physical_left
    # 逻辑左图使用物理右半瞳，确保近处满足 d=x_L-x_R>0。
    return physical_right, physical_left


def logical_subaperture_power_weights(
    pupil_x: Any,
    valid_mask: Any,
    *,
    split_x: float = 0.0,
    transition_width: float | None = None,
    cross_talk: float = 0.0,
) -> tuple[Any, Any]:
    """返回逻辑左右 DP 通道的连续出口瞳功率权重。

    ``transition_width`` 以归一化出口瞳半径为单位。``None`` 保留硬半瞳；正值
    用互补 sigmoid 表示微透镜/子像素的重叠角响应。``cross_talk`` 在功率域进行
    对称混合。两者都会真实改变渲染 PSF 的 centroid，而不会修改 label。
    """

    import jax.numpy as jnp

    x = jnp.asarray(pupil_x)
    valid = jnp.asarray(valid_mask, dtype=bool) & jnp.isfinite(x)
    amount = float(cross_talk)
    if not 0.0 <= amount < 0.5:
        raise ValueError("cross_talk 必须位于 [0, 0.5)")
    if transition_width is None:
        left_mask, right_mask = logical_subaperture_masks(x, valid, split_x=split_x)
        left = left_mask.astype(x.dtype)
        right = right_mask.astype(x.dtype)
    else:
        width = float(transition_width)
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError("transition_width 必须为有限正数或 null")
        centered = x - float(split_x)
        radius = jnp.maximum(jnp.max(jnp.where(valid, jnp.abs(centered), 0.0)), 1.0e-20)
        pupil_u = centered / radius
        # 逻辑左通道对应物理正 x 半瞳，保持近处 d=x_L-x_R>0。
        left = 0.5 * (jnp.tanh(0.5 * pupil_u / width) + 1.0)
        right = 1.0 - left
        left = jnp.where(valid, left, 0.0)
        right = jnp.where(valid, right, 0.0)
    if amount > 0.0:
        left, right = (
            (1.0 - amount) * left + amount * right,
            (1.0 - amount) * right + amount * left,
        )
    return jnp.maximum(left, 0.0), jnp.maximum(right, 0.0)


def _normalize_psf_jax(psf: Any, eps: float = 1.0e-20) -> Any:
    import jax.numpy as jnp

    value = jnp.nan_to_num(jnp.asarray(psf), nan=0.0, posinf=0.0, neginf=0.0)
    value = jnp.maximum(value, 0.0)
    energy = value.sum()
    height, width = value.shape
    identity = jnp.zeros_like(value).at[height // 2, width // 2].set(1.0)
    return jnp.where(energy > eps, value / jnp.maximum(energy, eps), identity)


def centroid_xy(psf: np.ndarray, eps: float = 1.0e-20) -> tuple[float, float]:
    """计算图像布局 PSF 的质心，原点位于核中心。"""

    value = np.asarray(psf, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"PSF 必须是二维数组，实际为 {value.shape}")
    value = np.maximum(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    energy = float(value.sum())
    if energy <= float(eps):
        raise ValueError("PSF 能量必须为正")
    value = value / energy
    height, width = value.shape
    xs = np.arange(width, dtype=np.float64) - (width - 1) / 2.0
    ys = np.arange(height, dtype=np.float64) - (height - 1) / 2.0
    return float((value * xs[None, :]).sum()), float((value * ys[:, None]).sum())


def analytic_centroid_disparities(bank: np.ndarray) -> np.ndarray:
    """从 ``[A,Ncoc,Gh,Gw,2,K,K]`` bank 计算唯一解析标签。"""

    value = np.asarray(bank)
    if value.ndim != 7 or value.shape[4] != 2:
        raise ValueError(
            "bank 必须是 [A,Ncoc,Gh,Gw,2,K,K]，"
            f"实际为 {value.shape}"
        )
    labels = np.empty(value.shape[:4], dtype=np.float64)
    for index in np.ndindex(value.shape[:4]):
        mu_left_x, _ = centroid_xy(value[index + (0,)])
        mu_right_x, _ = centroid_xy(value[index + (1,)])
        labels[index] = mu_left_x - mu_right_x
    return labels


def aperture_curve_diagnostics(
    signed_coc_bins_px: np.ndarray,
    analytic_disparity_bins_px: np.ndarray,
    *,
    exact_zero_coc_max_px: float = 1.0e-5,
    exact_zero_disparity_max_px: float = 0.02,
    monotonic_drop_max_px: float = 0.02,
    centroid_to_coc_ratio_min: float = 0.10,
) -> dict[str, Any]:
    """逐光圈核验 CoC→centroid 曲线，避免总体 NCC 指标掩盖坏光圈。

    ``centroid_to_coc_ratio`` 读取每档曲线 ``|CoC|`` 最大端点。它不参与
    label 标定，只用于发现传感器欠采样或半瞳传播的数值塌缩。
    """

    coc = np.asarray(signed_coc_bins_px, dtype=np.float64)
    disparity = np.asarray(analytic_disparity_bins_px, dtype=np.float64)
    if coc.ndim != 2 or disparity.shape != coc.shape:
        raise ValueError(
            "signed_coc_bins_px 与 analytic_disparity_bins_px 必须是同形 [A,N]，"
            f"实际为 {coc.shape} 和 {disparity.shape}"
        )
    if not np.isfinite(coc).all() or not np.isfinite(disparity).all():
        raise ValueError("CoC/centroid 曲线含 NaN 或 Inf")

    rows: list[dict[str, Any]] = []
    for aperture_index in range(coc.shape[0]):
        c = coc[aperture_index]
        d = disparity[aperture_index]
        if np.any(np.diff(c) <= 0.0):
            raise ValueError(f"第 {aperture_index} 档 signed CoC 必须严格递增")
        zero_index = int(np.argmin(np.abs(c)))
        endpoint_index = int(np.argmax(np.abs(c)))
        endpoint_coc = float(c[endpoint_index])
        endpoint_disp = float(d[endpoint_index])
        response_ratio = abs(endpoint_disp / endpoint_coc) if abs(endpoint_coc) > 1.0e-12 else 0.0
        drops = np.diff(d)
        monotonic_violation_count = int(np.count_nonzero(drops < -float(monotonic_drop_max_px)))
        exact_zero_pass = (
            abs(float(c[zero_index])) <= float(exact_zero_coc_max_px)
            and abs(float(d[zero_index])) <= float(exact_zero_disparity_max_px)
        )
        monotonic_pass = monotonic_violation_count == 0
        response_pass = response_ratio >= float(centroid_to_coc_ratio_min)
        rows.append(
            {
                "aperture_index": aperture_index,
                "zero_index": zero_index,
                "zero_coc_abs_px": abs(float(c[zero_index])),
                "zero_disparity_abs_px": abs(float(d[zero_index])),
                "label_span_px": float(np.ptp(d)),
                "endpoint_coc_px": endpoint_coc,
                "endpoint_disparity_px": endpoint_disp,
                "centroid_to_coc_ratio": float(response_ratio),
                "monotonic_min_delta_px": float(np.min(drops)),
                "monotonic_violation_count": monotonic_violation_count,
                "exact_zero_pass": bool(exact_zero_pass),
                "monotonic_pass": bool(monotonic_pass),
                "response_pass": bool(response_pass),
                "pass": bool(exact_zero_pass and monotonic_pass and response_pass),
            }
        )
    return {
        "thresholds": {
            "exact_zero_coc_max_px": float(exact_zero_coc_max_px),
            "exact_zero_disparity_max_px": float(exact_zero_disparity_max_px),
            "monotonic_drop_max_px": float(monotonic_drop_max_px),
            "centroid_to_coc_ratio_min": float(centroid_to_coc_ratio_min),
        },
        "apertures": rows,
        "pass": bool(all(row["pass"] for row in rows)),
    }


def stack_aperture_bank(per_aperture_pairs: Sequence[np.ndarray]) -> np.ndarray:
    """把每档 ``[Ncoc,Gh,Gw,2,K,K]`` 资产堆叠为标准 7 维 bank。"""

    if not per_aperture_pairs:
        raise ValueError("per_aperture_pairs 不能为空")
    arrays = [np.asarray(value, dtype=np.float32) for value in per_aperture_pairs]
    reference_shape = arrays[0].shape
    if len(reference_shape) != 6 or reference_shape[3] != 2:
        raise ValueError(
            "每档资产必须是 [Ncoc,Gh,Gw,2,K,K]，"
            f"实际为 {reference_shape}"
        )
    for index, value in enumerate(arrays):
        if value.shape != reference_shape:
            raise ValueError(
                f"第 {index} 档 shape={value.shape}，与首档 {reference_shape} 不一致"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"第 {index} 档含 NaN 或 Inf")
        energy = value.sum(axis=(-2, -1))
        if not np.allclose(energy, 1.0, atol=2.0e-5, rtol=2.0e-5):
            raise ValueError(f"第 {index} 档 PSF 未逐核归一")
    return np.stack(arrays, axis=0)


def _propagate_half_pupil_pair(
    wf: Any,
    *,
    s_prop: float,
    parax: Any,
    sensor: Any,
    kernel_size: int,
    pixel_pitch_m: float,
    split_x_m: float,
    split_transition_width: float | None,
    pdraw_cross_talk: float,
    upsample: int,
) -> dict[str, Any]:
    """将完整复合镜头出口波前按左右半瞳分别作 RS 传播。"""

    import jax.numpy as jnp
    import datasyn.jaxutils.nputils as nputils
    import datasyn.mathutils.vecop as vecop
    import datasyn.optics.safeop as safeop
    from datasyn.mathutils.complex import sqabs_complex
    from datasyn.optics.imaging.defocus import downsample_integer
    from datasyn.optics.imaging.imaging import (
        make_tilted_debye_coords,
        rough_scoc_debye_from_coords,
        wh_physics_to_image,
    )
    from datasyn.optics.imaging.pupil_function.pf import opd_to_phase_factor
    from datasyn.optics.imaging.rs import rayleigh_sommerfeld
    from datasyn.optics.ray import project_to_z

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size 必须是正奇数")
    if upsample <= 0:
        raise ValueError("upsample 必须为正数")

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

    left_power, right_power = logical_subaperture_power_weights(
        wf.wf.pts[..., 0],
        wf.wf.mask,
        split_x=split_x_m,
        transition_width=split_transition_width,
        cross_talk=pdraw_cross_talk,
    )
    phase = opd_to_phase_factor(wf.wf.wvl, wf.wf.opd)
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
    coords = make_tilted_debye_coords(parax.xp, wf.xp_sphere, z_obs=z_prop)
    upstream_signed_radius_m = rough_scoc_debye_from_coords(coords)
    project_signed_radius_m = -upstream_signed_radius_m
    return {
        "left": left,
        "right": right,
        "signed_coc_px": 2.0 * project_signed_radius_m / float(pixel_pitch_m),
        "signed_coc_radius_m": project_signed_radius_m,
        "left_ray_count": jnp.count_nonzero(left_power > 1.0e-10),
        "right_ray_count": jnp.count_nonzero(right_power > 1.0e-10),
        "left_effective_pupil_power": left_power.sum(),
        "right_effective_pupil_power": right_power.sum(),
        "pupil_overlap_fraction": (
            jnp.count_nonzero((left_power > 1.0e-10) & (right_power > 1.0e-10))
            / jnp.maximum(jnp.count_nonzero(jnp.asarray(wf.wf.mask, dtype=bool)), 1)
        ),
    }


def _signed_coc_px_at_propagation(
    wf: Any,
    *,
    s_prop: float,
    parax: Any,
    pixel_pitch_m: float,
) -> float:
    """只计算几何 signed CoC，不执行昂贵的半瞳波动传播。"""

    from datasyn.optics.imaging.imaging import (
        make_tilted_debye_coords,
        rough_scoc_debye_from_coords,
    )

    z_prop = parax.xp.z + float(s_prop)
    coords = make_tilted_debye_coords(parax.xp, wf.xp_sphere, z_obs=z_prop)
    project_signed_radius_m = -rough_scoc_debye_from_coords(coords)
    return float(2.0 * project_signed_radius_m / float(pixel_pitch_m))


def _exact_zero_coc_offset(
    wf: Any,
    *,
    base_s_prop: float,
    candidate_offsets: Sequence[float],
    parax: Any,
    pixel_pitch_m: float,
    tolerance_px: float = 1.0e-7,
) -> float:
    """在已声明 offset envelope 内二分求解 exact CoC=0 的传感器位置。"""

    rows = [
        (
            float(offset),
            _signed_coc_px_at_propagation(
                wf,
                s_prop=float(base_s_prop) + float(offset),
                parax=parax,
                pixel_pitch_m=pixel_pitch_m,
            ),
        )
        for offset in candidate_offsets
    ]
    nearest_offset, nearest_coc = min(rows, key=lambda row: abs(row[1]))
    if abs(nearest_coc) <= float(tolerance_px):
        return nearest_offset
    bracket: tuple[tuple[float, float], tuple[float, float]] | None = None
    for first, second in zip(rows[:-1], rows[1:], strict=True):
        if first[1] * second[1] < 0.0:
            bracket = (first, second)
            break
    if bracket is None:
        raise RuntimeError(
            "optics.defocus_offsets_m 没有包围 exact CoC=0："
            f"samples={rows}"
        )
    (low_offset, low_coc), (high_offset, high_coc) = bracket
    for _ in range(64):
        mid_offset = 0.5 * (low_offset + high_offset)
        mid_coc = _signed_coc_px_at_propagation(
            wf,
            s_prop=float(base_s_prop) + mid_offset,
            parax=parax,
            pixel_pitch_m=pixel_pitch_m,
        )
        if abs(mid_coc) <= float(tolerance_px):
            return mid_offset
        if low_coc * mid_coc <= 0.0:
            high_offset, high_coc = mid_offset, mid_coc
        else:
            low_offset, low_coc = mid_offset, mid_coc
    return 0.5 * (low_offset + high_offset)


def _deterministic_texture(size: int, seed: int) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    if size < 128:
        raise ValueError("NCC texture_size 至少为 128")
    rng = np.random.default_rng(int(seed))
    noise_fine = gaussian_filter(rng.standard_normal((size, size)), sigma=0.55)
    noise_coarse = gaussian_filter(rng.standard_normal((size, size)), sigma=2.0)
    yy, xx = np.mgrid[:size, :size]
    structured = 0.30 * np.sin(xx * 0.19 + yy * 0.07) + 0.22 * np.cos(xx * 0.05 - yy * 0.17)
    texture = 0.58 * noise_fine + 0.22 * noise_coarse + structured
    texture -= float(texture.min())
    texture /= max(float(texture.max()), 1.0e-12)
    return np.ascontiguousarray(texture, dtype=np.float32)


def _linear_fit(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(prediction, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError("线性拟合至少需要两个同长一维数组")
    design = np.column_stack((x, np.ones_like(x)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = slope * x + intercept
    residual = y - fitted
    sse = float(np.sum(residual**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "slope_ncc_vs_analytic": float(slope),
        "intercept_px": float(intercept),
        "r2": float(1.0 - sse / sst) if sst > 1.0e-12 else float("nan"),
        "fit_rmse_px": float(np.sqrt(np.mean(residual**2))),
    }


def run_ncc_centroid_diagnostic(
    bank: np.ndarray,
    *,
    texture_size: int = 320,
    seed: int = 20260811,
    tile_size: int = 64,
    tiles_per_axis: int = 3,
    search_x: int = 4,
    search_y: int = 2,
) -> dict[str, Any]:
    """在合成纹理上比较 NCC 与质心解析标签；不会修改或返回替代标签。"""

    from scipy.signal import fftconvolve
    from pdraw_benchmark.ncc_reference import NccProtocol, estimate_bidirectional_ncc

    value = np.asarray(bank, dtype=np.float32)
    labels = analytic_centroid_disparities(value)
    if value.shape[2:4] != (1, 1):
        raise ValueError("当前 NCC smoke 仅支持 Gh=Gw=1")
    if int(search_x) <= math.ceil(float(np.max(np.abs(labels)))):
        raise ValueError("NCC search_x 必须严格覆盖全部解析视差")

    texture = _deterministic_texture(int(texture_size), int(seed))
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
    half_kernel = value.shape[-1] // 2
    safe_margin = half_kernel + max(int(search_x), int(search_y)) + 8
    low = safe_margin
    high = int(texture_size) - safe_margin - int(tile_size)
    if high < low:
        raise ValueError("texture_size 无法容纳 kernel、tile 与 NCC 搜索边界")
    positions = np.linspace(low, high, num=int(tiles_per_axis), dtype=int)
    tiles = [
        {"x": int(x), "y": int(y), "width": int(tile_size), "height": int(tile_size)}
        for y in positions
        for x in positions
    ]

    records: list[dict[str, Any]] = []
    fit_labels: list[float] = []
    fit_predictions: list[float] = []
    for aperture_index in range(value.shape[0]):
        for coc_index in range(value.shape[1]):
            pair = value[aperture_index, coc_index, 0, 0]
            # fftconvolve 是真正的卷积；因此输出特征位移与 PSF 质心同号。
            left = fftconvolve(texture, pair[0], mode="same").astype(np.float32)
            right = fftconvolve(texture, pair[1], mode="same").astype(np.float32)
            estimates = [estimate_bidirectional_ncc(left, right, tile, protocol) for tile in tiles]
            valid = [row for row in estimates if bool(row["ncc_quality_pass"])]
            predictions = [float(row["pseudo_disp_model_px"]) for row in valid]
            analytic = float(labels[aperture_index, coc_index, 0, 0])
            prediction = float(np.median(predictions)) if predictions else None
            record = {
                "aperture_index": aperture_index,
                "coc_index": coc_index,
                "analytic_centroid_disparity_px": analytic,
                "ncc_disparity_px": prediction,
                "ncc_valid_tiles": len(valid),
                "ncc_total_tiles": len(estimates),
                "quality_reasons": sorted(
                    {reason for row in estimates for reason in row["quality_reasons"]}
                ),
            }
            if prediction is not None:
                record["ncc_minus_analytic_px"] = prediction - analytic
                fit_labels.append(analytic)
                fit_predictions.append(prediction)
            records.append(record)

    if len(fit_labels) < 2 or float(np.var(fit_labels)) <= 1.0e-12:
        raise RuntimeError("NCC 通过记录不足，无法拟合 small-disparity slope")
    reference = np.asarray(fit_labels, dtype=np.float64)
    prediction = np.asarray(fit_predictions, dtype=np.float64)
    error = prediction - reference
    fit = _linear_fit(reference, prediction)
    nonzero = np.abs(reference) > 0.10
    sign_agreement = (
        float(np.mean(np.sign(reference[nonzero]) == np.sign(prediction[nonzero])))
        if np.any(nonzero)
        else None
    )
    return {
        "role": "independent_diagnostic_only",
        "label_source": "PSF centroid mu_left_x-mu_right_x",
        "label_compensation_from_ncc": False,
        "texture": {"kind": "deterministic_synthetic", "size": int(texture_size), "seed": int(seed)},
        "protocol": asdict(protocol),
        "record_count": len(records),
        "fit_record_count": len(fit_labels),
        "epe_px": float(np.mean(np.abs(error))),
        "rmse_px": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_px": float(np.max(np.abs(error))),
        "sign_agreement_nonzero": sign_agreement,
        **fit,
        "records": records,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_output(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_preview(path: Path, bank: np.ndarray) -> None:
    from PIL import Image, ImageDraw

    value = np.asarray(bank)
    kernel_size = value.shape[-1]
    pad = 2
    canvas = Image.new(
        "L",
        (value.shape[1] * 2 * (kernel_size + pad), value.shape[0] * (kernel_size + pad)),
        color=0,
    )
    for aperture_index in range(value.shape[0]):
        for coc_index in range(value.shape[1]):
            for side in range(2):
                psf = value[aperture_index, coc_index, 0, 0, side]
                visible = np.power(np.clip(psf / max(float(psf.max()), 1.0e-20), 0.0, 1.0), 0.25)
                cell = Image.fromarray((visible * 255.0 + 0.5).astype(np.uint8), mode="L")
                x = (2 * coc_index + side) * (kernel_size + pad)
                y = aperture_index * (kernel_size + pad)
                canvas.paste(cell, (x, y))
    # 保留一个显式 draw 对象，防止 Pillow 在极小 montage 上延迟初始化。
    ImageDraw.Draw(canvas)
    canvas.save(path)


def _resolve_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    project_root = Path(__file__).resolve().parents[3]
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or int(loaded.get("schema_version", 0)) != 1:
        raise ValueError("配置必须是 schema_version: 1 的 YAML 映射")
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


def _lens_with_stop_radius(lens: Any, stop_radius_m: float) -> Any:
    from datasyn.optics.complens.tabular_lens import SurfCol

    surfaces = lens.surfaces.at[lens.stop_idx, SurfCol.APER].set(float(stop_radius_m))
    return lens._replace(surfaces=surfaces)


def _ordered_offsets(values: Iterable[float]) -> list[float]:
    offsets = [float(value) for value in values]
    if not offsets:
        raise ValueError("optics.defocus_offsets_m 不能为空")
    if offsets != sorted(offsets) or len(set(offsets)) != len(offsets):
        raise ValueError("optics.defocus_offsets_m 必须严格递增")
    return offsets


def generate(config_path: Path) -> dict[str, Any]:
    """执行多光圈物理导出并写出来源、解析标签和独立 NCC 诊断。"""

    resolved, project_root = _resolve_config(config_path)
    source_root = Path(resolved["source"]["checkout"])
    datasyn_src = source_root / "datasyn" / "src"
    for import_root in (project_root, datasyn_src):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

    from datasyn.jaxutils.configs import easy_optics_setup

    easy_optics_setup()

    import jax.numpy as jnp
    import torch
    from datasyn.optics.complens.tabular_lens import load_tabular_lens_from_mytable
    from datasyn.optics.imaging.imaging import HFProper, make_imaging

    output_root = Path(resolved["output"]["root"])
    if output_root == project_root or project_root in output_root.parents:
        raise ValueError(f"输出目录必须位于仓库外部，实际为 {output_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_commit = _git_output(source_root, "rev-parse", "HEAD")
    expected_commit = str(resolved["source"]["commit"])
    if source_commit != expected_commit:
        raise RuntimeError(f"CLDefocus commit 不匹配：expected={expected_commit}, actual={source_commit}")

    lens_path = Path(resolved["lens"]["root"]) / str(resolved["lens"]["relative_path"])
    if not lens_path.is_file():
        raise FileNotFoundError(f"镜头处方不存在：{lens_path}")
    lens_sha256 = _sha256_file(lens_path)
    if lens_sha256 != str(resolved["lens"]["sha256"]):
        raise RuntimeError("镜头处方 SHA256 不匹配")

    sensor_cfg = resolved["sensor"]
    optics = resolved["optics"]
    kernel_size = int(optics["kernel_size"])
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("optics.kernel_size 必须是正奇数")
    fields = list(optics["field_heights"])
    if len(fields) != 1:
        raise ValueError("冻结 V3 当前只允许一个 field_height")
    offsets = _ordered_offsets(optics["defocus_offsets_m"])
    wavelength_m = float(optics["wavelength_m"])

    source_lens = load_tabular_lens_from_mytable(lens_path)
    native_imaging = make_imaging(
        lens=source_lens,
        sen_wh=(int(sensor_cfg["width_px"]), int(sensor_cfg["height_px"])),
        proper=HFProper(),
        wvl_ref=wavelength_m,
        pixsize=float(sensor_cfg["pixel_pitch_m"]),
    )
    native_f_number = float(native_imaging.parax.fnum)
    native_stop_radius_m = float(source_lens.get_aperture(source_lens.stop_idx))
    aperture_plan = build_aperture_plan(
        native_f_number,
        native_stop_radius_m,
        resolved["apertures"]["f_numbers"],
    )

    per_aperture_assets: list[np.ndarray] = []
    aperture_rows: list[dict[str, Any]] = []
    all_coc_bins: list[list[float]] = []
    for aperture in aperture_plan:
        lens = _lens_with_stop_radius(source_lens, aperture.stop_radius_m)
        imaging = make_imaging(
            lens=lens,
            sen_wh=(int(sensor_cfg["width_px"]), int(sensor_cfg["height_px"])),
            proper=HFProper(),
            wvl_ref=wavelength_m,
            pixsize=float(sensor_cfg["pixel_pitch_m"]),
        )
        actual_f_number = float(imaging.parax.fnum)
        if abs(actual_f_number - aperture.requested_f_number) > 1.0e-6:
            raise RuntimeError(
                f"f-number 构造漂移：requested={aperture.requested_f_number}, actual={actual_f_number}"
            )
        field_xy = jnp.asarray([0.0, -float(fields[0])])
        wavefront = imaging.generate_wavefront(
            float(optics["object_depth_m"]), field_xy, wvl=wavelength_m
        )
        focus_reference = str(optics.get("focus_reference", "lens_imgpos"))
        if focus_reference == "lens_imgpos":
            base_s_prop = float(lens.imgpos - imaging.xp.z)
        elif focus_reference == "wavefront_best_focus":
            base_s_prop = float(wavefront.xp_sphere.c[2] - imaging.xp.z)
        else:
            raise ValueError(f"未知 focus_reference：{focus_reference}")

        aperture_offsets = list(offsets)
        if bool(optics.get("include_exact_coc_zero", False)):
            zero_offset = _exact_zero_coc_offset(
                wavefront,
                base_s_prop=base_s_prop,
                candidate_offsets=aperture_offsets,
                parax=imaging.parax,
                pixel_pitch_m=float(sensor_cfg["pixel_pitch_m"]),
            )
            if all(abs(zero_offset - value) > 1.0e-12 for value in aperture_offsets):
                aperture_offsets.append(zero_offset)
                aperture_offsets.sort()

        records: list[dict[str, Any]] = []
        for offset in aperture_offsets:
            propagated = _propagate_half_pupil_pair(
                wavefront,
                s_prop=base_s_prop + offset,
                parax=imaging.parax,
                sensor=imaging.sen,
                kernel_size=kernel_size,
                pixel_pitch_m=float(sensor_cfg["pixel_pitch_m"]),
                split_x_m=float(optics.get("pupil_split_x_m", 0.0)),
                split_transition_width=optics.get("pupil_split_transition_width"),
                pdraw_cross_talk=float(optics.get("pdraw_cross_talk", 0.0)),
                upsample=int(optics.get("upsample", 1)),
            )
            left = np.asarray(propagated["left"], dtype=np.float32)
            right = np.asarray(propagated["right"], dtype=np.float32)
            mu_left_x, mu_left_y = centroid_xy(left)
            mu_right_x, mu_right_y = centroid_xy(right)
            disparity = mu_left_x - mu_right_x
            signed_coc_px = float(propagated["signed_coc_px"])
            sign_evaluated = abs(signed_coc_px) > 0.25 and abs(disparity) > 0.05
            if sign_evaluated and disparity * signed_coc_px <= 0.0:
                raise RuntimeError(
                    "PDraw 符号失败："
                    f"f/{actual_f_number:.3f}, coc={signed_coc_px:.6f}, centroid={disparity:.6f}"
                )
            records.append(
                {
                    "offset_m": offset,
                    "signed_coc_px": signed_coc_px,
                    "left": left,
                    "right": right,
                    "analytic_centroid_disparity_px": disparity,
                    "mu_left_x_px": mu_left_x,
                    "mu_left_y_px": mu_left_y,
                    "mu_right_x_px": mu_right_x,
                    "mu_right_y_px": mu_right_y,
                    "left_ray_count": int(propagated["left_ray_count"]),
                    "right_ray_count": int(propagated["right_ray_count"]),
                    "left_effective_pupil_power": float(
                        propagated["left_effective_pupil_power"]
                    ),
                    "right_effective_pupil_power": float(
                        propagated["right_effective_pupil_power"]
                    ),
                    "pupil_overlap_fraction": float(propagated["pupil_overlap_fraction"]),
                    "sign_evaluated": sign_evaluated,
                    "sign_pass": (disparity * signed_coc_px > 0.0) if sign_evaluated else None,
                    "exact_zero_coc_trace": abs(signed_coc_px) <= 1.0e-6,
                }
            )
        records.sort(key=lambda row: float(row["signed_coc_px"]))
        coc_bins = [float(row["signed_coc_px"]) for row in records]
        if np.any(np.diff(coc_bins) <= 0.0):
            raise RuntimeError(f"f/{actual_f_number:g} 的 signed CoC bins 非严格递增")
        pairs = np.stack(
            [np.stack((row.pop("left"), row.pop("right")), axis=0) for row in records],
            axis=0,
        )[:, None, None]
        per_aperture_assets.append(pairs)
        all_coc_bins.append(coc_bins)
        aperture_rows.append(
            {
                **asdict(aperture),
                "actual_f_number": actual_f_number,
                "actual_stop_radius_m": float(lens.get_aperture(lens.stop_idx)),
                "entrance_pupil_radius_m": float(imaging.parax.ep.r),
                "exit_pupil_radius_m": float(imaging.parax.xp.r),
                "effective_focal_length_m": float(imaging.parax.efl),
                "sensor_plane_z_m": float(lens.imgpos),
                "base_propagation_m": base_s_prop,
                "bins": records,
            }
        )

    bank = stack_aperture_bank(per_aperture_assets)
    analytic_labels = analytic_centroid_disparities(bank)
    recorded_labels = np.asarray(
        [[row["analytic_centroid_disparity_px"] for row in aperture["bins"]] for aperture in aperture_rows],
        dtype=np.float64,
    )
    if not np.array_equal(analytic_labels[:, :, 0, 0], recorded_labels):
        raise RuntimeError("解析质心标签在 bank 堆叠后发生漂移")

    diagnostic_cfg = dict(resolved.get("diagnostic", {}))
    ncc = run_ncc_centroid_diagnostic(
        bank,
        texture_size=int(diagnostic_cfg.get("texture_size", 320)),
        seed=int(diagnostic_cfg.get("seed", 20260811)),
        tile_size=int(diagnostic_cfg.get("tile_size", 64)),
        tiles_per_axis=int(diagnostic_cfg.get("tiles_per_axis", 3)),
        search_x=int(diagnostic_cfg.get("search_x", 4)),
        search_y=int(diagnostic_cfg.get("search_y", 2)),
    )
    gates = dict(diagnostic_cfg.get("gates", {}))
    gate_results = {
        "slope_abs_error": abs(float(ncc["slope_ncc_vs_analytic"]) - 1.0)
        <= float(gates.get("slope_abs_error_max", 0.25)),
        "r2": float(ncc["r2"]) >= float(gates.get("r2_min", 0.90)),
        "epe": float(ncc["epe_px"]) <= float(gates.get("epe_px_max", 0.20)),
        "sign": ncc["sign_agreement_nonzero"] is not None
        and float(ncc["sign_agreement_nonzero"]) >= float(gates.get("sign_agreement_min", 1.0)),
    }
    ncc["gates"] = {
        "thresholds": {
            "slope_abs_error_max": float(gates.get("slope_abs_error_max", 0.25)),
            "r2_min": float(gates.get("r2_min", 0.90)),
            "epe_px_max": float(gates.get("epe_px_max", 0.20)),
            "sign_agreement_min": float(gates.get("sign_agreement_min", 1.0)),
        },
        "results": gate_results,
        "pass": bool(all(gate_results.values())),
    }
    curve_gates = aperture_curve_diagnostics(
        np.asarray(all_coc_bins, dtype=np.float64),
        analytic_labels[:, :, 0, 0],
        exact_zero_coc_max_px=float(gates.get("exact_zero_coc_max_px", 1.0e-5)),
        exact_zero_disparity_max_px=float(gates.get("exact_zero_disparity_max_px", 0.02)),
        monotonic_drop_max_px=float(gates.get("monotonic_drop_max_px", 0.02)),
        centroid_to_coc_ratio_min=float(gates.get("centroid_to_coc_ratio_min", 0.10)),
    )

    asset_id = str(resolved["run"]["id"])
    bank_path = output_root / "psf_bank.pt"
    torch.save(
        {
            "asset_id": asset_id,
            "asset_spec": {
                "kind": "cldefocus_compound_lens_multi_aperture_half_pupil_rs",
                "source_repository": str(resolved["source"]["repository"]),
                "source_commit": source_commit,
                "lens_relative_path": str(resolved["lens"]["relative_path"]),
                "lens_sha256": lens_sha256,
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
                "pupil_split_transition_width": optics.get("pupil_split_transition_width"),
                "pdraw_cross_talk": float(optics.get("pdraw_cross_talk", 0.0)),
            },
            "f_numbers": torch.tensor([row["actual_f_number"] for row in aperture_rows]),
            "stop_scales": torch.tensor([row["stop_scale"] for row in aperture_rows]),
            "stop_radii_m": torch.tensor([row["actual_stop_radius_m"] for row in aperture_rows]),
            "signed_coc_bins_px": torch.tensor(all_coc_bins),
            "analytic_disparity_bins_px": torch.from_numpy(analytic_labels[:, :, 0, 0].copy()),
            "field_grid_hw": (1, 1),
            "kernel_size": kernel_size,
            "psf_bank": torch.from_numpy(bank),
        },
        bank_path,
    )

    resolved_path = output_root / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8")
    aperture_path = output_root / "aperture_manifest.json"
    aperture_path.write_text(
        json.dumps(
            {
                "axis_order": ["aperture", "signed_coc", "field_y", "field_x", "side", "y", "x"],
                "shape": list(bank.shape),
                "native_f_number": native_f_number,
                "native_stop_radius_m": native_stop_radius_m,
                "apertures": aperture_rows,
                "analytic_label_source": "PSF centroid mu_left_x-mu_right_x",
                "ncc_updates_labels": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    ncc_path = output_root / "ncc_diagnostic.json"
    ncc_path.write_text(json.dumps(ncc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preview_path = output_root / "psf_preview.png"
    _write_preview(preview_path, bank)
    admission_pass = bool(ncc["gates"]["pass"] and curve_gates["pass"])
    metrics = {
        "asset_id": asset_id,
        "status": "pass" if admission_pass else "diagnostic_fail",
        "shape": list(bank.shape),
        "finite": bool(np.isfinite(bank).all()),
        "energy_min": float(bank.sum(axis=(-2, -1)).min()),
        "energy_max": float(bank.sum(axis=(-2, -1)).max()),
        "f_numbers": [row["actual_f_number"] for row in aperture_rows],
        "signed_coc_bins_px": all_coc_bins,
        "analytic_disparity_bins_px": analytic_labels[:, :, 0, 0].tolist(),
        "signed_disparity_convention_pass": True,
        "ncc_diagnostic_gate_pass": bool(ncc["gates"]["pass"]),
        "aperture_curve_gate_pass": bool(curve_gates["pass"]),
        "aperture_curve_diagnostics": curve_gates,
        "ncc_slope_vs_analytic": float(ncc["slope_ncc_vs_analytic"]),
        "ncc_r2": float(ncc["r2"]),
        "ncc_epe_px": float(ncc["epe_px"]),
        "ncc_label_compensation": False,
    }
    metrics_path = output_root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    source_paths = [Path(__file__).resolve(), Path(__file__).with_name("__init__.py").resolve()]
    provenance = {
        "source_repository": str(resolved["source"]["repository"]),
        "source_commit": source_commit,
        "project_commit": _git_output(project_root, "rev-parse", "HEAD"),
        "project_dirty_paths": _git_output(project_root, "status", "--porcelain").splitlines(),
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256_file(config_path),
        "resolved_config_sha256": _sha256_file(resolved_path),
        "resolved_config_content_sha256": _sha256_json(resolved),
        "implementation_source_sha256": {
            str(path.relative_to(project_root)): _sha256_file(path) for path in source_paths
        },
        "lens_path": str(lens_path),
        "lens_sha256": lens_sha256,
        "runtime": "Genfocus",
        "google_dev_accessed": False,
        "google_holdout_accessed": False,
        "real_pdraw_accessed": False,
    }
    provenance_path = output_root / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    artifact_paths = (
        bank_path,
        resolved_path,
        aperture_path,
        ncc_path,
        preview_path,
        metrics_path,
        provenance_path,
    )
    artifact_manifest = {
        path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
        for path in artifact_paths
    }
    manifest_path = output_root / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(artifact_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "output_root": str(output_root),
        "metrics": metrics,
        "artifact_manifest_sha256": _sha256_file(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 CLDefocus 多光圈左右半瞳 PDraw PSF bank")
    parser.add_argument("--config", type=Path, required=True, help="版本化 YAML 配置")
    args = parser.parse_args()
    print(json.dumps(generate(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
