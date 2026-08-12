"""多镜头处方 PDraw device-atom 的纯 CPU 规划器。

本模块只读取 ``.mytable`` 光学处方，执行近轴兼容性筛选，并冻结后续 GPU
传播所需的 device-atom manifest。它不会生成 PSF、不会执行 NCC，也不会把
PDOFFSET 写入任何计划资产。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

# 规划器是刻意的 CPU-only 工具。必须在任何 JAX import 之前冻结平台，避免与正式
# propagation 或同机训练争用 GPU。保留系统可见设备仅供 JAX 插件完成发现；
# ``JAX_PLATFORMS=cpu`` 才是实际的执行平台硬约束。
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import yaml


@dataclass(frozen=True)
class LensParaxialRecord:
    relative_path: str
    lens_id: str
    sha256: str
    native_f_number: float
    effective_focal_length_m: float
    image_radius_m: float
    back_focal_position_m: float
    entrance_pupil_radius_m: float
    exit_pupil_radius_m: float
    stop_radius_m: float
    total_track_m: float
    surface_count: int
    aspheric_surface_count: int
    stop_index: int
    aperture_stop_scales: tuple[float, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _git_output(checkout: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args], text=True
    ).strip()


def _cauchy_2term(nd: float, vd: float) -> tuple[float, float]:
    wave_d_um = 0.5875618
    wave_f_um = 0.4861327
    wave_c_um = 0.6562725
    dispersion = (float(nd) - 1.0) / float(vd)
    denominator = (1.0 / wave_f_um**2) - (1.0 / wave_c_um**2)
    coefficient_b = dispersion / denominator
    coefficient_a = float(nd) - coefficient_b / wave_d_um**2
    return coefficient_a, coefficient_b


def _ior_curvature_thickness_sequence(
    indices: np.ndarray,
    curvatures: np.ndarray,
    thicknesses: np.ndarray,
    z_enter: float,
) -> tuple[np.ndarray, float, float]:
    ns = np.asarray(indices, dtype=np.float64)
    cs = np.asarray(curvatures, dtype=np.float64)
    ts = np.asarray(thicknesses, dtype=np.float64)
    if cs.size < 1 or ns.size != cs.size + 1 or ts.size != cs.size - 1:
        raise ValueError("近轴 IOR/curvature/thickness 序列长度不一致")

    def refraction(n1: float, n2: float, curvature: float) -> np.ndarray:
        return np.asarray(
            [[1.0, 0.0], [curvature * (n1 - n2), 1.0]], dtype=np.float64
        )

    matrix = refraction(float(ns[0]), float(ns[1]), float(cs[0]))
    distance = 0.0
    for index in range(1, cs.size):
        n1 = float(ns[index])
        propagation = np.asarray(
            [[1.0, float(ts[index - 1]) / n1], [0.0, 1.0]],
            dtype=np.float64,
        )
        matrix = (
            refraction(n1, float(ns[index + 1]), float(cs[index]))
            @ propagation
            @ matrix
        )
        distance += float(ts[index - 1])
    return matrix, float(z_enter), float(z_enter + distance)


def _obj2img_s0(matrix: np.ndarray, *, n_image: float) -> tuple[float, float]:
    denominator = float(matrix[1, 1])
    if abs(denominator) <= 1.0e-15:
        raise ValueError("pupil 近轴矩阵奇异")
    return -float(n_image) * float(matrix[0, 1]) / denominator, 1.0 / denominator


def _parse_paraxial_record(
    path: Path,
    *,
    root: Path,
    wavelength_m: float,
    requested_f_numbers: Sequence[float],
) -> LensParaxialRecord:
    from datasyn.optics.parse_mytable import parse_tables

    tables = parse_tables(path.read_text(encoding="utf-8", errors="strict"))
    if "SURFACES" not in tables:
        raise ValueError("缺少 SURFACES table")
    source = tables["SURFACES"]
    if len(source) < 4:
        raise ValueError("处方 surface 数不足")
    image_row = source.iloc[-1]
    if "APERDIAM" in source:
        image_radius_m = 0.5e-3 * float(image_row["APERDIAM"])
    elif "APERSEMI" in source:
        image_radius_m = 1.0e-3 * float(image_row["APERSEMI"])
    else:
        raise ValueError("处方没有 image aperture 半径")

    surfaces = source.iloc[:-1].reset_index(drop=True)
    curvature: list[float] = []
    z_positions: list[float] = []
    apertures: list[float] = []
    cauchy_a: list[float] = []
    cauchy_b: list[float] = []
    stop_index: int | None = None
    z_cumulative = 0.0
    for index, row in surfaces.iterrows():
        radius_mm = row["R"]
        curvature.append(
            0.0
            if radius_mm is None or bool(np.asarray(np.isnan(radius_mm)).item())
            else 1.0 / (1.0e-3 * float(radius_mm))
        )
        z_positions.append(z_cumulative)
        thickness_mm = row["D"]
        if index != 0 and thickness_mm is not None and not bool(
            np.asarray(np.isnan(thickness_mm)).item()
        ):
            z_cumulative += 1.0e-3 * float(thickness_mm)
        aperture_mm = row["APERDIAM"] / 2.0 if "APERDIAM" in surfaces else row["APERSEMI"]
        apertures.append(
            math.inf
            if aperture_mm is None or bool(np.asarray(np.isnan(aperture_mm)).item())
            else 1.0e-3 * float(aperture_mm)
        )
        nd, vd = row.get("Nd", None), row.get("Vd", None)
        if nd is None or vd is None or bool(np.asarray(np.isnan(nd)).item()) or bool(
            np.asarray(np.isnan(vd)).item()
        ):
            coefficient_a, coefficient_b = 1.0, 0.0
        else:
            coefficient_a, coefficient_b = _cauchy_2term(float(nd), float(vd))
        cauchy_a.append(coefficient_a)
        cauchy_b.append(coefficient_b)
        if "STOP" in surfaces and row.get("STOP", None) == "X":
            stop_index = int(index)
    if stop_index is None:
        stop_index = 1
    if not 1 <= stop_index < len(surfaces) - 1:
        raise ValueError(f"stop index 不支持：{stop_index}")

    wavelength_um = float(wavelength_m) * 1.0e6
    indices = np.asarray(cauchy_a, dtype=np.float64) + np.asarray(
        cauchy_b, dtype=np.float64
    ) / wavelength_um**2
    curvatures = np.asarray(curvature[1:], dtype=np.float64)
    z_values = np.asarray(z_positions, dtype=np.float64)
    thicknesses = np.diff(z_values[1:])
    z0 = float(z_values[1])
    stop_radius_m = float(apertures[stop_index])
    if not math.isfinite(stop_radius_m) or stop_radius_m <= 0.0:
        raise ValueError("stop radius 非有限正数")

    front_curvature = curvatures[:stop_index].copy()
    front_curvature[-1] = 0.0
    front = _ior_curvature_thickness_sequence(
        indices[: stop_index + 1],
        front_curvature,
        thicknesses[: stop_index - 1],
        z0,
    )
    stop = _ior_curvature_thickness_sequence(
        indices[stop_index - 1 : stop_index + 1],
        curvatures[stop_index - 1 : stop_index],
        np.empty((0,), dtype=np.float64),
        front[2],
    )
    back_curvature = curvatures[stop_index - 1 :].copy()
    back_curvature[0] = 0.0
    back = _ior_curvature_thickness_sequence(
        indices[stop_index - 1 :],
        back_curvature,
        thicknesses[stop_index - 1 :],
        front[2],
    )
    total_matrix = back[0] @ stop[0] @ front[0]
    optical_power = -float(total_matrix[1, 0])
    if optical_power <= 0.0 or not math.isfinite(optical_power):
        raise ValueError("处方不是有限正光焦度")
    effective_focal_length_m = 1.0 / optical_power

    front_matrix = front[0]
    front_reverse = np.asarray(
        [
            [front_matrix[1, 1], -front_matrix[0, 1]],
            [-front_matrix[1, 0], front_matrix[0, 0]],
        ],
        dtype=np.float64,
    )
    entrance_relative, entrance_magnification = _obj2img_s0(
        front_reverse, n_image=float(indices[0])
    )
    entrance_pupil_radius_m = abs(entrance_magnification) * stop_radius_m
    entrance_pupil_z_m = float(front[1] + entrance_relative)
    del entrance_pupil_z_m
    exit_relative, exit_magnification = _obj2img_s0(
        back[0], n_image=float(indices[-1])
    )
    exit_pupil_radius_m = abs(exit_magnification) * stop_radius_m
    exit_pupil_z_m = float(back[2] + exit_relative)
    native_f_number = effective_focal_length_m / (2.0 * entrance_pupil_radius_m)
    back_focal_position_m = float(back[2] - total_matrix[0, 0] / total_matrix[1, 0])
    total_track_m = float(z_cumulative)

    numbers = (
        native_f_number,
        effective_focal_length_m,
        image_radius_m,
        back_focal_position_m,
        entrance_pupil_radius_m,
        exit_pupil_radius_m,
        exit_pupil_z_m,
        total_track_m,
    )
    if not all(math.isfinite(value) for value in numbers) or min(
        native_f_number,
        effective_focal_length_m,
        image_radius_m,
        entrance_pupil_radius_m,
        exit_pupil_radius_m,
        total_track_m,
    ) <= 0.0:
        raise ValueError("处方近轴量存在非有限值或非正尺度")
    aperture_scales = tuple(native_f_number / float(value) for value in requested_f_numbers)
    if any(scale > 1.0 + 1.0e-6 for scale in aperture_scales):
        raise ValueError("目标五光圈需要放大原生 stop")

    aspheric_count = 0
    if "ASPHERIC" in tables:
        aspheric_count = int(len(tables["ASPHERIC"]))
    relative_path = path.relative_to(root).as_posix()
    return LensParaxialRecord(
        relative_path=relative_path,
        lens_id=path.stem,
        sha256=_sha256_file(path),
        native_f_number=float(native_f_number),
        effective_focal_length_m=float(effective_focal_length_m),
        image_radius_m=float(image_radius_m),
        back_focal_position_m=float(back_focal_position_m),
        entrance_pupil_radius_m=float(entrance_pupil_radius_m),
        exit_pupil_radius_m=float(exit_pupil_radius_m),
        stop_radius_m=float(stop_radius_m),
        total_track_m=float(total_track_m),
        surface_count=int(len(surfaces)),
        aspheric_surface_count=aspheric_count,
        stop_index=int(stop_index),
        aperture_stop_scales=aperture_scales,
    )


def _selection_features(record: LensParaxialRecord, reference: LensParaxialRecord) -> np.ndarray:
    return np.asarray(
        [
            math.log(record.effective_focal_length_m / reference.effective_focal_length_m),
            math.log(record.image_radius_m / reference.image_radius_m),
            math.log(record.native_f_number / reference.native_f_number),
            (record.surface_count - reference.surface_count) / 12.0,
            (record.aspheric_surface_count - reference.aspheric_surface_count) / 6.0,
            record.stop_index / record.surface_count
            - reference.stop_index / reference.surface_count,
            math.log(record.total_track_m / reference.total_track_m),
        ],
        dtype=np.float64,
    )


def select_lens_prescriptions(
    records: Sequence[LensParaxialRecord],
    *,
    reference_relative_path: str,
    selected_count: int,
    efl_ratio_range: Sequence[float],
    image_radius_ratio_range: Sequence[float],
    native_f_number_max: float,
    compatibility_penalty: float,
) -> tuple[list[LensParaxialRecord], list[LensParaxialRecord]]:
    by_path = {row.relative_path: row for row in records}
    if len(by_path) != len(records):
        raise ValueError("lens relative_path 不唯一")
    if reference_relative_path not in by_path:
        raise ValueError("reference lens 未通过近轴解析/五光圈初筛")
    reference = by_path[reference_relative_path]
    efl_min, efl_max = map(float, efl_ratio_range)
    radius_min, radius_max = map(float, image_radius_ratio_range)
    compatible = [
        row
        for row in records
        if row.native_f_number <= float(native_f_number_max) + 1.0e-9
        and efl_min
        <= row.effective_focal_length_m / reference.effective_focal_length_m
        <= efl_max
        and radius_min
        <= row.image_radius_m / reference.image_radius_m
        <= radius_max
    ]
    if len(compatible) < int(selected_count):
        raise RuntimeError(
            f"兼容处方只有 {len(compatible)}，不足 selected_count={selected_count}"
        )

    selected = [reference]
    candidates = [row for row in compatible if row.relative_path != reference_relative_path]
    reference_feature = _selection_features(reference, reference)
    while len(selected) < int(selected_count):
        selected_features = [_selection_features(row, reference) for row in selected]

        def score(row: LensParaxialRecord) -> tuple[float, str]:
            feature = _selection_features(row, reference)
            diversity = min(float(np.linalg.norm(feature - value)) for value in selected_features)
            envelope_distance = float(np.linalg.norm(feature[:3] - reference_feature[:3]))
            value = diversity - float(compatibility_penalty) * envelope_distance
            # max() 使用路径的反向排序不直观，返回负 tie key 后改用显式排序。
            return value, row.relative_path

        ranked = sorted(candidates, key=lambda row: (-score(row)[0], score(row)[1]))
        winner = ranked[0]
        selected.append(winner)
        candidates.remove(winner)
    return selected, compatible


def _jax_cpu_validate(
    records: Sequence[LensParaxialRecord],
    *,
    root: Path,
    wavelength_m: float,
    requested_f_numbers: Sequence[float],
    relative_tolerance: float,
) -> list[dict[str, Any]]:
    from datasyn.optics.complens.tabular_lens import load_tabular_lens_from_mytable
    from datasyn.optics.imaging.imaging import HFProper, make_imaging

    try:
        from .multi_aperture import build_aperture_plan
    except ImportError:  # pragma: no cover - 兼容直接执行。
        from multi_aperture import build_aperture_plan  # type: ignore[no-redef]

    rows: list[dict[str, Any]] = []
    for record in records:
        lens = load_tabular_lens_from_mytable(str(root / record.relative_path))
        imaging = make_imaging(
            lens=lens,
            sen_wh=(2560, 2560),
            proper=HFProper(),
            wvl_ref=float(wavelength_m),
            pixsize=1.0e-5,
        )
        actual_f_number = float(imaging.parax.fnum)
        actual_efl = float(imaging.parax.efl)
        build_aperture_plan(
            actual_f_number,
            float(lens.get_aperture(lens.stop_idx)),
            requested_f_numbers,
        )
        f_error = abs(actual_f_number - record.native_f_number) / record.native_f_number
        efl_error = abs(actual_efl - record.effective_focal_length_m) / record.effective_focal_length_m
        passed = max(f_error, efl_error) <= float(relative_tolerance)
        rows.append(
            {
                "lens_id": record.lens_id,
                "relative_path": record.relative_path,
                "numpy_native_f_number": record.native_f_number,
                "jax_native_f_number": actual_f_number,
                "f_number_relative_error": f_error,
                "numpy_effective_focal_length_m": record.effective_focal_length_m,
                "jax_effective_focal_length_m": actual_efl,
                "efl_relative_error": efl_error,
                "all_requested_apertures_valid": True,
                "pass": passed,
            }
        )
    if not all(row["pass"] for row in rows):
        raise RuntimeError("selected lens 的 NumPy/JAX CPU 近轴等价门禁失败")
    return rows


def build_device_atom_plan(
    selected: Sequence[LensParaxialRecord], config: dict[str, Any]
) -> list[dict[str, Any]]:
    atom_cfg = config["device_atom"]
    dcc = float(atom_cfg["dcc_px_per_coc"])
    if not math.isfinite(dcc) or dcc <= 0.0:
        raise ValueError("device_atom.dcc_px_per_coc 必须为有限正数")
    if atom_cfg["pdoffset_embedded"] is not False:
        raise ValueError("multi-prescription v1 禁止把 PDOFFSET 写入 bank")
    if str(atom_cfg["coc_zero_anchor"]) != "preserve_native_optical_bias":
        raise ValueError("multi-prescription v1 必须保留原生 CoC0 optical bias")
    output_root = Path(config["output"]["gpu_artifact_root"])
    requested_f_numbers = [float(value) for value in config["optics"]["f_numbers"]]
    signed_coc = [float(value) for value in config["optics"]["signed_coc_bins_px"]]
    field_x = [float(value) for value in config["optics"]["field_x_normalized"]]
    field_y = [float(value) for value in config["optics"]["field_y_normalized"]]
    atoms: list[dict[str, Any]] = []
    for index, lens in enumerate(selected):
        atom_id = f"MP{index:02d}_{lens.lens_id}_DCC6"
        lineage_id = f"{config['run']['id']}::{lens.sha256[:16]}::dcc6_head0"
        atom_root = output_root / atom_id
        atoms.append(
            {
                "atom_id": atom_id,
                "lens_id": lens.lens_id,
                "lens_relative_path": lens.relative_path,
                "lens_sha256": lens.sha256,
                "shared_lineage_id": lineage_id,
                "shared_lineage_across_apertures": True,
                "f_numbers": requested_f_numbers,
                "signed_coc_bins_px": signed_coc,
                "field_x_normalized": field_x,
                "field_y_normalized": field_y,
                "sensor_head_id": str(atom_cfg["sensor_head_id"]),
                "response_profile": config["response_profile"],
                "dcc_px_per_coc": dcc,
                "dcc_anchor_mode": "preserve_native_coc0",
                "pair_common": config["pair_common"],
                "analytic_label_source": "final_kernel_centroid_mu_left_x_minus_mu_right_x",
                "pdoffset_embedded": False,
                "pdoffset_contract": "disp_gt=analytic_centroid_disp+independent_pd_offset_px",
                "ncc_role": "diagnostic_admission_only_never_label_writeback",
                "planned_outputs": {
                    "raw_family_root": str((atom_root / "raw_family").resolve()),
                    "raw_optical_root": str((atom_root / "raw_optical").resolve()),
                    "k64_audit_root": str((atom_root / "audit_k64").resolve()),
                    "k128_audit_root": str((atom_root / "audit_k128").resolve()),
                    "final_asset_root": str((atom_root / "paircommon_dcc6").resolve()),
                },
                "stage_status": {
                    "cpu_lens_selection": "pass",
                    "gpu_raw_propagation": "pending_user_authorization",
                    "pair_common_dcc6_export": "pending_raw_asset",
                    "continuous_ncc_k64": "pending_export",
                    "continuous_ncc_k128": "pending_k64",
                    "training_admission": "not_admitted",
                },
            }
        )
    return atoms


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "run",
        "source",
        "selection",
        "optics",
        "response_profile",
        "device_atom",
        "pair_common",
        "gates",
        "output",
        "data_isolation",
    }
    if set(config) != required:
        raise ValueError(
            f"multi-prescription config 顶层字段不一致：expected={sorted(required)}"
        )
    if int(config["schema_version"]) != 1:
        raise ValueError("只支持 schema_version=1")
    if config["run"].get("mode") != "cpu_prepare_only":
        raise ValueError("当前实现只允许 run.mode=cpu_prepare_only")
    isolation = config["data_isolation"]
    for key in (
        "real_pdraw_accessed",
        "google_dev_accessed",
        "google_holdout_accessed",
        "dp5k_accessed",
        "stereo_training_run",
    ):
        if isolation.get(key) is not False:
            raise ValueError(f"CPU planner 要求 data_isolation.{key}=false")
    f_numbers = [float(value) for value in config["optics"]["f_numbers"]]
    if f_numbers != sorted(f_numbers) or f_numbers != [1.8, 2.0, 2.8, 4.0, 5.6]:
        raise ValueError("v1 五光圈必须精确为 [1.8,2.0,2.8,4.0,5.6]")
    coc = [float(value) for value in config["optics"]["signed_coc_bins_px"]]
    if coc != sorted(coc) or coc.count(0.0) != 1:
        raise ValueError("signed CoC 必须严格递增并包含唯一 0")


def prepare(config_path: Path, *, write_outputs: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    source_root = Path(config["source"]["root"]).resolve()
    patterns = list(config["source"]["include_globs"])
    paths = sorted({path for pattern in patterns for path in source_root.glob(pattern)})
    expected_count = int(config["source"]["expected_file_count"])
    if len(paths) != expected_count:
        raise RuntimeError(
            f"source lens 数量不一致：expected={expected_count}, actual={len(paths)}"
        )

    requested_f_numbers = config["optics"]["f_numbers"]
    wavelength_m = float(config["optics"]["wavelength_m"])
    records: list[LensParaxialRecord] = []
    rejected: list[dict[str, str]] = []
    for path in paths:
        try:
            records.append(
                _parse_paraxial_record(
                    path,
                    root=source_root,
                    wavelength_m=wavelength_m,
                    requested_f_numbers=requested_f_numbers,
                )
            )
        except Exception as error:  # noqa: BLE001 - catalog 必须保留逐文件拒绝原因。
            rejected.append(
                {
                    "relative_path": path.relative_to(source_root).as_posix(),
                    "status": "rejected",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
    selection = config["selection"]
    selected, compatible = select_lens_prescriptions(
        records,
        reference_relative_path=str(selection["reference_relative_path"]),
        selected_count=int(selection["selected_count"]),
        efl_ratio_range=selection["effective_focal_length_ratio_range"],
        image_radius_ratio_range=selection["image_radius_ratio_range"],
        native_f_number_max=float(selection["native_f_number_max"]),
        compatibility_penalty=float(selection["compatibility_penalty"]),
    )
    expected_paths = list(selection["expected_selected_relative_paths"])
    selected_paths = [row.relative_path for row in selected]
    if write_outputs and (
        len(expected_paths) != int(selection["selected_count"])
        or len(selection["expected_selected_sha256"]) != int(selection["selected_count"])
    ):
        raise RuntimeError("写正式 CPU plan 前必须冻结全部 selected lens 路径与 SHA256")
    if expected_paths and selected_paths != expected_paths:
        raise RuntimeError(
            "selected lens 与冻结顺序不一致："
            f"expected={expected_paths}, actual={selected_paths}"
        )
    expected_sha = selection["expected_selected_sha256"]
    if expected_sha:
        actual_sha = {row.relative_path: row.sha256 for row in selected}
        if actual_sha != expected_sha:
            raise RuntimeError("selected lens SHA256 与冻结配置不一致")

    jax_rows = _jax_cpu_validate(
        selected,
        root=source_root,
        wavelength_m=wavelength_m,
        requested_f_numbers=requested_f_numbers,
        relative_tolerance=float(config["gates"]["numpy_jax_relative_error_max"]),
    )
    atoms = build_device_atom_plan(selected, config)
    catalog_rows = [
        {"status": "paraxial_aperture_pass", **asdict(row)} for row in records
    ] + rejected
    catalog_rows.sort(key=lambda row: str(row["relative_path"]))
    selected_payload = {
        "schema_version": 1,
        "selection_policy": "reference_plus_deterministic_compatible_maximin_v1",
        "reference_relative_path": selection["reference_relative_path"],
        "source_file_count": len(paths),
        "paraxial_aperture_pass_count": len(records),
        "compatible_envelope_count": len(compatible),
        "rejected_count": len(rejected),
        "selected": [asdict(row) for row in selected],
        "jax_cpu_validation": jax_rows,
    }
    result = {
        "status": "cpu_prepared_pending_gpu_propagation",
        "selected_lens_relative_paths": selected_paths,
        "selected_lens_sha256": {row.relative_path: row.sha256 for row in selected},
        "compatible_envelope_count": len(compatible),
        "paraxial_aperture_pass_count": len(records),
        "rejected_count": len(rejected),
        "device_atom_count": len(atoms),
        "device_atoms": atoms,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if not write_outputs:
        return result

    output_root = Path(config["output"]["plan_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    catalog_path = output_root / "lens_catalog.jsonl"
    with catalog_path.open("w", encoding="utf-8") as stream:
        for row in catalog_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    selected_path = output_root / "selected_lenses.json"
    selected_path.write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    atom_plan_path = output_root / "device_atom_plan.json"
    atom_plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": result["status"],
                "atom_count": len(atoms),
                "atoms": atoms,
                **config["data_isolation"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    resolved_path = output_root / "resolved_config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    provenance_path = output_root / "provenance.json"
    checkout = Path(__file__).resolve().parents[1]
    provenance = {
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": _sha256_file(Path(__file__).resolve()),
        "source_root": str(source_root),
        "source_file_count": len(paths),
        "source_catalog_identity_sha256": _sha256_json(
            [(row.relative_path, row.sha256) for row in records]
        ),
        "source_commit": _git_output(checkout, "rev-parse", "HEAD"),
        "source_dirty_paths": _git_output(checkout, "status", "--short").splitlines(),
        "python_executable": sys.executable,
        "execution_platform": "cpu_only",
        "jax_platforms": os.environ["JAX_PLATFORMS"],
        **config["data_isolation"],
    }
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    files = {
        path.name: _sha256_file(path)
        for path in (
            catalog_path,
            selected_path,
            atom_plan_path,
            resolved_path,
            provenance_path,
            summary_path,
        )
    }
    manifest = {
        "schema_version": 1,
        "status": result["status"],
        "plan_id": config["run"]["id"],
        "selected_lens_ids": [row.lens_id for row in selected],
        "selected_lens_sha256": result["selected_lens_sha256"],
        "device_atom_count": len(atoms),
        "shared_aperture_lineage_required": True,
        "dcc_px_per_coc": float(config["device_atom"]["dcc_px_per_coc"]),
        "coc_zero_anchor": "preserve_native_optical_bias",
        "pdoffset_embedded": False,
        "gpu_propagation_started": False,
        "training_admitted": False,
        "files": files,
        **config["data_isolation"],
    }
    manifest_path = output_root / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result.update(
        {
            "output_root": str(output_root),
            "artifact_manifest_sha256": _sha256_file(manifest_path),
            "summary_sha256": _sha256_file(summary_path),
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印选镜结果，不写外部 plan 产物",
    )
    args = parser.parse_args(argv)
    result = prepare(args.config, write_outputs=not args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
