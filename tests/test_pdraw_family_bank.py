from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from pdraw_adapter.family_bank import (
    _EXPORT_RETARGET_AXES,
    _PROFILE_AXES,
    _RAW_ACTIVE_AXES,
    _atomic_torch_save,
    _chunk_cache_identity,
    _load_config,
    _load_verified_chunk_cache,
    _select_family_medoid,
    analytic_centroid_labels,
    compose_analytic_label_with_pdoffset,
    family_profile_document,
    retarget_centroid_slope,
    sample_family_profiles,
    _export_pdraw_mono_v1_by_aperture,
)
from pdraw_adapter.multi_profile import (
    LowOrderAberrationProfile,
    low_order_aberration_waves,
)
from pdraw_adapter.raw_optical_export import build_raw_profile_payload


def _randomization_config() -> dict[str, object]:
    return {
        "master_seed": 20260812,
        "profile_count": 4,
        "profile_id_prefix": "T",
        "ranges": {axis: [0.0, 0.0] for axis in _PROFILE_AXES},
    }


def _valid_randomization_config() -> dict[str, object]:
    config = _randomization_config()
    ranges = config["ranges"]
    assert isinstance(ranges, dict)
    ranges.update(
        {
            "transition_width": [0.35, 0.55],
            "cross_talk": [0.0, 0.04],
            "split_bias_norm": [-0.02, 0.02],
            "boundary_curvature_norm": [-0.01, 0.01],
            "microlens_edge_rolloff": [0.0, 0.08],
            "field_split_slope": [-0.02, 0.02],
            "field_acceptance_slope": [-0.05, 0.05],
            "lr_throughput_delta": [-0.02, 0.02],
            "field_lr_throughput_slope": [-0.01, 0.01],
            "aperture_lr_throughput_slope": [-0.01, 0.01],
            "astigmatism_0_waves": [-0.03, 0.03],
            "astigmatism_45_waves": [-0.03, 0.03],
            "coma_x_waves": [-0.03, 0.03],
            "coma_y_waves": [-0.03, 0.03],
            "spherical_waves": [-0.02, 0.02],
            "field_astigmatism_0_waves_per_norm": [-0.03, 0.03],
            "field_astigmatism_45_waves_per_norm": [-0.03, 0.03],
            "field_coma_x_waves_per_norm": [-0.03, 0.03],
            "field_coma_y_waves_per_norm": [-0.03, 0.03],
            "centroid_slope_px_per_coc": [0.14, 0.18],
        }
    )
    return config


def test_named_random_axes_are_independent_and_deterministic() -> None:
    base_config = _valid_randomization_config()
    base = sample_family_profiles(base_config)
    repeated = sample_family_profiles(base_config)
    assert base == repeated

    changed_config = _valid_randomization_config()
    changed_ranges = changed_config["ranges"]
    assert isinstance(changed_ranges, dict)
    changed_ranges["coma_x_waves"] = [0.10, 0.12]
    changed = sample_family_profiles(changed_config)

    for before, after in zip(base, changed, strict=True):
        assert asdict(before.response) == asdict(after.response)
        before_aberration = asdict(before.aberration)
        after_aberration = asdict(after.aberration)
        for key in before_aberration:
            if key != "coma_x_waves":
                assert before_aberration[key] == after_aberration[key]
        assert before.centroid_slope_px_per_coc == after.centroid_slope_px_per_coc
        assert family_profile_document(before)["parameter_sha256"] != (
            family_profile_document(after)["parameter_sha256"]
        )


def test_family_axis_contract_separates_raw_and_export_retarget_axes() -> None:
    assert set(_RAW_ACTIVE_AXES).isdisjoint(_EXPORT_RETARGET_AXES)
    assert _PROFILE_AXES == (*_RAW_ACTIVE_AXES, *_EXPORT_RETARGET_AXES)
    profile = sample_family_profiles(_valid_randomization_config())[0]
    contract = family_profile_document(profile)["axis_contract"]
    assert contract["raw_active_axes"] == list(_RAW_ACTIVE_AXES)
    assert contract["export_retarget_axes"] == list(_EXPORT_RETARGET_AXES)
    assert contract["centroid_slope_px_per_coc_role"] == "export_retarget_only"


def test_low_order_screen_has_zero_piston_and_field_variation() -> None:
    coordinates = np.linspace(-1.0, 1.0, 9, dtype=np.float32)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    valid = xx**2 + yy**2 <= 1.0
    profile = LowOrderAberrationProfile(
        coma_x_waves=0.02,
        field_coma_x_waves_per_norm=0.04,
        field_astigmatism_0_waves_per_norm=0.03,
        field_coma_y_waves_per_norm=0.02,
    )
    left, left_meta = low_order_aberration_waves(
        xx, yy, valid, profile, field_x=-0.5
    )
    right, right_meta = low_order_aberration_waves(
        xx, yy, valid, profile, field_x=0.5
    )
    left_np = np.asarray(left)
    right_np = np.asarray(right)
    assert abs(float(left_np[valid].mean())) < 1.0e-7
    assert abs(float(right_np[valid].mean())) < 1.0e-7
    assert not np.allclose(left_np, right_np)
    assert left_meta["effective_coma_x_waves"] == pytest.approx(0.0)
    assert right_meta["effective_coma_x_waves"] == pytest.approx(0.04)


def test_pdraw_mono_export_is_shape_only_and_loader_contract_compatible(
    tmp_path: Path,
) -> None:
    bank = np.zeros((1, 3, 1, 1, 2, 7, 7), dtype=np.float32)
    bank[:, :, :, :, 0, 3, 2] = 1.0
    bank[:, :, :, :, 1, 3, 4] = 1.0
    rows = _export_pdraw_mono_v1_by_aperture(
        tmp_path,
        resource_prefix="fixture",
        bank=bank,
        signed_coc=np.asarray([[-1.0, 0.0, 1.0]], dtype=np.float64),
        f_numbers=np.asarray([2.0]),
        field_x=[0.0],
        field_y=[0.0],
    )
    assert len(rows) == 1
    array = np.load(rows[0]["array_path"], allow_pickle=False)
    metadata = __import__("json").loads(
        Path(rows[0]["metadata_path"]).read_text(encoding="utf-8")
    )
    assert array.ndim == 5 and array.shape[:3] == (1, 1, 2)
    assert metadata["format_version"] == "pdraw_mono_v1"
    assert metadata["coc_values_px"] == [1.0]
    assert metadata["centroid_residual_abs_max_px"] < 1.0e-7
    assert metadata["pdoffset_embedded"] is False


def test_centroid_retarget_and_pdoffset_are_independent() -> None:
    kernel = np.zeros((9, 9), dtype=np.float32)
    kernel[4, 4] = 1.0
    bank = np.tile(kernel, (1, 3, 1, 1, 2, 1, 1))
    original = bank.copy()
    signed_coc = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    output, labels, metadata = retarget_centroid_slope(
        bank,
        signed_coc,
        [0.2],
        support_padding_px=2,
    )
    np.testing.assert_array_equal(bank, original)
    np.testing.assert_allclose(
        labels[:, :, 0, 0], [[-0.2, 0.0, 0.2]], atol=2.0e-7, rtol=0.0
    )
    np.testing.assert_allclose(labels, analytic_centroid_labels(output), atol=0.0, rtol=0.0)
    assert metadata["pd_offset_embedded"] is False
    pdoffset = compose_analytic_label_with_pdoffset(labels, 0.125)
    np.testing.assert_allclose(pdoffset[:, 1], 0.125, atol=2.0e-7, rtol=0.0)
    np.testing.assert_allclose(labels[:, 1], 0.0, atol=2.0e-7, rtol=0.0)


def test_centroid_retarget_preserves_native_coc0_optical_bias() -> None:
    bank = np.zeros((1, 3, 1, 2, 2, 11, 11), dtype=np.float32)
    # 两个 field 的原生 CoC=0 disparity 分别为 +1 与 -1 px；它们不是 PDOFFSET。
    for coc_index in range(3):
        bank[0, coc_index, 0, 0, 0, 5, 6] = 1.0
        bank[0, coc_index, 0, 0, 1, 5, 5] = 1.0
        bank[0, coc_index, 0, 1, 0, 5, 4] = 1.0
        bank[0, coc_index, 0, 1, 1, 5, 5] = 1.0
    output, labels, metadata = retarget_centroid_slope(
        bank,
        np.asarray([-1.0, 0.0, 1.0]),
        [0.25],
        support_padding_px=2,
        anchor_mode="preserve_native_coc0",
    )
    np.testing.assert_allclose(
        labels[0, :, 0],
        [[0.75, -1.25], [1.0, -1.0], [1.25, -0.75]],
        atol=2.0e-7,
        rtol=0.0,
    )
    np.testing.assert_allclose(labels, analytic_centroid_labels(output), atol=0.0, rtol=0.0)
    assert metadata["anchor_mode"] == "preserve_native_coc0"
    assert metadata["native_coc0_optical_bias_preserved"] is True
    assert metadata["pd_offset_embedded"] is False


def test_centroid_retarget_accepts_frozen_common_anchor() -> None:
    kernel = np.zeros((9, 9), dtype=np.float32)
    kernel[4, 4] = 1.0
    bank = np.tile(kernel, (1, 3, 1, 2, 2, 1, 1))
    _, labels, metadata = retarget_centroid_slope(
        bank,
        np.asarray([-1.0, 0.0, 1.0]),
        [0.2],
        support_padding_px=2,
        anchor_mode="common",
        common_anchor_disparity_px=np.asarray([[[0.1, -0.2]]]),
    )
    np.testing.assert_allclose(
        labels[0, :, 0],
        [[-0.1, -0.4], [0.1, -0.2], [0.3, 0.0]],
        atol=2.0e-7,
        rtol=0.0,
    )
    assert metadata["common_anchor_frozen_by_caller"] is True


def test_raw_optical_payload_recomputes_labels_and_declares_native_contract() -> None:
    bank = torch.zeros(2, 3, 1, 1, 2, 9, 9)
    bank[:, :, 0, 0, 0, 4, 5] = 1.0
    bank[:, :, 0, 0, 1, 4, 3] = 1.0
    analytic = torch.full((1, 2, 3, 1, 1), 2.0, dtype=torch.float64)
    source = {
        "asset_id": "fixture_family",
        "family_id": "fixture",
        "family_sha256": "a" * 64,
        "profile_ids": ["F000"],
        "family_profiles": [
            {"profile_id": "F000", "parameter_sha256": "b" * 64}
        ],
        "physical_centroid_vs_coc_by_profile": [{"source": "fixture"}],
        "f_numbers": torch.tensor([1.8, 2.8]),
        "signed_coc_bins_px": torch.tensor(
            [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]], dtype=torch.float64
        ),
        "raw_analytic_disparity_bins_px": analytic,
        "field_grid_hw": (1, 1),
        "field_x_normalized": torch.tensor([0.0]),
        "field_y_normalized": torch.tensor([0.0]),
        "kernel_size": 9,
        "side_throughput_grid": torch.ones(1, 2, 1, 1, 2),
    }
    payload, stats = build_raw_profile_payload(
        source_payload=source,
        profile_index=0,
        raw_bank=bank,
        analytic_tolerance_px=1.0e-7,
    )
    assert payload["centroid_policy"] == "native"
    assert payload["centroid_variant"] == "raw_optical_v1"
    assert payload["pdoffset_embedded"] is False
    assert payload["profile_axis_contract"]["raw_active_axes"] == list(
        _RAW_ACTIVE_AXES
    )
    assert payload["profile_axis_contract"][
        "inactive_profile_parameters_for_raw_export"
    ] == ["centroid_slope_px_per_coc"]
    assert payload["profile_axis_contract"][
        "centroid_slope_px_per_coc_active"
    ] is False
    torch.testing.assert_close(
        payload["analytic_disparity_bins_px"], analytic[0], atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        payload["raw_analytic_disparity_bins_px"], analytic[0], atol=0.0, rtol=0.0
    )
    assert stats["analytic_agreement_abs_max_px"] == 0.0


def test_family_config_rejects_any_real_access_or_training(tmp_path: Path) -> None:
    config = {
        "schema_version": 1,
        "run": {},
        "source": {},
        "lens": {},
        "sensor": {},
        "apertures": {},
        "optics": {},
        "family": {},
        "pdoffset": {"sample_range_px": [-0.2, 0.2], "smoke_probe_px": 0.125},
        "gates": {},
        "output": {},
        "data_isolation": {
            "real_pdraw_accessed": False,
            "google_dev_accessed": False,
            "google_holdout_accessed": False,
            "dp5k_accessed": False,
            "stereo_training_run": False,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert _load_config(path)["data_isolation"]["google_holdout_accessed"] is False
    config["data_isolation"]["google_dev_accessed"] = True
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="禁止访问真实数据"):
        _load_config(path)


def test_medoid_uses_hard_gate_slack_not_arbitrary_half_gate() -> None:
    randomization = _valid_randomization_config()
    profiles = sample_family_profiles(randomization)[:2]
    gates = {
        "reference_crop_retained_mass_min": 0.995,
        "retarget_retained_mass_min": 0.995,
        "combined_support_retained_mass_min": 0.99,
        "centroid_target_error_abs_max_px": 5.0e-5,
        "zero_centroid_abs_max_px": 5.0e-5,
        "energy_abs_max": 2.0e-6,
        "adjacent_psf_l1_max": 1.999,
    }
    diagnostic = {
        "pass": True,
        "reference_crop_retained_mass_min": 0.996,
        "combined_support_retained_mass_min": 0.991,
        "energy_min": 0.9999999,
        "energy_max": 1.0000001,
        "zero_centroid_disparity_abs_max_px": 3.75e-5,
        "adjacent_coc_psf_l1_max": 1.98,
        "retarget": {
            "retained_mass_min": 0.996,
            "analytic_target_error_abs_max_px": 3.75e-5,
        },
    }
    # 旧版的 centroid<=half gate 和 continuity<=95% gate 都会拒绝该 profile；
    # 新版按距正式 hard gate 的归一化余量判断，仍保留 10% 以上余量。
    selected = _select_family_medoid(
        profiles,
        [diagnostic, diagnostic],
        randomization,
        gates,
    )
    assert selected["eligible_profile_count"] == 2
    assert selected["selected_is_not_support_or_centroid_gate_edge"] is True
    assert (
        selected["selected"]["minimum_support_centroid_normalized_slack"]
        == pytest.approx(0.1)
    )


def test_chunk_cache_is_atomic_and_rejects_identity_mismatch(tmp_path: Path) -> None:
    profiles = sample_family_profiles(_valid_randomization_config())[:2]
    identity = _chunk_cache_identity(
        config_sha="a" * 64,
        family_sha="b" * 64,
        lens_sha="c" * 64,
        profiles=profiles,
        chunk_start=0,
        reference_kernel_size=9,
        export_kernel_size=7,
        expected_shape=(1, 1, 1, 1, 2, 7, 7),
    )
    path = tmp_path / "raw_chunks" / "chunk_000_002.pt"
    payload = {
        "identity": identity,
        "source_seconds": 1.25,
        "raw_banks": {
            profile.profile_id: torch.ones((1, 1, 1, 1, 2, 7, 7))
            for profile in profiles
        },
        "reference_crop_retained": {
            profile.profile_id: torch.ones((2,), dtype=torch.float64)
            for profile in profiles
        },
    }
    sha = _atomic_torch_save(path, payload)
    loaded = _load_verified_chunk_cache(path, identity)
    assert loaded is not None and loaded["file_sha256"] == sha
    mismatched = {**identity, "family_sha256": "d" * 64}
    with pytest.raises(RuntimeError, match="身份不匹配"):
        _load_verified_chunk_cache(path, mismatched)
