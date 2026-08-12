from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pdraw_adapter.family_bank import analytic_centroid_labels
from pdraw_adapter.raw_family_candidate_export import _load_verified_audit
from pdraw_adapter.raw_family_continuous_audit import (
    common_sensor_mtf,
    pair_common_morphology,
    transform_candidate,
)


def _fixture_bank() -> tuple[np.ndarray, np.ndarray]:
    bank = np.zeros((1, 3, 1, 1, 2, 15, 15), dtype=np.float32)
    # 两侧形态不同、能量不同，并含一个不应被吞掉的 +1 px optical bias。
    for coc_index in range(3):
        bank[0, coc_index, 0, 0, 0, 7, 8] = 1.5
        bank[0, coc_index, 0, 0, 0, 7, 9] = 0.5
        bank[0, coc_index, 0, 0, 1, 7, 7] = 1.0
        bank[0, coc_index, 0, 0, 1, 8, 7] = 2.0
    return bank, np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)


def test_pair_common_preserves_side_energy_and_native_centroid() -> None:
    bank, coc = _fixture_bank()
    source_labels = analytic_centroid_labels(bank)
    output, metadata = pair_common_morphology(
        bank,
        coc,
        transition_coc_px=4.0,
        alignment="direct_mean",
    )
    np.testing.assert_allclose(
        output.sum(axis=(-2, -1)),
        bank.sum(axis=(-2, -1)),
        atol=1.0e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        analytic_centroid_labels(output), source_labels, atol=2.0e-6, rtol=0.0
    )
    assert metadata["native_optical_centroid_preserved"] is True
    assert metadata["side_throughput_unchanged"] is True
    assert metadata["ncc_used"] is False


def test_common_mtf_is_pair_common_and_labels_are_recomputed() -> None:
    bank, coc = _fixture_bank()
    output, metadata = common_sensor_mtf(bank, sigma_px=0.75)
    assert output.shape == bank.shape
    assert metadata["same_kernel_left_right"] is True
    assert metadata["ncc_used"] is False
    transformed, labels, operations = transform_candidate(
        bank,
        coc,
        {
            "id": "fixture",
            "operations": [
                {"kind": "pair_common", "transition_coc_px": 4.0},
                {"kind": "common_mtf", "sigma_px": 0.75},
            ],
        },
        centroid_tolerance_px=0.002,
        retained_mass_min=0.98,
    )
    np.testing.assert_allclose(
        labels, analytic_centroid_labels(transformed), atol=0.0, rtol=0.0
    )
    assert [row["operation"] for row in operations] == [
        "small_coc_pair_common_morphology_v1",
        "pair_common_zero_centroid_gaussian_sensor_mtf_v1",
    ]
    assert all(row["pdoffset_embedded"] is False for row in operations)


def test_pair_common_hybrid_fourier_place_keeps_nonnegative_centroid_contract() -> None:
    bank, coc = _fixture_bank()
    output, metadata = pair_common_morphology(
        bank,
        coc,
        transition_coc_px=16.0,
        alignment="direct_mean",
        translation_mode="bilinear_center_fourier_place",
        common_shape_mtf_sigma_px=1.0,
        clipped_negative_fraction_max=0.02,
    )
    assert np.isfinite(output).all()
    assert np.all(output >= 0.0)
    np.testing.assert_allclose(
        analytic_centroid_labels(output),
        analytic_centroid_labels(bank),
        atol=2.0e-6,
        rtol=0.0,
    )
    assert metadata["center_translation_mode"] == "bilinear_splat"
    assert metadata["place_translation_mode"] == "fourier_nonnegative"
    assert metadata["common_shape_mtf_sigma_px"] == 1.0
    assert metadata["fourier_clipped_negative_fraction_max"] <= 0.02


def test_dcc_retarget_before_pair_common_preserves_native_zero_anchor() -> None:
    bank, coc = _fixture_bank()
    target_slope = 1.0 / 6.0
    transformed, labels, operations = transform_candidate(
        bank,
        coc,
        {
            "id": "dcc6_pair_common",
            "operations": [
                {
                    "kind": "dcc_retarget",
                    "target_slopes_by_aperture": [target_slope],
                    "support_padding_px": 0,
                    "anchor_mode": "preserve_native_coc0",
                },
                {
                    "kind": "pair_common",
                    "transition_coc_px": 16.0,
                    "alignment": "direct_mean",
                    "translation_mode": "bilinear_center_fourier_place",
                    "common_shape_mtf_sigma_px": 1.0,
                },
            ],
        },
        centroid_tolerance_px=0.002,
        retained_mass_min=0.98,
    )
    source_labels = analytic_centroid_labels(bank)
    expected = source_labels[:, 1:2] + target_slope * coc[None, :, None, None]
    np.testing.assert_allclose(labels, expected, atol=2.0e-6, rtol=0.0)
    np.testing.assert_allclose(
        labels, analytic_centroid_labels(transformed), atol=0.0, rtol=0.0
    )
    assert operations[0]["native_coc0_optical_bias_preserved"] is True
    assert operations[1]["native_optical_centroid_preserved"] is True
    assert all(operation["ncc_used"] is False for operation in operations)


def test_candidate_export_selects_one_candidate_from_multi_candidate_audit(
    tmp_path: Path,
) -> None:
    profiles = []
    for profile_id in ("F000", "F001"):
        fields = []
        for aperture_index in range(5):
            aperture_fields = [
                {
                    "field_y_index": field_y,
                    "field_x_index": field_x,
                    "pass": True,
                    "fit": {"slope_ncc_vs_analytic": 1.0, "r2": 1.0},
                }
                for field_y in range(3)
                for field_x in range(3)
            ]
            fields.append({"aperture_index": aperture_index, "fields": aperture_fields})
        profiles.append(
            {
                "profile_id": profile_id,
                "all_apertures_pass": True,
                "all_aperture_fields_pass": True,
                "per_aperture": fields,
            }
        )
    screens = [
        {
            "candidate_id": candidate_id,
            "tile_size": 64,
            "field_mode": "full_flattened",
            "fields_per_profile": 9,
            "profiles": profiles,
            "seconds": 1.0,
        }
        for candidate_id in ("reject_me", "select_me")
    ]
    payloads = {
        "summary.json": {"status": "pass"},
        "provenance.json": {"source": "fixture"},
        "screens.json": screens,
    }
    row: dict[str, object] = {"root": str(tmp_path), "tile_size": 64}
    for name, payload in payloads.items():
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        row[f"{name.removesuffix('.json')}_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

    verified, fields = _load_verified_audit(
        row,
        candidate_id="select_me",
        expected_profile_ids=["F000", "F001"],
    )
    assert verified["tile_size"] == 64
    assert len(fields) == 2 * 5 * 9

    with pytest.raises(RuntimeError, match="精确包含一个匹配"):
        _load_verified_audit(
            row,
            candidate_id="missing",
            expected_profile_ids=["F000", "F001"],
        )
