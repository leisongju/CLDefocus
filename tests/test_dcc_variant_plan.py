from __future__ import annotations

from pdraw_adapter.dcc_variant_plan import candidate_operations


def test_dcc57_operations_preserve_native_anchor_and_paircommon_contract() -> None:
    pair_common = {
        "common_shape_mtf_sigma_px": 1.0,
        "clipped_negative_fraction_max": 0.02,
        "transition_coc_px": 16.0,
        "transition_power": 1.0,
    }
    for slope in (0.2, 1.0 / 7.0):
        operations = candidate_operations(
            slope_px_per_coc=slope,
            aperture_count=5,
            pair_common=pair_common,
        )
        assert operations[0]["target_slopes_by_aperture"] == [slope] * 5
        assert operations[0]["anchor_mode"] == "preserve_native_coc0"
        assert operations[1] == {
            "kind": "pair_common",
            "alignment": "direct_mean",
            "translation_mode": "bilinear_center_fourier_place",
            "common_shape_mtf_sigma_px": 1.0,
            "clipped_negative_fraction_max": 0.02,
            "transition_coc_px": 16.0,
            "transition_power": 1.0,
        }
