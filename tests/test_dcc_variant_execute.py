from __future__ import annotations

from pathlib import Path

import pytest

from pdraw_adapter.dcc_variant_execute import (
    _convergence,
    build_candidate_config,
    load_config,
)


def _isolation() -> dict[str, bool]:
    return {
        "real_pdraw_accessed": False,
        "google_dev_accessed": False,
        "google_holdout_accessed": False,
        "dp5k_accessed": False,
        "stereo_training_run": False,
    }


def test_formal_config_freezes_cpu_only_and_order() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/pdraw_dcc57_variant_execute_2a_v1.yaml")
    assert config["run"]["audit_order"] == [64, 128]
    assert config["run"]["gpu_allowed"] is False
    assert config["run"]["new_optical_propagation_allowed"] is False
    assert config["data_isolation"] == _isolation()


def test_candidate_config_uses_verified_audits_and_recipe() -> None:
    execution = {
        "run": {"id": "fixture"},
        "data_isolation": _isolation(),
    }
    candidate = {"id": "atom__dcc5", "operations": []}
    variant = {
        "candidate_id": candidate["id"],
        "output_root": "/formal/atom/dcc5/paircommon",
        "recipe": {
            "source": {
                "asset_root": "/raw",
                "artifact_manifest_sha256": "a" * 64,
                "profile_ids": ["H000"],
            },
            "candidate": candidate,
            "parent": {
                "repository": "/parent",
                "loader_sha256": "b" * 64,
                "renderer_sha256": "c" * 64,
            },
            "gates": {
                "centroid_abs_max_error_px": 0.002,
                "parent_centroid_abs_max_error_px": 1e-6,
                "retained_mass_min": 0.98,
            },
        },
    }
    audits = [
        {
            "tile_size": tile,
            "root": f"/audit/{tile}",
            "summary_sha256": "d" * 64,
            "provenance_sha256": "e" * 64,
            "screens_sha256": "f" * 64,
            "_fields": {},
        }
        for tile in (64, 128)
    ]
    result = build_candidate_config(execution, variant, audits)
    assert [row["tile_size"] for row in result["audits"]] == [64, 128]
    assert result["candidate"] == candidate
    assert result["output"]["root"] == variant["output_root"]
    assert result["data_isolation"] == _isolation()


def test_convergence_pairs_the_same_fields() -> None:
    def stage(offset: float) -> dict:
        return {
            "_fields": {
                ("H000", 0, 0, index): {
                    "fit": {
                        "slope_ncc_vs_analytic": 1.0 + offset * index,
                        "r2": 0.999 - offset * index,
                    }
                }
                for index in range(3)
            }
        }

    result = _convergence(stage(0.0), stage(0.001))
    assert result["field_count"] == 3
    assert result["slope_delta_min"] == 0.0
    assert result["slope_delta_max"] == pytest.approx(0.002)
    assert result["slope_delta_abs_mean"] == pytest.approx(0.001)
    assert result["r2_delta_abs_mean"] == pytest.approx(0.001)
