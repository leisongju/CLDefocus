from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from pdraw_adapter.multi_prescription_execute import (
    _CANDIDATE_ID,
    _write_aggregate,
    build_audit_config,
    build_candidate_config,
    build_family_config,
    load_verified_plan,
    _screen_atom,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _isolation() -> dict[str, bool]:
    return {
        "real_pdraw_accessed": False,
        "google_dev_accessed": False,
        "google_holdout_accessed": False,
        "dp5k_accessed": False,
        "stereo_training_run": False,
    }


def _atom(tmp_path: Path) -> dict:
    root = tmp_path / "gpu" / "MP00_lens_DCC6"
    return {
        "atom_id": "MP00_lens_DCC6",
        "lens_id": "lens",
        "lens_relative_path": "prime/lens.mytable",
        "lens_sha256": "a" * 64,
        "shared_lineage_id": "fixture::lens::dcc6_head0",
        "shared_lineage_across_apertures": True,
        "f_numbers": [1.8, 2.0, 2.8, 4.0, 5.6],
        "signed_coc_bins_px": [-8, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8],
        "field_x_normalized": [-0.65, 0, 0.65],
        "field_y_normalized": [-0.65, 0, 0.65],
        "response_profile": {
            "transition_width": 0.44,
            "cross_talk": 0.025,
            "split_bias_norm": 0,
            "boundary_curvature_norm": 0,
            "microlens_edge_rolloff": 0.06,
            "field_split_slope": 0,
            "field_acceptance_slope": 0,
            "lr_throughput_delta": 0,
            "field_lr_throughput_slope": 0,
            "aperture_lr_throughput_slope": 0,
            "low_order_aberration_waves": {
                "astigmatism_0": 0,
                "astigmatism_45": 0,
                "coma_x": 0,
                "coma_y": 0,
                "spherical": 0,
                "field_astigmatism_0_per_norm": 0,
                "field_astigmatism_45_per_norm": 0,
                "field_coma_x_per_norm": 0,
                "field_coma_y_per_norm": 0,
            },
        },
        "dcc_px_per_coc": 1 / 6,
        "pair_common": {
            "alignment": "direct_mean",
            "common_shape_mtf_sigma_px": 1,
            "clipped_negative_fraction_max": 0.02,
            "transition_coc_px": 16,
            "transition_power": 1,
        },
        "pdoffset_embedded": False,
        "planned_outputs": {
            "raw_family_root": str(root / "raw_family"),
            "raw_optical_root": str(root / "raw_optical"),
            "k64_audit_root": str(root / "audit_k64"),
            "k128_audit_root": str(root / "audit_k128"),
            "final_asset_root": str(root / "paircommon_dcc6"),
        },
    }


def _config(tmp_path: Path, atom: dict) -> dict:
    plan_root = tmp_path / "plan"
    plan_root.mkdir()
    plan = {"atoms": [atom]}
    plan_path = plan_root / "device_atom_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    manifest = {
        "status": "cpu_prepared_pending_gpu_propagation",
        "plan_id": "fixture-plan",
        "gpu_propagation_started": False,
        "training_admitted": False,
        "pdoffset_embedded": False,
        "device_atom_count": 1,
        "files": {"device_atom_plan.json": _sha(plan_path)},
    }
    manifest_path = plan_root / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "run": {"id": "fixture"},
        "source_plan": {
            "root": str(plan_root),
            "artifact_manifest_sha256": _sha(manifest_path),
            "device_atom_plan_sha256": _sha(plan_path),
        },
        "cldefocus": {
            "repository": "repo",
            "checkout": str(tmp_path / "checkout"),
            "commit": "commit",
            "license": "MIT",
        },
        "lens": {"root": str(tmp_path / "lenses")},
        "parent": {
            "repository": str(tmp_path / "parent"),
            "loader_sha256": "b" * 64,
            "renderer_sha256": "c" * 64,
        },
        "propagation": {
            "wavelength_m": 5.875618e-7,
            "object_depth_m": 1000,
            "sensor_width_px": 2560,
            "sensor_height_px": 2560,
            "pixel_pitch_m": 1e-5,
            "sensor_offset_envelope_m": 0.0015,
            "target_coc_tolerance_px": 1e-6,
            "reference_kernel_size": 57,
            "export_kernel_size": 49,
            "upsample": 1,
            "batch_sensor_chunk_size": 32,
            "master_seed": 1,
            "profile_id_prefix": "H",
            "pdoffset_sample_range_px": [-0.25, 0.25],
            "pdoffset_smoke_probe_px": 0.125,
        },
        "raw_family_gates": {
            "energy_abs_max": 2e-6,
            "raw_support_border_width_px": 2,
            "raw_support_inner_mass_min": 0.98,
        },
        "audit": {
            "texture_size_by_tile": {"64": 256, "128": 384},
            "coc_abs_max_px": 1,
            "seeds": [1],
            "tiles_per_axis": 1,
            "search_x": 4,
            "search_y": 2,
            "lanczos_radius": 24,
            "refinement_radius_px": 0.999,
            "coordinate_iterations": 1,
            "optimizer_xatol_px": 1e-5,
            "min_texture_std": 0.008,
            "min_score": 0.2,
            "max_lr_error_px": 0.6,
            "max_vertical_shift_px": 0.75,
            "peak_exclusion_x": 2,
            "peak_exclusion_y": 1,
            "final_score_tolerance": 1e-6,
            "worker_count": 4,
            "gates": {"slope_min": 0.95, "slope_max": 1.05},
        },
        "candidate_gates": {
            "centroid_abs_max_error_px": 0.002,
            "parent_centroid_abs_max_error_px": 1e-6,
            "retained_mass_min": 0.98,
        },
        "output": {"root": str(tmp_path / "gpu")},
        "data_isolation": _isolation(),
    }


def test_verified_plan_and_family_config_preserve_lineage(tmp_path: Path) -> None:
    atom = _atom(tmp_path)
    config = _config(tmp_path, atom)
    _, atoms = load_verified_plan(config)
    child = build_family_config(config, atoms[0])
    assert child["family"]["family_id"] == atom["shared_lineage_id"]
    assert child["family"]["randomization"]["profile_count"] == 1
    assert child["family"]["randomization"]["ranges"]["cross_talk"] == [0.025, 0.025]
    assert child["gates"]["expected_aperture_count"] == 5
    assert child["data_isolation"] == _isolation()

    config["propagation"]["sensor_offset_envelope_m_by_atom"] = {
        atom["atom_id"]: 0.004
    }
    overridden = build_family_config(config, atoms[0])
    assert overridden["optics"]["sensor_offset_envelope_m"] == 0.004

    screen = _screen_atom(atoms[0])
    screen_child = build_family_config(config, screen)
    assert screen_child["apertures"]["f_numbers"] == [1.8]
    assert screen_child["optics"]["target_signed_coc_bins_px"] == [
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
    ]
    assert screen_child["gates"]["expected_coc_count"] == 5


def test_audit_and_candidate_freeze_dcc6_paircommon(tmp_path: Path) -> None:
    atom = _atom(tmp_path)
    config = _config(tmp_path, atom)
    k64 = build_audit_config(
        config, atom, raw_manifest_sha256="d" * 64, tile_size=64
    )
    assert k64["source"]["profile_ids"] == ["H000"]
    assert k64["screen"]["field_mode"] == "full_flattened"
    assert k64["screen"]["tile_size"] == 64
    assert k64["candidates"][0]["id"] == _CANDIDATE_ID
    operations = k64["candidates"][0]["operations"]
    assert operations[0]["anchor_mode"] == "preserve_native_coc0"
    assert operations[0]["target_slopes_by_aperture"] == [1 / 6] * 5
    assert operations[1]["translation_mode"] == "bilinear_center_fourier_place"

    audit_rows = [
        {
            "tile_size": tile,
            "root": f"/audit/{tile}",
            "summary_sha256": "a" * 64,
            "provenance_sha256": "b" * 64,
            "screens_sha256": "c" * 64,
        }
        for tile in (64, 128)
    ]
    candidate = build_candidate_config(
        config,
        atom,
        raw_manifest_sha256="d" * 64,
        audit_rows=audit_rows,
    )
    assert [row["tile_size"] for row in candidate["audits"]] == [64, 128]
    assert candidate["candidate"]["operations"] == operations
    assert candidate["data_isolation"] == _isolation()


def test_aggregate_never_admits_partial_atoms(tmp_path: Path, monkeypatch) -> None:
    atom = _atom(tmp_path)
    config = _config(tmp_path, atom)
    checkout = Path(config["cldefocus"]["checkout"])
    checkout.mkdir()
    config_path = tmp_path / "execute.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        "pdraw_adapter.multi_prescription_execute._git_output",
        lambda *_args: "fixture",
    )
    result = _write_aggregate(
        config_path,
        config,
        {"plan_id": "fixture-plan"},
        [atom],
    )
    assert result["status"] == "in_progress_not_admitted"
    assert result["device_atom_pass_count"] == 0
    assert result["training_admitted"] is False
    manifest = json.loads(
        (Path(config["output"]["root"]) / "artifact_manifest.json").read_text()
    )
    assert manifest["training_admitted"] is False
    assert manifest["pdoffset_embedded"] is False
