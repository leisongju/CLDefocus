from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from pdraw_adapter.multi_prescription_plan import (
    LensParaxialRecord,
    _parse_paraxial_record,
    build_device_atom_plan,
    prepare,
    select_lens_prescriptions,
)


def _record(
    relative_path: str,
    *,
    f_number: float = 1.4,
    focal_length_m: float = 0.016,
    image_radius_m: float = 0.014,
    surfaces: int = 20,
    aspheres: int = 2,
) -> LensParaxialRecord:
    return LensParaxialRecord(
        relative_path=relative_path,
        lens_id=Path(relative_path).stem,
        sha256=(Path(relative_path).stem[0] * 64).lower(),
        native_f_number=f_number,
        effective_focal_length_m=focal_length_m,
        image_radius_m=image_radius_m,
        back_focal_position_m=0.05,
        entrance_pupil_radius_m=focal_length_m / (2.0 * f_number),
        exit_pupil_radius_m=0.01,
        stop_radius_m=0.005,
        total_track_m=0.06,
        surface_count=surfaces,
        aspheric_surface_count=aspheres,
        stop_index=max(1, surfaces // 2),
        aperture_stop_scales=tuple(f_number / value for value in (1.8, 2.0, 2.8, 4.0, 5.6)),
    )


def _config(tmp_path: Path) -> dict:
    source_root = tmp_path / "lenses"
    plan_root = tmp_path / "plan"
    return {
        "schema_version": 1,
        "run": {"id": "fixture", "purpose": "测试", "mode": "cpu_prepare_only"},
        "source": {
            "root": str(source_root),
            "include_globs": ["prime/*.mytable"],
            "expected_file_count": 4,
        },
        "selection": {
            "policy": "reference_plus_deterministic_compatible_maximin_v1",
            "reference_relative_path": "prime/ref.mytable",
            "selected_count": 4,
            "native_f_number_max": 1.8,
            "effective_focal_length_ratio_range": [0.5, 2.0],
            "image_radius_ratio_range": [0.5, 2.0],
            "compatibility_penalty": 0.0,
            "expected_selected_relative_paths": [
                "prime/ref.mytable",
                "prime/a.mytable",
                "prime/b.mytable",
                "prime/c.mytable",
            ],
            "expected_selected_sha256": {
                "prime/ref.mytable": "r" * 64,
                "prime/a.mytable": "a" * 64,
                "prime/b.mytable": "b" * 64,
                "prime/c.mytable": "c" * 64,
            },
        },
        "optics": {
            "wavelength_m": 5.875618e-7,
            "f_numbers": [1.8, 2.0, 2.8, 4.0, 5.6],
            "signed_coc_bins_px": [-1.0, 0.0, 1.0],
            "field_x_normalized": [-0.65, 0.0, 0.65],
            "field_y_normalized": [-0.65, 0.0, 0.65],
        },
        "response_profile": {"profile_id": "HEAD0"},
        "device_atom": {
            "sensor_head_id": "HEAD0",
            "dcc_px_per_coc": 1.0 / 6.0,
            "coc_zero_anchor": "preserve_native_optical_bias",
            "pdoffset_embedded": False,
        },
        "pair_common": {"transition_coc_px": 16.0},
        "gates": {"numpy_jax_relative_error_max": 1.0e-9},
        "output": {
            "plan_root": str(plan_root),
            "gpu_artifact_root": str(tmp_path / "gpu"),
        },
        "data_isolation": {
            "real_pdraw_accessed": False,
            "google_dev_accessed": False,
            "google_holdout_accessed": False,
            "dp5k_accessed": False,
            "stereo_training_run": False,
        },
    }


def test_reference_numpy_paraxial_matches_frozen_jp2018_values() -> None:
    root = Path("/mnt/data/lsj/data/optics/cldefocus/lens-mytable-original")
    record = _parse_paraxial_record(
        root / "prime/JP2018-205527_Example01P.mytable",
        root=root,
        wavelength_m=5.875618e-7,
        requested_f_numbers=[1.8, 2.0, 2.8, 4.0, 5.6],
    )
    assert record.native_f_number == pytest.approx(1.3253461158063504, rel=1.0e-12)
    assert record.effective_focal_length_m == pytest.approx(
        0.016446230957878647, rel=1.0e-12
    )
    assert record.stop_index == 18
    assert max(record.aperture_stop_scales) < 1.0


def test_selection_is_deterministic_and_keeps_reference_first() -> None:
    records = [
        _record("prime/ref.mytable"),
        _record("prime/a.mytable", surfaces=12, aspheres=0),
        _record("prime/b.mytable", focal_length_m=0.022, surfaces=30, aspheres=5),
        _record("prime/c.mytable", focal_length_m=0.011, image_radius_m=0.010),
        _record("prime/reject.mytable", focal_length_m=0.05),
    ]
    first, compatible = select_lens_prescriptions(
        records,
        reference_relative_path="prime/ref.mytable",
        selected_count=4,
        efl_ratio_range=[0.6, 1.5],
        image_radius_ratio_range=[0.6, 1.5],
        native_f_number_max=1.8,
        compatibility_penalty=0.35,
    )
    second, _ = select_lens_prescriptions(
        records,
        reference_relative_path="prime/ref.mytable",
        selected_count=4,
        efl_ratio_range=[0.6, 1.5],
        image_radius_ratio_range=[0.6, 1.5],
        native_f_number_max=1.8,
        compatibility_penalty=0.35,
    )
    assert [row.relative_path for row in first] == [row.relative_path for row in second]
    assert first[0].relative_path == "prime/ref.mytable"
    assert len(compatible) == 4


def test_device_atom_contract_rejects_embedded_pdoffset(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["device_atom"]["pdoffset_embedded"] = True
    with pytest.raises(ValueError, match="PDOFFSET"):
        build_device_atom_plan([_record("prime/ref.mytable")], config)


def test_prepare_writes_frozen_cpu_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    source_root = Path(config["source"]["root"])
    for relative_path in config["selection"]["expected_selected_relative_paths"]:
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative_path, encoding="utf-8")
    records = {
        relative_path: _record(relative_path)
        for relative_path in config["selection"]["expected_selected_relative_paths"]
    }

    def fake_parse(path: Path, **_: object) -> LensParaxialRecord:
        return records[path.relative_to(source_root).as_posix()]

    monkeypatch.setattr(
        "pdraw_adapter.multi_prescription_plan._parse_paraxial_record", fake_parse
    )
    monkeypatch.setattr(
        "pdraw_adapter.multi_prescription_plan._jax_cpu_validate",
        lambda selected, **_: [
            {"lens_id": row.lens_id, "pass": True} for row in selected
        ],
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    result = prepare(config_path)
    assert result["status"] == "cpu_prepared_pending_gpu_propagation"
    manifest = json.loads(
        (Path(config["output"]["plan_root"]) / "artifact_manifest.json").read_text()
    )
    assert manifest["gpu_propagation_started"] is False
    assert manifest["training_admitted"] is False
    assert manifest["pdoffset_embedded"] is False
    assert manifest["device_atom_count"] == 4
