from __future__ import annotations

from experiments.randomized_sort_benchmark.randomization import (
    build_matched_randomization,
    classify_application,
)
from sorting_core.synthetic_data import load_synthetic_data_spec


def test_independent_seed_domains_are_combined_by_consumer() -> None:
    spec = load_synthetic_data_spec()
    matched = build_matched_randomization(
        spec,
        condition_id="condition-test",
        appearance_seed=11,
        physics_seed=22,
        sensor_seed=33,
    )
    assert matched.combined.value("materials.table_roughness") == matched.appearance.value(
        "materials.table_roughness"
    )
    assert matched.combined.value("physics.object_static_friction") == matched.physics.value(
        "physics.object_static_friction"
    )
    assert matched.combined.value("sensor.exposure_stops") == matched.sensor.value(
        "sensor.exposure_stops"
    )
    assert matched.combined.sensor_noise_seed == matched.sensor.sensor_noise_seed
    assert len(set(matched.as_dict()["source_manifest_sha256"].values())) == 3


def test_application_ledger_replaces_legacy_fixed_scene_parameters() -> None:
    spec = load_synthetic_data_spec()
    matched = build_matched_randomization(
        spec,
        condition_id="condition-test",
        appearance_seed=11,
        physics_seed=22,
        sensor_seed=33,
    )
    applied = {
        name
        for name in matched.combined.parameters
        if not name.startswith("scene.")
    }
    ledger = classify_application(matched.combined, applied=sorted(applied))
    assert ledger["all_parameters_classified"] is True
    assert all(
        entry["application_status"] == "skipped"
        for name, entry in ledger["parameters"].items()
        if name.startswith("scene.")
    )
