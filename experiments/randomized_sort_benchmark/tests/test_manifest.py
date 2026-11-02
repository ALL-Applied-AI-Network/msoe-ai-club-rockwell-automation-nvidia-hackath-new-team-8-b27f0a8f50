from __future__ import annotations

import json

import pytest

from experiments.randomized_sort_benchmark.manifest import (
    BenchmarkManifest,
    BenchmarkManifestError,
    PickupRegion,
    generate_condition,
    generate_matched_suite,
    read_manifest,
    write_manifest,
)


def test_condition_is_reproducible_and_seed_domains_are_separate() -> None:
    first = generate_condition(27, object_count=10)
    second = generate_condition(27, object_count=10)

    assert first == second
    assert first.sha256 == second.sha256
    assert len(set(first.seeds.as_dict().values())) == 5
    assert first.object_count == 10


def test_random_count_stays_in_supported_range_and_destinations_are_correct() -> None:
    seen_counts: set[int] = set()
    seen_classes: set[str] = set()
    for seed in range(80):
        manifest = generate_condition(seed)
        seen_counts.add(manifest.object_count)
        for obj in manifest.objects:
            seen_classes.add(obj.class_name)
            if obj.class_name == "screw_iso10642_m6x30":
                assert obj.destination_colour == "red"
            else:
                assert obj.destination_colour == obj.colour

    assert seen_counts == set(range(1, 11))
    assert seen_classes == {"cube_10mm", "screw_iso10642_m6x30"}


def test_all_spawn_footprints_are_validated_in_pickup_region() -> None:
    for seed in range(30):
        generate_condition(seed, object_count=10).validate()


def test_default_suite_has_40_conditions_and_exact_count_coverage() -> None:
    suite = generate_matched_suite(20261101)

    assert len(suite.conditions) == 40
    assert len({condition.sha256 for condition in suite.conditions}) == 40
    for object_count in range(1, 11):
        assert sum(c.object_count == object_count for c in suite.conditions) == 4
    # The four policy combinations replay this suite for 160 matched runs.
    assert len(suite.conditions) * 4 == 160


def test_manifest_round_trip_and_hash_tamper_detection(tmp_path) -> None:
    expected = generate_condition(51, object_count=6)
    path = tmp_path / "condition.json"
    write_manifest(path, expected)
    assert read_manifest(path) == expected

    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["manifest"]["objects"][0]["yaw_deg"] += 1.0
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(BenchmarkManifestError, match="SHA-256 mismatch"):
        read_manifest(path)


def test_parser_rejects_count_mismatch_and_out_of_region_object() -> None:
    manifest = generate_condition(12, object_count=3)
    payload = manifest.as_dict()
    payload["object_count"] = 4
    with pytest.raises(BenchmarkManifestError, match="object_count"):
        BenchmarkManifest.from_dict(payload)

    payload = manifest.as_dict()
    payload["objects"][0]["position_base_m"][0] = 9.0
    with pytest.raises(BenchmarkManifestError, match="leaves the pickup region"):
        BenchmarkManifest.from_dict(payload)


def test_generator_rejects_invalid_inputs() -> None:
    with pytest.raises(BenchmarkManifestError, match=r"\[1, 10\]"):
        generate_condition(1, object_count=0)
    with pytest.raises(BenchmarkManifestError, match="increasing"):
        generate_condition(1, pickup_region=PickupRegion((1.0, 0.0), (-1.0, 1.0)))
    with pytest.raises(BenchmarkManifestError, match="non-negative"):
        generate_condition(-1)
