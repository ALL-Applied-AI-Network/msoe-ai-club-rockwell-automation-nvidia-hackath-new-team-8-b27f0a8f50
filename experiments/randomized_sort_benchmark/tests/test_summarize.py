from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.randomized_sort_benchmark.manifest import (
    BenchmarkManifest,
    generate_condition,
    write_manifest,
)
from experiments.randomized_sort_benchmark.summarize import CELLS, build_summary


def _capacity_stress_condition() -> BenchmarkManifest:
    for seed in range(10_000):
        manifest = generate_condition(seed, object_count=6)
        screws = sum(
            obj.class_name == "screw_iso10642_m6x30" for obj in manifest.objects
        )
        if screws > 2 and any(obj.class_name == "cube_10mm" for obj in manifest.objects):
            return manifest
    raise AssertionError("could not find deterministic mixed capacity-stress fixture")


def _write_suite(root: Path, manifests: list[BenchmarkManifest]) -> Path:
    manifest_root = root / "manifests"
    manifest_root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for index, manifest in enumerate(manifests):
        filename = f"{index:04d}-{manifest.condition_id}.json"
        write_manifest(manifest_root / filename, manifest)
        rows.append(
            {
                "index": index,
                "condition_id": manifest.condition_id,
                "object_count": manifest.object_count,
                "manifest_relpath": f"manifests/{filename}",
                "manifest_sha256": manifest.sha256,
            }
        )
    payload = {
        "schema_version": "randomized-sort-suite-index-v1",
        "base_seed": 123,
        "trials_per_object_count": 1,
        "full_suite_sha256": "fixture",
        "selected_counts": sorted({manifest.object_count for manifest in manifests}),
        "condition_count": len(rows),
        "planned_policy_cells": list(CELLS),
        "planned_end_to_end_runs": len(rows) * len(CELLS),
        "conditions": rows,
    }
    path = root / "suite_index.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _runtime_provenance(token: str = "frozen") -> dict[str, Any]:
    return {
        "comparison_key": {
            "repository_commit": token,
            "isaac_version": "6.0.1",
            "synthetic_data_spec_sha256": "spec-sha",
            "camera_calibration_sha256": "calibration-sha",
        },
        "execution_host": {"brev_instance": "intentionally-ignored"},
    }


def _write_report(
    runs: Path,
    cell: str,
    manifest: BenchmarkManifest,
    *,
    correct_count: int,
    randomization_hash: str | None = None,
    runtime_token: str = "frozen",
) -> None:
    correct_ids = [obj.instance_id for obj in manifest.objects[:correct_count]]
    run = runs / cell / manifest.condition_id
    run.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "randomized-sort-episode-v1",
        "condition_id": manifest.condition_id,
        "condition_manifest_sha256": manifest.sha256,
        "policy_cell": {"id": cell},
        "realistic_mode": True,
        "runtime_provenance": _runtime_provenance(runtime_token),
        "randomization_manifest_sha256": (
            randomization_hash or f"randomization-{manifest.sha256}"
        ),
        "object_count": manifest.object_count,
        "correctly_sorted_object_ids": correct_ids,
        "correctly_sorted_objects": len(correct_ids),
        "complete_scene_success": len(correct_ids) == manifest.object_count,
        "simulated_episode_time_s": 30.0 + CELLS.index(cell),
        "wall_time_s": 4.0,
        "events": [],
        "vision_reports": [
            {
                "status": "accepted",
                "total_latency_ms": 5.0 + CELLS.index(cell),
            }
        ],
    }
    (run / "episode_report.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_failure(runs: Path, cell: str, manifest: BenchmarkManifest) -> None:
    run = runs / cell / manifest.condition_id
    run.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "randomized-sort-episode-failure-v1",
        "condition_id": manifest.condition_id,
        "condition_manifest_sha256": manifest.sha256,
        "error_type": "RuntimeError",
        "error_message": "fixture infrastructure failure",
    }
    (run / "episode_failure.json").write_text(json.dumps(payload), encoding="utf-8")


def test_complete_matrix_emits_all_pairs_cluster_intervals_and_inventory(tmp_path) -> None:
    manifests = [generate_condition(101, object_count=2), _capacity_stress_condition()]
    suite = _write_suite(tmp_path / "suite", manifests)
    runs = tmp_path / "runs"
    correct_by_cell = {"DD": 0, "YD": 1, "DA": 1, "YA": 2}
    for cell in CELLS:
        for manifest in manifests:
            _write_report(
                runs,
                cell,
                manifest,
                correct_count=min(correct_by_cell[cell], manifest.object_count),
            )

    result = build_summary(runs, suite)

    assert result["matched_conclusion_eligible"] is True
    assert result["matrix_audit"]["expected_run_count"] == 8
    assert result["matrix_audit"]["runtime_suite_frozen"] is True
    assert set(result["all_pairwise_comparisons"]) == {
        "DD__YD",
        "DD__DA",
        "DD__YA",
        "YD__DA",
        "YD__YA",
        "DA__YA",
    }
    assert all(
        row["conclusion_eligible"]
        for row in result["all_pairwise_comparisons"].values()
    )
    assert result["factorial_interactions"] is not None
    dd = result["cell_summaries"][0]
    assert dd["object_completion_cluster_bootstrap_95"] is not None
    assert dd["throughput_cluster_bootstrap_95"] is not None
    assert (
        dd["completion_by_class"]["screw_iso10642_m6x30"]
        ["planned_reference_opportunities"]
        > 2
    )
    assert dd["capacity_strata"]["capacity_stress_candidate"]["completed_runs"] == 1


def test_failures_and_missing_runs_cannot_produce_matched_conclusions(tmp_path) -> None:
    manifest = generate_condition(202, object_count=3)
    suite = _write_suite(tmp_path / "suite", [manifest])
    runs = tmp_path / "runs"
    _write_report(runs, "DD", manifest, correct_count=2)
    _write_failure(runs, "YD", manifest)

    result = build_summary(runs, suite)

    assert result["matched_conclusion_eligible"] is False
    assert result["matrix_audit"]["cell_status"]["YD"][
        "infrastructure_failure_runs"
    ] == 1
    assert result["matrix_audit"]["cell_status"]["DA"]["missing_runs"] == 1
    assert len(result["infrastructure_failures"]) == 1
    assert all(
        row["estimates"] is None
        for row in result["all_pairwise_comparisons"].values()
    )
    dd = result["cell_summaries"][0]
    assert dd["object_completion_fraction_all_planned_conservative"] == 2 / 3
    assert dd["completion_by_class"]


def test_randomization_or_runtime_mismatch_invalidates_full_matrix(tmp_path) -> None:
    manifest = generate_condition(303, object_count=2)
    suite = _write_suite(tmp_path / "suite", [manifest])
    runs = tmp_path / "runs"
    for cell in CELLS:
        _write_report(
            runs,
            cell,
            manifest,
            correct_count=1,
            randomization_hash=(
                "different-randomization"
                if cell == "YA"
                else f"randomization-{manifest.sha256}"
            ),
            runtime_token="different-runtime" if cell == "DA" else "frozen",
        )

    result = build_summary(runs, suite)

    assert result["matrix_audit"]["matrix_complete"] is True
    assert result["matrix_audit"]["provenance_complete"] is False
    assert manifest.condition_id in result["matrix_audit"][
        "randomization_hash_mismatches"
    ]
    assert manifest.condition_id in result["matrix_audit"][
        "runtime_provenance_mismatches"
    ]
    assert result["matched_conclusion_eligible"] is False


def test_missing_runtime_provenance_is_explicitly_reported(tmp_path) -> None:
    manifest = generate_condition(404, object_count=1)
    suite = _write_suite(tmp_path / "suite", [manifest])
    runs = tmp_path / "runs"
    for cell in CELLS:
        _write_report(runs, cell, manifest, correct_count=1)
    path = runs / "YA" / manifest.condition_id / "episode_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    del report["runtime_provenance"]
    path.write_text(json.dumps(report), encoding="utf-8")

    result = build_summary(runs, suite)

    assert result["matrix_audit"]["runtime_provenance_missing"] == [
        {"cell": "YA", "condition_id": manifest.condition_id}
    ]
    assert result["matched_conclusion_eligible"] is False
