"""Audit and compare the four matched randomized benchmark policy cells.

The suite index is authoritative. A partial or failed run matrix is useful
diagnostic evidence, but it is not silently promoted into a matched policy
comparison. Every inferential interval resamples whole scene conditions so the
several objects in one scene are not treated as independent trials.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

from experiments.randomized_sort_benchmark.manifest import (
    BenchmarkManifest,
    read_manifest,
)
from experiments.randomized_sort_benchmark.randomization import write_json_atomic


CELLS = ("DD", "YD", "DA", "YA")
PAIRWISE_CELL_KEYS = tuple(combinations(CELLS, 2))
BOOTSTRAP_SAMPLES = 5_000


def _canonical_sha256(value: Any) -> str:
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    """Wilson interval for genuinely independent Bernoulli scene outcomes."""

    if total <= 0:
        return None
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def _percentile_interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    values.sort()
    return [
        values[int(0.025 * (len(values) - 1))],
        values[int(0.975 * (len(values) - 1))],
    ]


def _cluster_ratio_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    numerator: Callable[[Mapping[str, Any]], float],
    denominator: Callable[[Mapping[str, Any]], float],
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
) -> list[float] | None:
    """Bootstrap a ratio while resampling complete condition/scene clusters."""

    if not rows:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        selected = [rows[rng.randrange(len(rows))] for _ in rows]
        total_denominator = sum(denominator(row) for row in selected)
        if total_denominator > 0.0:
            values.append(sum(numerator(row) for row in selected) / total_denominator)
    return _percentile_interval(values)


def _paired_bootstrap(
    pairs: Sequence[tuple[float, float]],
    *,
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
) -> list[float] | None:
    if not pairs:
        return None
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(samples):
        selected = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        differences.append(mean(right - left for left, right in selected))
    return _percentile_interval(differences)


def _mean_bootstrap(
    values: Sequence[float],
    *,
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    estimates = [
        mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(samples)
    ]
    return _percentile_interval(estimates)


def _capacity_classification(manifest: BenchmarkManifest) -> dict[str, Any]:
    class_counts = Counter(obj.class_name for obj in manifest.objects)
    destination_counts = Counter(obj.destination_colour for obj in manifest.objects)
    screw_count = class_counts["screw_iso10642_m6x30"]
    reasons: list[str] = []
    # The benchmark's screw release layout has two alternating slots. More
    # than two screws therefore guarantees commanded-slot reuse, independent
    # of action order. This is a capacity-stress *candidate*, not proof that
    # the physical bin is full.
    if screw_count > 2:
        reasons.append("more_than_two_screws_reuse_two_commanded_screw_slots")
    if max(destination_counts.values(), default=0) > 9:
        reasons.append("destination_load_exceeds_nine_cube_grid_slots")
    return {
        "capacity_stress_candidate": bool(reasons),
        "capacity_stress_reasons": reasons,
        "screw_count": screw_count,
        "destination_counts": dict(sorted(destination_counts.items())),
    }


def _load_suite(suite_path: Path) -> dict[str, Any]:
    raw = json.loads(suite_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "randomized-sort-suite-index-v1":
        raise ValueError(f"unsupported suite index: {suite_path}")
    planned_cells = tuple(raw.get("planned_policy_cells", ()))
    if set(planned_cells) != set(CELLS) or len(planned_cells) != len(CELLS):
        raise ValueError(f"suite must plan exactly the four cells {CELLS!r}")
    raw_rows = raw.get("conditions")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("suite index must contain at least one condition")
    if int(raw.get("condition_count", -1)) != len(raw_rows):
        raise ValueError("suite condition_count does not match conditions")

    conditions: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        condition_id = str(row.get("condition_id", "")).strip()
        if not condition_id or condition_id in conditions:
            raise ValueError(f"invalid or duplicate suite condition {condition_id!r}")
        manifest_path = (suite_path.parent / str(row["manifest_relpath"])).resolve()
        manifest = read_manifest(manifest_path)
        expected_hash = str(row.get("manifest_sha256", ""))
        if manifest.condition_id != condition_id:
            raise ValueError(f"condition ID mismatch in {manifest_path}")
        if manifest.sha256 != expected_hash:
            raise ValueError(f"manifest hash mismatch in suite index for {condition_id}")
        if manifest.object_count != int(row.get("object_count", -1)):
            raise ValueError(f"object count mismatch in suite index for {condition_id}")
        conditions[condition_id] = {
            "condition_id": condition_id,
            "manifest_sha256": manifest.sha256,
            "manifest_path": str(manifest_path),
            "manifest": manifest,
            "capacity": _capacity_classification(manifest),
        }
    return {
        "path": str(suite_path),
        "sha256": _canonical_sha256(raw),
        "base_seed": raw.get("base_seed"),
        "planned_policy_cells": list(CELLS),
        "conditions": conditions,
    }


def _load_artifacts(root: Path) -> dict[str, Any]:
    reports: dict[str, dict[str, dict[str, Any]]] = {cell: {} for cell in CELLS}
    failures: dict[str, dict[str, dict[str, Any]]] = {cell: {} for cell in CELLS}
    conflicts: list[dict[str, Any]] = []
    for cell in CELLS:
        cell_root = root / cell
        for report_path in cell_root.glob("*/episode_report.json"):
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("policy_cell", {}).get("id") != cell:
                raise ValueError(f"cell mismatch in {report_path}")
            condition_id = str(report.get("condition_id", "")).strip()
            if not condition_id or condition_id in reports[cell]:
                raise ValueError(f"duplicate/invalid report {cell}:{condition_id}")
            report["_path"] = str(report_path)
            reports[cell][condition_id] = report
        for failure_path in cell_root.glob("*/episode_failure.json"):
            row = json.loads(failure_path.read_text(encoding="utf-8"))
            condition_id = str(row.get("condition_id", "")).strip()
            if not condition_id or condition_id in failures[cell]:
                raise ValueError(f"duplicate/invalid failure {cell}:{condition_id}")
            row["cell"] = cell
            row["path"] = str(failure_path)
            failures[cell][condition_id] = row
        for condition_id in sorted(set(reports[cell]) & set(failures[cell])):
            conflicts.append(
                {
                    "cell": cell,
                    "condition_id": condition_id,
                    "report_path": reports[cell][condition_id]["_path"],
                    "failure_path": failures[cell][condition_id]["path"],
                }
            )
    return {"reports": reports, "failures": failures, "conflicts": conflicts}


def _runtime_comparison_key(report: Mapping[str, Any]) -> Mapping[str, Any] | None:
    provenance = report.get("runtime_provenance")
    if not isinstance(provenance, Mapping):
        return None
    comparison_key = provenance.get("comparison_key")
    if isinstance(comparison_key, Mapping) and comparison_key:
        return dict(comparison_key)
    # Compatibility for a flat mapping. Host identity and GPU metadata are
    # excluded because multiple equivalent L40S instances may run the cells.
    ignored = {
        "execution_host",
        "host",
        "hostname",
        "brev_instance",
        "gpu",
        "gpu_name",
        "gpu_uuid",
        "started_at",
    }
    stable = {str(key): value for key, value in provenance.items() if key not in ignored}
    return stable or None


def _matrix_audit(
    suite: Mapping[str, Any], artifacts: Mapping[str, Any]
) -> dict[str, Any]:
    conditions = suite["conditions"]
    expected_ids = set(conditions)
    reports = artifacts["reports"]
    failures = artifacts["failures"]
    cell_status: dict[str, Any] = {}
    unexpected: list[dict[str, Any]] = []
    hash_mismatches: list[dict[str, Any]] = []
    realism_issues: list[dict[str, Any]] = []
    runtime_missing: list[dict[str, str]] = []
    runtime_hashes_by_condition: dict[str, set[str]] = defaultdict(set)
    randomization_missing: list[dict[str, str]] = []
    randomization_hashes_by_condition: dict[str, set[str]] = defaultdict(set)

    for cell in CELLS:
        report_ids = set(reports[cell])
        failure_ids = set(failures[cell])
        for condition_id in sorted((report_ids | failure_ids) - expected_ids):
            unexpected.append(
                {
                    "cell": cell,
                    "condition_id": condition_id,
                    "artifact": "report" if condition_id in report_ids else "failure",
                }
            )
        missing = expected_ids - report_ids - failure_ids
        cell_status[cell] = {
            "expected_runs": len(expected_ids),
            "completed_runs": len(report_ids & expected_ids),
            "infrastructure_failure_runs": len(failure_ids & expected_ids),
            "missing_runs": len(missing),
            "completed_condition_ids": sorted(report_ids & expected_ids),
            "infrastructure_failure_condition_ids": sorted(failure_ids & expected_ids),
            "missing_condition_ids": sorted(missing),
        }
        for condition_id in sorted(report_ids & expected_ids):
            report = reports[cell][condition_id]
            expected_hash = conditions[condition_id]["manifest_sha256"]
            if report.get("condition_manifest_sha256") != expected_hash:
                hash_mismatches.append(
                    {
                        "cell": cell,
                        "condition_id": condition_id,
                        "expected": expected_hash,
                        "observed": report.get("condition_manifest_sha256"),
                    }
                )
            if report.get("realistic_mode") is not True:
                realism_issues.append(
                    {
                        "cell": cell,
                        "condition_id": condition_id,
                        "reason": "realistic_mode_not_true",
                    }
                )
            randomization_hash = report.get("randomization_manifest_sha256")
            if not isinstance(randomization_hash, str) or not randomization_hash:
                randomization_missing.append({"cell": cell, "condition_id": condition_id})
            else:
                randomization_hashes_by_condition[condition_id].add(randomization_hash)
            runtime_key = _runtime_comparison_key(report)
            if runtime_key is None:
                runtime_missing.append({"cell": cell, "condition_id": condition_id})
            else:
                runtime_hashes_by_condition[condition_id].add(_canonical_sha256(runtime_key))

    randomization_mismatches = {
        condition_id: sorted(values)
        for condition_id, values in randomization_hashes_by_condition.items()
        if len(values) != 1
    }
    runtime_mismatches = {
        condition_id: sorted(values)
        for condition_id, values in runtime_hashes_by_condition.items()
        if len(values) != 1
    }
    global_runtime_hashes = sorted(
        {value for values in runtime_hashes_by_condition.values() for value in values}
    )
    matrix_complete = (
        not artifacts["conflicts"]
        and not unexpected
        and all(
            row["completed_runs"] == row["expected_runs"]
            and row["infrastructure_failure_runs"] == 0
            and row["missing_runs"] == 0
            for row in cell_status.values()
        )
    )
    provenance_complete = not any(
        (
            hash_mismatches,
            realism_issues,
            randomization_missing,
            randomization_mismatches,
            runtime_missing,
            runtime_mismatches,
        )
    ) and len(global_runtime_hashes) == 1
    return {
        "expected_condition_count": len(expected_ids),
        "expected_run_count": len(expected_ids) * len(CELLS),
        "cell_status": cell_status,
        "artifact_conflicts": artifacts["conflicts"],
        "unexpected_artifacts": unexpected,
        "condition_manifest_hash_mismatches": hash_mismatches,
        "realistic_mode_issues": realism_issues,
        "randomization_hash_missing": randomization_missing,
        "randomization_hash_mismatches": randomization_mismatches,
        "runtime_provenance_missing": runtime_missing,
        "runtime_provenance_mismatches": runtime_mismatches,
        "runtime_comparison_key_sha256": global_runtime_hashes,
        "runtime_suite_frozen": len(global_runtime_hashes) == 1,
        "matrix_complete": matrix_complete,
        "provenance_complete": provenance_complete,
        "matched_conclusion_eligible": matrix_complete and provenance_complete,
    }


def _correct_ids(report: Mapping[str, Any], manifest: BenchmarkManifest) -> set[str]:
    known = {obj.instance_id for obj in manifest.objects}
    values = {str(value) for value in report.get("correctly_sorted_object_ids", ())}
    unknown = values - known
    if unknown:
        raise ValueError(
            f"report {report.get('_path')} claims unknown correctly sorted IDs: {sorted(unknown)}"
        )
    if int(report.get("correctly_sorted_objects", -1)) != len(values):
        raise ValueError(f"correctly_sorted_objects disagrees with IDs in {report.get('_path')}")
    return values


def _report_row(report: Mapping[str, Any], condition: Mapping[str, Any]) -> dict[str, Any]:
    manifest: BenchmarkManifest = condition["manifest"]
    correct = _correct_ids(report, manifest)
    return {
        "condition_id": manifest.condition_id,
        "report": report,
        "manifest": manifest,
        "correct_ids": correct,
        "correct": len(correct),
        "objects": manifest.object_count,
        "scene_success": float(len(correct) == manifest.object_count),
        "simulated_time_s": float(report["simulated_episode_time_s"]),
        "capacity_stress_candidate": condition["capacity"]["capacity_stress_candidate"],
    }


def _outcome_metrics(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    objects = sum(int(row["objects"]) for row in rows)
    correct = sum(int(row["correct"]) for row in rows)
    scene_successes = sum(int(row["scene_success"]) for row in rows)
    simulated_time = sum(float(row["simulated_time_s"]) for row in rows)
    return {
        "completed_runs": len(rows),
        "reference_objects": objects,
        "correctly_sorted_objects": correct,
        "object_completion_fraction": None if objects == 0 else correct / objects,
        "object_completion_cluster_bootstrap_95": _cluster_ratio_bootstrap(
            rows,
            numerator=lambda row: float(row["correct"]),
            denominator=lambda row: float(row["objects"]),
            seed=seed,
        ),
        "complete_scene_successes": scene_successes,
        "complete_scene_success_fraction": None if not rows else scene_successes / len(rows),
        "complete_scene_wilson_95": _wilson(scene_successes, len(rows)),
        "total_simulated_time_s": simulated_time,
        "correctly_sorted_per_simulated_minute": (
            None if simulated_time <= 0.0 else correct * 60.0 / simulated_time
        ),
        "throughput_cluster_bootstrap_95": _cluster_ratio_bootstrap(
            rows,
            numerator=lambda row: float(row["correct"]) * 60.0,
            denominator=lambda row: float(row["simulated_time_s"]),
            seed=seed + 1,
        ),
    }


def _inventory_breakdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: Callable[[Any], str],
    seed: int,
    planned_manifests: Sequence[Any],
) -> dict[str, Any]:
    labels = sorted(
        {key(obj) for manifest in planned_manifests for obj in manifest.objects}
        | {key(obj) for row in rows for obj in row["manifest"].objects}
    )
    result: dict[str, Any] = {}
    for index, label in enumerate(labels):
        clusters: list[dict[str, float]] = []
        for row in rows:
            matching = [obj for obj in row["manifest"].objects if key(obj) == label]
            if not matching:
                continue
            correct = sum(obj.instance_id in row["correct_ids"] for obj in matching)
            clusters.append({"opportunities": float(len(matching)), "correct": float(correct)})
        opportunities = int(sum(item["opportunities"] for item in clusters))
        correct = int(sum(item["correct"] for item in clusters))
        planned_opportunities = sum(
            key(obj) == label
            for manifest in planned_manifests
            for obj in manifest.objects
        )
        result[label] = {
            "condition_clusters": len(clusters),
            "completed_run_reference_opportunities": opportunities,
            "planned_reference_opportunities": planned_opportunities,
            "correctly_sorted": correct,
            "completion_fraction_completed_runs": (
                None if not opportunities else correct / opportunities
            ),
            "completion_fraction_all_planned_conservative": (
                None if not planned_opportunities else correct / planned_opportunities
            ),
            "cluster_bootstrap_95": _cluster_ratio_bootstrap(
                clusters,
                numerator=lambda item: item["correct"],
                denominator=lambda item: item["opportunities"],
                seed=seed + index,
            ),
        }
    return result


class _ObjectSubsetManifest:
    def __init__(self, objects: Sequence[Any]) -> None:
        self.objects = tuple(objects)


def _cube_only_manifest(manifest: BenchmarkManifest) -> _ObjectSubsetManifest:
    return _ObjectSubsetManifest(
        [obj for obj in manifest.objects if obj.class_name == "cube_10mm"]
    )


def _cell_summary(
    cell: str,
    reports: Mapping[str, Mapping[str, Any]],
    failures: Mapping[str, Mapping[str, Any]],
    suite: Mapping[str, Any],
    matrix_audit: Mapping[str, Any],
) -> dict[str, Any]:
    conditions = suite["conditions"]
    rows = [
        _report_row(reports[condition_id], conditions[condition_id])
        for condition_id in sorted(set(reports) & set(conditions))
    ]
    completed_metrics = _outcome_metrics(rows, seed=20261101 + CELLS.index(cell) * 100)
    planned_objects = sum(
        condition["manifest"].object_count for condition in conditions.values()
    )
    events = [event for row in rows for event in row["report"].get("events", ())]
    outcomes = Counter(str(event.get("status", "unknown")) for event in events)
    failure_reasons = Counter(
        str(event["reason"])
        for event in events
        if event.get("reason") not in (None, "")
    )
    vision_calls = [item for row in rows for item in row["report"].get("vision_reports", ())]
    ai_events = [
        event["grasp_decision"]
        for event in events
        if isinstance(event.get("grasp_decision"), Mapping)
        and event["grasp_decision"].get("mode") == "ai"
    ]
    cell_status = matrix_audit["cell_status"][cell]
    cell_complete = (
        cell_status["completed_runs"] == cell_status["expected_runs"]
        and cell_status["infrastructure_failure_runs"] == 0
        and cell_status["missing_runs"] == 0
    )
    capacity_rows = [row for row in rows if row["capacity_stress_candidate"]]
    ordinary_rows = [row for row in rows if not row["capacity_stress_candidate"]]
    planned_manifests = [condition["manifest"] for condition in conditions.values()]
    planned_cube_manifests = [
        _cube_only_manifest(manifest) for manifest in planned_manifests
    ]
    return {
        "cell": cell,
        "planned_runs": len(conditions),
        "infrastructure_failure_runs": len(set(failures) & set(conditions)),
        "missing_runs": cell_status["missing_runs"],
        "planned_reference_objects": planned_objects,
        **completed_metrics,
        "object_completion_fraction_all_planned_conservative": (
            None
            if planned_objects == 0
            else completed_metrics["correctly_sorted_objects"] / planned_objects
        ),
        "complete_scene_success_fraction_all_planned_conservative": (
            completed_metrics["complete_scene_successes"] / len(conditions)
        ),
        "submission_metrics_eligible": cell_complete and matrix_audit["provenance_complete"],
        "mean_wall_time_s": (
            None if not rows else mean(float(row["report"]["wall_time_s"]) for row in rows)
        ),
        "event_outcomes": dict(sorted(outcomes.items())),
        "event_failure_reasons": dict(sorted(failure_reasons.items())),
        "vision_calls": len(vision_calls),
        "vision_abstentions": sum(item.get("status") == "abstained" for item in vision_calls),
        "vision_abstention_fraction": (
            None
            if not vision_calls
            else sum(item.get("status") == "abstained" for item in vision_calls)
            / len(vision_calls)
        ),
        "mean_vision_latency_ms": (
            None
            if not vision_calls
            else mean(float(item["total_latency_ms"]) for item in vision_calls)
        ),
        "ai_decisions": len(ai_events),
        "ai_bound_decisions": sum(bool(item.get("ai_bound")) for item in ai_events),
        "ai_fallback_decisions": sum(bool(item.get("used_fallback")) for item in ai_events),
        "ai_selection_changed_from_baseline": sum(
            bool(item.get("selection_changed_from_baseline")) for item in ai_events
        ),
        "multi_feasible_candidate_decisions": sum(
            int(event["grasp_decision"].get("feasible_candidate_count", 0)) > 1
            for event in events
            if isinstance(event.get("grasp_decision"), Mapping)
        ),
        "completion_by_class": _inventory_breakdown(
            rows,
            key=lambda obj: obj.class_name,
            seed=20261200 + CELLS.index(cell) * 100,
            planned_manifests=planned_manifests,
        ),
        "completion_by_cube_colour": _inventory_breakdown(
            [
                {
                    **row,
                    "manifest": _cube_only_manifest(row["manifest"]),
                    "correct_ids": row["correct_ids"],
                }
                for row in rows
                if any(obj.class_name == "cube_10mm" for obj in row["manifest"].objects)
            ],
            key=lambda obj: obj.colour,
            seed=20261300 + CELLS.index(cell) * 100,
            planned_manifests=planned_cube_manifests,
        ),
        "completion_by_destination": _inventory_breakdown(
            rows,
            key=lambda obj: obj.destination_colour,
            seed=20261400 + CELLS.index(cell) * 100,
            planned_manifests=planned_manifests,
        ),
        "capacity_strata": {
            "ordinary": _outcome_metrics(
                ordinary_rows, seed=20261500 + CELLS.index(cell) * 100
            ),
            "capacity_stress_candidate": _outcome_metrics(
                capacity_rows, seed=20261600 + CELLS.index(cell) * 100
            ),
        },
    }


def _paired_comparison(
    left_cell: str,
    right_cell: str,
    reports: Mapping[str, Mapping[str, Mapping[str, Any]]],
    suite: Mapping[str, Any],
    eligible: bool,
) -> dict[str, Any]:
    conditions = suite["conditions"]
    expected_ids = set(conditions)
    common = sorted(expected_ids & set(reports[left_cell]) & set(reports[right_cell]))
    result: dict[str, Any] = {
        "left_cell": left_cell,
        "right_cell": right_cell,
        "difference_direction": "right_minus_left",
        "expected_matched_conditions": len(expected_ids),
        "available_matched_conditions": len(common),
        "missing_from_left": sorted(expected_ids - set(reports[left_cell])),
        "missing_from_right": sorted(expected_ids - set(reports[right_cell])),
        "conclusion_eligible": eligible and len(common) == len(expected_ids),
    }
    if not result["conclusion_eligible"]:
        result["estimates"] = None
        result["reason"] = "incomplete run matrix or unmatched provenance"
        return result
    pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for condition_id in common:
        condition = conditions[condition_id]
        left = _report_row(reports[left_cell][condition_id], condition)
        right = _report_row(reports[right_cell][condition_id], condition)
        pairs["object_completion"].append(
            (left["correct"] / left["objects"], right["correct"] / right["objects"])
        )
        pairs["scene_success"].append((left["scene_success"], right["scene_success"]))
        pairs["throughput_per_simulated_minute"].append(
            (
                left["correct"] * 60.0 / left["simulated_time_s"],
                right["correct"] * 60.0 / right["simulated_time_s"],
            )
        )
        pairs["simulated_time_s"].append(
            (left["simulated_time_s"], right["simulated_time_s"])
        )
    estimates: dict[str, Any] = {}
    for index, (metric, values) in enumerate(sorted(pairs.items())):
        wins = sum(right > left for left, right in values)
        losses = sum(right < left for left, right in values)
        estimates[metric] = {
            "mean_difference": mean(right - left for left, right in values),
            "paired_cluster_bootstrap_95": _paired_bootstrap(
                values,
                seed=(
                    20262000
                    + index
                    + CELLS.index(left_cell) * 100
                    + CELLS.index(right_cell)
                ),
            ),
            "right_wins_ties_losses": {
                "wins": wins,
                "ties": len(values) - wins - losses,
                "losses": losses,
            },
        }
    result["estimates"] = estimates
    return result


def _factorial_interactions(
    reports: Mapping[str, Mapping[str, Mapping[str, Any]]],
    suite: Mapping[str, Any],
    eligible: bool,
) -> dict[str, Any] | None:
    if not eligible:
        return None
    conditions = suite["conditions"]
    values: dict[str, list[float]] = defaultdict(list)
    for condition_id, condition in sorted(conditions.items()):
        rows = {
            cell: _report_row(reports[cell][condition_id], condition) for cell in CELLS
        }
        metrics = {
            "object_completion": {
                cell: row["correct"] / row["objects"] for cell, row in rows.items()
            },
            "scene_success": {cell: row["scene_success"] for cell, row in rows.items()},
            "throughput_per_simulated_minute": {
                cell: row["correct"] * 60.0 / row["simulated_time_s"]
                for cell, row in rows.items()
            },
        }
        for metric, cell_values in metrics.items():
            values[metric].append(
                (cell_values["YA"] - cell_values["DA"])
                - (cell_values["YD"] - cell_values["DD"])
            )
    return {
        metric: {
            "definition": "(YA - DA) - (YD - DD)",
            "mean_interaction": mean(rows),
            "condition_cluster_bootstrap_95": _mean_bootstrap(
                rows, seed=20263000 + index
            ),
        }
        for index, (metric, rows) in enumerate(sorted(values.items()))
    }


def build_summary(runs: Path, suite_path: Path) -> dict[str, Any]:
    suite = _load_suite(suite_path)
    artifacts = _load_artifacts(runs)
    audit = _matrix_audit(suite, artifacts)
    reports = artifacts["reports"]
    failures = artifacts["failures"]
    summaries = [
        _cell_summary(cell, reports[cell], failures[cell], suite, audit) for cell in CELLS
    ]
    pairwise = {
        f"{left}__{right}": _paired_comparison(
            left,
            right,
            reports,
            suite,
            audit["matched_conclusion_eligible"],
        )
        for left, right in PAIRWISE_CELL_KEYS
    }
    infrastructure_failures = [
        row
        for cell in CELLS
        for condition_id, row in sorted(failures[cell].items())
        if condition_id in suite["conditions"]
    ]
    return {
        "schema_version": "randomized-sort-comparison-v2",
        "cell_definitions": {
            "DD": "deterministic hybrid vision + deterministic geometric grasp ranking",
            "YD": "YOLO11n corroboration gate + calibrated geometry + deterministic grasp ranking",
            "DA": "deterministic hybrid vision + guarded pretrained-AI grasp ranking",
            "YA": "YOLO11n corroboration gate + calibrated geometry + guarded pretrained-AI grasp ranking",
        },
        "suite": {
            "path": suite["path"],
            "suite_index_sha256": suite["sha256"],
            "base_seed": suite["base_seed"],
            "condition_count": len(suite["conditions"]),
        },
        "matched_design": True,
        "matched_conclusion_eligible": audit["matched_conclusion_eligible"],
        "matrix_audit": audit,
        "infrastructure_failures": infrastructure_failures,
        "cell_summaries": summaries,
        "all_pairwise_comparisons": pairwise,
        "factorial_interactions": _factorial_interactions(
            reports, suite, audit["matched_conclusion_eligible"]
        ),
        "capacity_stress_definition": {
            "is_candidate_not_proven_capacity_failure": True,
            "rules": [
                "more than two screws guarantees reuse of the benchmark's two commanded screw slots",
                "a destination load above nine exceeds the cube grid's unique commanded slots",
            ],
        },
        "interpretation_constraints": [
            "YOLO supplies a corroborating gate; calibrated geometry supplies metric pose.",
            "AI provider confidence is not a calibrated complete-grasp probability.",
            "Fallback executions are not counted as AI-bound selections.",
            "Object-completion intervals resample scene conditions, not individual objects.",
            "Missing and infrastructure-failed runs make matched conclusions ineligible.",
            "Capacity-stress tags identify commanded-slot reuse risk, not proven physical bin capacity.",
            "Results are Isaac evidence, not physical-robot proof.",
        ],
    }


def _write_outputs(out: Path, payload: Mapping[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "comparison.json", payload)
    fieldnames = [
        "cell",
        "planned_runs",
        "completed_runs",
        "infrastructure_failure_runs",
        "missing_runs",
        "planned_reference_objects",
        "reference_objects",
        "correctly_sorted_objects",
        "object_completion_fraction",
        "object_completion_fraction_all_planned_conservative",
        "complete_scene_success_fraction",
        "correctly_sorted_per_simulated_minute",
        "vision_abstention_fraction",
        "mean_vision_latency_ms",
        "ai_decisions",
        "ai_bound_decisions",
        "ai_fallback_decisions",
        "ai_selection_changed_from_baseline",
        "submission_metrics_eligible",
    ]
    with (out / "cell_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["cell_summaries"]:
            writer.writerow({name: row.get(name) for name in fieldnames})

    markdown = [
        "# Randomized realistic Isaac benchmark",
        "",
        (
            "Matched policy conclusions are **eligible**."
            if payload["matched_conclusion_eligible"]
            else "Matched policy conclusions are **not eligible**: inspect `matrix_audit` in `comparison.json`."
        ),
        "",
        "| Cell | Runs complete/planned | Infra/missing | Objects sorted/reference | Object success | Scene success | Sorted/sim min |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(value: Any, eligible: bool) -> str:
        if value is None or not eligible:
            return "n/a"
        return f"{float(value):.3f}"

    for row in payload["cell_summaries"]:
        eligible = bool(row["submission_metrics_eligible"])
        markdown.append(
            f"| {row['cell']} | {row['completed_runs']}/{row['planned_runs']} | "
            f"{row['infrastructure_failure_runs']}/{row['missing_runs']} | "
            f"{row['correctly_sorted_objects']}/{row['reference_objects']} | "
            f"{fmt(row['object_completion_fraction'], eligible)} | "
            f"{fmt(row['complete_scene_success_fraction'], eligible)} | "
            f"{fmt(row['correctly_sorted_per_simulated_minute'], eligible)} |"
        )
    markdown.extend(
        [
            "",
            "DD is the pre-registered baseline. `comparison.json` contains all six pairwise comparisons, the 2×2 interaction, cluster-aware confidence intervals, class/destination breakdowns, capacity-stress strata, failures, and provenance checks.",
            "",
            "Partial-run conservative completion fractions remain in JSON for diagnostics; they are not submission-ready policy estimates.",
            "",
        ]
    )
    (out / "README.md").write_text("\n".join(markdown), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--suite", required=True, help="authoritative suite_index.json")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="return success after writing an explicitly ineligible partial audit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = Path(args.runs).expanduser().resolve()
    suite_path = Path(args.suite).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    payload = build_summary(runs, suite_path)
    _write_outputs(out, payload)
    print(json.dumps(payload, sort_keys=True))
    if not payload["matched_conclusion_eligible"] and not args.allow_partial:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
