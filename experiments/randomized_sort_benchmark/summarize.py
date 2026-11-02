"""Aggregate and pair the four randomized benchmark policy cells."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

from experiments.randomized_sort_benchmark.randomization import write_json_atomic


CELLS = ("DD", "YD", "DA", "YA")


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def _paired_bootstrap(
    pairs: Sequence[tuple[float, float]], *, seed: int = 20261101, samples: int = 5000
) -> list[float] | None:
    if not pairs:
        return None
    rng = random.Random(seed)
    differences = []
    for _ in range(samples):
        resampled = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        differences.append(mean(right - left for left, right in resampled))
    differences.sort()
    return [
        differences[int(0.025 * (samples - 1))],
        differences[int(0.975 * (samples - 1))],
    ]


def _load_reports(root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    reports: dict[str, dict[str, Any]] = {}
    failures = []
    for cell in CELLS:
        cell_root = root / cell
        for report_path in cell_root.glob("*/episode_report.json"):
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["policy_cell"]["id"] != cell:
                raise ValueError(f"cell mismatch in {report_path}")
            condition_id = report["condition_id"]
            key = f"{cell}:{condition_id}"
            if key in reports:
                raise ValueError(f"duplicate report {key}")
            report["_path"] = str(report_path)
            reports[key] = report
        for failure_path in cell_root.glob("*/episode_failure.json"):
            row = json.loads(failure_path.read_text(encoding="utf-8"))
            row["cell"] = cell
            row["path"] = str(failure_path)
            failures.append(row)
    return reports, failures


def _cell_summary(cell: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reference_objects = sum(int(row["object_count"]) for row in rows)
    correct_objects = sum(int(row["correctly_sorted_objects"]) for row in rows)
    scene_successes = sum(bool(row["complete_scene_success"]) for row in rows)
    simulated_time = sum(float(row["simulated_episode_time_s"]) for row in rows)
    events = [event for row in rows for event in row["events"]]
    ai_events = [
        event["grasp_decision"]
        for event in events
        if isinstance(event.get("grasp_decision"), Mapping)
        and event["grasp_decision"]["mode"] == "ai"
    ]
    outcomes = Counter(str(event.get("status", "unknown")) for event in events)
    class_attempts = Counter(str(event.get("target_class_id", "none")) for event in events)
    class_success = Counter(
        str(event.get("target_class_id", "none"))
        for event in events
        if event.get("complete_grasp_success") is True
    )
    vision_calls = [item for row in rows for item in row["vision_reports"]]
    vision_abstentions = sum(item["status"] == "abstained" for item in vision_calls)
    summary = {
        "cell": cell,
        "completed_runs": len(rows),
        "reference_objects": reference_objects,
        "correctly_sorted_objects": correct_objects,
        "object_completion_fraction": (
            None if reference_objects == 0 else correct_objects / reference_objects
        ),
        "object_completion_wilson_95": _wilson(correct_objects, reference_objects),
        "complete_scene_successes": scene_successes,
        "complete_scene_success_fraction": (
            None if not rows else scene_successes / len(rows)
        ),
        "complete_scene_wilson_95": _wilson(scene_successes, len(rows)),
        "total_simulated_time_s": simulated_time,
        "correctly_sorted_per_simulated_minute": (
            None if simulated_time <= 0.0 else correct_objects * 60.0 / simulated_time
        ),
        "mean_wall_time_s": None if not rows else mean(float(row["wall_time_s"]) for row in rows),
        "event_outcomes": dict(sorted(outcomes.items())),
        "attempts_by_class": dict(sorted(class_attempts.items())),
        "successes_by_class": dict(sorted(class_success.items())),
        "vision_calls": len(vision_calls),
        "vision_abstentions": vision_abstentions,
        "vision_abstention_fraction": (
            None if not vision_calls else vision_abstentions / len(vision_calls)
        ),
        "mean_vision_latency_ms": (
            None if not vision_calls else mean(float(item["total_latency_ms"]) for item in vision_calls)
        ),
        "ai_decisions": len(ai_events),
        "ai_bound_decisions": sum(bool(item["ai_bound"]) for item in ai_events),
        "ai_fallback_decisions": sum(bool(item["used_fallback"]) for item in ai_events),
        "ai_selection_changed_from_baseline": sum(
            bool(item["selection_changed_from_baseline"]) for item in ai_events
        ),
        "multi_feasible_candidate_decisions": sum(
            int(item["feasible_candidate_count"]) > 1
            for item in (
                event["grasp_decision"]
                for event in events
                if isinstance(event.get("grasp_decision"), Mapping)
            )
        ),
    }
    return summary


def _paired_comparison(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    common = sorted(set(baseline) & set(candidate))
    completion_pairs = [
        (
            float(baseline[key]["object_completion_fraction"]),
            float(candidate[key]["object_completion_fraction"]),
        )
        for key in common
    ]
    time_pairs = [
        (
            float(baseline[key]["simulated_episode_time_s"]),
            float(candidate[key]["simulated_episode_time_s"]),
        )
        for key in common
    ]
    wins = sum(right > left for left, right in completion_pairs)
    losses = sum(right < left for left, right in completion_pairs)
    ties = len(completion_pairs) - wins - losses
    return {
        "matched_conditions": len(common),
        "completion_wins_ties_losses": {"wins": wins, "ties": ties, "losses": losses},
        "mean_object_completion_difference": (
            None if not completion_pairs else mean(right - left for left, right in completion_pairs)
        ),
        "object_completion_difference_bootstrap_95": _paired_bootstrap(completion_pairs),
        "mean_simulated_time_difference_s": (
            None if not time_pairs else mean(right - left for left, right in time_pairs)
        ),
        "simulated_time_difference_bootstrap_95_s": _paired_bootstrap(time_pairs, seed=20261102),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = Path(args.runs).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    reports, failures = _load_reports(runs)
    by_cell: dict[str, list[dict[str, Any]]] = {cell: [] for cell in CELLS}
    by_cell_condition: dict[str, dict[str, dict[str, Any]]] = {
        cell: {} for cell in CELLS
    }
    condition_hashes: dict[str, set[str]] = defaultdict(set)
    for key, report in reports.items():
        cell = report["policy_cell"]["id"]
        by_cell[cell].append(report)
        by_cell_condition[cell][report["condition_id"]] = report
        condition_hashes[report["condition_id"]].add(report["condition_manifest_sha256"])
    mismatched = {
        condition: sorted(hashes)
        for condition, hashes in condition_hashes.items()
        if len(hashes) != 1
    }
    if mismatched:
        raise SystemExit(f"condition manifest hashes differ across cells: {mismatched}")
    summaries = [_cell_summary(cell, by_cell[cell]) for cell in CELLS]
    comparisons = {
        cell: _paired_comparison(by_cell_condition["DD"], by_cell_condition[cell])
        for cell in CELLS
        if cell != "DD"
    }
    payload = {
        "schema_version": "randomized-sort-comparison-v1",
        "cell_definitions": {
            "DD": "deterministic hybrid vision + deterministic geometric grasp ranking",
            "YD": "YOLO11n-assisted hybrid vision + deterministic geometric grasp ranking",
            "DA": "deterministic hybrid vision + pretrained AI grasp ranking",
            "YA": "YOLO11n-assisted hybrid vision + pretrained AI grasp ranking",
        },
        "matched_design": True,
        "completed_report_count": len(reports),
        "infrastructure_failure_count": len(failures),
        "infrastructure_failures": failures,
        "cell_summaries": summaries,
        "paired_against_DD": comparisons,
        "condition_manifest_hash_mismatches": mismatched,
        "interpretation_constraints": [
            "YOLO supplies corroborating boxes; calibrated geometry supplies pose.",
            "AI provider confidence is not a calibrated complete-grasp probability.",
            "Fallback executions are not counted as AI-bound selections.",
            "Results are Isaac evidence, not physical-robot proof.",
        ],
    }
    write_json_atomic(out / "comparison.json", payload)
    with (out / "cell_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "cell",
                "completed_runs",
                "reference_objects",
                "correctly_sorted_objects",
                "object_completion_fraction",
                "complete_scene_success_fraction",
                "correctly_sorted_per_simulated_minute",
                "vision_abstention_fraction",
                "mean_vision_latency_ms",
                "ai_decisions",
                "ai_bound_decisions",
                "ai_fallback_decisions",
                "ai_selection_changed_from_baseline",
            ],
        )
        writer.writeheader()
        for row in summaries:
            writer.writerow({name: row.get(name) for name in writer.fieldnames})
    markdown = [
        "# Randomized realistic Isaac benchmark",
        "",
        "| Cell | Runs | Objects sorted/reference | Object success | Scene success | Sorted/sim min | Vision abstention | AI bound/fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        def fmt(value: Any) -> str:
            return "n/a" if value is None else f"{float(value):.3f}"
        markdown.append(
            f"| {row['cell']} | {row['completed_runs']} | "
            f"{row['correctly_sorted_objects']}/{row['reference_objects']} | "
            f"{fmt(row['object_completion_fraction'])} | "
            f"{fmt(row['complete_scene_success_fraction'])} | "
            f"{fmt(row['correctly_sorted_per_simulated_minute'])} | "
            f"{fmt(row['vision_abstention_fraction'])} | "
            f"{row['ai_bound_decisions']}/{row['ai_fallback_decisions']} |"
        )
    markdown.extend(
        [
            "",
            "DD is the pre-registered baseline. See `comparison.json` for paired confidence intervals, failure counts, and interpretation constraints.",
            "",
        ]
    )
    (out / "README.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
