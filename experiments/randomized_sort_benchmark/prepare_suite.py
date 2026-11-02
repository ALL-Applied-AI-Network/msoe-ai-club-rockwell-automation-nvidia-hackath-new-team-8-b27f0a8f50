"""Pre-register matched condition manifests before launching any policy cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.randomized_sort_benchmark.manifest import (
    BenchmarkSuite,
    generate_condition,
    generate_matched_suite,
    write_manifest,
)
from experiments.randomized_sort_benchmark.randomization import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-seed", type=int, default=20261101)
    parser.add_argument("--trials-per-object-count", type=int, default=4)
    parser.add_argument(
        "--counts",
        help="optional comma-separated subset such as 1,3,6,10 for smoke tests",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.out).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty suite directory {output}")
    manifests = output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    full = generate_matched_suite(
        args.base_seed,
        trials_per_object_count=args.trials_per_object_count,
    )
    selected = full.conditions
    if args.counts:
        counts = {int(value.strip()) for value in args.counts.split(",") if value.strip()}
        if not counts or not counts <= set(range(1, 11)):
            raise SystemExit("--counts must contain values in 1..10")
        selected = tuple(condition for condition in selected if condition.object_count in counts)
    rows = []
    for index, condition in enumerate(selected):
        filename = f"{index:04d}-{condition.condition_id}.json"
        write_manifest(manifests / filename, condition)
        rows.append(
            {
                "index": index,
                "condition_id": condition.condition_id,
                "object_count": condition.object_count,
                "manifest_relpath": f"manifests/{filename}",
                "manifest_sha256": condition.sha256,
            }
        )
    payload = {
        "schema_version": "randomized-sort-suite-index-v1",
        "base_seed": args.base_seed,
        "trials_per_object_count": args.trials_per_object_count,
        "full_suite_sha256": full.sha256,
        "selected_counts": sorted({condition.object_count for condition in selected}),
        "condition_count": len(rows),
        "planned_policy_cells": ["DD", "YD", "DA", "YA"],
        "planned_end_to_end_runs": len(rows) * 4,
        "conditions": rows,
    }
    write_json_atomic(output / "suite_index.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
