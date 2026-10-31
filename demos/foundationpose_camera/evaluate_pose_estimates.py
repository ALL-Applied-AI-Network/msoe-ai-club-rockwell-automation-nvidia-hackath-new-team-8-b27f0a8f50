#!/usr/bin/env python3
"""Write JSON and CSV error reports for FoundationPose camera estimates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sorting_core.pose_evaluation import evaluate_pose_documents  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimates", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    estimates = json.loads(args.estimates.read_text(encoding="utf-8"))
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    report = evaluate_pose_documents(estimates, truth)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "object_id",
        "class_id",
        "estimated_x_base_m",
        "estimated_y_base_m",
        "estimated_z_base_m",
        "truth_x_base_m",
        "truth_y_base_m",
        "truth_z_base_m",
        "position_error_mm",
        "estimated_qx",
        "estimated_qy",
        "estimated_qz",
        "estimated_qw",
        "truth_qx",
        "truth_qy",
        "truth_qz",
        "truth_qw",
        "rotation_error_deg",
        "axis_error_deg",
        "confidence",
        "confidence_source",
        "foundationpose_raw_score",
        "confidence_is_calibrated_probability",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in report["objects"]:
            flat = dict(row)
            for prefix, source in (
                ("estimated", "estimated_position_base_m"),
                ("truth", "truth_position_base_m"),
            ):
                flat.update(
                    {
                        f"{prefix}_x_base_m": row[source][0],
                        f"{prefix}_y_base_m": row[source][1],
                        f"{prefix}_z_base_m": row[source][2],
                    }
                )
            for prefix, source in (
                ("estimated", "estimated_orientation_xyzw"),
                ("truth", "truth_orientation_xyzw"),
            ):
                flat.update(
                    {
                        f"{prefix}_qx": row[source][0],
                        f"{prefix}_qy": row[source][1],
                        f"{prefix}_qz": row[source][2],
                        f"{prefix}_qw": row[source][3],
                    }
                )
            writer.writerow(flat)
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
