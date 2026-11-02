"""Resume-safe launcher for one 2x2 policy cell on a Brev Isaac runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.randomized_sort_benchmark.randomization import write_json_atomic


CELLS = {
    "DD": ("deterministic_hybrid", "deterministic"),
    "YD": ("yolo_assisted_hybrid", "deterministic"),
    "DA": ("deterministic_hybrid", "ai"),
    "YA": ("yolo_assisted_hybrid", "ai"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, help="suite_index.json")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--cell", choices=tuple(CELLS), required=True)
    parser.add_argument("--repo", default=os.environ.get("ROCKWELL_REPO_ROOT"))
    parser.add_argument("--runtime", default=os.environ.get("ROCKWELL_TWIN_RUNTIME"))
    parser.add_argument("--isaac", default=os.environ.get("ISAAC"))
    parser.add_argument("--reference-assets", default=os.environ.get("ROCKWELL_REFERENCE_ASSETS"))
    parser.add_argument("--yolo-checkpoint", default=os.environ.get("ROCKWELL_VISION_CHECKPOINT"))
    parser.add_argument("--yolo-endpoint", default=os.environ.get("ROCKWELL_YOLO_ENDPOINT"))
    parser.add_argument("--ai-endpoint", default=os.environ.get("ROCKWELL_AI_GRASP_ENDPOINT"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--record-first", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite_path = Path(args.suite).expanduser().resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    conditions = list(suite["conditions"])[args.start_index :: args.stride]
    if args.limit is not None:
        conditions = conditions[: args.limit]
    repo = Path(args.repo or "").expanduser().resolve()
    runtime = Path(args.runtime or "").expanduser().resolve()
    isaac = Path(args.isaac or "").expanduser().resolve()
    if not (repo / "experiments/randomized_sort_benchmark/isaac_episode.py").is_file():
        raise SystemExit("--repo/ROCKWELL_REPO_ROOT does not identify this checkout")
    if not runtime.is_dir() or not (isaac / "python.sh").is_file():
        raise SystemExit("source the commit-specific env.sh or pass --runtime and --isaac")
    vision, grasp = CELLS[args.cell]
    if vision.startswith("yolo") and not (args.yolo_checkpoint or args.yolo_endpoint):
        raise SystemExit("YOLO cells require --yolo-endpoint or --yolo-checkpoint")
    if grasp == "ai" and not args.ai_endpoint:
        raise SystemExit("AI cells require --ai-endpoint")
    run_root = Path(args.runs).expanduser().resolve() / args.cell
    run_root.mkdir(parents=True, exist_ok=True)
    status_rows = []
    for ordinal, condition in enumerate(conditions):
        condition_id = condition["condition_id"]
        run_dir = run_root / condition_id
        report_path = run_dir / "episode_report.json"
        if report_path.is_file():
            status_rows.append({"condition_id": condition_id, "status": "already_complete"})
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            status_rows.append(
                {
                    "condition_id": condition_id,
                    "status": "blocked_partial_output",
                    "run_dir": str(run_dir),
                }
            )
            continue
        manifest_path = (suite_path.parent / condition["manifest_relpath"]).resolve()
        command = [
            str(isaac / "python.sh"),
            str(repo / "experiments/randomized_sort_benchmark/isaac_episode.py"),
            "--manifest",
            str(manifest_path),
            "--run",
            str(run_dir),
            "--vision",
            vision,
            "--grasp",
            grasp,
        ]
        if args.reference_assets:
            command.extend(
                [
                    "--bin-model",
                    str(Path(args.reference_assets) / "models/manipbench-bin-50x50x55.stl"),
                    "--screw-model",
                    str(Path(args.reference_assets) / "models/ISO10642-M6x30-countersunk-screw.stl"),
                ]
            )
        if args.yolo_endpoint and vision.startswith("yolo"):
            command.extend(["--yolo-endpoint", str(args.yolo_endpoint)])
        elif args.yolo_checkpoint and vision.startswith("yolo"):
            command.extend(["--yolo-checkpoint", str(args.yolo_checkpoint)])
        if args.ai_endpoint and grasp == "ai":
            command.extend(["--ai-endpoint", str(args.ai_endpoint)])
        if args.record_first and ordinal == 0:
            command.append("--record-video")
        started = time.perf_counter()
        env = os.environ.copy()
        env["ROCKWELL_REPO_ROOT"] = str(repo)
        if args.reference_assets:
            env["ROCKWELL_REFERENCE_ASSETS"] = str(args.reference_assets)
        completed = subprocess.run(
            command,
            cwd=runtime,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "launcher.log").write_text(completed.stdout, encoding="utf-8")
        status_rows.append(
            {
                "condition_id": condition_id,
                "status": "complete" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "wall_time_s": time.perf_counter() - started,
            }
        )
        write_json_atomic(
            run_root / "shard_status.json",
            {
                "schema_version": "randomized-sort-shard-status-v1",
                "cell": args.cell,
                "suite": str(suite_path),
                "conditions_selected": len(conditions),
                "rows": status_rows,
            },
        )
        print(json.dumps(status_rows[-1], sort_keys=True), flush=True)
    return 0 if all(row["status"] in {"complete", "already_complete"} for row in status_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
