"""Prepare matched 12/16/20 mm screw trials outside the Git checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Iterable

from demos.scripted_cube_screw_bucket_sort.scenario import (
    build_sort_items,
    load_scenario_config,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFSETS_MM = (12, 16, 20)
SOURCE_FOLDERS = ("config", "demos", "sorting_core")


def _tree_sha256(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(ROOT)
        if (
            not path.is_file()
            or "__pycache__" in relative.parts
            or path.suffix == ".pyc"
        ):
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _source_state(*, require_clean: bool) -> tuple[str, bool, tuple[str, ...]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *SOURCE_FOLDERS,
            "team_plugins/grasping",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dirty_entries = tuple(line for line in status.splitlines() if line)
    if require_clean and dirty_entries:
        raise RuntimeError(
            "refusing to prepare trials from dirty relevant source paths"
        )
    return commit, not dirty_entries, dirty_entries


def _offsets(values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError("offsets must be non-empty and unique")
    if any(value <= 4 or value >= 30 for value in result):
        raise ValueError("offsets must remain on the 4-30 mm shaft region")
    return result


def prepare(
    output: Path,
    offsets_mm=DEFAULT_OFFSETS_MM,
    *,
    require_clean: bool = True,
) -> Path:
    """Create isolated external repo roots and a machine-readable manifest."""

    output = output.expanduser().resolve()
    offsets = _offsets(offsets_mm)
    source_commit, source_tree_clean, dirty_entries = _source_state(
        require_clean=require_clean
    )
    if output == ROOT or ROOT in output.parents:
        raise ValueError("trial output must be outside the Git checkout")
    output.mkdir(parents=True, exist_ok=False)

    source_config_path = ROOT / "config" / "cube_screw_bucket_sort.json"
    source = json.loads(source_config_path.read_text(encoding="utf-8"))
    baseline_offset_m = float(source["screw"]["grasp_offset_from_head_m"])
    manifest = {
        "schema_version": 1,
        "experiment_id": "team8-screw-pinch-matched-v1",
        "source_commit": source_commit,
        "relevant_source_tree_clean": source_tree_clean,
        "relevant_source_dirty_entries": list(dirty_entries),
        "source_tree_sha256": {
            folder: _tree_sha256(ROOT / folder) for folder in SOURCE_FOLDERS
        },
        "target_item_id": "screw-red-01",
        "offsets_mm": list(offsets),
        "cases": [],
    }

    for offset_mm in offsets:
        case_root = output / f"pinch-{offset_mm:02d}mm"
        for folder in SOURCE_FOLDERS:
            shutil.copytree(
                ROOT / folder,
                case_root / folder,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

        scenario_path = case_root / "config" / "cube_screw_bucket_sort.json"
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        new_offset_m = offset_mm / 1000.0
        delta_m = new_offset_m - baseline_offset_m
        scenario["screw"]["grasp_offset_from_head_m"] = new_offset_m
        for screw in scenario["task"]["screws"]:
            yaw = math.radians(float(screw["target_axis_yaw_deg"]))
            point = screw["placement_grasp_point_offset_from_bin_centre_m"]
            point[0] = float(point[0]) + delta_m * math.cos(yaw)
            point[1] = float(point[1]) + delta_m * math.sin(yaw)

        # Preserve the validated full-demo order. The screw release lanes were
        # designed around cubes that have already been placed; moving the screw
        # first would confound the final label with a later cube interaction.
        scenario_path.write_text(
            json.dumps(scenario, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        validated = load_scenario_config(scenario_path)
        target = next(
            item
            for item in build_sort_items(validated)
            if item.item_id == "screw-red-01"
        )
        manifest["cases"].append(
            {
                "case_id": f"pinch-{offset_mm:02d}mm",
                "repo_root": str(case_root),
                "config_path": str(scenario_path),
                "scenario_sha256": hashlib.sha256(
                    scenario_path.read_bytes()
                ).hexdigest(),
                "sort_order": list(scenario["task"]["sort_order"]),
                "pinch_from_head_mm": offset_mm,
                "spawn_grasp_point_base_m": list(
                    target.spawn_grasp_point_base_m
                ),
                "placement_head_base_m": list(
                    target.placement_reference_base_m
                ),
            }
        )

    manifest_path = output / "experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--offset-mm",
        type=int,
        action="append",
        dest="offsets_mm",
        help="repeat to override the default 12/16/20 mm candidates",
    )
    args = parser.parse_args()
    path = prepare(
        args.output,
        DEFAULT_OFFSETS_MM if args.offsets_mm is None else args.offsets_mm,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
