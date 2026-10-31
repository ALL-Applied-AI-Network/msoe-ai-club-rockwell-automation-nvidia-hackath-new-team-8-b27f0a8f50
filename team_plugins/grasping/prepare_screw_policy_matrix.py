"""Prepare a small, matched screw-policy pilot outside the Git checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping

from demos.scripted_cube_screw_bucket_sort.scenario import (
    build_sort_items,
    load_scenario_config,
)
from sorting_core import DetectedObject, Pose3D, SceneObservation, TipPoseBase

from team_plugins.grasping.isolated_sim_runner import (
    _screw_orientation_xyzw,
)
from team_plugins.grasping.multigrasp_pipeline import (
    SCREW_CLASS_ID,
    build_offline_screw_planner,
)
from team_plugins.grasping.prepare_screw_trials import (
    ROOT,
    SOURCE_FOLDERS,
    _source_state,
    _tree_sha256,
)


TARGET_ITEM_ID = "screw-red-01"
CANDIDATES = (
    (12, "screw-shaft-012mm-0"),
    (16, "screw-shaft-016mm-1"),
    (20, "screw-shaft-020mm-2"),
)
TARGET_HEAD_M = (0.44, 0.0, 0.0014)
PARKED_CUBES = {
    "red": (0.32, 0.0, 0.0),
    "yellow": (0.32, -0.24, 0.0),
    "blue": (0.32, 0.24, 0.0),
}
LAYOUTS = (
    {
        "layout_family_id": "pilot-clear-y020",
        "group_id": "pilot-clear-y020-v1",
        "split": "train",
        "target_yaw_deg": 20.0,
        "tip_gap_m": None,
    },
    {
        "layout_family_id": "pilot-tipgap9-y020",
        "group_id": "pilot-tipgap9-y020-v1",
        "split": "train",
        "target_yaw_deg": 20.0,
        "tip_gap_m": 0.009,
    },
    {
        "layout_family_id": "pilot-tipgap8-y020",
        "group_id": "pilot-tipgap8-y020-v1",
        "split": "train",
        "target_yaw_deg": 20.0,
        "tip_gap_m": 0.008,
    },
    {
        "layout_family_id": "pilot-tipgap8-y160",
        "group_id": "pilot-tipgap8-y160-v1",
        "split": "test",
        "target_yaw_deg": 160.0,
        "tip_gap_m": 0.008,
    },
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _candidate_offset(
    scenario: dict[str, Any],
    *,
    offset_m: float,
    baseline_offset_m: float,
) -> None:
    delta_m = offset_m - baseline_offset_m
    scenario["screw"]["grasp_offset_from_head_m"] = offset_m
    for screw in scenario["task"]["screws"]:
        yaw = math.radians(float(screw["target_axis_yaw_deg"]))
        point = screw["placement_grasp_point_offset_from_bin_centre_m"]
        point[0] = float(point[0]) + delta_m * math.cos(yaw)
        point[1] = float(point[1]) + delta_m * math.sin(yaw)


def _layout_configs(
    source_scenario: Mapping[str, Any],
    source_cube: Mapping[str, Any],
    layout: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scenario = json.loads(json.dumps(source_scenario))
    cube = json.loads(json.dumps(source_cube))
    yaw_deg = float(layout["target_yaw_deg"])
    yaw_rad = math.radians(yaw_deg)
    axis = (math.cos(yaw_rad), math.sin(yaw_rad))

    target = next(
        row
        for row in scenario["task"]["screws"]
        if row["id"] == TARGET_ITEM_ID
    )
    target["spawn_head_base_m"] = list(TARGET_HEAD_M)
    target["spawn_yaw_deg"] = yaw_deg
    parked = next(
        row
        for row in scenario["task"]["screws"]
        if row["id"] != TARGET_ITEM_ID
    )
    parked["spawn_head_base_m"] = [0.48, 0.22, TARGET_HEAD_M[2]]
    parked["spawn_yaw_deg"] = 180.0

    cube_spawns = {
        colour: list(values) for colour, values in PARKED_CUBES.items()
    }
    gap = layout["tip_gap_m"]
    obstacle = None
    if gap is not None:
        # Cube near face is 5 mm from its centre, so 30 mm screw length plus
        # 5 mm cube half-extent plus the requested free gap locates the centre.
        distance_m = 0.035 + float(gap)
        obstacle = (
            TARGET_HEAD_M[0] + distance_m * axis[0],
            TARGET_HEAD_M[1] + distance_m * axis[1],
        )
        cube_spawns["red"] = [obstacle[0], obstacle[1], yaw_deg]
    cube["task"]["cube_spawns_base_xy_yaw_deg"] = cube_spawns

    layout_record = {
        "layout_family_id": layout["layout_family_id"],
        "group_id": layout["group_id"],
        "split": layout["split"],
        "target_item_id": TARGET_ITEM_ID,
        "target_head_base_m": list(TARGET_HEAD_M),
        "target_yaw_deg": yaw_deg,
        "tip_obstacle_gap_m": gap,
        "tip_obstacle_cube": "cube-red" if gap is not None else None,
        "tip_obstacle_centre_base_m": (
            list(obstacle) if obstacle is not None else None
        ),
        "parked_screw_head_base_m": parked["spawn_head_base_m"],
        "parked_screw_yaw_deg": parked["spawn_yaw_deg"],
        "cube_spawns_base_xy_yaw_deg": cube_spawns,
    }
    layout_record["layout_fingerprint_sha256"] = _canonical_sha256(
        layout_record
    )
    return scenario, cube, layout_record


def _configured_observation(config: Any, group_id: str) -> SceneObservation:
    cube_config = config.cube_config
    capture_time_s = 0.0
    objects: list[DetectedObject] = []
    cube_z = (
        cube_config.frames.surface_base_z_m + cube_config.cube_edge_m / 2.0
    )
    for colour in cube_config.task.colour_order:
        x, y, yaw_deg = cube_config.task.cube_spawns_base_xy_yaw_deg[
            colour
        ]
        objects.append(
            DetectedObject(
                track_id=f"cube-{colour}",
                class_id="cube",
                sort_label=colour,
                capture_time_s=capture_time_s,
                pose=Pose3D.from_yaw(
                    (x, y, cube_z), math.radians(yaw_deg)
                ),
                dimensions_m=(
                    cube_config.cube_edge_m,
                    cube_config.cube_edge_m,
                    cube_config.cube_edge_m,
                ),
                confidence=1.0,
                pose_quality=1.0,
                visibility=1.0,
                source_camera_ids=("configured-isaac-truth",),
            )
        )
    for screw in config.screws:
        objects.append(
            DetectedObject(
                track_id=screw.item_id,
                class_id=SCREW_CLASS_ID,
                sort_label=screw.destination_colour,
                capture_time_s=capture_time_s,
                pose=Pose3D(
                    position_base_m=screw.spawn_head_base_m,
                    orientation_xyzw=_screw_orientation_xyzw(
                        math.radians(screw.spawn_yaw_deg)
                    ),
                ),
                dimensions_m=(0.012, 0.01199627, 0.030),
                confidence=1.0,
                pose_quality=1.0,
                visibility=1.0,
                source_camera_ids=("configured-isaac-truth",),
            )
        )
    return SceneObservation(
        scene_id=group_id,
        capture_time_s=capture_time_s,
        objects=tuple(objects),
        provider_name="configured-isaac-truth",
        provider_version="1",
        calibration_ids=("simulation-only",),
    )


def _planned_candidates(config: Any, group_id: str) -> dict[str, Any]:
    observation = _configured_observation(config, group_id)
    cube_config = config.cube_config
    red_bin = cube_config.task.bin_centres_base_m["red"]
    planner = build_offline_screw_planner(cube_config)
    decision = planner.decide(
        observation,
        TARGET_ITEM_ID,
        pick_tip_z_m=cube_config.pick_tip_base_z_m,
        approach_clearance_m=cube_config.task.approach_clearance_m,
        release_tip_pose=TipPoseBase(
            (
                red_bin[0],
                red_bin[1],
                cube_config.bin_release_tip_base_z_m,
            ),
            0.0,
        ),
    )
    result: dict[str, Any] = {}
    for evaluation in decision.evaluations:
        result[evaluation.candidate.candidate_id] = {
            "accepted": True,
            "rejection_reasons": [],
            "checker_id": evaluation.feasibility.checker_id,
            "metrics": dict(evaluation.feasibility.metrics),
            "feature_schema_id": evaluation.features.schema_id,
            "features": dict(evaluation.features.values),
            "uncalibrated_simulation_score": (
                evaluation.score.success_probability
            ),
        }
    for rejection in decision.rejections:
        result[rejection.candidate.candidate_id] = {
            "accepted": False,
            "rejection_reasons": list(
                rejection.feasibility.rejection_reasons
            ),
            "checker_id": rejection.feasibility.checker_id,
            "metrics": dict(rejection.feasibility.metrics),
            "feature_schema_id": None,
            "features": None,
            "uncalibrated_simulation_score": None,
        }
    return result


def prepare(
    output: Path,
    *,
    require_clean: bool = True,
) -> Path:
    output = output.expanduser().resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("pilot output must be outside the Git checkout")
    if output.exists():
        raise FileExistsError(f"pilot output already exists: {output}")

    source_commit, source_tree_clean, dirty_entries = _source_state(
        require_clean=require_clean
    )
    output.mkdir(parents=True)
    source_scenario_path = ROOT / "config" / "cube_screw_bucket_sort.json"
    source_cube_path = ROOT / "config" / "colour_bucket_sort.json"
    source_scenario = json.loads(source_scenario_path.read_text("utf-8"))
    source_cube = json.loads(source_cube_path.read_text("utf-8"))
    baseline_offset_m = float(
        source_scenario["screw"]["grasp_offset_from_head_m"]
    )
    split_by_family = {
        str(layout["layout_family_id"]): str(layout["split"])
        for layout in LAYOUTS
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "team8-isolated-screw-policy-pilot-v1",
        "source_commit": source_commit,
        "relevant_source_tree_clean": source_tree_clean,
        "relevant_source_dirty_entries": list(dirty_entries),
        "source_tree_sha256": {
            folder: _tree_sha256(ROOT / folder)
            for folder in (*SOURCE_FOLDERS, "team_plugins/grasping")
        },
        "target_item_id": TARGET_ITEM_ID,
        "candidate_offsets_mm": [value for value, _ in CANDIDATES],
        "split_by_layout_family": split_by_family,
        "split_manifest_sha256": _canonical_sha256(split_by_family),
        "layout_count": len(LAYOUTS),
        "attempt_count": len(LAYOUTS) * len(CANDIDATES),
        "notes": [
            "configured truth is provisional until pre-action settling",
            "collector must recompute features from event.source_truth",
            "each case requires a fresh Isaac process and run directory",
        ],
        "cases": [],
    }

    for layout in LAYOUTS:
        scenario_template, cube_config, layout_record = _layout_configs(
            source_scenario,
            source_cube,
            layout,
        )
        layout_dir = output / "cases" / str(layout["group_id"])
        layout_dir.mkdir(parents=True)
        _write_json(layout_dir / "layout.json", layout_record)

        planned_config_path = layout_dir / "planning_scenario.json"
        planning_scenario = json.loads(json.dumps(scenario_template))
        planning_scenario["base_colour_sort_config"] = (
            "planning_colour_bucket_sort.json"
        )
        _write_json(
            layout_dir / "planning_colour_bucket_sort.json",
            cube_config,
        )
        _write_json(planned_config_path, planning_scenario)
        validated_planning = load_scenario_config(planned_config_path)
        planned = _planned_candidates(
            validated_planning,
            str(layout["group_id"]),
        )

        for offset_mm, candidate_id in CANDIDATES:
            evidence = planned.get(candidate_id)
            if evidence is None or not evidence["accepted"]:
                raise RuntimeError(
                    f"pilot case {layout['group_id']} {candidate_id} "
                    "is not deterministically feasible"
                )
            case_dir = layout_dir / candidate_id
            config_dir = case_dir / "config"
            config_dir.mkdir(parents=True)
            shutil.copy2(
                layout_dir / "planning_colour_bucket_sort.json",
                config_dir / "colour_bucket_sort.json",
            )
            scenario = json.loads(json.dumps(scenario_template))
            scenario["base_colour_sort_config"] = "colour_bucket_sort.json"
            _candidate_offset(
                scenario,
                offset_m=offset_mm / 1000.0,
                baseline_offset_m=baseline_offset_m,
            )
            scenario_path = config_dir / "cube_screw_bucket_sort.json"
            _write_json(scenario_path, scenario)
            validated = load_scenario_config(scenario_path)
            target = next(
                item
                for item in build_sort_items(validated)
                if item.item_id == TARGET_ITEM_ID
            )
            case = {
                **layout_record,
                "case_id": (
                    f"{layout['group_id']}--{candidate_id}"
                ),
                "candidate_id": candidate_id,
                "pinch_from_head_mm": offset_mm,
                "scenario_path": str(scenario_path),
                "scenario_sha256": hashlib.sha256(
                    scenario_path.read_bytes()
                ).hexdigest(),
                "candidate_spawn_grasp_point_base_m": list(
                    target.spawn_grasp_point_base_m
                ),
                "placement_head_base_m": list(
                    target.placement_reference_base_m
                ),
                "configured_pre_settle_planning": evidence,
            }
            _write_json(case_dir / "case.json", case)
            manifest["cases"].append(case)

    manifest_path = output / "experiment_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="development only; dirty entries remain recorded",
    )
    args = parser.parse_args()
    print(
        prepare(
            args.output,
            require_clean=not args.allow_dirty_source,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
