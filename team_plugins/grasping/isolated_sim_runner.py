"""Execute one generated grasp candidate in a fresh Isaac mixed scene.

This simulation-only adapter keeps every configured object in the physics scene
as an obstacle, but patches the validated mixed-object demo so exactly one
target is acted on and evaluated. Each candidate must run in a fresh process
and run directory, removing the earlier sequential-action confound.

The adapter consumes the checked-in candidate generators and maps TipPose yaw
onto the twin's jaw-closing axis. It does not claim camera perception,
calibrated probabilities, or hardware support.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

from sorting_core import (
    BinTarget,
    CubeDetection,
    DetectedObject,
    Pose3D,
    SceneDetections,
    SceneObservation,
    TipPoseBase,
)
from sorting_core.grasping import GraspCandidate, GraspPlanningContext

from team_plugins.grasping.multigrasp_pipeline import (
    CUBE_GENERATOR_ID,
    SCREW_CLASS_ID,
    SCREW_GENERATOR_ID,
    CubeYawCandidateGenerator,
    ScrewShaftCandidateGenerator,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER_ID = "team8-isolated-isaac-candidate-v1"
HELD_RETENTION_PHASES = ("lift", "carry", "lower")
MAXIMUM_HELD_GRASP_POINT_ERROR_M = 0.008


@dataclass(frozen=True)
class AttemptIdentity:
    layout_family_id: str
    group_id: str
    candidate_id: str
    split: str

    def __post_init__(self) -> None:
        for name in ("layout_family_id", "group_id", "candidate_id", "split"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)


def spread_from_tip_yaw(yaw_rad: float) -> tuple[float, float, float]:
    """Map TipPose yaw to the twin's parallel-jaw closing direction."""

    yaw = float(yaw_rad)
    if not math.isfinite(yaw):
        raise ValueError("candidate yaw must be finite")
    return (-math.sin(yaw), math.cos(yaw), 0.0)


def tip_yaw_from_spread(spread: Sequence[float]) -> float:
    """Invert the twin convention, modulo the parallel jaw's 180-degree axis."""

    if len(spread) != 3:
        raise ValueError("spread axis must have three components")
    x, y, z = (float(value) for value in spread)
    if not all(math.isfinite(value) for value in (x, y, z)):
        raise ValueError("spread axis must be finite")
    if math.hypot(x, y) <= 0.0 or abs(z) > 1e-9:
        raise ValueError("spread axis must be nonzero and planar")
    return math.atan2(-x, y)


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _parallel_axis_distance(first: float, second: float) -> float:
    return min(
        abs(_wrap_angle(first - second)),
        abs(_wrap_angle(first - second + math.pi)),
        abs(_wrap_angle(first - second - math.pi)),
    )


def same_parallel_axis(
    first: Sequence[float],
    second: Sequence[float],
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Return true when two planar axes agree modulo 180 degrees."""

    if len(first) != 3 or len(second) != 3:
        raise ValueError("axes must have three components")
    a = tuple(float(value) for value in first)
    b = tuple(float(value) for value in second)
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        raise ValueError("axes must be nonzero")
    cosine = sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)
    return abs(abs(cosine) - 1.0) <= tolerance


def _finite_xyz(value: Any, name: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise ValueError(f"{name} must contain three coordinates")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain finite coordinates")
    return result


def _screw_orientation_xyzw(yaw_rad: float) -> tuple[float, float, float, float]:
    """Quaternion for Rz(yaw) * Ry(-90 degrees), matching the screw CAD."""

    half_yaw = float(yaw_rad) / 2.0
    half = math.sqrt(0.5)
    return (
        half * math.sin(half_yaw),
        -half * math.cos(half_yaw),
        half * math.sin(half_yaw),
        half * math.cos(half_yaw),
    )


def _candidate_by_id(
    candidates: Sequence[GraspCandidate], candidate_id: str
) -> GraspCandidate:
    matches = tuple(
        candidate
        for candidate in candidates
        if candidate.candidate_id == candidate_id
    )
    if len(matches) != 1:
        available = ", ".join(candidate.candidate_id for candidate in candidates)
        raise ValueError(
            f"candidate_id {candidate_id!r} is not unique; available: {available}"
        )
    return matches[0]


def build_candidate(
    scenario: Any,
    target_item: Any,
    identity: AttemptIdentity,
) -> GraspCandidate:
    """Regenerate the requested catalog candidate from the frozen scenario."""

    config = scenario.cube_config
    capture_time_s = 0.0
    if target_item.kind == "cube":
        target = CubeDetection(
            track_id=target_item.item_id,
            colour=target_item.destination_colour,
            capture_time_s=capture_time_s,
            position_base_m=target_item.spawn_reference_base_m,
            yaw_rad=math.radians(target_item.spawn_yaw_deg),
            confidence=1.0,
        )
        scene = SceneDetections(identity.group_id, capture_time_s, (target,))
        destination = BinTarget(
            target.colour,
            config.task.bin_centres_base_m[target.colour],
        )
        context = GraspPlanningContext(
            destination=destination,
            release_tip_pose=TipPoseBase(
                (
                    destination.centre_base_m[0],
                    destination.centre_base_m[1],
                    config.bin_release_tip_base_z_m,
                ),
                0.0,
            ),
            pick_tip_z_m=config.pick_tip_base_z_m,
            approach_clearance_m=config.task.approach_clearance_m,
        )
        candidates = CubeYawCandidateGenerator(
            cube_edge_m=config.cube_edge_m
        ).generate(scene, target, context)
        expected_generator_id = CUBE_GENERATOR_ID
        selected_offset_m = None
    elif target_item.kind == "screw":
        target = next(
            screw
            for screw in scenario.screws
            if screw.item_id == target_item.item_id
        )
        detected = DetectedObject(
            track_id=target.item_id,
            class_id=SCREW_CLASS_ID,
            sort_label=target.destination_colour,
            capture_time_s=capture_time_s,
            pose=Pose3D(
                position_base_m=target.spawn_head_base_m,
                orientation_xyzw=_screw_orientation_xyzw(
                    math.radians(target.spawn_yaw_deg)
                ),
            ),
            dimensions_m=(0.012, 0.01199627, 0.030),
            confidence=1.0,
            pose_quality=1.0,
            visibility=1.0,
            source_camera_ids=("configured-isaac-truth",),
        )
        scene = SceneObservation(
            scene_id=identity.group_id,
            capture_time_s=capture_time_s,
            objects=(detected,),
            provider_name="configured-isaac-truth",
            provider_version="1",
            calibration_ids=("simulation-only",),
        )
        generator = ScrewShaftCandidateGenerator()
        candidates = generator.generate(
            scene,
            detected,
            pick_tip_z_m=config.pick_tip_base_z_m,
            approach_clearance_m=config.task.approach_clearance_m,
        )
        expected_generator_id = SCREW_GENERATOR_ID
        selected_offset_m = generator.offsets[
            candidates.index(_candidate_by_id(candidates, identity.candidate_id))
        ]
    else:
        raise ValueError(f"unsupported isolated target kind {target_item.kind!r}")

    candidate = _candidate_by_id(candidates, identity.candidate_id)
    if candidate.generator_id != expected_generator_id:
        raise AssertionError("candidate came from an unexpected generator")
    if candidate.required_opening_m is not None:
        if candidate.required_opening_m > config.gripper.hook_open_gap_m:
            raise ValueError("candidate opening exceeds the configured hook gap")
    if (
        math.hypot(*candidate.pick_tip_pose.xyz_m[:2])
        > config.hardware.maximum_radial_reach_m
    ):
        raise ValueError("candidate pick exceeds configured radial reach")
    if selected_offset_m is not None:
        configured = float(scenario.screw.grasp_offset_from_head_m)
        if not math.isclose(
            configured, selected_offset_m, abs_tol=1e-12, rel_tol=0.0
        ):
            raise ValueError(
                "scenario screw offset does not match the requested candidate: "
                f"{configured} != {selected_offset_m}"
            )
    return candidate


def target_only_builder(
    original: Callable[[Any], Sequence[Any]], target_item_id: str
) -> Callable[[Any], tuple[Any, ...]]:
    def build(config: Any) -> tuple[Any, ...]:
        matches = tuple(
            item for item in original(config) if item.item_id == target_item_id
        )
        if len(matches) != 1:
            raise ValueError(
                f"scenario must contain target {target_item_id!r} exactly once"
            )
        return matches

    return build


def candidate_pick(
    original_pick: dict[str, Any],
    candidate: GraspCandidate,
    *,
    configured_target_yaw_rad: float,
    maximum_position_disagreement_m: float = 0.002,
    maximum_yaw_disagreement_rad: float = math.radians(5.0),
) -> dict[str, Any]:
    """Regenerate candidate yaw around the demo's settled, scene-bound pick."""

    original_xy = tuple(float(value) for value in original_pick["xy"])
    if len(original_xy) != 2:
        raise ValueError("demo pick xy must contain two coordinates")
    configured_xy = candidate.pick_tip_pose.xyz_m[:2]
    movement_m = math.dist(original_xy, configured_xy)
    if movement_m > maximum_position_disagreement_m:
        raise ValueError(
            "settled target disagrees with configured candidate position: "
            f"{movement_m} > {maximum_position_disagreement_m}"
        )

    observed_target_yaw = tip_yaw_from_spread(original_pick["spread_dir"])
    yaw_disagreement = _parallel_axis_distance(
        observed_target_yaw,
        configured_target_yaw_rad,
    )
    if yaw_disagreement > maximum_yaw_disagreement_rad:
        raise ValueError(
            "settled target disagrees with configured candidate yaw: "
            f"{yaw_disagreement} > {maximum_yaw_disagreement_rad}"
        )
    yaw_offset = _wrap_angle(
        candidate.pick_tip_pose.yaw_rad - configured_target_yaw_rad
    )
    executed_yaw = _wrap_angle(observed_target_yaw + yaw_offset)

    clearance = (
        candidate.approach_tip_pose.xyz_m[2]
        - candidate.pick_tip_pose.xyz_m[2]
    )
    if clearance <= 0.0:
        raise ValueError("candidate approach must be above the pick pose")

    patched = dict(original_pick)
    # The shared demo computed this position from settled pre-action truth.
    # Preserve it while applying the catalog candidate's relative yaw.
    patched["xy"] = list(original_xy)
    patched["pick_z"] = float(original_pick["pick_z"])
    patched["approach_clearance_m"] = float(clearance)
    patched["spread_dir"] = list(spread_from_tip_yaw(executed_yaw))
    patched["_team8_executed_tip_yaw_rad"] = executed_yaw
    patched["_team8_position_disagreement_m"] = movement_m
    patched["_team8_yaw_disagreement_rad"] = yaw_disagreement
    return patched


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_state() -> tuple[str, bool, list[str]]:
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
            "config",
            "demos",
            "sorting_core",
            "team_plugins/grasping",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dirty = [line for line in status.splitlines() if line]
    return commit, not dirty, dirty


def _resolved_outcome(
    report: dict[str, Any],
    target_item_id: str,
) -> tuple[str, bool | None]:
    events = report.get("events")
    final_items = report.get("final_scene_verification", {}).get("items", {})
    event = (
        events[0]
        if isinstance(events, list)
        and len(events) == 1
        and isinstance(events[0], dict)
        and events[0].get("item_id") == target_item_id
        else None
    )
    final_item = (
        final_items.get(target_item_id)
        if isinstance(final_items, dict)
        else None
    )
    if event is None or not isinstance(final_item, dict):
        return "unresolved", None
    success = bool(
        event.get("pickup_verified")
        and event.get("placement_verified")
        and not event.get("trajectory_ik_failures")
        and final_item.get("passed")
        and report.get("task_complete")
    )
    return ("resolved_success" if success else "resolved_failure"), success


def _screw_retention_evidence(
    report: dict[str, Any],
    target_item_id: str,
    execution_trace: Mapping[str, Any],
    *,
    maximum_error_m: float = MAXIMUM_HELD_GRASP_POINT_ERROR_M,
) -> dict[str, Any] | None:
    """Compare observed grasp points with commanded held-pose endpoints."""

    if not target_item_id.startswith("screw-"):
        return None
    maximum_error = float(maximum_error_m)
    if not math.isfinite(maximum_error) or maximum_error <= 0.0:
        raise ValueError("maximum retention error must be finite and positive")
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "required_held_phases": list(HELD_RETENTION_PHASES),
        "maximum_grasp_point_error_m": maximum_error,
        "status": "unresolved",
        "passed": None,
        "phase_grasp_point_error_m": {},
    }
    try:
        events = report["events"]
        if (
            not isinstance(events, list)
            or len(events) != 1
            or not isinstance(events[0], Mapping)
            or events[0].get("item_id") != target_item_id
        ):
            raise ValueError("target event is absent or ambiguous")
        event = events[0]
        source_truth = event["source_truth"]
        phase_truth = event["phase_truth"]
        commanded_ends = execution_trace["held_phase_tip_end_xyz_m"]
        if not all(
            isinstance(value, Mapping)
            for value in (source_truth, phase_truth, commanded_ends)
        ):
            raise ValueError("retention truth or command evidence is malformed")
        source_grasp = _finite_xyz(
            source_truth.get("grasp_point_world_m"), "source grasp point"
        )
        pick_tip = _finite_xyz(
            execution_trace.get("pick_tip_xyz_m"), "executed pick tip"
        )
        offset = tuple(
            grasp - tip for grasp, tip in zip(source_grasp, pick_tip)
        )
        phase_errors: dict[str, float] = {}
        for phase in HELD_RETENTION_PHASES:
            sample = phase_truth.get(phase)
            if not isinstance(sample, Mapping):
                raise ValueError(f"{phase} phase truth is absent")
            commanded_tip = _finite_xyz(
                commanded_ends.get(phase), f"{phase} commanded tip endpoint"
            )
            observed = _finite_xyz(
                sample.get("grasp_point_world_m"),
                f"{phase} observed grasp point",
            )
            expected = tuple(
                tip + component for tip, component in zip(commanded_tip, offset)
            )
            phase_errors[phase] = math.dist(observed, expected)
    except (KeyError, TypeError, ValueError) as exc:
        evidence["reason"] = str(exc)
        return evidence
    evidence["source_grasp_point_to_pick_tip_offset_m"] = list(offset)
    evidence["phase_grasp_point_error_m"] = phase_errors
    passed = all(error <= maximum_error for error in phase_errors.values())
    evidence["status"] = "passed" if passed else "failed"
    evidence["passed"] = passed
    return evidence


def _screw_trainer_label(
    report: dict[str, Any],
    target_item_id: str,
    retention_evidence: Mapping[str, Any] | None = None,
) -> bool | None:
    """Return a target-only screw label only with resolved retention evidence."""

    if not target_item_id.startswith("screw-"):
        return None
    status, success = _resolved_outcome(report, target_item_id)
    if status == "unresolved":
        return None
    events = report["events"]
    event = events[0]
    revalidation = event.get("pre_descent_revalidation")
    if not isinstance(revalidation, dict) or not revalidation.get("passed"):
        return None
    phase_truth = event.get("phase_truth")
    required_phases = {
        "lift",
        "carry",
        "lower",
        "release",
        "settle",
        "retreat",
    }
    if not isinstance(phase_truth, dict):
        return None
    if not required_phases.issubset(phase_truth):
        return None
    if not isinstance(retention_evidence, Mapping):
        return None
    retention_status = retention_evidence.get("status")
    retention_passed = retention_evidence.get("passed")
    if retention_status not in {"passed", "failed"}:
        return None
    if type(retention_passed) is not bool:
        return None
    return bool(success and retention_passed)


def _validate_execution_trace(execution_trace: Mapping[str, Any]) -> None:
    required = {
        "pick_tip_xyz_m",
        "pick_tip_yaw_rad",
        "spread_axis",
        "position_disagreement_m",
        "yaw_disagreement_rad",
        "held_phase_tip_end_xyz_m",
    }
    missing = sorted(required - set(execution_trace))
    if missing:
        raise ValueError(
            "execution trace is incomplete: " + ", ".join(missing)
        )
    _finite_xyz(execution_trace["pick_tip_xyz_m"], "executed pick tip")
    _finite_xyz(execution_trace["spread_axis"], "commanded spread axis")
    for key in (
        "pick_tip_yaw_rad",
        "position_disagreement_m",
        "yaw_disagreement_rad",
    ):
        value = float(execution_trace[key])
        if not math.isfinite(value):
            raise ValueError(f"execution trace {key} must be finite")
    endpoints = execution_trace["held_phase_tip_end_xyz_m"]
    if not isinstance(endpoints, Mapping):
        raise ValueError("held-phase command endpoints must be a mapping")
    for phase in HELD_RETENTION_PHASES:
        _finite_xyz(endpoints.get(phase), f"{phase} commanded tip endpoint")


def augment_report(
    report_path: Path,
    *,
    identity: AttemptIdentity,
    candidate: GraspCandidate,
    execution_trace: Mapping[str, Any],
    target_item_id: str,
    scenario_path: Path,
    source_commit: str,
    source_tree_clean: bool,
    source_dirty_entries: Sequence[str],
) -> tuple[str, bool | None]:
    if not report_path.is_file():
        return "unresolved", None
    _validate_execution_trace(execution_trace)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    status, smoke_success = _resolved_outcome(report, target_item_id)
    retention_evidence = _screw_retention_evidence(
        report, target_item_id, execution_trace
    )
    target_only_label = _screw_trainer_label(
        report, target_item_id, retention_evidence
    )
    if target_item_id.startswith("screw-") and target_only_label is not None:
        status = (
            "resolved_success"
            if target_only_label
            else "resolved_failure"
        )
    report["isolated_candidate_execution"] = {
        "runner_id": RUNNER_ID,
        "simulation_only": True,
        "decision_source": (
            "configured candidate transformed onto settled simulator target truth"
        ),
        "candidate_generation_source": "configured scenario target-only truth",
        "candidate_generation_capture_time_s": 0.0,
        "runtime_transform_source": "settled simulator target truth",
        "simulator_obstacle_truth_used_for_candidate_generation": False,
        "simulator_truth_affects_runtime_control": True,
        "camera_perception": False,
        "hardware_integration": False,
        "target_item_id": target_item_id,
        "layout_family_id": identity.layout_family_id,
        "group_id": identity.group_id,
        "candidate_id": identity.candidate_id,
        "split": identity.split,
        "candidate_generator_id": candidate.generator_id,
        "candidate_scene_id": candidate.scene_id,
        "configured_candidate_pick_tip_xyz_m": list(
            candidate.pick_tip_pose.xyz_m
        ),
        "configured_candidate_pick_tip_yaw_rad": (
            candidate.pick_tip_pose.yaw_rad
        ),
        "candidate_approach_tip_xyz_m": list(
            candidate.approach_tip_pose.xyz_m
        ),
        "candidate_required_opening_m": candidate.required_opening_m,
        "executed_pick_tip_xyz_m": list(execution_trace["pick_tip_xyz_m"]),
        "executed_pick_tip_yaw_rad": execution_trace["pick_tip_yaw_rad"],
        "commanded_spread_axis": list(execution_trace["spread_axis"]),
        "configured_to_settled_position_disagreement_m": (
            execution_trace["position_disagreement_m"]
        ),
        "configured_to_settled_yaw_disagreement_rad": (
            execution_trace["yaw_disagreement_rad"]
        ),
        "other_configured_objects_remained_as_physics_obstacles": True,
        "fresh_process_required_per_candidate": True,
        "source_commit": source_commit,
        "relevant_source_tree_clean": source_tree_clean,
        "relevant_source_dirty_entries": list(source_dirty_entries),
        "scenario_path": str(scenario_path),
        "scenario_sha256": hashlib.sha256(
            scenario_path.read_bytes()
        ).hexdigest(),
        "outcome_status": status,
        "simulation_smoke_success": smoke_success,
        "label_scope": "target_only",
        "non_target_obstacle_final_state_evaluated": False,
        "target_only_complete_grasp_label": target_only_label,
        "target_only_label_eligible": target_only_label is not None,
        "collision_safe_label_eligible": False,
        "screw_retention_evidence": retention_evidence,
        "cube_training_eligible": False,
    }
    _atomic_json(report_path, report)
    return status, smoke_success


def _load_demo_modules() -> tuple[ModuleType, ModuleType]:
    from demos.scripted_cube_screw_bucket_sort import (
        cube_screw_bucket_demo as demo,
    )
    from demos.scripted_cube_screw_bucket_sort import scenario

    return demo, scenario


def run(args: argparse.Namespace) -> None:
    scenario_path = Path(args.scenario).expanduser().resolve()
    run_path = Path(args.run).expanduser().resolve()
    if not scenario_path.is_file():
        raise FileNotFoundError(f"scenario does not exist: {scenario_path}")
    if run_path.exists():
        raise FileExistsError(
            f"run directory must not already exist: {run_path}"
        )
    identity = AttemptIdentity(
        args.layout_family_id,
        args.group_id,
        args.candidate_id,
        args.split,
    )
    source_commit, source_tree_clean, source_dirty_entries = _git_state()
    if not source_tree_clean and not args.allow_dirty_source:
        raise RuntimeError(
            "refusing isolated physics collection from dirty relevant sources"
        )

    demo, scenario_module = _load_demo_modules()
    configured = scenario_module.load_scenario_config(scenario_path)
    original_builder = scenario_module.build_sort_items
    target_matches = tuple(
        item
        for item in original_builder(configured)
        if item.item_id == args.target_item
    )
    if len(target_matches) != 1:
        raise ValueError(
            f"scenario must contain target {args.target_item!r} exactly once"
        )
    target_item = target_matches[0]
    candidate = build_candidate(configured, target_item, identity)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.mkdir(exist_ok=False)
    target_only = target_only_builder(original_builder, args.target_item)

    original_demo_builder = demo.build_sort_items
    original_scenario_builder = scenario_module.build_sort_items
    original_loader = demo.load_scenario_config
    original_motion = demo._motion_segments
    original_repo_root = demo.REPO_ROOT
    original_argv = sys.argv[:]
    configured_target_yaw_rad = math.radians(target_item.spawn_yaw_deg)
    execution_trace: dict[str, Any] = {}

    def candidate_motion(pick: dict[str, Any], home: Any, scene_config: Any):
        patched = candidate_pick(
            pick,
            candidate,
            configured_target_yaw_rad=configured_target_yaw_rad,
        )
        segments = original_motion(patched, home, scene_config)
        trace = {
            "pick_tip_xyz_m": [
                *patched["xy"],
                patched["pick_z"],
            ],
            "pick_tip_yaw_rad": patched["_team8_executed_tip_yaw_rad"],
            "spread_axis": list(patched["spread_dir"]),
            "position_disagreement_m": (
                patched["_team8_position_disagreement_m"]
            ),
            "yaw_disagreement_rad": patched[
                "_team8_yaw_disagreement_rad"
            ],
        }
        held_phase_tip_end_xyz_m: dict[str, list[float]] = {}
        expected_axis = patched["spread_dir"]
        for phase, _start, end, _duration, _jaw0, _jaw1, spread in segments:
            if phase in HELD_RETENTION_PHASES:
                if phase in held_phase_tip_end_xyz_m:
                    raise RuntimeError(
                        f"motion contains duplicate {phase!r} segments"
                    )
                held_phase_tip_end_xyz_m[phase] = list(
                    _finite_xyz(end, f"{phase} commanded tip endpoint")
                )
            if phase in {
                "approach_pick",
                "descend_pick",
                "close",
                "dwell",
                "lift",
            }:
                if not same_parallel_axis(spread, expected_axis):
                    raise RuntimeError(
                        f"{phase} did not preserve the selected candidate yaw"
                    )
        missing_held_phases = sorted(
            set(HELD_RETENTION_PHASES) - set(held_phase_tip_end_xyz_m)
        )
        if missing_held_phases:
            raise RuntimeError(
                "motion omitted held phases: " + ", ".join(missing_held_phases)
            )
        trace["held_phase_tip_end_xyz_m"] = held_phase_tip_end_xyz_m
        if execution_trace and execution_trace != trace:
            raise RuntimeError("candidate motion was regenerated inconsistently")
        execution_trace.update(trace)
        return segments

    delegated = [str(Path(demo.__file__).resolve()), "--run", str(run_path)]
    if args.bin_model:
        delegated.extend(["--bin-model", args.bin_model])
    if args.screw_model:
        delegated.extend(["--screw-model", args.screw_model])
    if args.physics_only:
        delegated.append("--physics-only")

    caught: BaseException | None = None
    try:
        scenario_module.build_sort_items = target_only
        demo.build_sort_items = target_only
        demo.load_scenario_config = lambda: scenario_module.load_scenario_config(
            scenario_path
        )
        demo._motion_segments = candidate_motion
        demo.REPO_ROOT = scenario_path.parent
        sys.argv = delegated
        demo.main()
    except BaseException as exc:
        caught = exc
    finally:
        sys.argv = original_argv
        demo.REPO_ROOT = original_repo_root
        demo._motion_segments = original_motion
        demo.load_scenario_config = original_loader
        demo.build_sort_items = original_demo_builder
        scenario_module.build_sort_items = original_scenario_builder

    execution_caught = caught
    report_path = run_path / "execution_report.json"
    augmentation_caught: BaseException | None = None
    status: str = "unresolved"
    success: bool | None = None
    try:
        status, success = augment_report(
            report_path,
            identity=identity,
            candidate=candidate,
            execution_trace=execution_trace,
            target_item_id=args.target_item,
            scenario_path=scenario_path,
            source_commit=source_commit,
            source_tree_clean=source_tree_clean,
            source_dirty_entries=source_dirty_entries,
        )
    except BaseException as exc:
        augmentation_caught = exc
        if caught is None:
            caught = exc
    if caught is not None:
        status = "unresolved"
        success = None
    metadata = {
        "runner_id": RUNNER_ID,
        "identity": {
            "layout_family_id": identity.layout_family_id,
            "group_id": identity.group_id,
            "candidate_id": identity.candidate_id,
            "split": identity.split,
        },
        "target_item_id": args.target_item,
        "scenario_path": str(scenario_path),
        "scenario_sha256": hashlib.sha256(
            scenario_path.read_bytes()
        ).hexdigest(),
        "source_commit": source_commit,
        "relevant_source_tree_clean": source_tree_clean,
        "relevant_source_dirty_entries": source_dirty_entries,
        "execution_trace": execution_trace or None,
        "outcome_status": status,
        "simulation_smoke_success": success,
        "execution_exception": (
            None
            if execution_caught is None
            else {
                "type": type(execution_caught).__name__,
                "message": str(execution_caught),
            }
        ),
        "augmentation_exception": (
            None
            if augmentation_caught is None
            else {
                "type": type(augmentation_caught).__name__,
                "message": str(augmentation_caught),
            }
        ),
        "exception": (
            None
            if caught is None
            else {
                "type": type(caught).__name__,
                "message": str(caught),
            }
        ),
    }
    _atomic_json(run_path / "attempt_metadata.json", metadata)
    if caught is not None:
        raise caught


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--target-item", required=True)
    parser.add_argument("--layout-family-id", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--split", default="pilot")
    parser.add_argument("--bin-model")
    parser.add_argument("--screw-model")
    parser.add_argument("--physics-only", action="store_true")
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="development only; recorded in every attempt",
    )
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
