"""Deterministic, fail-closed colour-to-bin selection and semantic planning."""

from __future__ import annotations

from itertools import combinations
import math

from .config import SortConfig
from .contracts import (
    BinTarget,
    CubeDetection,
    MotionPhase,
    PickPlacePlan,
    SceneDetections,
    TipPoseBase,
)


class PlanningError(ValueError):
    """Raised when camera observations are not unambiguous enough to execute."""


def build_colour_sort_plans(
    scene: SceneDetections, config: SortConfig
) -> tuple[PickPlacePlan, ...]:
    """Return exactly one plan per configured colour, in configured order.

    The initial task deliberately requires one red, one yellow, and one blue
    cube.  Missing colours, unknown colours, duplicate same-colour targets, and
    centres closer than the ambiguity threshold all fail closed.  The function
    consumes only camera detections and known bin configuration.
    """

    known = set(config.task.colour_order)
    unknown = sorted({cube.colour for cube in scene.cubes if cube.colour not in known})
    if unknown:
        raise PlanningError(f"unknown cube colours: {', '.join(unknown)}")

    by_colour: dict[str, list[CubeDetection]] = {colour: [] for colour in known}
    track_ids: set[str] = set()
    for cube in scene.cubes:
        if cube.track_id in track_ids:
            raise PlanningError(f"duplicate target identity: {cube.track_id}")
        track_ids.add(cube.track_id)
        by_colour[cube.colour].append(cube)

    missing = [colour for colour in config.task.colour_order if not by_colour[colour]]
    if missing:
        raise PlanningError(f"missing cube colours: {', '.join(missing)}")
    duplicates = [
        colour for colour in config.task.colour_order if len(by_colour[colour]) != 1
    ]
    if duplicates:
        raise PlanningError(
            "ambiguous duplicate cube colours: " + ", ".join(duplicates)
        )

    selected = [by_colour[colour][0] for colour in config.task.colour_order]
    reach = config.hardware.maximum_radial_reach_m
    for cube in selected:
        radius = math.hypot(cube.position_base_m[0], cube.position_base_m[1])
        if radius > reach:
            raise PlanningError(
                f"target {cube.track_id} exceeds radial reach: "
                f"{radius:.4f} m > {reach:.4f} m"
            )
    for colour in config.task.colour_order:
        bx, by = config.task.bin_centres_base_m[colour]
        radius = math.hypot(bx, by)
        if radius > reach:
            raise PlanningError(
                f"{colour} bin exceeds radial reach: {radius:.4f} m > {reach:.4f} m"
            )
    minimum = config.task.minimum_distinct_centres_m
    for first, second in combinations(selected, 2):
        separation = math.dist(first.position_base_m[:2], second.position_base_m[:2])
        if separation < minimum:
            raise PlanningError(
                f"targets {first.track_id} and {second.track_id} are too close: "
                f"{separation:.4f} m < {minimum:.4f} m"
            )

    plans = []
    for colour, cube in zip(config.task.colour_order, selected):
        bx, by = config.task.bin_centres_base_m[colour]
        destination = BinTarget(colour=colour, centre_base_m=(bx, by))
        plans.append(
            PickPlacePlan(
                plan_id=f"{scene.scene_id}:{cube.track_id}:{colour}",
                scene_id=scene.scene_id,
                capture_time_s=scene.capture_time_s,
                target=cube,
                destination=destination,
                pick_tip_pose=TipPoseBase(
                    xyz_m=(
                        cube.position_base_m[0],
                        cube.position_base_m[1],
                        config.pick_tip_base_z_m,
                    ),
                    yaw_rad=cube.yaw_rad,
                ),
                release_tip_pose=TipPoseBase(
                    xyz_m=(bx, by, config.bin_release_tip_base_z_m),
                    yaw_rad=0.0,
                ),
            )
        )
    return tuple(plans)


def semantic_phase_plan(
    plan: PickPlacePlan, config: SortConfig
) -> tuple[MotionPhase, ...]:
    """Build the shared task semantics for simulator and hardware adapters.

    Backends may use different controllers and timing, but they must preserve
    this order and stop at the revalidation gate before descending.
    """

    open_gripper = config.gripper.open_command
    closed_gripper = config.gripper.closed_command
    clearance = config.task.approach_clearance_m
    px, py, pz = plan.pick_tip_pose.xyz_m
    dx, dy, dz = plan.release_tip_pose.xyz_m
    above_pick = TipPoseBase((px, py, pz + clearance), plan.pick_tip_pose.yaw_rad)
    above_bin = TipPoseBase((dx, dy, dz + clearance), plan.release_tip_pose.yaw_rad)

    return (
        MotionPhase("approach_pick", above_pick, open_gripper),
        MotionPhase(
            "revalidate_before_descent",
            None,
            open_gripper,
            requires_reobservation_before=True,
        ),
        MotionPhase("descend_pick", plan.pick_tip_pose, open_gripper),
        MotionPhase("close", plan.pick_tip_pose, closed_gripper, dwell_s=0.6),
        MotionPhase("dwell", plan.pick_tip_pose, closed_gripper, dwell_s=0.5),
        MotionPhase("lift", above_pick, closed_gripper),
        MotionPhase("carry", above_bin, closed_gripper),
        MotionPhase("lower", plan.release_tip_pose, closed_gripper),
        MotionPhase("release", plan.release_tip_pose, open_gripper, dwell_s=0.6),
        MotionPhase("settle", plan.release_tip_pose, open_gripper, dwell_s=0.6),
        MotionPhase("retreat", above_bin, open_gripper),
    )
