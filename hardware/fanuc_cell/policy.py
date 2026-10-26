"""Colour-cube sorting policy for the official SIDC Lab FANUC cell client.

This module is the hardware adapter for the backend-neutral task contract in
``sorting_core``. It deliberately uses the official HTTPS ``Policy.act``
interface; it does not connect to FANUC, ROS, a PLC, or a safety controller.

The checked-in default is a hold-only diagnostic: it requires a fresh measured
TCP and gripper state and repeats exactly one measured row without running
perception or planning. Any planned movement requires
``ROCKWELL_ENABLE_PHYSICAL_MOTION=1`` at rollout reset. Descent additionally
requires ``ROCKWELL_ENABLE_PHYSICAL_DESCENT=1``. A repeated TCP target can still
cause small controller corrections and is not an emergency stop; the cell
operator/controller remains responsible for stopping.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


# ``run.py`` is normally launched from this directory. Tests and copied
# clients are not, so make both the vendored calibration package and repository
# root explicit before importing either one.
HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
for path in (HERE, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calibration.geometry import load as load_calibration  # noqa: E402
from calibration.geometry import pixel_to_base  # noqa: E402
from sorting_core import (  # noqa: E402
    ActionValidationError,
    BinTarget,
    CubeDetection,
    MotionPhase,
    PickPlacePlan,
    PlanningError,
    SceneDetections,
    TipPoseBase,
    VisionError,
    base_tip_pose_to_uf0_tcp,
    build_colour_sort_plans,
    detection_in_pickup_region,
    detect_coloured_cubes,
    interpolate_tcp_rows,
    load_config,
    semantic_phase_plan,
    validate_tcp_chunk,
    validate_reobserved_scene,
)


def _environment_flag(name: str) -> bool:
    return os.environ.get(name, "").strip() == "1"


def _finite_environment_float(name: str, default: float, *, minimum: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def _shortest_angle_degrees(first: float, second: float) -> float:
    return (float(first) - float(second) + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class _SceneViews:
    """Broad outcome observations and the narrower pick-eligible subset."""

    unplaced: SceneDetections
    pickable: SceneDetections


class Policy:
    """Persistent, fail-closed colour-to-bucket state machine.

    The camera supplies target positions. Known dimensions, bin positions,
    phase semantics, and action limits come from the shared checked-in config.
    Each ``act`` call returns at most one short absolute-TCP action chunk.
    """

    CAMERAS = ["table"]
    IMAGE_SIZE = (1920, 1080)  # native table RGB; calibration pixels stay valid
    IMAGE_FORMAT = "png"  # preserve tiny cube edges until JPEG is validated on real images
    ACTION = "tcp"
    PREFETCH = False  # every short phase is followed by a fresh observation
    DEPTH: list[str] = []  # depth is not recorded; start with reproducible RGB only
    DT = 0.1

    MOTION_INTERLOCK_ENV = "ROCKWELL_ENABLE_PHYSICAL_MOTION"
    DESCENT_INTERLOCK_ENV = "ROCKWELL_ENABLE_PHYSICAL_DESCENT"
    MAX_ACTIONS_PER_CALL = 8
    MAX_MOTION_CALLS_PER_PHASE = 20
    MAX_NO_PROGRESS_CALLS = 4
    MIN_MEASURED_PROGRESS_MM = 0.5
    MIN_MEASURED_ANGULAR_PROGRESS_DEG = 0.5
    GRIPPER_FEEDBACK_TOLERANCE = 0.1
    _OPEN_FEEDBACK_PHASES = frozenset(
        {"approach_pick", "descend_pick", "settle", "retreat"}
    )
    _CLOSED_FEEDBACK_PHASES = frozenset({"dwell", "lift", "carry", "lower"})

    def __init__(self, checkpoint: str | None = None):
        self.config = load_config()
        calibration = load_calibration()
        if "table" not in calibration:
            raise RuntimeError("official calibration does not contain the table camera")
        self._table_camera = calibration["table"]
        if tuple(int(v) for v in self._table_camera["size"]) != self.IMAGE_SIZE:
            raise RuntimeError(
                "Policy.IMAGE_SIZE must match the official table-camera calibration"
            )

        self._max_actions_per_call = min(
            self.MAX_ACTIONS_PER_CALL,
            self.config.hardware.max_chunk_actions,
        )
        requested_max = os.environ.get("ROCKWELL_MAX_ACTIONS_PER_ACT")
        if requested_max is not None:
            try:
                parsed = int(requested_max)
            except ValueError as exc:
                raise ValueError("ROCKWELL_MAX_ACTIONS_PER_ACT must be an integer") from exc
            if not 1 <= parsed <= self._max_actions_per_call:
                raise ValueError(
                    "ROCKWELL_MAX_ACTIONS_PER_ACT must be between 1 and "
                    f"{self._max_actions_per_call}; larger open-loop chunks are disabled"
                )
            self._max_actions_per_call = parsed

        self._maximum_observation_age_s = _finite_environment_float(
            "ROCKWELL_MAX_OBSERVATION_AGE_S", 3.0, minimum=0.1
        )
        self._maximum_motion_m = _finite_environment_float(
            "ROCKWELL_MAX_TARGET_MOTION_M", 0.005, minimum=0.0
        )
        self._maximum_yaw_motion_deg = _finite_environment_float(
            "ROCKWELL_MAX_TARGET_YAW_MOTION_DEG", 15.0, minimum=0.0
        )
        self._minimum_hardware_clearance_m = _finite_environment_float(
            "ROCKWELL_MIN_HARDWARE_CENTRE_CLEARANCE_M", 0.030, minimum=0.030
        )
        self._minimum_confidence = _finite_environment_float(
            "ROCKWELL_MIN_DETECTION_CONFIDENCE",
            self.config.vision.minimum_detection_confidence,
            minimum=self.config.vision.minimum_detection_confidence,
        )
        if self._maximum_observation_age_s > 3.0:
            raise ValueError("ROCKWELL_MAX_OBSERVATION_AGE_S cannot exceed 3.0")
        if self._maximum_motion_m > 0.005:
            raise ValueError("ROCKWELL_MAX_TARGET_MOTION_M cannot exceed 0.005")
        if self._maximum_yaw_motion_deg > 15.0:
            raise ValueError("ROCKWELL_MAX_TARGET_YAW_MOTION_DEG cannot exceed 15.0")
        if not (
            self.config.vision.minimum_detection_confidence
            <= self._minimum_confidence
            <= 1.0
        ):
            raise ValueError(
                "ROCKWELL_MIN_DETECTION_CONFIDENCE must be in "
                f"[{self.config.vision.minimum_detection_confidence:.2f}, 1]"
            )
        self._position_tolerance_mm = 2.0
        self._angle_tolerance_deg = 2.0

        # Used only when the official mock has no initial TCP pose. A real
        # observation with missing/invalid current TCP is rejected rather than
        # reusing stale state or inventing a new move.
        scan_xy = self.config.task.cube_spawns_base_xy_yaw_deg["yellow"][:2]
        scan_tip = TipPoseBase(
            (
                scan_xy[0],
                scan_xy[1],
                self.config.pick_tip_base_z_m
                + self.config.task.approach_clearance_m,
            ),
            0.0,
        )
        self._scan_tcp = tuple(
            base_tip_pose_to_uf0_tcp(
                scan_tip, self.config.gripper.open_command, self.config
            )
        )
        validate_tcp_chunk([self._scan_tcp], self.DT, self.config)

        self._checkpoint = checkpoint
        self._emit(
            "policy_startup",
            backend="sidclab_https_tcp",
            cube_edge_mm=self.config.cube_edge_m * 1000.0,
            tcp_beyond_fingertips_mm=self.config.frames.tcp_beyond_fingertips_mm,
            motion_interlock_env=self.MOTION_INTERLOCK_ENV,
            descent_interlock_env=self.DESCENT_INTERLOCK_ENV,
            default_mode="hold-only diagnostic; no perception or planning",
            hover_requires_motion_interlock=True,
            descent_requires_both_interlocks=True,
            minimum_hardware_centre_clearance_mm=(
                self._minimum_hardware_clearance_m * 1000.0
            ),
            warning=self.config.frames.tcp_conflict_note,
        )
        self.reset()

    def reset(self) -> None:
        """Reset all task state at the start of an official rollout."""

        self._active_plan: PickPlacePlan | None = None
        self._phases: tuple[MotionPhase, ...] = ()
        self._phase_index = 0
        self._phase_dwell_rows_remaining: int | None = None
        self._reset_phase_watchdog()
        self._completed_colours: set[str] = set()
        self._pending_verification_colour: str | None = None
        self._reference_detections: dict[str, CubeDetection] = {}
        self._blocked_reason: str | None = None
        self._last_observation_time_s: float | None = None
        self._clock_offset_estimate_s: float | None = None
        self._last_valid_tcp: tuple[float, ...] | None = None
        self._last_valid_gripper_feedback: float | None = None
        self._last_commanded_gripper: float | None = None
        self._last_hold_log: str | None = None
        self._scene_number = 0
        self._motion_enabled = _environment_flag(self.MOTION_INTERLOCK_ENV)
        self._descent_enabled = _environment_flag(self.DESCENT_INTERLOCK_ENV)
        self._emit(
            "rollout_reset",
            physical_motion_enabled=self._motion_enabled,
            physical_descent_enabled=self._descent_enabled,
            active_assumption="10 mm cubes; UTOOL1 TCP at fingertips (0 mm extra)",
            warning=(
                "The pinned client labels its action point as the fingertip. "
                "Separately, an organizer confirmed standard-finger table contact "
                "near UF0 z=-335 mm and said the former -345 mm FAQ value was for "
                "shorter fingers. Verify the installed fingers and active "
                "UTOOL1/UF0 before setting the descent interlock."
            ),
        )

    def act(self, obs: dict):
        """Return one bounded TCP chunk for the latest official observation.

        Invalid perception never produces a grasp command. Once motion has
        started, a stale/moved/ambiguous scene latches the rollout into a hold;
        there is no blind retry. The returned hold is not an emergency stop.
        """

        if self._blocked_reason is not None:
            return self._hold_action(obs, self._blocked_reason)

        try:
            self._accept_observation_timestamp(obs)
        except (TypeError, ValueError) as exc:
            if not self._motion_enabled:
                raise ActionValidationError(
                    "hold-only diagnostic requires a fresh observation timestamp; "
                    "no action was constructed"
                ) from exc
            if self._active_plan is not None:
                self._block(f"stale or invalid observation during motion: {exc}")
            return self._hold_action(obs, f"observation rejected: {exc}")

        if not self._motion_enabled:
            return self._strict_diagnostic_hold_action(
                obs,
                "physical motion interlock is closed; repeating one fresh "
                "measured TCP/gripper row without perception or planning. Set "
                f"{self.MOTION_INTERLOCK_ENV}=1 only for a supervised motion session",
            )

        # Record a valid measured TCP before perception. Only the official mock
        # may reuse it if the mock omits TCP later; a real malformed/missing TCP
        # is rejected instead of continuing from stale state.
        self._measured_tcp_row(obs)

        try:
            if self._active_plan is None:
                scenes = self._detect_scene_views(obs)
                if self._pending_verification_colour is not None:
                    self._verify_post_place(scenes.unplaced)
                if len(self._completed_colours) == len(self.config.task.colour_order):
                    return self._hold_action(
                        obs, "all configured colours camera-checked after placement"
                    )
                self._start_next_plan(scenes.pickable)

            phase = self._phases[self._phase_index]
            if phase.requires_reobservation_before:
                scenes = self._detect_scene_views(obs)
                self._revalidate_before_descent(scenes.pickable)
                if not self._descent_enabled:
                    return self._hold_action(
                        obs,
                        "physical descent interlock is closed; holding at hover. "
                        f"Set {self.DESCENT_INTERLOCK_ENV}=1 only after supervised "
                        "10 mm cube and fingertip-TCP verification",
                    )
                # The revalidated observation directly supplies the next
                # descent command; no queued action separates the gate from it.
                phase = self._phases[self._phase_index]

            return self._execute_phase(obs, phase)
        except (ActionValidationError, PlanningError, VisionError, ValueError) as exc:
            if (
                self._active_plan is not None
                or self._pending_verification_colour is not None
            ):
                self._block(str(exc))
            return self._hold_action(obs, f"policy failed closed: {exc}")

    def _accept_observation_timestamp(self, obs: dict) -> None:
        timestamp = float(obs.get("t"))
        if not math.isfinite(timestamp):
            raise ValueError("obs['t'] is not finite")
        # obs['t'] is on the cell clock. CellClient measures its offset but does
        # not pass that value into Policy, so raw wall-clock subtraction would
        # incorrectly reject a healthy cell whose clock has a constant offset.
        # Establish that offset from the first freshly fetched observation and
        # only reject *additional* age thereafter.
        observed_offset = time.time() - timestamp
        if self._clock_offset_estimate_s is None:
            self._clock_offset_estimate_s = observed_offset
        relative_age = observed_offset - self._clock_offset_estimate_s
        if relative_age > self._maximum_observation_age_s:
            raise ValueError(
                f"obs['t'] gained {relative_age:.2f} s of age "
                f"(limit {self._maximum_observation_age_s:.2f} s)"
            )
        if (
            self._last_observation_time_s is not None
            and timestamp <= self._last_observation_time_s
        ):
            raise ValueError("obs['t'] did not advance")
        self._last_observation_time_s = timestamp

    def _detect_scene_views(self, obs: dict) -> _SceneViews:
        images = obs.get("images")
        if not isinstance(images, dict) or "table" not in images:
            raise VisionError("obs['images']['table'] is required")
        image = np.asarray(images["table"])
        if image.shape != (self.IMAGE_SIZE[1], self.IMAGE_SIZE[0], 3):
            raise VisionError(
                "table RGB must be the uncropped native 1920x1080 frame for calibration"
            )

        capture_time = float(obs["t"])
        trial = obs.get("trial") or {}
        trial_id = str(trial.get("trial_id") or "unknown-trial")
        self._scene_number += 1
        raw = detect_coloured_cubes(
            image,
            lambda uv, z: pixel_to_base(self._table_camera, uv, z=z),
            self.config,
            scene_id=f"{trial_id}:scene-{self._scene_number}",
            capture_time_s=capture_time,
        )

        # The bins are themselves red/yellow/blue. Exclude only a component in
        # its same-colour destination footprint. A cube of one colour near a
        # different bin is a wrong-bin outcome and must remain visible/fail.
        # Confident non-bin detections remain in the broader outcome view so a
        # misplaced/completed colour fails verification. Only the calibrated
        # pickup-region subset may become a target. This partition is not an
        # obstacle map or swept-volume collision check.
        unplaced: list[CubeDetection] = []
        pickable: list[CubeDetection] = []
        bin_components_excluded = 0
        outside_pickup_region_excluded = 0
        low_confidence_outside_region_retained = 0
        for detection in raw.cubes:
            bin_membership = {
                colour: (
                    abs(
                        detection.position_base_m[0]
                        - self.config.task.bin_centres_base_m[colour][0]
                    )
                    <= self.config.bin.outer_m / 2.0
                    and abs(
                        detection.position_base_m[1]
                        - self.config.task.bin_centres_base_m[colour][1]
                    )
                    <= self.config.bin.outer_m / 2.0
                )
                for colour in self.config.task.colour_order
            }
            if bin_membership[detection.colour]:
                bin_components_excluded += 1
                continue
            wrong_bins = [
                colour for colour, inside in bin_membership.items()
                if colour != detection.colour and inside
            ]
            if wrong_bins:
                raise PlanningError(
                    f"{detection.colour} cube detected in/near the "
                    f"{wrong_bins[0]} bin; wrong-bin placements are not ignored"
                )
            in_pickup_region = detection_in_pickup_region(detection, self.config)
            if detection.confidence < self._minimum_confidence:
                if in_pickup_region:
                    raise PlanningError(
                        f"low-confidence {detection.colour} detection "
                        f"({detection.confidence:.2f} < {self._minimum_confidence:.2f})"
                    )
                # It is never pickable, but erasing weak evidence here could
                # turn a dropped cube into a false successful placement. Keep
                # it in the outcome view so matching/completed-colour checks
                # fail closed until fixture colour is calibrated explicitly.
                unplaced.append(detection)
                low_confidence_outside_region_retained += 1
                continue
            unplaced.append(detection)
            if not in_pickup_region:
                # Preserve a confident non-bin observation for outcome checks,
                # but never promote it to a pick target. This catches a dropped
                # cube outside the pickup region instead of declaring success.
                outside_pickup_region_excluded += 1
                continue
            pickable.append(detection)

        unplaced_scene = SceneDetections(
            raw.scene_id, raw.capture_time_s, tuple(unplaced)
        )
        pickable_scene = SceneDetections(
            raw.scene_id, raw.capture_time_s, tuple(pickable)
        )
        self._emit(
            "camera_scene",
            scene_id=raw.scene_id,
            unplaced_detections=len(unplaced_scene.cubes),
            pickable_detections=len(pickable_scene.cubes),
            raw_detections=len(raw.cubes),
            bin_components_excluded=bin_components_excluded,
            outside_pickup_region_excluded=outside_pickup_region_excluded,
            low_confidence_outside_region_retained=(
                low_confidence_outside_region_retained
            ),
            pickable_colours=[cube.colour for cube in pickable_scene.cubes],
        )
        return _SceneViews(unplaced=unplaced_scene, pickable=pickable_scene)

    def _start_next_plan(self, scene: SceneDetections) -> None:
        self._validate_hardware_clearance(scene.cubes)
        if not self._reference_detections:
            plans = build_colour_sort_plans(scene, self.config)
            self._reference_detections = {
                plan.target.colour: plan.target for plan in plans
            }
            plan = plans[0]
        else:
            current = self._validate_scene_against_reference(scene)
            next_colour = next(
                colour
                for colour in self.config.task.colour_order
                if colour not in self._completed_colours
            )
            plan = self._plan_for_detection(current[next_colour], scene.scene_id)
        self._set_active_plan(plan)

    def _set_active_plan(self, plan: PickPlacePlan) -> None:
        self._active_plan = plan
        self._phases = semantic_phase_plan(plan, self.config)
        self._phase_index = 0
        self._phase_dwell_rows_remaining = None
        self._reset_phase_watchdog()
        self._last_hold_log = None
        self._emit(
            "plan_selected",
            plan_id=plan.plan_id,
            colour=plan.target.colour,
            target_xyz_m=list(plan.target.position_base_m),
            phase_names=[phase.name for phase in self._phases],
        )

    def _plan_for_detection(
        self, detection: CubeDetection, scene_id: str
    ) -> PickPlacePlan:
        colour = detection.colour
        bin_x, bin_y = self.config.task.bin_centres_base_m[colour]
        return PickPlacePlan(
            plan_id=f"{scene_id}:{detection.track_id}:{colour}",
            scene_id=scene_id,
            capture_time_s=detection.capture_time_s,
            target=detection,
            destination=BinTarget(colour, (bin_x, bin_y)),
            pick_tip_pose=TipPoseBase(
                (
                    detection.position_base_m[0],
                    detection.position_base_m[1],
                    self.config.pick_tip_base_z_m,
                ),
                detection.yaw_rad,
            ),
            release_tip_pose=TipPoseBase(
                (bin_x, bin_y, self.config.bin_release_tip_base_z_m), 0.0
            ),
        )

    def _validate_scene_against_reference(
        self, scene: SceneDetections
    ) -> dict[str, CubeDetection]:
        selected = validate_reobserved_scene(
            scene,
            self._reference_detections,
            self._completed_colours,
            self.config,
            maximum_motion_m=self._maximum_motion_m,
            maximum_yaw_motion_deg=self._maximum_yaw_motion_deg,
        )
        selected_values = tuple(selected.values())
        self._validate_hardware_clearance(selected_values)
        return selected

    def _validate_hardware_clearance(
        self, detections: tuple[CubeDetection, ...] | list[CubeDetection]
    ) -> None:
        values = tuple(detections)
        for index, first in enumerate(values):
            for second in values[index + 1 :]:
                distance = math.dist(
                    first.position_base_m[:2], second.position_base_m[:2]
                )
                if distance < self._minimum_hardware_clearance_m:
                    raise PlanningError(
                        f"{first.colour} and {second.colour} violate the "
                        f"hardware grasp clearance: {distance * 1000.0:.1f} mm "
                        f"< {self._minimum_hardware_clearance_m * 1000.0:.1f} mm"
                    )

    def _revalidate_before_descent(self, scene: SceneDetections) -> None:
        if self._active_plan is None:
            raise PlanningError("revalidation has no active plan")
        current = self._validate_scene_against_reference(scene)
        colour = self._active_plan.target.colour
        refreshed = self._plan_for_detection(current[colour], scene.scene_id)
        # Keep the original identity in logs while binding motion to the fresh
        # scene. The detector's scene-local track IDs are not persistent IDs.
        # Keep the ID bounded if the closed descent interlock causes repeated
        # revalidation polls at the hover pose.
        refreshed = PickPlacePlan(
            plan_id=f"{refreshed.plan_id}:revalidated",
            scene_id=refreshed.scene_id,
            capture_time_s=refreshed.capture_time_s,
            target=refreshed.target,
            destination=refreshed.destination,
            pick_tip_pose=refreshed.pick_tip_pose,
            release_tip_pose=refreshed.release_tip_pose,
        )
        self._active_plan = refreshed
        self._phases = semantic_phase_plan(refreshed, self.config)
        gate_index = next(
            index
            for index, phase in enumerate(self._phases)
            if phase.requires_reobservation_before
        )
        self._phase_index = gate_index
        self._phase_dwell_rows_remaining = None
        self._reset_phase_watchdog()
        self._emit(
            "pre_descent_revalidation",
            plan_id=refreshed.plan_id,
            result="accepted",
            physical_descent_enabled=self._descent_enabled,
        )
        if self._descent_enabled:
            self._phase_index += 1

    def _execute_phase(self, obs: dict, phase: MotionPhase):
        if phase.target is None:
            raise PlanningError(f"phase {phase.name} has no TCP target")
        current = np.asarray(self._current_tcp_row(obs), dtype=float)
        if phase.name in self._OPEN_FEEDBACK_PHASES:
            self._require_gripper_feedback(
                obs, self.config.gripper.open_command, phase.name
            )
        elif phase.name in self._CLOSED_FEEDBACK_PHASES:
            self._require_gripper_feedback(
                obs, self.config.gripper.closed_command, phase.name
            )
        target = np.asarray(
            base_tip_pose_to_uf0_tcp(phase.target, phase.gripper, self.config),
            dtype=float,
        )
        target = self._canonicalise_target_angles(target, current)
        validate_tcp_chunk([target], self.DT, self.config)

        position_error = float(np.linalg.norm(current[:3] - target[:3]))
        angle_error = max(
            abs(_shortest_angle_degrees(current[index], target[index]))
            for index in range(3, 6)
        )
        at_target = (
            position_error <= self._position_tolerance_mm
            and angle_error <= self._angle_tolerance_deg
        )

        if not at_target:
            self._phase_motion_calls += 1
            position_improved = position_error < (
                self._phase_best_position_error_mm - self.MIN_MEASURED_PROGRESS_MM
            )
            angle_improved = angle_error < (
                self._phase_best_angle_error_deg
                - self.MIN_MEASURED_ANGULAR_PROGRESS_DEG
            )
            self._phase_best_position_error_mm = min(
                self._phase_best_position_error_mm, position_error
            )
            self._phase_best_angle_error_deg = min(
                self._phase_best_angle_error_deg, angle_error
            )
            if position_improved or angle_improved:
                self._phase_no_progress_calls = 0
            else:
                self._phase_no_progress_calls += 1
            if self._phase_motion_calls > self.MAX_MOTION_CALLS_PER_PHASE:
                raise PlanningError(
                    f"phase {phase.name} exceeded "
                    f"{self.MAX_MOTION_CALLS_PER_PHASE} measured motion calls"
                )
            if self._phase_no_progress_calls >= self.MAX_NO_PROGRESS_CALLS:
                raise PlanningError(
                    f"phase {phase.name} made no measured TCP progress for "
                    f"{self._phase_no_progress_calls} calls"
                )
            rows = interpolate_tcp_rows(current, target, self.config)
            rows = rows[: self._max_actions_per_call]
            rows = validate_tcp_chunk(rows, self.DT, self.config)
            self._emit(
                "phase_action",
                phase=phase.name,
                action_rows=len(rows),
                position_error_mm=round(position_error, 2),
                angle_error_deg=round(angle_error, 2),
            )
            self._last_commanded_gripper = float(rows[-1, 6])
            return rows.tolist()

        if self._phase_dwell_rows_remaining is None:
            self._phase_dwell_rows_remaining = max(
                1, math.ceil(phase.dwell_s / self.DT)
            )
        count = min(
            self._max_actions_per_call, self._phase_dwell_rows_remaining
        )
        rows = np.repeat(target[None, :], count, axis=0)
        rows = validate_tcp_chunk(rows, self.DT, self.config)
        self._phase_dwell_rows_remaining -= count
        phase_finished = self._phase_dwell_rows_remaining == 0
        self._emit(
            "phase_action",
            phase=phase.name,
            action_rows=len(rows),
            target_reached=True,
            phase_finished=phase_finished,
        )
        if phase_finished:
            self._phase_index += 1
            self._phase_dwell_rows_remaining = None
            self._reset_phase_watchdog()
            if self._phase_index >= len(self._phases):
                self._finish_active_plan()
        self._last_commanded_gripper = float(rows[-1, 6])
        return rows.tolist()

    def _finish_active_plan(self) -> None:
        if self._active_plan is None:
            raise PlanningError("cannot finish a missing plan")
        colour = self._active_plan.target.colour
        self._pending_verification_colour = colour
        self._emit(
            "plan_motion_complete",
            colour=colour,
            verification="pending fresh table-camera observation",
        )
        self._active_plan = None
        self._phases = ()
        self._phase_index = 0
        self._phase_dwell_rows_remaining = None
        self._reset_phase_watchdog()

    def _reset_phase_watchdog(self) -> None:
        self._phase_motion_calls = 0
        self._phase_no_progress_calls = 0
        self._phase_best_position_error_mm = math.inf
        self._phase_best_angle_error_deg = math.inf

    @staticmethod
    def _canonicalise_target_angles(
        target: np.ndarray, current: np.ndarray
    ) -> np.ndarray:
        """Choose equivalent W/P/R values closest to the measured values.

        FANUC accepts angles within +-360 degrees. Keeping the numeric target
        close avoids a nominal +180 -> -180 representation change becoming an
        unintended 360-degree command at the end of interpolation.
        """

        result = np.asarray(target, dtype=float).copy()
        for index in range(3, 6):
            equivalents = [
                result[index] + 360.0 * turns
                for turns in range(-2, 3)
                if abs(result[index] + 360.0 * turns) <= 360.0
            ]
            result[index] = min(
                equivalents, key=lambda value: abs(value - current[index])
            )
        return result

    def _verify_post_place(self, scene: SceneDetections) -> None:
        colour = self._pending_verification_colour
        if colour is None:
            raise PlanningError("post-place verification has no pending colour")

        forbidden = self._completed_colours | {colour}
        visible = sorted({cube.colour for cube in scene.cubes if cube.colour in forbidden})
        if visible:
            raise PlanningError(
                "post-place verification found sorted colour(s) outside their "
                f"matching bin: {', '.join(visible)}; no automatic retry"
            )

        self._completed_colours.add(colour)
        self._pending_verification_colour = None
        self._emit(
            "plan_complete",
            colour=colour,
            completed_colours=sorted(self._completed_colours),
            verification=(
                "fresh table RGB found no matching-colour cube outside the "
                "configured same-colour bin footprint"
            ),
            limitation=(
                "This is not independent grasp/force confirmation; occlusion or "
                "a missed detection can still hide a failed placement."
            ),
        )

    def _measured_tcp_row(self, obs: dict) -> tuple[float, ...] | None:
        pose = obs.get("tcp_pose")
        try:
            values = np.asarray(pose, dtype=float)
        except (TypeError, ValueError):
            return None
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            return None
        feedback = self._current_gripper_feedback(obs)
        if feedback is not None:
            self._last_valid_gripper_feedback = feedback

        # A measured gripper value may lag the command that is still keeping a
        # part clamped.  Reissuing that stale value during a hold could release
        # the part, so preserve the last action command first.  Before any
        # command, use current/last valid feedback.  If neither has ever been
        # valid, refuse the action instead of inventing a pneumatic command.
        if self._last_commanded_gripper is not None:
            gripper = self._last_commanded_gripper
        elif feedback is not None:
            gripper = feedback
        elif self._last_valid_gripper_feedback is not None:
            gripper = self._last_valid_gripper_feedback
        else:
            # There is no universally safe invented pneumatic state: opening
            # can drop a part, while closing can pinch an unknown obstruction.
            # With neither a command nor feedback, refuse to construct a row.
            return None
        row = tuple(float(v) for v in values) + (gripper,)
        try:
            checked = validate_tcp_chunk([row], self.DT, self.config)[0]
        except ActionValidationError:
            return None
        self._last_valid_tcp = tuple(float(v) for v in checked)
        return self._last_valid_tcp

    @staticmethod
    def _current_gripper_feedback(obs: dict) -> float | None:
        """Return fresh bounded gripper feedback without clipping corruption."""

        try:
            joints = np.asarray(obs.get("joints"), dtype=float)
        except (TypeError, ValueError):
            return None
        if joints.shape != (7,) or not math.isfinite(float(joints[6])):
            return None
        feedback = float(joints[6])
        if not 0.0 <= feedback <= 1.0:
            return None
        return feedback

    def _require_gripper_feedback(
        self, obs: dict, expected: float, phase_name: str
    ) -> float:
        """Reject hazardous motion unless this observation confirms gripper state."""

        feedback = self._current_gripper_feedback(obs)
        if feedback is None:
            raise PlanningError(
                f"phase {phase_name} has no valid fresh gripper feedback; "
                "hazardous motion is blocked"
            )
        if abs(feedback - float(expected)) > self.GRIPPER_FEEDBACK_TOLERANCE:
            state = "closed" if expected >= 0.5 else "open"
            raise PlanningError(
                f"phase {phase_name} requires confirmed {state} gripper "
                f"feedback {expected:.2f}, observed {feedback:.2f}"
            )
        return feedback

    def _strict_diagnostic_hold_action(self, obs: dict, reason: str):
        """Repeat one current measured row without fallback, clipping, or planning.

        The official TCP contract has no no-op. Repeating the measured target is
        the narrowest available diagnostic action, but it may still cause small
        controller corrections and therefore is not an emergency stop.
        """

        try:
            pose = np.asarray(obs.get("tcp_pose"), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ActionValidationError(
                "hold-only diagnostic requires current obs['tcp_pose']; no action sent"
            ) from exc
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise ActionValidationError(
                "hold-only diagnostic requires six finite current TCP values; "
                "no action sent"
            )

        try:
            joints = np.asarray(obs.get("joints"), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ActionValidationError(
                "hold-only diagnostic requires current gripper feedback; no action sent"
            ) from exc
        if joints.shape != (7,) or not math.isfinite(float(joints[6])):
            raise ActionValidationError(
                "hold-only diagnostic requires current gripper feedback in "
                "seven finite joint/gripper values; no action sent"
            )
        gripper = float(joints[6])
        if not 0.0 <= gripper <= 1.0:
            raise ActionValidationError(
                "hold-only diagnostic gripper feedback must be in [0, 1]; "
                "no action sent"
            )

        row = tuple(float(v) for v in pose) + (gripper,)
        checked = validate_tcp_chunk([row], self.DT, self.config)
        exact = tuple(float(v) for v in checked[0])
        self._last_valid_tcp = exact
        self._last_valid_gripper_feedback = gripper
        self._last_commanded_gripper = gripper
        if reason != self._last_hold_log:
            self._emit(
                "holding",
                reason=reason,
                diagnostic_hold_only=True,
                perception_and_planning_ran=False,
                hold_is_emergency_stop=False,
                operator_controller_remain_responsible_for_stop=True,
            )
            self._last_hold_log = reason
        return checked.tolist()

    def _current_tcp_row(self, obs: dict) -> tuple[float, ...]:
        measured = self._measured_tcp_row(obs)
        if measured is not None:
            return measured
        trial = obs.get("trial") or {}
        trial_id = str(trial.get("trial_id") or "")
        if trial_id.startswith("mock-t"):
            if self._last_valid_tcp is not None:
                return self._last_valid_tcp
            return self._scan_tcp
        raise ActionValidationError(
            "real observation has no valid current obs['tcp_pose']; refusing to "
            "continue from stale TCP state or invent a scan move"
        )

    def _hold_action(self, obs: dict, reason: str):
        row = self._current_tcp_row(obs)
        checked = validate_tcp_chunk([row], self.DT, self.config)
        self._last_commanded_gripper = float(checked[-1, 6])
        if reason != self._last_hold_log:
            self._emit(
                "holding",
                reason=reason,
                hold_is_emergency_stop=False,
                operator_controller_remain_responsible_for_stop=True,
            )
            self._last_hold_log = reason
        return checked.tolist()

    def _block(self, reason: str) -> None:
        if self._blocked_reason is None:
            self._blocked_reason = reason
            self._emit(
                "policy_blocked",
                reason=reason,
                automatic_retry=False,
                hold_is_emergency_stop=False,
            )

    @staticmethod
    def _emit(event: str, **fields) -> None:
        print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)
