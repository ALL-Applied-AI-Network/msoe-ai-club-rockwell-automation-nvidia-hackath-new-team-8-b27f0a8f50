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

from dataclasses import dataclass, replace
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
    validate_colour_sort_scene,
    validate_reobserved_scene,
)
from sorting_core.grasping import (  # noqa: E402
    GraspPlanningContext,
    GraspPlanningError,
    GraspPlanningPipeline,
)
from sorting_core.perception import (  # noqa: E402
    CameraFrame,
    DepthEncoding,
    DetectedObject,
    PerceptionRequest,
    RobotState,
    SceneObservation,
    VisionProvider,
    validate_scene_for_request,
)
from sorting_core.task_perception import cube_scene_from_observation  # noqa: E402
from sorting_core.team_integrations import (  # noqa: E402
    CUBE_TASK_CLASS_IDS,
    GRASP_CHECKPOINT_ENV,
    PERCEPTION_CALIBRATION_IDS,
    PERCEPTION_CAMERA_IDS,
    PERCEPTION_DEPTH_CALIBRATION_IDS,
    PERCEPTION_DEPTH_CAMERA_IDS,
    VISION_CHECKPOINT_ENV,
    create_grasp_pipeline,
    create_vision_provider,
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
    at_destination: dict[str, tuple[CubeDetection, ...]]
    unsupported_objects: tuple[DetectedObject, ...] = ()
    observation: SceneObservation | None = None


class Policy:
    """Persistent, fail-closed colour-to-bucket state machine.

    The camera supplies target positions. Known dimensions, bin positions,
    phase semantics, and action limits come from the shared checked-in config.
    Each ``act`` call returns at most one short absolute-TCP action chunk.
    """

    CAMERAS = list(PERCEPTION_CAMERA_IDS)
    IMAGE_SIZE = (1920, 1080)  # native table RGB; calibration pixels stay valid
    IMAGE_FORMAT = "png"  # preserve tiny cube edges until JPEG is validated on real images
    ACTION = "tcp"
    # Let the official client refill a long motion before its queue empties.
    # Phase transitions still require measured feedback; this only removes
    # network/camera pauses between chunks of the same phase.
    PREFETCH = True
    DEPTH = list(PERCEPTION_DEPTH_CAMERA_IDS)
    DT = 0.1

    MOTION_INTERLOCK_ENV = "ROCKWELL_ENABLE_PHYSICAL_MOTION"
    DESCENT_INTERLOCK_ENV = "ROCKWELL_ENABLE_PHYSICAL_DESCENT"
    VISION_CUBE_EDGE_ENV = "ROCKWELL_VISION_CUBE_EDGE_MM"
    DESTINATION_MODE_ENV = "ROCKWELL_DESTINATION_MODE"
    CONFIRM_GRASP_FIT_ENV = "ROCKWELL_CONFIRM_CUBE_GRASP_FIT"
    STABLE_OBSERVATIONS_ENV = "ROCKWELL_INITIAL_STABLE_OBSERVATIONS"
    MOTION_PROFILE_ENV = "ROCKWELL_HARDWARE_MOTION_PROFILE"
    MOTION_PROFILES = {
        "conservative": (5.0, 5.0),
        # Hardware-only: 10 mm / 0.1 s is 100 mm/s. This stays below the
        # current service limits while making every recorded motion phase fit
        # in the launcher's 32-row chunk. Isaac continues to use its shared
        # 5 mm trajectory configuration.
        "smooth": (10.0, 8.0),
    }
    MAX_ACTIONS_PER_CALL = 100
    ABSOLUTE_MAX_MOTION_CALLS_PER_PHASE = 200
    MAX_NO_PROGRESS_CALLS = 4
    MIN_MEASURED_PROGRESS_MM = 0.5
    MIN_MEASURED_ANGULAR_PROGRESS_DEG = 0.5
    GRIPPER_FEEDBACK_TOLERANCE = 0.1
    INSPECTION_OFFSET_X_M = -0.060
    MAX_REVALIDATION_VISIBILITY_RETRIES = 3
    _OPEN_FEEDBACK_PHASES = frozenset(
        {
            "approach_inspection",
            "align_above_pick",
            "approach_pick",
            "descend_pick",
            "settle",
            "retreat",
        }
    )
    _CLOSED_FEEDBACK_PHASES = frozenset({"dwell", "lift", "carry", "lower"})

    def __init__(
        self,
        checkpoint: str | None = None,
        *,
        perception_provider: VisionProvider | None = None,
        grasp_pipeline: GraspPlanningPipeline | None = None,
    ):
        self.config = load_config()
        calibration = load_calibration()
        if "table" not in calibration:
            raise RuntimeError("official calibration does not contain the table camera")
        self._table_camera = calibration["table"]
        if tuple(int(v) for v in self._table_camera["size"]) != self.IMAGE_SIZE:
            raise RuntimeError(
                "Policy.IMAGE_SIZE must match the official table-camera calibration"
            )

        cube_profile = os.environ.get(
            self.VISION_CUBE_EDGE_ENV,
            f"{self.config.cube_edge_m * 1000.0:g}",
        ).strip()
        if cube_profile not in {"10", "10.0", "16", "16.0"}:
            raise ValueError(
                f"{self.VISION_CUBE_EDGE_ENV} must be 10 or 16 millimetres"
            )
        self._vision_cube_edge_m = float(cube_profile) / 1000.0
        self._perception_provider = (
            perception_provider
            if perception_provider is not None
            else create_vision_provider(
                config=self.config,
                camera_projectors={
                    "table": lambda uv, z: pixel_to_base(
                        self._table_camera, uv, z=z
                    )
                },
                calibration_ids=PERCEPTION_CALIBRATION_IDS,
                expected_cube_edge_m=self._vision_cube_edge_m,
                checkpoint=os.environ.get(VISION_CHECKPOINT_ENV, checkpoint),
                # Preserve the old module-level patch seam used by the
                # hardware replay tests while routing production through the
                # explicit provider contract.
                detector=lambda *args, **kwargs: detect_coloured_cubes(
                    *args, **kwargs
                ),
            )
        )
        self._grasp_pipeline = (
            grasp_pipeline
            if grasp_pipeline is not None
            else create_grasp_pipeline(
                config=self.config,
                checkpoint=os.environ.get(GRASP_CHECKPOINT_ENV, checkpoint),
            )
        )

        self._destination_mode = os.environ.get(
            self.DESTINATION_MODE_ENV, "bins"
        ).strip().lower()
        if self._destination_mode not in {"bins", "virtual-table"}:
            raise ValueError(
                f"{self.DESTINATION_MODE_ENV} must be bins or virtual-table"
            )
        self._grasp_fit_confirmed = _environment_flag(self.CONFIRM_GRASP_FIT_ENV)

        raw_stable_observations = os.environ.get(self.STABLE_OBSERVATIONS_ENV, "3")
        try:
            self._stable_observations_required = int(raw_stable_observations)
        except ValueError as exc:
            raise ValueError(
                f"{self.STABLE_OBSERVATIONS_ENV} must be an integer"
            ) from exc
        if not 1 <= self._stable_observations_required <= 10:
            raise ValueError(
                f"{self.STABLE_OBSERVATIONS_ENV} must be between 1 and 10"
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

        self._motion_profile = os.environ.get(
            self.MOTION_PROFILE_ENV, "conservative"
        ).strip().lower()
        try:
            (
                self._max_linear_step_mm,
                self._max_angular_step_deg,
            ) = self.MOTION_PROFILES[self._motion_profile]
        except KeyError as exc:
            choices = ", ".join(sorted(self.MOTION_PROFILES))
            raise ValueError(
                f"{self.MOTION_PROFILE_ENV} must be one of: {choices}"
            ) from exc

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
            simulator_task_cube_edge_mm=self.config.cube_edge_m * 1000.0,
            hardware_vision_cube_edge_mm=self._vision_cube_edge_m * 1000.0,
            hardware_destination_mode=self._destination_mode,
            initial_stable_observations=self._stable_observations_required,
            hardware_motion_profile=self._motion_profile,
            maximum_linear_step_mm=self._max_linear_step_mm,
            maximum_angular_step_deg=self._max_angular_step_deg,
            nominal_linear_speed_mm_s=self._max_linear_step_mm / self.DT,
            tcp_beyond_fingertips_mm=self.config.frames.tcp_beyond_fingertips_mm,
            motion_interlock_env=self.MOTION_INTERLOCK_ENV,
            descent_interlock_env=self.DESCENT_INTERLOCK_ENV,
            default_mode="hold-only diagnostic; no perception or planning",
            hover_requires_motion_interlock=True,
            descent_requires_both_interlocks=True,
            minimum_hardware_centre_clearance_mm=(
                self._minimum_hardware_clearance_m * 1000.0
            ),
            cube_profile_warning=(
                "The recorded 640x360 table videos suggest approximately 16 mm "
                "silhouettes, but video resampling is not a dimensional "
                "measurement. Confirm the placed 10/16 mm cube with the "
                "operator or calipers before descent. The 16 mm nominal edge "
                "is also slightly larger than the modeled 15.99 mm hook-open "
                "gap and therefore requires explicit grasp-fit confirmation."
            ),
            warning=self.config.frames.tcp_conflict_note,
        )
        self.reset()

    def reset(self) -> None:
        """Reset all task state at the start of an official rollout."""

        self._active_plan: PickPlacePlan | None = None
        self._active_attempt_id: str | None = None
        self._active_attempt_started_at_s: float | None = None
        self._phases: tuple[MotionPhase, ...] = ()
        self._phase_index = 0
        self._phase_dwell_rows_remaining: int | None = None
        self._revalidation_visibility_failures = 0
        self._reset_phase_watchdog()
        self._completed_colours: set[str] = set()
        self._pending_verification_colour: str | None = None
        self._pending_verification_plan: PickPlacePlan | None = None
        self._pending_attempt_id: str | None = None
        self._pending_attempt_started_at_s: float | None = None
        self._reference_detections: dict[str, CubeDetection] = {}
        self._initial_stability_reference: dict[str, CubeDetection] = {}
        self._initial_stability_count = 0
        self._blocked_reason: str | None = None
        self._last_observation_time_s: float | None = None
        self._clock_offset_estimate_s: float | None = None
        self._last_valid_tcp: tuple[float, ...] | None = None
        self._last_valid_gripper_feedback: float | None = None
        self._last_commanded_gripper: float | None = None
        self._latest_robot_state: RobotState | None = None
        self._last_hold_log: str | None = None
        self._scene_number = 0
        self._motion_enabled = _environment_flag(self.MOTION_INTERLOCK_ENV)
        self._descent_enabled = _environment_flag(self.DESCENT_INTERLOCK_ENV)
        if (
            self._descent_enabled
            and self._vision_cube_edge_m >= self.config.gripper.hook_open_gap_m
            and not self._grasp_fit_confirmed
        ):
            raise ValueError(
                f"{self.VISION_CUBE_EDGE_ENV}={self._vision_cube_edge_m * 1000.0:g} "
                "mm is not nominally smaller than the modeled "
                f"{self.config.gripper.hook_open_gap_m * 1000.0:.2f} mm "
                "hook-open gap; hover-only inspection is allowed, but descent "
                f"requires {self.CONFIRM_GRASP_FIT_ENV}=1 after an operator "
                "confirms the actual rounded cube/finger fit"
            )
        self._emit(
            "rollout_reset",
            physical_motion_enabled=self._motion_enabled,
            physical_descent_enabled=self._descent_enabled,
            hardware_vision_cube_edge_mm=self._vision_cube_edge_m * 1000.0,
            hardware_destination_mode=self._destination_mode,
            hardware_motion_profile=self._motion_profile,
            maximum_linear_step_mm=self._max_linear_step_mm,
            maximum_angular_step_deg=self._max_angular_step_deg,
            nominal_linear_speed_mm_s=self._max_linear_step_mm / self.DT,
            grasp_fit_explicitly_confirmed=self._grasp_fit_confirmed,
            active_assumption=(
                "UTOOL1 TCP at fingertips (0 mm extra); cube size is an "
                "explicit 10/16 mm hardware vision profile"
            ),
            warning=(
                "The pinned client labels its action point as the fingertip. "
                "Separately, an organizer confirmed standard-finger table contact "
                "near UF0 z=-335 mm and said the former -345 mm FAQ value was for "
                "shorter fingers. Verify the installed fingers and active "
                "UTOOL1/UF0, placed cube size, and grasp fit before setting "
                "the descent interlock."
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

        started_with_active_plan = self._active_plan is not None
        revalidated_this_call = False
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
                    self._verify_post_place(scenes)
                if len(self._completed_colours) == len(self.config.task.colour_order):
                    return self._hold_action(
                        obs, "all configured colours camera-checked after placement"
                    )
                if not self._start_next_plan(
                    scenes.pickable,
                    generic_scene=scenes.observation,
                    unsupported_objects=scenes.unsupported_objects,
                ):
                    return self._hold_action(
                        obs,
                        "waiting for a stable complete camera scene: "
                        f"{self._initial_stability_count}/"
                        f"{self._stable_observations_required} accepted "
                        "observations",
                    )

            # A measured zero-dwell waypoint can hand directly to the next
            # semantic phase in this same act() call. The loop never crosses
            # the explicit camera-reobservation gate and remains bounded by
            # the finite phase list.
            for _ in range(len(self._phases) + 1):
                phase = self._phases[self._phase_index]
                if phase.requires_reobservation_before:
                    scenes = self._detect_scene_views(obs)
                    try:
                        self._revalidate_before_descent(
                            scenes.pickable,
                            generic_scene=scenes.observation,
                            unsupported_objects=scenes.unsupported_objects,
                        )
                        revalidated_this_call = True
                    except PlanningError as exc:
                        message = str(exc)
                        retryable = (
                            " target is missing" in message
                            or " target is ambiguous" in message
                        )
                        if (
                            retryable
                            and self._revalidation_visibility_failures
                            < self.MAX_REVALIDATION_VISIBILITY_RETRIES
                        ):
                            self._revalidation_visibility_failures += 1
                            self._emit(
                                "pre_descent_visibility_retry",
                                attempt=self._revalidation_visibility_failures,
                                maximum_attempts=(
                                    self.MAX_REVALIDATION_VISIBILITY_RETRIES
                                ),
                                reason=message,
                            )
                            return self._hold_action(
                                obs,
                                "inspection view temporarily incomplete; waiting "
                                "for a fresh table-camera frame",
                            )
                        raise
                    phase = self._phases[self._phase_index]

                processing_age_s = self._relative_observation_age_s(obs)
                if processing_age_s > self._maximum_observation_age_s:
                    reason = (
                        "camera observation expired during perception/planning: "
                        f"{processing_age_s:.2f} s > "
                        f"{self._maximum_observation_age_s:.2f} s"
                    )
                    self._emit(
                        "planning_deadline_missed",
                        **self._plan_evidence_fields(self._active_plan),
                        attempt_id=self._active_attempt_id,
                        observation_age_s=processing_age_s,
                        maximum_observation_age_s=(
                            self._maximum_observation_age_s
                        ),
                        forward_motion_emitted=False,
                    )
                    if not started_with_active_plan:
                        self._cancel_unexecuted_plan(reason)
                    elif revalidated_this_call:
                        self._phase_index = next(
                            index
                            for index, candidate_phase in enumerate(self._phases)
                            if candidate_phase.requires_reobservation_before
                        )
                        self._reset_phase_watchdog()
                    else:
                        self._block(reason)
                    return self._hold_action(obs, reason)

                # Revalidation is performed from a laterally offset inspection
                # pose. Align over the refreshed target first, then apply the
                # descent interlock; never ask the table camera to see through
                # the gripper at the direct-overhead pose.
                if phase.name == "descend_pick" and not self._descent_enabled:
                    return self._hold_action(
                        obs,
                        "physical descent interlock is closed; holding at hover. "
                        f"Set {self.DESCENT_INTERLOCK_ENV}=1 only after supervised "
                        f"{self._vision_cube_edge_m * 1000.0:g} mm cube, grasp-fit, "
                        "and fingertip-TCP verification",
                    )

                chunk = self._execute_phase(obs, phase)
                if chunk is not None:
                    return chunk
            raise PlanningError("phase transition loop exceeded the finite plan")
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

    def _relative_observation_age_s(self, obs: dict) -> float:
        if self._clock_offset_estimate_s is None:
            raise ValueError("observation clock offset has not been established")
        timestamp = float(obs.get("t"))
        if not math.isfinite(timestamp):
            raise ValueError("obs['t'] is not finite")
        return time.time() - timestamp - self._clock_offset_estimate_s

    def _cancel_unexecuted_plan(self, reason: str) -> None:
        plan = self._active_plan
        self._emit(
            "plan_cancelled_before_motion",
            **self._plan_evidence_fields(plan),
            attempt_id=self._active_attempt_id,
            attempt_started_at_s=self._active_attempt_started_at_s,
            outcome="cancelled_stale_observation",
            reason=reason,
        )
        self._active_plan = None
        self._active_attempt_id = None
        self._active_attempt_started_at_s = None
        self._phases = ()
        self._phase_index = 0
        self._phase_dwell_rows_remaining = None
        self._reset_phase_watchdog()

    def _detect_scene_views(self, obs: dict) -> _SceneViews:
        images = obs.get("images")
        if not isinstance(images, dict):
            raise VisionError("obs['images'] must contain the configured cameras")
        missing_cameras = [name for name in self.CAMERAS if name not in images]
        if missing_cameras:
            raise VisionError(
                "missing configured RGB cameras: " + ", ".join(missing_cameras)
            )
        for camera_id in self.CAMERAS:
            image = np.asarray(images[camera_id])
            if image.shape != (self.IMAGE_SIZE[1], self.IMAGE_SIZE[0], 3):
                raise VisionError(
                    f"{camera_id} RGB must have requested uncropped shape "
                    f"{(self.IMAGE_SIZE[1], self.IMAGE_SIZE[0], 3)}, got "
                    f"{image.shape}"
                )
            if image.dtype != np.uint8:
                raise VisionError(
                    f"{camera_id} RGB must be uint8 RGB, got {image.dtype}"
                )

        capture_time = float(obs["t"])
        trial = obs.get("trial") or {}
        trial_id = str(trial.get("trial_id") or "unknown-trial")
        self._scene_number += 1
        depth_images = obs.get("depth")
        if not isinstance(depth_images, dict):
            depth_images = {}
        missing_depth = [name for name in self.DEPTH if name not in depth_images]
        if missing_depth:
            raise VisionError(
                "missing configured depth cameras: " + ", ".join(missing_depth)
            )
        expected_depth_shapes = {
            "table": (400, 640),
            "wrist": (480, 848),
        }
        for camera_id in self.DEPTH:
            depth = np.asarray(depth_images[camera_id])
            expected_shape = expected_depth_shapes.get(camera_id)
            if expected_shape is None:
                raise VisionError(
                    f"no validated hardware depth geometry for {camera_id}"
                )
            if depth.shape != expected_shape or depth.dtype != np.uint16:
                raise VisionError(
                    f"{camera_id} depth must be uint16 with shape "
                    f"{expected_shape}, got {depth.dtype} {depth.shape}"
                )
        frame_camera_ids = tuple(
            dict.fromkeys((*self.CAMERAS, *self.DEPTH))
        )
        frames = []
        for camera_id in frame_camera_ids:
            rgb = (
                None
                if camera_id not in images
                else np.asarray(images[camera_id])
            )
            depth = (
                None
                if camera_id not in depth_images
                else np.asarray(depth_images[camera_id])
            )
            frames.append(
                CameraFrame(
                camera_id=camera_id,
                capture_time_s=capture_time,
                calibration_id=(
                    None
                    if rgb is None
                    else PERCEPTION_CALIBRATION_IDS.get(camera_id)
                ),
                rgb=rgb,
                depth=depth,
                depth_calibration_id=(
                    None
                    if depth is None
                    else PERCEPTION_DEPTH_CALIBRATION_IDS.get(camera_id)
                ),
                depth_encoding=(
                    None
                    if depth is None
                    else DepthEncoding(
                        meters_per_unit=0.001,
                        invalid_values=(0.0,),
                        aligned_to_rgb=False,
                    )
                ),
            )
            )
        frames = tuple(frames)
        robot_state = self._perception_robot_state(obs)
        request = PerceptionRequest(
            scene_id=f"{trial_id}:scene-{self._scene_number}",
            capture_time_s=capture_time,
            frames=frames,
            robot_state=robot_state,
            metadata={"trial_id": trial_id},
        )
        self._latest_robot_state = request.robot_state
        try:
            provider_result = self._perception_provider.observe(request)
            observation = validate_scene_for_request(request, provider_result)
        except VisionError:
            raise
        except Exception as exc:
            raise VisionError(
                f"vision provider failed: {type(exc).__name__}: {exc}"
            ) from exc
        projection = cube_scene_from_observation(
            observation, class_ids=CUBE_TASK_CLASS_IDS
        )
        raw = projection.scene

        # The bins are themselves red/yellow/blue. Exclude only a component in
        # its same-colour destination footprint. A cube of one colour near a
        # different bin is a wrong-bin outcome and must remain visible/fail.
        # Confident non-bin detections remain in the broader outcome view so a
        # misplaced/completed colour fails verification. Only the calibrated
        # pickup-region subset may become a target. This partition is not an
        # obstacle map or swept-volume collision check.
        unplaced: list[CubeDetection] = []
        pickable: list[CubeDetection] = []
        at_destination: dict[str, list[CubeDetection]] = {
            colour: [] for colour in self.config.task.colour_order
        }
        bin_components_excluded = 0
        outside_pickup_region_excluded = 0
        low_confidence_outside_region_retained = 0
        low_confidence_outside_region_ignored = 0
        outside_hardware_workspace_excluded = 0
        for detection in raw.cubes:
            if (
                self._destination_mode == "virtual-table"
                and not self._detection_in_outcome_workspace(detection)
            ):
                # The calibrated camera sees coloured robot/cell components
                # well beyond any physically reachable cube location. Do not
                # let geometrically impossible projections masquerade as
                # unsorted cubes. Virtual-table mode separately requires
                # positive detections for the current and all previously
                # completed colours at their destinations. Bin mode remains
                # fail-closed and retains this evidence because it cannot see
                # reliably into an opaque bin.
                outside_hardware_workspace_excluded += 1
                continue
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
                at_destination[detection.colour].append(detection)
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
                if self._destination_mode == "virtual-table":
                    # Positive destination evidence is mandatory in this mode.
                    # Keeping weak cell-colour evidence elsewhere can only
                    # create a false failure; a real missing/drop outcome still
                    # fails because the required destination cube is absent.
                    low_confidence_outside_region_ignored += 1
                    continue
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
        destination_detections = {
            colour: tuple(detections)
            for colour, detections in at_destination.items()
        }
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
            low_confidence_outside_region_ignored=(
                low_confidence_outside_region_ignored
            ),
            outside_hardware_workspace_excluded=(
                outside_hardware_workspace_excluded
            ),
            generic_objects=len(observation.objects),
            unsupported_generic_objects=len(projection.unsupported_objects),
            vision_provider=observation.provider_name,
            vision_provider_version=observation.provider_version,
            pickable_colours=[cube.colour for cube in pickable_scene.cubes],
            matching_destination_detections={
                colour: len(detections)
                for colour, detections in destination_detections.items()
            },
        )
        return _SceneViews(
            unplaced=unplaced_scene,
            pickable=pickable_scene,
            at_destination=destination_detections,
            unsupported_objects=projection.unsupported_objects,
            observation=observation,
        )

    def _perception_robot_state(self, obs: dict) -> RobotState | None:
        """Convert official mm/degrees feedback into the shared SI contract."""

        raw_joints = obs.get("joints")
        raw_tcp = obs.get("tcp_pose")
        if raw_joints is None and raw_tcp is None:
            return None

        arm_joints_rad = None
        gripper_state = None
        if raw_joints is not None:
            joints = np.asarray(raw_joints, dtype=float)
            if joints.shape != (7,) or not np.all(np.isfinite(joints)):
                raise VisionError(
                    "obs['joints'] must contain six finite degree values and "
                    "one finite gripper value"
                )
            if not 0.0 <= float(joints[6]) <= 1.0:
                raise VisionError("obs['joints'] gripper value must be in [0, 1]")
            arm_joints_rad = tuple(float(value) for value in np.radians(joints[:6]))
            gripper_state = float(joints[6])

        tcp_pose_base_m_wpr_rad = None
        if raw_tcp is not None:
            tcp = np.asarray(raw_tcp, dtype=float)
            if tcp.shape != (6,) or not np.all(np.isfinite(tcp)):
                raise VisionError(
                    "obs['tcp_pose'] must contain six finite mm/degree values"
                )
            tcp_pose_base_m_wpr_rad = (
                float(tcp[0]) / 1000.0,
                float(tcp[1]) / 1000.0,
                (
                    float(tcp[2]) - self.config.frames.uf0_z_offset_mm
                )
                / 1000.0,
                *(float(value) for value in np.radians(tcp[3:6])),
            )

        return RobotState(
            arm_joints_rad=arm_joints_rad,
            tcp_pose_base_m_wpr_rad=tcp_pose_base_m_wpr_rad,
            gripper_state=gripper_state,
        )

    def _detection_in_outcome_workspace(self, detection: CubeDetection) -> bool:
        """Whether a projected cube centre is physically plausible in the cell.

        Base and UF0 share XY. Use the same official TCP XY box and radial
        reach that constrain commanded targets. A virtual-table placement also
        needs positive destination evidence, so excluding an unreachable
        projected colour cannot by itself turn a drop into success.
        """

        x_m, y_m = detection.position_base_m[:2]
        bounds = self.config.hardware.tcp_bounds_uf0_mm
        x_lower, x_upper = (float(value) / 1000.0 for value in bounds["x"])
        y_lower, y_upper = (float(value) / 1000.0 for value in bounds["y"])
        return (
            x_lower <= x_m <= x_upper
            and y_lower <= y_m <= y_upper
            and math.hypot(x_m, y_m)
            <= self.config.hardware.maximum_radial_reach_m
        )

    def _start_next_plan(
        self,
        scene: SceneDetections,
        *,
        generic_scene: SceneObservation | None = None,
        unsupported_objects: tuple[DetectedObject, ...] | None = None,
    ) -> bool:
        self._validate_hardware_clearance(scene.cubes)
        if not self._reference_detections:
            try:
                targets = validate_colour_sort_scene(scene, self.config)
            except PlanningError:
                self._initial_stability_reference = {}
                self._initial_stability_count = 0
                raise
            current = {
                target.colour: target for target in targets
            }
            if not self._initial_stability_reference:
                self._initial_stability_reference = current
                self._initial_stability_count = 1
            else:
                try:
                    validate_reobserved_scene(
                        scene,
                        self._initial_stability_reference,
                        (),
                        self.config,
                        maximum_motion_m=self._maximum_motion_m,
                        maximum_yaw_motion_deg=self._maximum_yaw_motion_deg,
                    )
                except PlanningError as exc:
                    # Before any motion, a changed but otherwise valid scene is
                    # not a rollout fault. Start temporal confirmation over
                    # from this newest observation. Once motion starts, the
                    # stricter latched revalidation path below still applies.
                    self._initial_stability_reference = current
                    self._initial_stability_count = 1
                    self._emit(
                        "initial_scene_stability_reset",
                        reason=str(exc),
                        accepted_observations=1,
                        required_observations=self._stable_observations_required,
                    )
                    return False
                self._initial_stability_reference = current
                self._initial_stability_count += 1
            self._emit(
                "initial_scene_stability",
                accepted_observations=self._initial_stability_count,
                required_observations=self._stable_observations_required,
                colours=list(self.config.task.colour_order),
            )
            if self._initial_stability_count < self._stable_observations_required:
                return False
            self._reference_detections = current
            first_colour = self.config.task.colour_order[0]
            plan = self._plan_for_detection(
                current[first_colour],
                scene,
                generic_scene=generic_scene,
                unsupported_objects=unsupported_objects,
            )
        else:
            current = self._validate_scene_against_reference(scene)
            next_colour = next(
                colour
                for colour in self.config.task.colour_order
                if colour not in self._completed_colours
            )
            plan = self._plan_for_detection(
                current[next_colour],
                scene,
                generic_scene=generic_scene,
                unsupported_objects=unsupported_objects,
            )
        self._set_active_plan(plan)
        return True

    def _set_active_plan(self, plan: PickPlacePlan) -> None:
        self._active_plan = plan
        self._active_attempt_id = (
            f"{plan.plan_id}|attempt-{len(self._completed_colours) + 1}"
        )
        self._active_attempt_started_at_s = plan.capture_time_s
        self._phases = self._hardware_phase_plan(plan)
        self._phase_index = 0
        self._phase_dwell_rows_remaining = None
        self._revalidation_visibility_failures = 0
        self._reset_phase_watchdog()
        self._last_hold_log = None
        self._emit(
            "plan_selected",
            **self._plan_evidence_fields(plan),
            attempt_id=self._active_attempt_id,
            attempt_started_at_s=self._active_attempt_started_at_s,
            colour=plan.target.colour,
            target_xyz_m=list(plan.target.position_base_m),
            destination_mode=self._destination_mode,
            destination_xyz_m=list(plan.release_tip_pose.xyz_m),
            phase_names=[phase.name for phase in self._phases],
        )

    @staticmethod
    def _plan_evidence_fields(plan: PickPlacePlan | None) -> dict:
        if plan is None:
            return {}
        selection = plan.grasp_selection
        decision_key = None
        if selection is not None:
            decision_key = "|".join(
                (
                    plan.scene_id,
                    plan.target.track_id,
                    selection.candidate_id,
                    selection.selector_id,
                )
            )
        return {
            "plan_id": plan.plan_id,
            "plan_capture_time_s": plan.capture_time_s,
            "scene_id": plan.scene_id,
            "target_track_id": plan.target.track_id,
            "grasp_decision_key": decision_key,
            "grasp_candidate_id": (
                None if selection is None else selection.candidate_id
            ),
            "grasp_success_probability": (
                None if selection is None else selection.success_probability
            ),
            "grasp_expected_total_episode_time_s": (
                None
                if selection is None
                else selection.expected_total_episode_time_s
            ),
            "grasp_scorer": None if selection is None else selection.scorer_id,
            "grasp_model_version": (
                None if selection is None else selection.model_version
            ),
            "grasp_selector": (
                None if selection is None else selection.selector_id
            ),
            "grasp_selection_objective": (
                None if selection is None else selection.selection_objective
            ),
        }

    def _hardware_phase_plan(self, plan: PickPlacePlan) -> tuple[MotionPhase, ...]:
        """Add a visible inspection/alignment pair without changing Isaac."""

        shared = semantic_phase_plan(plan, self.config)
        above_pick = shared[0].target
        if above_pick is None:
            raise PlanningError("shared approach phase has no target")
        inspection = TipPoseBase(
            (
                above_pick.xyz_m[0] + self.INSPECTION_OFFSET_X_M,
                above_pick.xyz_m[1],
                above_pick.xyz_m[2],
            ),
            above_pick.yaw_rad,
        )
        return (
            MotionPhase(
                "approach_inspection",
                inspection,
                self.config.gripper.open_command,
            ),
            shared[1],
            MotionPhase(
                "align_above_pick",
                above_pick,
                self.config.gripper.open_command,
            ),
            *shared[2:],
        )

    def _plan_for_detection(
        self,
        detection: CubeDetection,
        scene: SceneDetections,
        *,
        generic_scene: SceneObservation | None = None,
        unsupported_objects: tuple[DetectedObject, ...] | None = None,
    ) -> PickPlacePlan:
        colour = detection.colour
        bin_x, bin_y = self.config.task.bin_centres_base_m[colour]
        release_z_m = (
            self.config.bin_release_tip_base_z_m
            if self._destination_mode == "bins"
            else self.config.pick_tip_base_z_m
        )
        context = GraspPlanningContext(
            destination=BinTarget(colour, (bin_x, bin_y)),
            release_tip_pose=TipPoseBase(
                (bin_x, bin_y, release_z_m), 0.0
            ),
            pick_tip_z_m=self.config.pick_tip_base_z_m,
            approach_clearance_m=self.config.task.approach_clearance_m,
            generic_scene=generic_scene,
            unsupported_objects=unsupported_objects,
            robot_state=self._latest_robot_state,
        )
        try:
            decision = self._grasp_pipeline.decide_for_target(
                scene, detection, context
            )
        except GraspPlanningError as exc:
            raise PlanningError(
                f"no executable grasp for {detection.track_id}: {exc}"
            ) from exc
        self._emit(
            "grasp_candidates_evaluated",
            **self._plan_evidence_fields(decision.plan),
            attempt_id=self._active_attempt_id,
            selected_candidate_id=decision.selected.candidate.candidate_id,
            accepted_candidates=[
                {
                    "candidate_id": item.candidate.candidate_id,
                    "pick_tip_pose": {
                        "xyz_m": list(item.candidate.pick_tip_pose.xyz_m),
                        "yaw_rad": item.candidate.pick_tip_pose.yaw_rad,
                    },
                    "approach_tip_pose": {
                        "xyz_m": list(item.candidate.approach_tip_pose.xyz_m),
                        "yaw_rad": item.candidate.approach_tip_pose.yaw_rad,
                    },
                    "checker_id": item.feasibility.checker_id,
                    "feasibility_metrics": dict(item.feasibility.metrics),
                    "feature_schema": item.features.schema_id,
                    "features": dict(item.features.values),
                    "success_probability": item.score.success_probability,
                    "expected_total_episode_time_s": (
                        item.score.expected_total_episode_time_s
                    ),
                    "expected_execution_time_s": (
                        item.score.expected_execution_time_s
                    ),
                    "expected_recovery_time_s": (
                        item.score.expected_recovery_time_s
                    ),
                    "scorer_id": item.score.scorer_id,
                    "model_version": item.score.model_version,
                }
                for item in decision.evaluations
            ],
            rejected_candidates=[
                {
                    "candidate_id": item.candidate.candidate_id,
                    "reasons": list(item.feasibility.rejection_reasons),
                    "checker_id": item.feasibility.checker_id,
                    "metrics": dict(item.feasibility.metrics),
                }
                for item in decision.rejections
            ],
        )
        return decision.plan

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

    def _revalidate_before_descent(
        self,
        scene: SceneDetections,
        *,
        generic_scene: SceneObservation | None = None,
        unsupported_objects: tuple[DetectedObject, ...] | None = None,
    ) -> None:
        if self._active_plan is None:
            raise PlanningError("revalidation has no active plan")
        current = self._validate_scene_against_reference(scene)
        colour = self._active_plan.target.colour
        refreshed = self._plan_for_detection(
            current[colour],
            scene,
            generic_scene=generic_scene,
            unsupported_objects=unsupported_objects,
        )
        # Keep the original identity in logs while binding motion to the fresh
        # scene. The detector's scene-local track IDs are not persistent IDs.
        # Keep the ID bounded if the closed descent interlock causes repeated
        # revalidation polls at the hover pose.
        refreshed = replace(
            refreshed, plan_id=f"{refreshed.plan_id}:revalidated"
        )
        self._active_plan = refreshed
        self._phases = self._hardware_phase_plan(refreshed)
        gate_index = next(
            index
            for index, phase in enumerate(self._phases)
            if phase.requires_reobservation_before
        )
        self._phase_index = gate_index
        self._phase_dwell_rows_remaining = None
        self._revalidation_visibility_failures = 0
        self._reset_phase_watchdog()
        self._emit(
            "pre_descent_revalidation",
            plan_id=refreshed.plan_id,
            result="accepted_from_offset_inspection",
            physical_descent_enabled=self._descent_enabled,
        )
        # Always move from the inspection offset to the direct-overhead hover.
        # The act loop gates descent only after this alignment is measured.
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
            all_rows = interpolate_tcp_rows(
                current,
                target,
                self.config,
                max_linear_step_mm=self._max_linear_step_mm,
                max_angular_step_deg=self._max_angular_step_deg,
            )
            if self._phase_motion_call_limit is None:
                expected_calls = math.ceil(
                    len(all_rows) / self._max_actions_per_call
                )
                self._phase_motion_call_limit = min(
                    self.ABSOLUTE_MAX_MOTION_CALLS_PER_PHASE,
                    expected_calls + self.MAX_NO_PROGRESS_CALLS + 2,
                )
                if expected_calls > self.ABSOLUTE_MAX_MOTION_CALLS_PER_PHASE:
                    raise PlanningError(
                        f"phase {phase.name} needs about {expected_calls} measured "
                        "motion calls, above the absolute bounded limit "
                        f"{self.ABSOLUTE_MAX_MOTION_CALLS_PER_PHASE}"
                    )
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
            if self._phase_motion_calls > self._phase_motion_call_limit:
                raise PlanningError(
                    f"phase {phase.name} exceeded "
                    f"{self._phase_motion_call_limit} measured motion calls"
                )
            if self._phase_no_progress_calls >= self.MAX_NO_PROGRESS_CALLS:
                raise PlanningError(
                    f"phase {phase.name} made no measured TCP progress for "
                    f"{self._phase_no_progress_calls} calls"
                )
            rows = all_rows[: self._max_actions_per_call]
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

        next_phase_index = self._phase_index + 1
        can_stream_next_phase = (
            phase.dwell_s <= 0.0
            and self._phase_index < len(self._phases)
            and self._phases[self._phase_index] is phase
            and next_phase_index < len(self._phases)
            and not self._phases[next_phase_index].requires_reobservation_before
        )
        if can_stream_next_phase:
            self._emit(
                "phase_complete",
                phase=phase.name,
                target_reached=True,
                next_phase=self._phases[next_phase_index].name,
                transition="same_observation",
            )
            self._phase_index = next_phase_index
            self._phase_dwell_rows_remaining = None
            self._reset_phase_watchdog()
            return None

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
        completed_plan = self._active_plan
        colour = completed_plan.target.colour
        self._pending_verification_colour = colour
        self._pending_verification_plan = completed_plan
        self._pending_attempt_id = self._active_attempt_id
        self._pending_attempt_started_at_s = self._active_attempt_started_at_s
        motion_finished_at_s = self._last_observation_time_s
        self._emit(
            "plan_motion_complete",
            **self._plan_evidence_fields(completed_plan),
            attempt_id=self._pending_attempt_id,
            attempt_started_at_s=self._pending_attempt_started_at_s,
            motion_finished_at_s=motion_finished_at_s,
            motion_elapsed_s=(
                None
                if motion_finished_at_s is None
                or self._pending_attempt_started_at_s is None
                else max(
                    0.0,
                    motion_finished_at_s - self._pending_attempt_started_at_s,
                )
            ),
            colour=colour,
            verification="pending fresh table-camera observation",
        )
        self._active_plan = None
        self._active_attempt_id = None
        self._active_attempt_started_at_s = None
        self._phases = ()
        self._phase_index = 0
        self._phase_dwell_rows_remaining = None
        self._reset_phase_watchdog()

    def _reset_phase_watchdog(self) -> None:
        self._phase_motion_calls = 0
        self._phase_no_progress_calls = 0
        self._phase_motion_call_limit: int | None = None
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

    def _verify_post_place(self, scenes: _SceneViews) -> None:
        colour = self._pending_verification_colour
        if colour is None:
            raise PlanningError("post-place verification has no pending colour")

        forbidden = self._completed_colours | {colour}
        visible = sorted(
            {
                cube.colour
                for cube in scenes.unplaced.cubes
                if cube.colour in forbidden
            }
        )
        if visible:
            raise PlanningError(
                "post-place verification found sorted colour(s) outside their "
                f"matching destination: {', '.join(visible)}; no automatic retry"
            )

        if self._destination_mode == "virtual-table":
            for expected_colour in sorted(forbidden):
                matching = tuple(
                    cube
                    for cube in scenes.at_destination.get(expected_colour, ())
                    if cube.confidence >= self._minimum_confidence
                )
                if len(matching) != 1:
                    raise PlanningError(
                        "virtual-table post-place verification requires exactly "
                        f"one confident {expected_colour} cube in its configured "
                        f"destination footprint, observed {len(matching)}"
                    )

        verified_plan = self._pending_verification_plan
        attempt_id = self._pending_attempt_id
        attempt_started_at_s = self._pending_attempt_started_at_s
        verification_time_s = scenes.pickable.capture_time_s
        self._completed_colours.add(colour)
        self._pending_verification_colour = None
        self._pending_verification_plan = None
        self._pending_attempt_id = None
        self._pending_attempt_started_at_s = None
        verification = (
            "fresh table RGB found exactly one matching-colour cube in the "
            "configured virtual-table destination for every completed colour "
            "and none outside them"
            if self._destination_mode == "virtual-table"
            else "fresh table RGB found no matching-colour cube outside the "
            "configured same-colour bin footprint"
        )
        self._emit(
            "plan_complete",
            **self._plan_evidence_fields(verified_plan),
            attempt_id=attempt_id,
            attempt_started_at_s=attempt_started_at_s,
            verification_capture_time_s=verification_time_s,
            total_episode_time_s=(
                None
                if attempt_started_at_s is None
                else max(0.0, verification_time_s - attempt_started_at_s)
            ),
            recovery_attempted=False,
            outcome="correct_placement_camera_verified",
            colour=colour,
            completed_colours=sorted(self._completed_colours),
            verification=verification,
            limitation=(
                "This is camera verification, not independent grasp/force "
                "confirmation; occlusion or a detector error can still hide a "
                "failed placement."
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
            attempt_id = self._active_attempt_id or self._pending_attempt_id
            attempt_started_at_s = (
                self._active_attempt_started_at_s
                if self._active_attempt_started_at_s is not None
                else self._pending_attempt_started_at_s
            )
            blocked_at_s = self._last_observation_time_s
            self._emit(
                "policy_blocked",
                **self._plan_evidence_fields(
                    self._active_plan or self._pending_verification_plan
                ),
                attempt_id=attempt_id,
                attempt_started_at_s=attempt_started_at_s,
                blocked_at_s=blocked_at_s,
                total_episode_time_s=(
                    None
                    if attempt_started_at_s is None or blocked_at_s is None
                    else max(0.0, blocked_at_s - attempt_started_at_s)
                ),
                outcome="unresolved_blocked",
                reason=reason,
                automatic_retry=False,
                hold_is_emergency_stop=False,
            )

    @staticmethod
    def _emit(event: str, **fields) -> None:
        print(
            json.dumps(
                {
                    "event": event,
                    "event_wall_time_s": time.time(),
                    **fields,
                },
                sort_keys=True,
            ),
            flush=True,
        )
