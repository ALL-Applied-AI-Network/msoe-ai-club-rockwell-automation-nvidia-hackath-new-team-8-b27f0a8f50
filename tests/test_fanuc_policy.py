from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import time
import unittest
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parents[1]
CELL = REPO / "hardware" / "fanuc_cell"
for path in (CELL, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import policy as policy_module  # noqa: E402
from policy import Policy  # noqa: E402
from sorting_core import (  # noqa: E402
    ActionValidationError,
    CubeDetection,
    DetectedObject,
    MotionPhase,
    PlanningError,
    Pose3D,
    SceneObservation,
    SceneDetections,
    TipPoseBase,
    VisionError,
    base_tip_pose_to_uf0_tcp,
    uf0_tcp_to_base_tip,
)


def _scene(obs: dict, *, red_dx_m: float = 0.0) -> SceneDetections:
    timestamp = float(obs["t"])
    return SceneDetections(
        scene_id=f"test:{timestamp}",
        capture_time_s=timestamp,
        cubes=(
            CubeDetection(
                "red",
                "red",
                timestamp,
                (0.44 + red_dx_m, -0.19, -0.0001),
                0.0,
                0.9,
            ),
            CubeDetection(
                "yellow",
                "yellow",
                timestamp,
                (0.44, 0.0, -0.0001),
                0.0,
                0.9,
            ),
            CubeDetection(
                "blue",
                "blue",
                timestamp,
                (0.44, 0.19, -0.0001),
                0.0,
                0.9,
            ),
        ),
    )


def _views(
    scene: SceneDetections,
    *,
    at_destination: dict[str, tuple[CubeDetection, ...]] | None = None,
):
    destinations = {"red": (), "yellow": (), "blue": ()}
    if at_destination:
        destinations.update(at_destination)
    return policy_module._SceneViews(
        unplaced=scene,
        pickable=scene,
        at_destination=destinations,
    )


def _observation(tcp_pose=None, *, timestamp=None, trial_id="mock-t00") -> dict:
    return {
        "t": time.time() if timestamp is None else float(timestamp),
        "images": {"table": np.zeros((1080, 1920, 3), dtype=np.uint8)},
        "joints": np.zeros(7, dtype=float),
        "tcp_pose": tcp_pose,
        "trial": {"trial_id": trial_id, "status": "running"},
    }


def _advance_to_gate(policy: Policy, obs: dict) -> dict:
    """Play returned chunks exactly, as the official mock does."""

    for _ in range(30):
        obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
        chunk = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
        obs["tcp_pose"] = chunk[-1, :6].tolist()
        obs["joints"][-1] = chunk[-1, 6]
        if policy._active_plan is not None and policy._phase_index == 1:
            return obs
    raise AssertionError("approach did not reach the reobservation gate")


class FanucPolicyTests(unittest.TestCase):
    def setUp(self):
        # Most unit tests focus on a downstream gate. Keep them single-frame;
        # a dedicated regression below exercises the real default of three
        # temporally consistent observations before any motion starts.
        patcher = mock.patch.dict(
            os.environ,
            {
                Policy.STABLE_OBSERVATIONS_ENV: "1",
                Policy.MOTION_INTERLOCK_ENV: "0",
                Policy.DESCENT_INTERLOCK_ENV: "0",
                Policy.DESTINATION_MODE_ENV: "bins",
                Policy.MOTION_PROFILE_ENV: "conservative",
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_injected_vision_provider_reaches_hardware_without_detector_patch(self):
        class FakeProvider:
            request = None

            def observe(self, request):
                self.request = request
                objects = tuple(
                    DetectedObject(
                        track_id=f"external-{colour}",
                        class_id="cube",
                        sort_label=colour,
                        capture_time_s=request.capture_time_s,
                        pose=Pose3D.from_yaw((0.44, y, -0.0001), 0.0),
                        dimensions_m=(0.010, 0.010, 0.010),
                        confidence=0.95,
                        source_camera_ids=("table",),
                    )
                    for colour, y in (
                        ("red", -0.19),
                        ("yellow", 0.0),
                        ("blue", 0.19),
                    )
                )
                return SceneObservation(
                    scene_id=request.scene_id,
                    capture_time_s=request.capture_time_s,
                    objects=objects,
                    provider_name="fake-team-vision",
                    provider_version="test",
                    calibration_ids=(request.frame("table").calibration_id,),
                )

        provider = FakeProvider()
        policy = Policy(perception_provider=provider)
        obs = _observation(policy._scan_tcp[:6])
        with mock.patch.object(
            policy_module,
            "detect_coloured_cubes",
            side_effect=AssertionError("legacy detector should not run"),
        ):
            views = policy._detect_scene_views(obs)
        self.assertEqual(
            [cube.colour for cube in views.pickable.cubes],
            ["red", "yellow", "blue"],
        )
        self.assertEqual(views.observation.provider_name, "fake-team-vision")
        self.assertIsNotNone(provider.request)
        self.assertEqual(provider.request.frame("table").calibration_id,
                         "sidclab-cameras.json:table")
        self.assertEqual(len(provider.request.robot_state.arm_joints_rad), 6)
        self.assertAlmostEqual(provider.request.robot_state.gripper_state, 0.0)
        self.assertAlmostEqual(
            provider.request.robot_state.tcp_pose_base_m_wpr_rad[2],
            (float(obs["tcp_pose"][2]) + 330.0) / 1000.0,
        )

    def test_hardware_feedback_is_converted_to_typed_si_robot_state(self):
        policy = Policy()
        state = policy._perception_robot_state(
            {
                "joints": [180.0, 90.0, -45.0, 30.0, 0.0, 15.0, 0.25],
                "tcp_pose": [420.0, -100.0, -260.0, -180.0, 0.0, 90.0],
            }
        )
        self.assertIsNotNone(state)
        np.testing.assert_allclose(
            state.arm_joints_rad,
            [math.pi, math.pi / 2.0, -math.pi / 4.0, math.pi / 6.0, 0.0, math.pi / 12.0],
        )
        np.testing.assert_allclose(
            state.tcp_pose_base_m_wpr_rad,
            [0.420, -0.100, 0.070, -math.pi, 0.0, math.pi / 2.0],
        )
        self.assertAlmostEqual(state.gripper_state, 0.25)

    def test_vision_provider_runtime_error_fails_closed(self):
        class ExplodingProvider:
            def observe(self, request):
                raise RuntimeError("inference device unavailable")

        policy = Policy(perception_provider=ExplodingProvider())
        obs = _observation(policy._scan_tcp[:6], trial_id="real-rollout")
        with self.assertRaisesRegex(VisionError, "inference device unavailable"):
            policy._detect_scene_views(obs)

    def test_slow_provider_cancels_plan_before_forward_motion(self):
        clock = {"now": 100.0}

        class SlowProvider:
            def observe(self, request):
                clock["now"] += 4.0
                return SceneObservation(
                    scene_id=request.scene_id,
                    capture_time_s=request.capture_time_s,
                    objects=tuple(
                        DetectedObject(
                            track_id=f"slow-{colour}",
                            class_id="cube",
                            sort_label=colour,
                            capture_time_s=request.capture_time_s,
                            pose=Pose3D.from_yaw((0.44, y, -0.0001), 0.0),
                            dimensions_m=(0.01, 0.01, 0.01),
                            source_camera_ids=("table",),
                        )
                        for colour, y in (
                            ("red", -0.19),
                            ("yellow", 0.0),
                            ("blue", 0.19),
                        )
                    ),
                    provider_name="slow-test-provider",
                    provider_version="1",
                    calibration_ids=("sidclab-cameras.json:table",),
                )

        with mock.patch.dict(
            os.environ,
            {
                Policy.MOTION_INTERLOCK_ENV: "1",
                Policy.DESCENT_INTERLOCK_ENV: "0",
                Policy.STABLE_OBSERVATIONS_ENV: "1",
            },
        ):
            policy = Policy(perception_provider=SlowProvider())
        obs = _observation(
            policy._scan_tcp[:6], timestamp=100.0, trial_id="real-rollout"
        )
        with mock.patch.object(
            policy_module.time, "time", side_effect=lambda: clock["now"]
        ), mock.patch.object(policy, "_emit", wraps=policy._emit) as emit:
            chunk = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
        self.assertEqual(len(chunk), 1)
        np.testing.assert_allclose(chunk[0, :6], obs["tcp_pose"])
        self.assertIsNone(policy._active_plan)
        self.assertIsNone(policy._blocked_reason)
        deadline_events = [
            call
            for call in emit.call_args_list
            if call.args and call.args[0] == "planning_deadline_missed"
        ]
        self.assertEqual(len(deadline_events), 1)
        self.assertFalse(deadline_events[0].kwargs["forward_motion_emitted"])

    def test_default_hold_only_repeats_current_measurement_without_perception(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.MOTION_INTERLOCK_ENV: "0",
                Policy.DESCENT_INTERLOCK_ENV: "1",
            },
        ):
            policy = Policy()
        self.assertEqual(policy.CAMERAS, ["table"])
        self.assertEqual(policy.IMAGE_SIZE, (1920, 1080))
        self.assertEqual(policy.ACTION, "tcp")
        self.assertTrue(policy.PREFETCH)
        self.assertEqual(policy.DEPTH, [])
        self.assertEqual(policy.IMAGE_FORMAT, "png")

        obs = _observation(policy._scan_tcp[:6], trial_id="real-rollout")
        obs["joints"][-1] = 0.65
        policy._detect_scene_views = mock.Mock(
            side_effect=AssertionError("diagnostic mode must not run perception")
        )
        chunk = np.asarray(policy.act(obs), dtype=float)
        self.assertEqual(chunk.shape, (1, 7))
        np.testing.assert_allclose(chunk[0, :6], obs["tcp_pose"])
        self.assertAlmostEqual(chunk[0, 6], 0.65)
        policy._detect_scene_views.assert_not_called()
        self.assertIsNone(policy._active_plan)
        self.assertIsNone(policy._blocked_reason)

    def test_motion_enabled_official_mock_image_fails_closed(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.MOTION_INTERLOCK_ENV: "1",
                Policy.DESCENT_INTERLOCK_ENV: "0",
            },
        ):
            policy = Policy()

        chunk = np.asarray(policy.act(_observation()), dtype=float)
        self.assertEqual(chunk.shape, (1, 7))
        self.assertAlmostEqual(chunk[0, 6], 0.0)
        self.assertIsNone(policy._active_plan)
        self.assertIsNone(policy._blocked_reason)

    def test_hold_only_rejects_missing_or_out_of_range_gripper_feedback(self):
        with mock.patch.dict(os.environ, {Policy.MOTION_INTERLOCK_ENV: "0"}):
            policy = Policy()
        obs = _observation(policy._scan_tcp[:6], trial_id="real-rollout")
        obs["joints"] = None
        with self.assertRaisesRegex(ActionValidationError, "gripper feedback"):
            policy.act(obs)

        policy.reset()
        obs = _observation(policy._scan_tcp[:6], trial_id="real-rollout")
        obs["joints"][-1] = 1.01
        with self.assertRaisesRegex(ActionValidationError, "must be in"):
            policy.act(obs)

    def test_hold_only_rejects_repeated_timestamp_without_stale_hold(self):
        with mock.patch.dict(os.environ, {Policy.MOTION_INTERLOCK_ENV: "0"}):
            policy = Policy()
        obs = _observation(policy._scan_tcp[:6], trial_id="real-rollout")
        policy.act(obs)
        with self.assertRaisesRegex(ActionValidationError, "fresh observation"):
            policy.act(obs)

    def test_missing_real_tcp_never_invents_a_scan_move(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.MOTION_INTERLOCK_ENV: "1",
                Policy.DESCENT_INTERLOCK_ENV: "0",
            },
        ):
            policy = Policy()
        with self.assertRaisesRegex(
            ActionValidationError, "refusing to.*invent a scan move"
        ):
            policy.act(_observation(trial_id="real-rollout"))

    def test_wrong_bin_detection_is_not_hidden_as_a_bin_component(self):
        with mock.patch.dict(os.environ, {Policy.DESCENT_INTERLOCK_ENV: "0"}):
            policy = Policy()
        obs = _observation(policy._scan_tcp[:6])
        timestamp = float(obs["t"])
        raw = SceneDetections(
            "wrong-bin",
            timestamp,
            (
                CubeDetection(
                    "red-in-yellow-bin",
                    "red",
                    timestamp,
                    (0.24, 0.0, -0.0001),
                    0.0,
                    0.9,
                ),
            ),
        )
        with mock.patch.object(policy_module, "detect_coloured_cubes", return_value=raw):
            with self.assertRaisesRegex(PlanningError, "wrong-bin"):
                policy._detect_scene_views(obs)

    def test_hardware_clearance_is_stricter_than_detection_ambiguity(self):
        with mock.patch.dict(os.environ, {Policy.DESCENT_INTERLOCK_ENV: "0"}):
            policy = Policy()
        timestamp = time.time()
        close_but_distinct = SceneDetections(
            "two-millimetre-gap",
            timestamp,
            (
                CubeDetection(
                    "red-close", "red", timestamp, (0.44, 0.0, -0.0001), 0.0, 0.9
                ),
                CubeDetection(
                    "yellow-close",
                    "yellow",
                    timestamp,
                    (0.44, 0.020, -0.0001),
                    0.0,
                    0.9,
                ),
                CubeDetection(
                    "blue-far", "blue", timestamp, (0.44, 0.19, -0.0001), 0.0, 0.9
                ),
            ),
        )
        with self.assertRaisesRegex(PlanningError, "hardware grasp clearance"):
            policy._start_next_plan(close_but_distinct)

    def test_open_loop_chunk_override_cannot_exceed_safe_policy_cap(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.DESCENT_INTERLOCK_ENV: "0",
                "ROCKWELL_MAX_ACTIONS_PER_ACT": "101",
            },
        ):
            with self.assertRaisesRegex(ValueError, "open-loop chunks are disabled"):
                Policy()

    def test_stalled_phase_latches_after_bounded_no_progress_calls(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.MOTION_INTERLOCK_ENV: "1",
                Policy.DESCENT_INTERLOCK_ENV: "1",
            },
        ):
            policy = Policy()
        policy._detect_scene_views = lambda obs: _views(_scene(obs))
        obs = _observation(policy._scan_tcp[:6], trial_id="real-rollout")
        for _ in range(policy.MAX_NO_PROGRESS_CALLS + 2):
            obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
            policy.act(obs)
            if policy._blocked_reason:
                break
        self.assertIn("no measured TCP progress", policy._blocked_reason)

    def test_in_place_angular_progress_does_not_trip_watchdog(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.DESCENT_INTERLOCK_ENV: "1",
                "ROCKWELL_MAX_ACTIONS_PER_ACT": "8",
            },
        ):
            policy = Policy()
        x, y = policy.config.task.cube_spawns_base_xy_yaw_deg["yellow"][:2]
        phase = MotionPhase(
            "rotate_in_place",
            TipPoseBase(
                (
                    x,
                    y,
                    policy.config.pick_tip_base_z_m
                    + policy.config.task.approach_clearance_m,
                ),
                math.radians(160.0),
            ),
            policy.config.gripper.open_command,
        )
        obs = _observation(policy._scan_tcp[:6], trial_id="real-rollout")

        for _ in range(4):
            chunk = np.asarray(policy._execute_phase(obs, phase), dtype=float)
            obs["tcp_pose"] = chunk[-1, :6].tolist()
            obs["joints"][-1] = chunk[-1, 6]

        self.assertLess(policy._phase_best_angle_error_deg, 160.0)
        self.assertLess(policy._phase_no_progress_calls, policy.MAX_NO_PROGRESS_CALLS)

    def test_smooth_hardware_profile_doubles_nominal_linear_rate(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.MOTION_PROFILE_ENV: "smooth",
                "ROCKWELL_MAX_ACTIONS_PER_ACT": "32",
            },
        ):
            policy = Policy()
        current = np.asarray(policy._scan_tcp, dtype=float)
        target = current.copy()
        target[2] -= 130.0
        phase = MotionPhase(
            "smooth-test",
            uf0_tcp_to_base_tip(target, policy.config),
            policy.config.gripper.open_command,
        )
        obs = _observation(current[:6], trial_id="real-rollout")
        chunk = np.asarray(policy._execute_phase(obs, phase), dtype=float)

        self.assertEqual(policy._max_linear_step_mm, 10.0)
        self.assertEqual(policy._max_angular_step_deg, 8.0)
        self.assertEqual(len(chunk), 13)
        deltas = np.linalg.norm(
            np.diff(np.vstack([current[:3], chunk[:, :3]]), axis=0), axis=1
        )
        self.assertLessEqual(float(deltas.max()), 10.0 + 1e-9)

    def test_zero_dwell_waypoint_streams_without_repeated_target_row(self):
        policy = Policy()
        target = TipPoseBase((0.40, 0.0, 0.12), 0.0)
        first = MotionPhase(
            "lift", target, policy.config.gripper.closed_command
        )
        second = MotionPhase(
            "carry",
            TipPoseBase((0.30, 0.0, 0.12), 0.0),
            policy.config.gripper.closed_command,
        )
        policy._phases = (first, second)
        policy._phase_index = 0
        row = base_tip_pose_to_uf0_tcp(
            target, policy.config.gripper.closed_command, policy.config
        )
        obs = _observation(row[:6], trial_id="real-rollout")
        obs["joints"][-1] = policy.config.gripper.closed_command

        self.assertIsNone(policy._execute_phase(obs, first))
        self.assertEqual(policy._phase_index, 1)

    def test_zero_dwell_waypoint_does_not_cross_reobservation_gate(self):
        policy = Policy()
        target = TipPoseBase((0.40, 0.0, 0.12), 0.0)
        approach = MotionPhase(
            "approach_inspection", target, policy.config.gripper.open_command
        )
        gate = MotionPhase(
            "revalidate_before_descent",
            None,
            policy.config.gripper.open_command,
            requires_reobservation_before=True,
        )
        policy._phases = (approach, gate)
        policy._phase_index = 0
        row = base_tip_pose_to_uf0_tcp(
            target, policy.config.gripper.open_command, policy.config
        )
        obs = _observation(row[:6], trial_id="real-rollout")

        chunk = np.asarray(policy._execute_phase(obs, approach), dtype=float)
        self.assertEqual(chunk.shape, (1, 7))
        self.assertEqual(policy._phase_index, 1)

    def test_equivalent_tool_down_angle_does_not_command_a_360_degree_jump(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.MOTION_INTERLOCK_ENV: "1",
                Policy.DESCENT_INTERLOCK_ENV: "0",
            },
        ):
            policy = Policy()
        obs = _observation(policy._scan_tcp[:6], trial_id="real-rollout")
        obs["tcp_pose"] = list(obs["tcp_pose"])
        obs["tcp_pose"][3] = 180.0  # equivalent to the configured -180 deg
        policy._detect_scene_views = lambda current: _views(_scene(current))
        chunk = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
        self.assertTrue(np.allclose(chunk[:, 3], 180.0))

    def test_bin_exclusion_uses_half_width_not_oversized_radius(self):
        with mock.patch.dict(os.environ, {Policy.DESCENT_INTERLOCK_ENV: "0"}):
            policy = Policy()
        obs = _observation(policy._scan_tcp[:6])
        timestamp = float(obs["t"])
        red_bin = policy.config.task.bin_centres_base_m["red"]
        raw = SceneDetections(
            "outside-red-bin",
            timestamp,
            (
                CubeDetection(
                    "red-outside",
                    "red",
                    timestamp,
                    (red_bin[0] + 0.040, red_bin[1], -0.0001),
                    0.0,
                    0.9,
                ),
            ),
        )
        with mock.patch.object(policy_module, "detect_coloured_cubes", return_value=raw):
            views = policy._detect_scene_views(obs)
        self.assertEqual(
            [cube.track_id for cube in views.unplaced.cubes], ["red-outside"]
        )
        self.assertEqual(views.pickable.cubes, ())

    def test_confident_outside_pickup_component_blocks_success_not_targeted(self):
        with mock.patch.dict(os.environ, {Policy.DESCENT_INTERLOCK_ENV: "0"}):
            policy = Policy()
        obs = _observation(policy._scan_tcp[:6])
        timestamp = float(obs["t"])
        raw = SceneDetections(
            "robot-colour",
            timestamp,
            (
                CubeDetection(
                    "yellow-robot-component",
                    "yellow",
                    timestamp,
                    (0.28, 0.06, -0.0001),
                    0.0,
                    0.9,
                ),
            ),
        )
        with mock.patch.object(policy_module, "detect_coloured_cubes", return_value=raw):
            views = policy._detect_scene_views(obs)
        self.assertEqual(
            [cube.track_id for cube in views.unplaced.cubes],
            ["yellow-robot-component"],
        )
        self.assertEqual(views.pickable.cubes, ())
        policy._pending_verification_colour = "yellow"
        with self.assertRaisesRegex(PlanningError, "outside.*matching destination"):
            policy._verify_post_place(views)

    def test_weak_outside_pickup_evidence_cannot_false_pass_outcome(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.DESTINATION_MODE_ENV: "bins",
                Policy.DESCENT_INTERLOCK_ENV: "0",
            },
        ):
            policy = Policy()
        obs = _observation(policy._scan_tcp[:6])
        timestamp = float(obs["t"])
        raw = SceneDetections(
            "weak-robot-colour",
            timestamp,
            (
                CubeDetection(
                    "yellow-robot-component",
                    "yellow",
                    timestamp,
                    (0.28, 0.06, -0.0001),
                    0.0,
                    0.10,
                ),
            ),
        )
        with mock.patch.object(policy_module, "detect_coloured_cubes", return_value=raw):
            views = policy._detect_scene_views(obs)
        self.assertEqual(
            [cube.track_id for cube in views.unplaced.cubes],
            ["yellow-robot-component"],
        )
        self.assertEqual(views.pickable.cubes, ())
        policy._pending_verification_colour = "yellow"
        with self.assertRaisesRegex(PlanningError, "outside.*matching destination"):
            policy._verify_post_place(views)

    def test_virtual_table_ignores_weak_non_destination_colour_evidence(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.DESTINATION_MODE_ENV: "virtual-table",
                Policy.DESCENT_INTERLOCK_ENV: "0",
            },
        ):
            policy = Policy()
        obs = _observation(policy._scan_tcp[:6])
        timestamp = float(obs["t"])
        raw = SceneDetections(
            "weak-cell-colour",
            timestamp,
            (
                CubeDetection(
                    "weak-blue-cell-component",
                    "blue",
                    timestamp,
                    (0.28, 0.06, -0.0001),
                    0.0,
                    0.08,
                ),
            ),
        )
        with mock.patch.object(
            policy_module, "detect_coloured_cubes", return_value=raw
        ):
            views = policy._detect_scene_views(obs)
        self.assertEqual(views.unplaced.cubes, ())
        self.assertEqual(views.pickable.cubes, ())

    def test_impossible_projected_cell_colour_does_not_block_virtual_destination(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.DESTINATION_MODE_ENV: "virtual-table",
                Policy.DESCENT_INTERLOCK_ENV: "0",
            },
        ):
            policy = Policy()
        obs = _observation(policy._scan_tcp[:6])
        timestamp = float(obs["t"])
        destination_xy = policy.config.task.bin_centres_base_m["red"]
        raw = SceneDetections(
            "outside-cell-false-positive",
            timestamp,
            (
                CubeDetection(
                    "red-cell-hardware",
                    "red",
                    timestamp,
                    (-0.758, -0.124, -0.0001),
                    0.0,
                    0.9,
                ),
                CubeDetection(
                    "red-at-virtual-destination",
                    "red",
                    timestamp,
                    (destination_xy[0], destination_xy[1], -0.0001),
                    0.0,
                    0.9,
                ),
            ),
        )
        with mock.patch.object(
            policy_module, "detect_coloured_cubes", return_value=raw
        ):
            views = policy._detect_scene_views(obs)
        self.assertEqual(views.unplaced.cubes, ())
        self.assertEqual(
            [cube.track_id for cube in views.at_destination["red"]],
            ["red-at-virtual-destination"],
        )
        policy._pending_verification_colour = "red"
        policy._verify_post_place(views)
        self.assertIn("red", policy._completed_colours)

    def test_bin_mode_retains_out_of_workspace_drop_evidence(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.DESTINATION_MODE_ENV: "bins",
                Policy.DESCENT_INTERLOCK_ENV: "0",
            },
        ):
            policy = Policy()
        obs = _observation(policy._scan_tcp[:6])
        timestamp = float(obs["t"])
        dropped = CubeDetection(
            "red-outside-workspace",
            "red",
            timestamp,
            (-0.758, -0.124, -0.0001),
            0.0,
            0.9,
        )
        raw = SceneDetections("outside-workspace-drop", timestamp, (dropped,))
        with mock.patch.object(
            policy_module, "detect_coloured_cubes", return_value=raw
        ):
            views = policy._detect_scene_views(obs)
        self.assertEqual(views.unplaced.cubes, (dropped,))
        policy._pending_verification_colour = "red"
        with self.assertRaisesRegex(PlanningError, "outside.*matching destination"):
            policy._verify_post_place(views)

    def test_virtual_table_rechecks_previously_completed_destinations(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.DESTINATION_MODE_ENV: "virtual-table",
                Policy.DESCENT_INTERLOCK_ENV: "0",
            },
        ):
            policy = Policy()
        timestamp = time.time()
        yellow_xy = policy.config.task.bin_centres_base_m["yellow"]
        yellow = CubeDetection(
            "yellow-at-destination",
            "yellow",
            timestamp,
            (yellow_xy[0], yellow_xy[1], -0.0001),
            0.0,
            0.9,
        )
        scenes = _views(
            SceneDetections("red-displaced", timestamp, ()),
            at_destination={"yellow": (yellow,)},
        )
        policy._completed_colours = {"red"}
        policy._pending_verification_colour = "yellow"
        with self.assertRaisesRegex(PlanningError, "confident red"):
            policy._verify_post_place(scenes)

    def test_invalid_real_tcp_during_motion_blocks_without_forward_chunk(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.MOTION_INTERLOCK_ENV: "1",
                Policy.DESCENT_INTERLOCK_ENV: "1",
            },
        ):
            policy = Policy()
        policy._detect_scene_views = lambda obs: _views(_scene(obs))
        obs = _observation(policy._scan_tcp[:6], trial_id="real-rollout")
        first = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
        self.assertIsNotNone(policy._active_plan)
        self.assertGreater(len(first), 0)

        obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
        obs["tcp_pose"] = None
        with self.assertRaisesRegex(ActionValidationError, "stale TCP state"):
            policy.act(obs)
        self.assertIsNotNone(policy._blocked_reason)

    def test_hold_without_any_gripper_state_refuses_to_invent_a_command(self):
        with mock.patch.dict(os.environ, {Policy.DESCENT_INTERLOCK_ENV: "0"}):
            policy = Policy()
        obs = _observation(policy._scan_tcp[:6], trial_id="real-rollout")
        obs["joints"] = np.zeros(6, dtype=float)

        with self.assertRaisesRegex(ActionValidationError, "no valid current"):
            policy._hold_action(obs, "test hold")

    def test_hold_preserves_last_valid_gripper_feedback(self):
        with mock.patch.dict(os.environ, {Policy.DESCENT_INTERLOCK_ENV: "0"}):
            policy = Policy()
        obs = _observation(policy._scan_tcp[:6], trial_id="real-rollout")
        obs["joints"][-1] = 0.65
        measured = policy._measured_tcp_row(obs)
        self.assertIsNotNone(measured)
        self.assertAlmostEqual(measured[-1], 0.65)

        obs["joints"] = None
        measured_without_feedback = policy._measured_tcp_row(obs)

        self.assertIsNotNone(measured_without_feedback)
        self.assertAlmostEqual(measured_without_feedback[-1], 0.65)

    def test_hold_preserves_close_command_over_lagging_open_feedback(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.DESCENT_INTERLOCK_ENV: "0",
                "ROCKWELL_MAX_ACTIONS_PER_ACT": "8",
            },
        ):
            policy = Policy()
        x, y = policy.config.task.cube_spawns_base_xy_yaw_deg["yellow"][:2]
        close = MotionPhase(
            "close",
            TipPoseBase(
                (
                    x,
                    y,
                    policy.config.pick_tip_base_z_m
                    + policy.config.task.approach_clearance_m,
                ),
                0.0,
            ),
            policy.config.gripper.closed_command,
            dwell_s=1.0,
        )
        obs = _observation(policy._scan_tcp[:6], trial_id="real-rollout")
        obs["joints"][-1] = policy.config.gripper.open_command
        close_chunk = np.asarray(policy._execute_phase(obs, close), dtype=float)
        self.assertEqual(close_chunk[-1, 6], policy.config.gripper.closed_command)

        # Feedback can lag a pneumatic command.  A fault hold must not turn that
        # stale open reading back into an open command.
        obs["joints"][-1] = policy.config.gripper.open_command
        held = np.asarray(policy._hold_action(obs, "fault after close"), dtype=float)

        self.assertEqual(held[-1, 6], policy.config.gripper.closed_command)

        # Preservation is not a permanent latch: the semantic release phase
        # can still issue and remember an intentional open command.
        policy._phase_dwell_rows_remaining = None
        release = MotionPhase(
            "release",
            close.target,
            policy.config.gripper.open_command,
            dwell_s=1.0,
        )
        release_chunk = np.asarray(policy._execute_phase(obs, release), dtype=float)
        self.assertEqual(release_chunk[-1, 6], policy.config.gripper.open_command)
        released_hold = np.asarray(
            policy._hold_action(obs, "fault after release"), dtype=float
        )
        self.assertEqual(released_hold[-1, 6], policy.config.gripper.open_command)

        # A new rollout must adopt its own measured state, not a command from
        # the preceding trial.
        policy.reset()
        after_reset = np.asarray(
            policy._hold_action(obs, "new rollout hold"), dtype=float
        )
        self.assertEqual(after_reset[-1, 6], policy.config.gripper.open_command)

    def test_hazardous_phases_require_fresh_gripper_confirmation(self):
        policy = Policy()
        x, y = policy.config.task.cube_spawns_base_xy_yaw_deg["yellow"][:2]
        high_target = TipPoseBase(
            (
                x,
                y,
                policy.config.pick_tip_base_z_m
                + policy.config.task.approach_clearance_m,
            ),
            0.0,
        )
        obs = _observation(policy._scan_tcp[:6], trial_id="real-rollout")

        # A previously requested close is an intent, not proof that the
        # pneumatic gripper actually closed.  Every object-carrying phase must
        # inspect the fresh feedback independently.
        policy._last_commanded_gripper = policy.config.gripper.closed_command
        obs["joints"][-1] = policy.config.gripper.open_command
        for phase_name in ("dwell", "lift", "carry", "lower"):
            with self.subTest(phase=phase_name):
                phase = MotionPhase(
                    phase_name,
                    high_target,
                    policy.config.gripper.closed_command,
                )
                with self.assertRaisesRegex(PlanningError, "confirmed closed"):
                    policy._execute_phase(obs, phase)

        obs["joints"] = None
        with self.assertRaisesRegex(PlanningError, "no valid fresh gripper"):
            policy._execute_phase(
                obs,
                MotionPhase(
                    "lift", high_target, policy.config.gripper.closed_command
                ),
            )

        # The reverse transition matters too: every nominally open motion can
        # pinch an obstruction or drag a part if the fingers are still closed.
        policy._last_commanded_gripper = policy.config.gripper.open_command
        obs["joints"] = np.zeros(7, dtype=float)
        obs["joints"][-1] = policy.config.gripper.closed_command
        for phase_name in ("approach_pick", "descend_pick", "settle", "retreat"):
            with self.subTest(phase=phase_name):
                with self.assertRaisesRegex(PlanningError, "confirmed open"):
                    policy._execute_phase(
                        obs,
                        MotionPhase(
                            phase_name,
                            high_target,
                            policy.config.gripper.open_command,
                        ),
                    )

    def test_default_interlock_stops_at_freshly_revalidated_hover(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.MOTION_INTERLOCK_ENV: "1",
                Policy.DESCENT_INTERLOCK_ENV: "0",
            },
        ):
            policy = Policy()
        policy._detect_scene_views = lambda obs: _views(_scene(obs))
        obs = _advance_to_gate(policy, _observation(policy._scan_tcp[:6]))

        hover_z = float(obs["tcp_pose"][2])
        obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
        chunk = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
        self.assertEqual(policy._phase_index, 2)
        self.assertAlmostEqual(float(chunk[-1, 2]), hover_z)
        self.assertGreater(float(chunk[-1, 2]), -332.6)
        self.assertIsNone(policy._blocked_reason)

        for _ in range(5):
            obs["tcp_pose"] = chunk[-1, :6].tolist()
            obs["joints"][-1] = chunk[-1, 6]
            obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
            chunk = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
            if policy._phase_index == 3:
                break
        self.assertEqual(policy._phase_index, 3)

        obs["tcp_pose"] = chunk[-1, :6].tolist()
        obs["joints"][-1] = chunk[-1, 6]
        obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
        held = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
        np.testing.assert_allclose(held[-1, :6], obs["tcp_pose"])
        self.assertEqual(policy._active_plan.plan_id.count("revalidated"), 1)

    def test_real_default_requires_three_consistent_scenes_before_motion(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.MOTION_INTERLOCK_ENV: "1",
                Policy.DESCENT_INTERLOCK_ENV: "0",
                Policy.STABLE_OBSERVATIONS_ENV: "3",
            },
        ):
            policy = Policy()
        policy._detect_scene_views = lambda obs: _views(_scene(obs))
        obs = _observation(policy._scan_tcp[:6], trial_id="real-rollout")

        for accepted in (1, 2):
            obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
            held = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
            self.assertIsNone(policy._active_plan)
            self.assertEqual(policy._initial_stability_count, accepted)
            np.testing.assert_allclose(held[-1, :6], obs["tcp_pose"])

        obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
        first_motion = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
        self.assertIsNotNone(policy._active_plan)
        self.assertEqual(policy._initial_stability_count, 3)
        self.assertFalse(np.allclose(first_motion[-1, :6], obs["tcp_pose"]))

    def test_one_row_chunks_get_distance_derived_watchdog_budget(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.MOTION_INTERLOCK_ENV: "1",
                Policy.DESCENT_INTERLOCK_ENV: "0",
                "ROCKWELL_MAX_ACTIONS_PER_ACT": "1",
            },
        ):
            policy = Policy()
        policy._detect_scene_views = lambda obs: _views(_scene(obs))
        obs = _observation(policy._scan_tcp[:6], trial_id="real-rollout")
        maximum_motion_calls = 0

        for _ in range(60):
            obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
            chunk = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
            obs["tcp_pose"] = chunk[-1, :6].tolist()
            obs["joints"][-1] = chunk[-1, 6]
            maximum_motion_calls = max(
                maximum_motion_calls, policy._phase_motion_calls
            )
            if policy._active_plan is not None and policy._phase_index == 1:
                break
        else:
            self.fail("one-row supervised mode did not reach the hover gate")

        self.assertIsNone(policy._blocked_reason)
        self.assertGreater(maximum_motion_calls, 20)

    def test_enabled_interlock_descends_only_after_reobservation(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.MOTION_INTERLOCK_ENV: "1",
                Policy.DESCENT_INTERLOCK_ENV: "1",
            },
        ):
            policy = Policy()
        calls = []

        def detect(obs):
            calls.append(float(obs["t"]))
            return _scene(obs)

        policy._detect_scene_views = lambda obs: _views(detect(obs))
        obs = _advance_to_gate(policy, _observation(policy._scan_tcp[:6]))
        calls_before_gate = len(calls)
        hover_z = float(obs["tcp_pose"][2])
        obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
        chunk = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)

        self.assertGreater(len(calls), calls_before_gate)
        self.assertEqual(policy._phase_index, 2)
        self.assertAlmostEqual(float(chunk[-1, 2]), hover_z)

        for _ in range(5):
            obs["tcp_pose"] = chunk[-1, :6].tolist()
            obs["joints"][-1] = chunk[-1, 6]
            obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
            chunk = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
            if policy._phase_index == 3:
                break
        self.assertEqual(policy._phase_index, 3)

        obs["tcp_pose"] = chunk[-1, :6].tolist()
        obs["joints"][-1] = chunk[-1, 6]
        obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
        descent = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
        self.assertLess(float(descent[-1, 2]), hover_z)
        self.assertIsNone(policy._blocked_reason)

    def test_moved_target_latches_hold_and_never_blindly_retries(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.MOTION_INTERLOCK_ENV: "1",
                Policy.DESCENT_INTERLOCK_ENV: "1",
            },
        ):
            policy = Policy()
        policy._detect_scene_views = lambda obs: _views(_scene(obs))
        obs = _advance_to_gate(policy, _observation(policy._scan_tcp[:6]))
        hover = np.asarray(obs["tcp_pose"], dtype=float)

        policy._detect_scene_views = lambda current: _views(
            _scene(current, red_dx_m=0.010)
        )
        obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
        rejected = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
        self.assertIn("moved", policy._blocked_reason)
        np.testing.assert_allclose(rejected[-1, :6], hover)

        policy._detect_scene_views = lambda current: _views(_scene(current))
        obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
        still_holding = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
        np.testing.assert_allclose(still_holding[-1, :6], hover)
        self.assertEqual(policy._phase_index, 1)

    def test_offset_inspection_retries_transient_camera_occlusion(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.MOTION_INTERLOCK_ENV: "1",
                Policy.DESCENT_INTERLOCK_ENV: "1",
            },
        ):
            policy = Policy()
        policy._detect_scene_views = lambda current: _views(_scene(current))
        obs = _advance_to_gate(policy, _observation(policy._scan_tcp[:6]))

        def without_red(current):
            complete = _scene(current)
            return _views(
                SceneDetections(
                    complete.scene_id,
                    complete.capture_time_s,
                    tuple(c for c in complete.cubes if c.colour != "red"),
                )
            )

        policy._detect_scene_views = without_red
        for expected_attempt in (1, 2, 3):
            obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
            held = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
            np.testing.assert_allclose(held[-1, :6], obs["tcp_pose"])
            self.assertEqual(
                policy._revalidation_visibility_failures, expected_attempt
            )
            self.assertIsNone(policy._blocked_reason)
            self.assertEqual(policy._phase_index, 1)

        policy._detect_scene_views = lambda current: _views(_scene(current))
        obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
        alignment = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
        self.assertEqual(policy._phase_index, 2)
        self.assertEqual(policy._revalidation_visibility_failures, 0)
        self.assertAlmostEqual(alignment[-1, 2], obs["tcp_pose"][2])
        self.assertIsNone(policy._blocked_reason)

    def test_staleness_uses_estimated_cell_clock_offset(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.MOTION_INTERLOCK_ENV: "1",
                Policy.DESCENT_INTERLOCK_ENV: "0",
            },
        ):
            policy = Policy()
        old_cell_time = time.time() - 1000.0
        policy._detect_scene_views = lambda obs: _views(
            SceneDetections("empty", float(obs["t"]), ())
        )
        policy.act(_observation(policy._scan_tcp[:6], timestamp=old_cell_time))
        self.assertIsNone(policy._blocked_reason)

        # The fixed server offset is accepted, but a repeated frame timestamp
        # is stale and therefore never starts motion.
        policy.act(_observation(policy._scan_tcp[:6], timestamp=old_cell_time))
        self.assertIsNone(policy._active_plan)

    def test_command_completion_requires_fresh_post_place_camera_check(self):
        with mock.patch.dict(os.environ, {Policy.DESCENT_INTERLOCK_ENV: "1"}):
            policy = Policy()
        initial = _observation(policy._scan_tcp[:6])
        initial_scene = _scene(initial)
        policy._start_next_plan(initial_scene)
        policy._finish_active_plan()

        self.assertEqual(policy._pending_verification_colour, "red")
        self.assertNotIn("red", policy._completed_colours)

        timestamp = float(initial["t"]) + 1.0
        failed_scene = SceneDetections(
            "failed-red",
            timestamp,
            (
                CubeDetection(
                    "red-still-on-table",
                    "red",
                    timestamp,
                    (0.44, -0.19, -0.0001),
                    0.0,
                    0.9,
                ),
            ),
        )
        with self.assertRaisesRegex(PlanningError, "post-place verification"):
            policy._verify_post_place(_views(failed_scene))
        self.assertNotIn("red", policy._completed_colours)

        clear_scene = SceneDetections("red-in-bin", timestamp, ())
        policy._verify_post_place(_views(clear_scene))
        self.assertIn("red", policy._completed_colours)
        self.assertIsNone(policy._pending_verification_colour)

    def test_virtual_table_mode_sets_down_and_requires_positive_destination_view(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.DESTINATION_MODE_ENV: "virtual-table",
                Policy.DESCENT_INTERLOCK_ENV: "0",
            },
        ):
            policy = Policy()
        initial = _observation(policy._scan_tcp[:6])
        initial_scene = _scene(initial)
        self.assertTrue(policy._start_next_plan(initial_scene))
        self.assertAlmostEqual(
            policy._active_plan.release_tip_pose.xyz_m[2],
            policy.config.pick_tip_base_z_m,
        )
        self.assertLess(
            policy._active_plan.release_tip_pose.xyz_m[2],
            policy.config.bin_release_tip_base_z_m,
        )
        policy._finish_active_plan()

        timestamp = float(initial["t"]) + 1.0
        empty = SceneDetections("missing-placement", timestamp, ())
        with self.assertRaisesRegex(PlanningError, "exactly one confident red"):
            policy._verify_post_place(_views(empty))

        destination_xy = policy.config.task.bin_centres_base_m["red"]
        placed = CubeDetection(
            "red-in-virtual-zone",
            "red",
            timestamp,
            (destination_xy[0], destination_xy[1], -0.0001),
            0.0,
            0.9,
        )
        policy._verify_post_place(
            _views(empty, at_destination={"red": (placed,)})
        )
        self.assertIn("red", policy._completed_colours)

    def test_complete_three_colour_virtual_table_state_machine(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.MOTION_INTERLOCK_ENV: "1",
                Policy.DESCENT_INTERLOCK_ENV: "1",
                Policy.DESTINATION_MODE_ENV: "virtual-table",
                Policy.VISION_CUBE_EDGE_ENV: "10",
                "ROCKWELL_MAX_ACTIONS_PER_ACT": "8",
            },
        ):
            policy = Policy()

        def detected_views(obs):
            timestamp = float(obs["t"])
            base_scene = _scene(obs)
            absent = set(policy._completed_colours)
            if policy._pending_verification_colour is not None:
                absent.add(policy._pending_verification_colour)
            remaining = SceneDetections(
                f"full-sort:{timestamp}",
                timestamp,
                tuple(
                    cube for cube in base_scene.cubes if cube.colour not in absent
                ),
            )
            at_destination: dict[str, tuple[CubeDetection, ...]] = {}
            for colour in absent:
                x_m, y_m = policy.config.task.bin_centres_base_m[colour]
                at_destination[colour] = (
                    CubeDetection(
                        f"{colour}-placed",
                        colour,
                        timestamp,
                        (x_m, y_m, -0.0001),
                        0.0,
                        0.9,
                    ),
                )
            return _views(remaining, at_destination=at_destination)

        policy._detect_scene_views = detected_views
        obs = _observation(policy._scan_tcp[:6], trial_id="real-rollout")
        with mock.patch.object(policy, "_emit", wraps=policy._emit) as emit:
            for _ in range(500):
                obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
                chunk = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
                self.assertGreaterEqual(len(chunk), 1)
                self.assertLessEqual(len(chunk), 8)
                obs["tcp_pose"] = chunk[-1, :6].tolist()
                obs["joints"][-1] = chunk[-1, 6]
                if len(policy._completed_colours) == 3:
                    break
            else:
                self.fail("three-colour virtual-table state machine did not complete")
        phases_seen = {
            call.kwargs["phase"]
            for call in emit.call_args_list
            if call.args
            and call.args[0] in {"phase_action", "phase_complete"}
            and "phase" in call.kwargs
        }

        self.assertIsNone(policy._blocked_reason)
        self.assertEqual(policy._completed_colours, {"red", "yellow", "blue"})
        self.assertTrue(
            {"descend_pick", "close", "lift", "carry", "lower", "release"}
            <= phases_seen
        )

    def test_sixteen_mm_hardware_profile_is_vision_only_and_gates_descent(self):
        with mock.patch.dict(
            os.environ,
            {
                Policy.VISION_CUBE_EDGE_ENV: "16",
                Policy.MOTION_INTERLOCK_ENV: "1",
                Policy.DESCENT_INTERLOCK_ENV: "0",
                Policy.CONFIRM_GRASP_FIT_ENV: "0",
            },
        ):
            policy = Policy()
        self.assertAlmostEqual(policy._vision_cube_edge_m, 0.016)
        self.assertAlmostEqual(policy.config.cube_edge_m, 0.010)

        obs = _observation(policy._scan_tcp[:6])
        raw = SceneDetections("empty", float(obs["t"]), ())
        with mock.patch.object(
            policy_module, "detect_coloured_cubes", return_value=raw
        ) as detector:
            policy._detect_scene_views(obs)
        self.assertAlmostEqual(
            detector.call_args.kwargs["expected_cube_edge_m"], 0.016
        )

        with mock.patch.dict(
            os.environ,
            {
                Policy.VISION_CUBE_EDGE_ENV: "16",
                Policy.MOTION_INTERLOCK_ENV: "1",
                Policy.DESCENT_INTERLOCK_ENV: "1",
                Policy.CONFIRM_GRASP_FIT_ENV: "0",
            },
        ):
            with self.assertRaisesRegex(
                ValueError, Policy.CONFIRM_GRASP_FIT_ENV
            ):
                Policy()

        with mock.patch.dict(
            os.environ,
            {
                Policy.VISION_CUBE_EDGE_ENV: "16",
                Policy.MOTION_INTERLOCK_ENV: "1",
                Policy.DESCENT_INTERLOCK_ENV: "1",
                Policy.CONFIRM_GRASP_FIT_ENV: "1",
            },
        ):
            confirmed = Policy()
        self.assertTrue(confirmed._grasp_fit_confirmed)


if __name__ == "__main__":
    unittest.main()
