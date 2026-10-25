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
    MotionPhase,
    PlanningError,
    SceneDetections,
    TipPoseBase,
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
    def test_official_contract_and_random_mock_image_fail_closed(self):
        with mock.patch.dict(os.environ, {Policy.DESCENT_INTERLOCK_ENV: "0"}):
            policy = Policy()
        self.assertEqual(policy.CAMERAS, ["table"])
        self.assertEqual(policy.IMAGE_SIZE, (1920, 1080))
        self.assertEqual(policy.ACTION, "tcp")
        self.assertFalse(policy.PREFETCH)
        self.assertEqual(policy.DEPTH, [])
        self.assertEqual(policy.IMAGE_FORMAT, "png")

        chunk = np.asarray(policy.act(_observation()), dtype=float)
        self.assertEqual(chunk.shape, (1, 7))
        self.assertAlmostEqual(chunk[0, 6], 0.0)
        self.assertIsNone(policy._active_plan)
        self.assertIsNone(policy._blocked_reason)

    def test_missing_real_tcp_never_invents_a_scan_move(self):
        with mock.patch.dict(os.environ, {Policy.DESCENT_INTERLOCK_ENV: "0"}):
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
                policy._detect_work_scene(obs)

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
                "ROCKWELL_MAX_ACTIONS_PER_ACT": "9",
            },
        ):
            with self.assertRaisesRegex(ValueError, "open-loop chunks are disabled"):
                Policy()

    def test_stalled_phase_latches_after_bounded_no_progress_calls(self):
        with mock.patch.dict(os.environ, {Policy.DESCENT_INTERLOCK_ENV: "1"}):
            policy = Policy()
        policy._detect_work_scene = lambda obs: _scene(obs)
        obs = _observation(policy._scan_tcp[:6], trial_id="real-rollout")
        for _ in range(policy.MAX_NO_PROGRESS_CALLS + 2):
            obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
            policy.act(obs)
            if policy._blocked_reason:
                break
        self.assertIn("no measured TCP progress", policy._blocked_reason)

    def test_in_place_angular_progress_does_not_trip_watchdog(self):
        with mock.patch.dict(os.environ, {Policy.DESCENT_INTERLOCK_ENV: "1"}):
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

    def test_equivalent_tool_down_angle_does_not_command_a_360_degree_jump(self):
        with mock.patch.dict(os.environ, {Policy.DESCENT_INTERLOCK_ENV: "0"}):
            policy = Policy()
        obs = _observation(policy._scan_tcp[:6], trial_id="real-rollout")
        obs["tcp_pose"] = list(obs["tcp_pose"])
        obs["tcp_pose"][3] = 180.0  # equivalent to the configured -180 deg
        policy._detect_work_scene = lambda current: _scene(current)
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
            filtered = policy._detect_work_scene(obs)
        self.assertEqual([cube.track_id for cube in filtered.cubes], ["red-outside"])

    def test_invalid_real_tcp_during_motion_blocks_without_forward_chunk(self):
        with mock.patch.dict(os.environ, {Policy.DESCENT_INTERLOCK_ENV: "1"}):
            policy = Policy()
        policy._detect_work_scene = lambda obs: _scene(obs)
        obs = _observation(policy._scan_tcp[:6], trial_id="real-rollout")
        first = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
        self.assertIsNotNone(policy._active_plan)
        self.assertGreater(len(first), 0)

        obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
        obs["tcp_pose"] = None
        with self.assertRaisesRegex(ActionValidationError, "stale TCP state"):
            policy.act(obs)
        self.assertIsNotNone(policy._blocked_reason)

    def test_default_interlock_stops_at_freshly_revalidated_hover(self):
        with mock.patch.dict(os.environ, {Policy.DESCENT_INTERLOCK_ENV: "0"}):
            policy = Policy()
        policy._detect_work_scene = lambda obs: _scene(obs)
        obs = _advance_to_gate(policy, _observation(policy._scan_tcp[:6]))

        hover_z = float(obs["tcp_pose"][2])
        obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
        chunk = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
        self.assertEqual(policy._phase_index, 1)
        self.assertAlmostEqual(float(chunk[-1, 2]), hover_z)
        self.assertGreater(float(chunk[-1, 2]), -332.6)
        self.assertIsNone(policy._blocked_reason)

        obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
        policy.act(obs)
        self.assertEqual(policy._active_plan.plan_id.count("revalidated"), 1)

    def test_enabled_interlock_descends_only_after_reobservation(self):
        with mock.patch.dict(os.environ, {Policy.DESCENT_INTERLOCK_ENV: "1"}):
            policy = Policy()
        calls = []

        def detect(obs):
            calls.append(float(obs["t"]))
            return _scene(obs)

        policy._detect_work_scene = detect
        obs = _advance_to_gate(policy, _observation(policy._scan_tcp[:6]))
        calls_before_gate = len(calls)
        hover_z = float(obs["tcp_pose"][2])
        obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
        chunk = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)

        self.assertGreater(len(calls), calls_before_gate)
        self.assertEqual(policy._phase_index, 2)
        self.assertLess(float(chunk[-1, 2]), hover_z)
        self.assertIsNone(policy._blocked_reason)

    def test_moved_target_latches_hold_and_never_blindly_retries(self):
        with mock.patch.dict(os.environ, {Policy.DESCENT_INTERLOCK_ENV: "1"}):
            policy = Policy()
        policy._detect_work_scene = lambda obs: _scene(obs)
        obs = _advance_to_gate(policy, _observation(policy._scan_tcp[:6]))
        hover = np.asarray(obs["tcp_pose"], dtype=float)

        policy._detect_work_scene = lambda current: _scene(current, red_dx_m=0.010)
        obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
        rejected = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
        self.assertIn("moved", policy._blocked_reason)
        np.testing.assert_allclose(rejected[-1, :6], hover)

        policy._detect_work_scene = lambda current: _scene(current)
        obs["t"] = max(time.time(), float(obs["t"]) + 0.001)
        still_holding = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
        np.testing.assert_allclose(still_holding[-1, :6], hover)
        self.assertEqual(policy._phase_index, 1)

    def test_staleness_uses_estimated_cell_clock_offset(self):
        with mock.patch.dict(os.environ, {Policy.DESCENT_INTERLOCK_ENV: "0"}):
            policy = Policy()
        old_cell_time = time.time() - 1000.0
        policy._detect_work_scene = lambda obs: SceneDetections(
            "empty", float(obs["t"]), ()
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
            policy._verify_post_place(failed_scene)
        self.assertNotIn("red", policy._completed_colours)

        clear_scene = SceneDetections("red-in-bin", timestamp, ())
        policy._verify_post_place(clear_scene)
        self.assertIn("red", policy._completed_colours)
        self.assertIsNone(policy._pending_verification_colour)


if __name__ == "__main__":
    unittest.main()
