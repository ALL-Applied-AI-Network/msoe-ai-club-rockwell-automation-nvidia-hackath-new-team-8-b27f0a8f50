from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sorting_core.hardware_replay import (  # noqa: E402
    HardwareReplayError,
    allowed_frame_ranges,
    build_replay_sample,
    frame_is_allowed,
    initial_joint_action_sentinel_length,
    predicted_tcp_wpr_uf0_deg,
    predicted_tcp_xyz_uf0_mm,
    validate_replay_document,
    validate_replay_samples,
)


def _state(joints=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), gripper=0.0):
    return [*joints, gripper]


def _tcp(joints):
    return [*predicted_tcp_xyz_uf0_mm(joints), *predicted_tcp_wpr_uf0_deg(joints)]


class HardwareReplayTests(unittest.TestCase):
    def test_zero_pose_fk_uses_fingertip_and_uf0_offset(self):
        tcp = predicted_tcp_xyz_uf0_mm((0.0,) * 6)
        self.assertAlmostEqual(tcp[0], 583.0, places=9)
        self.assertAlmostEqual(tcp[1], 0.0, places=9)
        self.assertAlmostEqual(tcp[2], 280.0, places=9)

    def test_recorded_joints_are_already_in_twin_convention(self):
        direct = (10.0, 30.0, -25.0, 5.0, -70.0, 20.0)
        incorrectly_converted = list(direct)
        incorrectly_converted[2] += incorrectly_converted[1]
        separation = math.dist(
            predicted_tcp_xyz_uf0_mm(direct),
            predicted_tcp_xyz_uf0_mm(incorrectly_converted),
        )
        self.assertGreater(separation, 25.0)

    def test_initial_joint_like_actions_are_detected_and_not_ambiguous(self):
        states = [_state(), _state((1.0, 0, 0, 0, 0, 0)), _state((2.0, 0, 0, 0, 0, 0))]
        actions = [states[0], states[1], [300.0, 0.0, -100.0, -180.0, 0.0, 0.0, 0.0]]
        self.assertEqual(initial_joint_action_sentinel_length(states, actions), 2)
        with self.assertRaisesRegex(HardwareReplayError, "row counts differ"):
            initial_joint_action_sentinel_length(states, actions[:-1])

    def test_sample_and_sequence_validation(self):
        joints_a = (0.0, 20.0, -30.0, 0.0, -70.0, 0.0)
        joints_b = (1.0, 21.0, -29.0, 0.0, -69.0, 1.0)
        first = build_replay_sample(
            frame_index=20,
            timestamp_s=2.0,
            observation_state=_state(joints_a),
            tcp_pose=_tcp(joints_a),
        )
        second = build_replay_sample(
            frame_index=30,
            timestamp_s=3.0,
            observation_state=_state(joints_b, 1.0),
            tcp_pose=_tcp(joints_b),
        )
        report = validate_replay_samples((first, second))
        self.assertEqual(report["sample_count"], 2)
        self.assertLess(report["maximum_fk_position_error_mm"], 1e-9)
        self.assertLess(report["maximum_fk_orientation_error_deg"], 1e-6)
        with self.assertRaisesRegex(HardwareReplayError, "strictly increasing"):
            validate_replay_samples((second, first))

    def test_limits_and_fk_mismatch_fail_closed(self):
        joints = (171.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        with self.assertRaisesRegex(HardwareReplayError, "J1"):
            build_replay_sample(
                frame_index=0,
                timestamp_s=0.0,
                observation_state=_state(joints),
                tcp_pose=(0.0,) * 6,
            )

        valid_joints = (0.0,) * 6
        bad_tcp = list(_tcp(valid_joints))
        bad_tcp[0] += 20.0
        bad = build_replay_sample(
            frame_index=1,
            timestamp_s=0.1,
            observation_state=_state(valid_joints),
            tcp_pose=bad_tcp,
        )
        with self.assertRaisesRegex(HardwareReplayError, "FK/TCP"):
            validate_replay_samples((bad,))

        bad_orientation_tcp = list(_tcp(valid_joints))
        bad_orientation_tcp[5] += 5.0
        bad_orientation = build_replay_sample(
            frame_index=1,
            timestamp_s=0.1,
            observation_state=_state(valid_joints),
            tcp_pose=bad_orientation_tcp,
        )
        with self.assertRaisesRegex(HardwareReplayError, "orientation"):
            validate_replay_samples((bad_orientation,))

    def test_privacy_curation_is_default_deny_by_split(self):
        curation = {
            "default_policy": "deny",
            "splits": {
                "appearance_fit": [
                    {
                        "dataset": "run-a",
                        "episode": 2,
                        "ranges": [[10, 20], [30, 40]],
                    }
                ],
                "excluded_human_or_setup": [
                    {"dataset": "run-a", "episode": 3, "ranges": [[0, 50]]}
                ],
            }
        }
        ranges = allowed_frame_ranges(
            curation,
            dataset_id="run-a",
            episode_index=2,
            split_name="appearance_fit",
        )
        self.assertTrue(frame_is_allowed(10, ranges))
        self.assertFalse(frame_is_allowed(20, ranges))
        with self.assertRaisesRegex(HardwareReplayError, "no unique record"):
            allowed_frame_ranges(
                curation,
                dataset_id="run-a",
                episode_index=3,
                split_name="appearance_fit",
            )

        curation["splits"]["excluded_human_or_setup"].append(
            {"dataset": "run-a", "episode": 2, "ranges": [[15, 16]]}
        )
        with self.assertRaisesRegex(HardwareReplayError, "overlaps excluded"):
            allowed_frame_ranges(
                curation,
                dataset_id="run-a",
                episode_index=2,
                split_name="appearance_fit",
            )

    def test_compact_replay_document_forbids_physical_commands(self):
        joints_a = (0.0, 20.0, -30.0, 0.0, -70.0, 0.0)
        joints_b = (1.0, 21.0, -29.0, 0.0, -69.0, 1.0)
        document = {
            "schema_version": 1,
            "replay_kind": "measured_state_shadow_replay",
            "physical_robot_commands_permitted": False,
            "samples": [
                {
                    "frame_index": 20,
                    "timestamp_s": 2.0,
                    "joints_deg": list(joints_a),
                    "gripper_state": 0.0,
                    "tcp_pose_uf0_mm_deg": _tcp(joints_a),
                },
                {
                    "frame_index": 30,
                    "timestamp_s": 3.0,
                    "joints_deg": list(joints_b),
                    "gripper_state": 1.0,
                    "tcp_pose_uf0_mm_deg": _tcp(joints_b),
                },
            ],
        }
        self.assertEqual(len(validate_replay_document(document)), 2)
        document["physical_robot_commands_permitted"] = True
        with self.assertRaisesRegex(HardwareReplayError, "forbid physical"):
            validate_replay_document(document)


if __name__ == "__main__":
    unittest.main()
