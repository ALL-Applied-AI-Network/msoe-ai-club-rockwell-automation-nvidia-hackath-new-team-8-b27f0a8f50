from __future__ import annotations

import math
import unittest

from sorting_core.pose_evaluation import (
    PoseEvaluationError,
    axis_error_deg,
    evaluate_pose_documents,
    rotation_error_deg,
)


def _yaw(degrees: float) -> list[float]:
    half = math.radians(degrees) / 2.0
    return [0.0, 0.0, math.sin(half), math.cos(half)]


class PoseEvaluationTests(unittest.TestCase):
    def test_position_rotation_axis_and_confidence_are_reported(self):
        estimates = {
            "schema_version": 1,
            "coordinate_frame": "robot_base",
            "provider": "foundationpose",
            "objects": {
                "screw-red-01": {
                    "position_base_m": [0.402, -0.1, 0.02],
                    "orientation_xyzw": _yaw(25.0),
                    "confidence": 0.87,
                }
            },
        }
        truth = {
            "schema_version": 1,
            "coordinate_frame": "robot_base",
            "truth_source": "isaac_sim_rigid_body_state",
            "objects": {
                "screw-red-01": {
                    "class_id": "screw",
                    "position_base_m": [0.4, -0.1, 0.02],
                    "orientation_xyzw": _yaw(20.0),
                    "evaluation_axis_local": [1.0, 0.0, 0.0],
                    "axis_undirected": False,
                }
            },
        }
        report = evaluate_pose_documents(estimates, truth)
        row = report["objects"][0]
        self.assertAlmostEqual(row["position_error_mm"], 2.0)
        self.assertAlmostEqual(row["rotation_error_deg"], 5.0)
        self.assertAlmostEqual(row["axis_error_deg"], 5.0)
        self.assertEqual(row["confidence"], 0.87)
        self.assertTrue(report["summary"]["complete"])

    def test_quaternion_double_cover_has_zero_rotation_error(self):
        quaternion = _yaw(31.0)
        self.assertAlmostEqual(
            rotation_error_deg(quaternion, [-value for value in quaternion]), 0.0
        )

    def test_undirected_axis_treats_180_degrees_as_equivalent(self):
        self.assertAlmostEqual(
            axis_error_deg(_yaw(0.0), _yaw(180.0), [1, 0, 0], undirected=True),
            0.0,
        )

    def test_missing_estimate_is_explicit(self):
        estimates = {
            "schema_version": 1,
            "coordinate_frame": "robot_base",
            "objects": {},
        }
        truth = {
            "schema_version": 1,
            "coordinate_frame": "robot_base",
            "objects": {
                "cube-red": {
                    "position_base_m": [0.4, 0.0, 0.0],
                    "orientation_xyzw": [0, 0, 0, 1],
                }
            },
        }
        report = evaluate_pose_documents(estimates, truth)
        self.assertEqual(report["missing_estimates"], ["cube-red"])
        self.assertFalse(report["summary"]["complete"])

    def test_coordinate_frame_mismatch_fails_closed(self):
        with self.assertRaisesRegex(PoseEvaluationError, "frames must match"):
            evaluate_pose_documents(
                {
                    "schema_version": 1,
                    "coordinate_frame": "camera",
                    "objects": {},
                },
                {
                    "schema_version": 1,
                    "coordinate_frame": "robot_base",
                    "objects": {},
                },
            )


if __name__ == "__main__":
    unittest.main()
