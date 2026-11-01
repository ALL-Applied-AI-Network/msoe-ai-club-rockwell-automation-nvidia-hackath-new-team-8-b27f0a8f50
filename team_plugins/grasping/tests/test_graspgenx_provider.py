import hashlib
import math
import unittest

from team_plugins.grasping.ai_providers.graspgenx import (
    GraspGenXCatalogProvider,
    PROPOSAL_SCHEMA_VERSION,
)
from team_plugins.grasping.ai_selection import canonical_json

MODEL_VERSION = "nvidia-graspgenx@fixture123"
DEPTH_M = 0.05


def _request_json(candidates=None):
    if candidates is None:
        candidates = [
            {
                "candidate_id": "candidate-a",
                "pick_tip_pose": {
                    "xyz_m": [0.4, 0.0, 0.1],
                    "yaw_rad": 0.0,
                },
            },
            {
                "candidate_id": "candidate-b",
                "pick_tip_pose": {
                    "xyz_m": [0.4, 0.0, 0.1],
                    "yaw_rad": math.pi / 2.0,
                },
            },
        ]
    return canonical_json(
        {
            "schema_version": "team8-feasible-grasp-ranking-request-v1",
            "scene_id": "scene-1",
            "target_track_id": "target-1",
            "candidates": candidates,
        }
    )


def _top_down_transform(tip_xyz, yaw_rad, *, depth_m=DEPTH_M):
    closing = (-math.sin(yaw_rad), math.cos(yaw_rad), 0.0)
    approach = (0.0, 0.0, -1.0)
    second = (
        approach[1] * closing[2] - approach[2] * closing[1],
        approach[2] * closing[0] - approach[0] * closing[2],
        approach[0] * closing[1] - approach[1] * closing[0],
    )
    base = (
        tip_xyz[0] - depth_m * approach[0],
        tip_xyz[1] - depth_m * approach[1],
        tip_xyz[2] - depth_m * approach[2],
    )
    return [
        [closing[0], second[0], approach[0], base[0]],
        [closing[1], second[1], approach[1], base[1]],
        [closing[2], second[2], approach[2], base[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _proposal(
    proposal_id="proposal-1",
    *,
    tip_xyz=(0.4, 0.0, 0.1),
    yaw_rad=0.0,
    confidence=0.8,
    branch_tag="diff",
):
    return {
        "proposal_id": proposal_id,
        "transform_input_from_gripper": _top_down_transform(tip_xyz, yaw_rad),
        "confidence": confidence,
        "branch_tag": branch_tag,
    }


def _envelope(request_json, proposals, **overrides):
    value = {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "request_sha256": hashlib.sha256(request_json.encode("utf-8")).hexdigest(),
        "scene_id": "scene-1",
        "model_version": MODEL_VERSION,
        "input_frame_id": "camera-depth",
        "calibration_id": "camera-to-base-v1",
        "fingertip_depth_m": DEPTH_M,
        "transform_robot_base_from_input": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "proposals": proposals,
    }
    value.update(overrides)
    return value


class _Source:
    def __init__(self, builder):
        self.builder = builder
        self.calls = []

    def infer(self, request_json, *, timeout_s):
        self.calls.append((request_json, timeout_s))
        if isinstance(self.builder, BaseException):
            raise self.builder
        if callable(self.builder):
            return self.builder(request_json)
        return self.builder


class GraspGenXCatalogProviderTests(unittest.TestCase):
    def _provider(self, source, **kwargs):
        defaults = {
            "model_version": MODEL_VERSION,
            "input_frame_id": "camera-depth",
            "calibration_id": "camera-to-base-v1",
            "fingertip_depth_m": DEPTH_M,
        }
        defaults.update(kwargs)
        return GraspGenXCatalogProvider(source, **defaults)

    def test_valid_pose_maps_to_existing_candidate_and_preserves_score(self):
        request_json = _request_json()
        source = _Source(
            lambda value: _envelope(
                value,
                [
                    _proposal(
                        yaw_rad=math.pi / 2.0,
                        confidence=0.91,
                        branch_tag="diff",
                    )
                ],
            )
        )
        response = self._provider(source).select(request_json, timeout_s=7.0)
        self.assertEqual(response["selected_candidate_id"], "candidate-b")
        self.assertEqual(response["confidence"], 0.91)
        self.assertEqual(
            [item["candidate_id"] for item in response["ranking"]],
            ["candidate-b", "candidate-a"],
        )
        self.assertEqual(source.calls, [(request_json, 7.0)])

    def test_input_frame_transform_is_applied_before_matching(self):
        request_json = _request_json()
        transform = [
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        source = _Source(
            lambda value: _envelope(
                value,
                [_proposal(tip_xyz=(0.3, 0.0, 0.1))],
                transform_robot_base_from_input=transform,
            )
        )
        response = self._provider(source).select(request_json, timeout_s=3.0)
        self.assertEqual(response["selected_candidate_id"], "candidate-a")

    def test_wrong_approach_or_distant_pose_abstains(self):
        request_json = _request_json()
        wrong_approach = _proposal()
        wrong_approach["transform_input_from_gripper"] = [
            [1.0, 0.0, 0.0, 0.4],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
        proposals = [
            wrong_approach,
            _proposal(
                "proposal-2",
                tip_xyz=(0.5, 0.0, 0.1),
                branch_tag="obb",
            ),
        ]
        source = _Source(lambda value: _envelope(value, proposals))
        response = self._provider(source).select(request_json, timeout_s=3.0)
        self.assertIsNone(response["selected_candidate_id"])
        self.assertEqual(response["confidence"], 0.0)
        self.assertTrue(all(item["score"] == 0.0 for item in response["ranking"]))

    def test_equidistant_catalog_match_is_not_bound(self):
        candidates = [
            {
                "candidate_id": "candidate-a",
                "pick_tip_pose": {
                    "xyz_m": [0.4, 0.0, 0.1],
                    "yaw_rad": 0.0,
                },
            },
            {
                "candidate_id": "candidate-b",
                "pick_tip_pose": {
                    "xyz_m": [0.4, 0.0, 0.1],
                    "yaw_rad": 0.0,
                },
            },
        ]
        request_json = _request_json(candidates)
        source = _Source(lambda value: _envelope(value, [_proposal()]))
        response = self._provider(source).select(request_json, timeout_s=3.0)
        self.assertIsNone(response["selected_candidate_id"])
        self.assertIn("1 ambiguous", response["rationale"])

    def test_stale_frame_model_calibration_and_descriptor_fail(self):
        request_json = _request_json()
        cases = {
            "stale": {"request_sha256": "0" * 64},
            "scene": {"scene_id": "other-scene"},
            "model": {"model_version": "other-model@revision"},
            "frame": {"input_frame_id": "other-camera"},
            "calibration": {"calibration_id": "other-calibration"},
            "depth": {"fingertip_depth_m": DEPTH_M + 0.001},
            "schema": {"schema_version": "wrong-schema"},
        }
        for name, overrides in cases.items():
            source = _Source(
                lambda value, overrides=overrides: _envelope(
                    value, [_proposal()], **overrides
                )
            )
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self._provider(source).select(request_json, timeout_s=3.0)

    def test_invalid_se3_is_rejected(self):
        request_json = _request_json()
        valid = _top_down_transform((0.4, 0.0, 0.1), 0.0)
        matrices = {
            "shape": [[1.0]],
            "homogeneous": [row[:] for row in valid],
            "nonfinite": [row[:] for row in valid],
            "nonorthogonal": [row[:] for row in valid],
            "reflection": [row[:] for row in valid],
        }
        matrices["homogeneous"][3][3] = 2.0
        matrices["nonfinite"][0][0] = math.nan
        matrices["nonorthogonal"][0][0] = 2.0
        for row in range(3):
            matrices["reflection"][row][0] *= -1.0
        for name, matrix in matrices.items():
            proposal = _proposal()
            proposal["transform_input_from_gripper"] = matrix
            source = _Source(
                lambda value, proposal=proposal: _envelope(value, [proposal])
            )
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self._provider(source).select(request_json, timeout_s=3.0)

    def test_proposal_schema_and_numeric_types_are_strict(self):
        request_json = _request_json()
        mutations = {}

        extra = _proposal()
        extra["jaw_width_m"] = 0.05
        mutations["extra_field"] = [extra]

        duplicate_a = _proposal("same")
        duplicate_b = _proposal("same", yaw_rad=math.pi / 2.0)
        mutations["duplicate_ids"] = [duplicate_a, duplicate_b]

        bad_branch = _proposal()
        bad_branch["branch_tag"] = "unknown"
        mutations["branch"] = [bad_branch]

        bool_score = _proposal()
        bool_score["confidence"] = True
        mutations["boolean"] = [bool_score]

        string_score = _proposal()
        string_score["confidence"] = "0.8"
        mutations["string"] = [string_score]

        bool_matrix = _proposal()
        bool_matrix["transform_input_from_gripper"][0][0] = True
        mutations["boolean_matrix"] = [bool_matrix]

        string_matrix = _proposal()
        string_matrix["transform_input_from_gripper"][0][0] = "0.0"
        mutations["string_matrix"] = [string_matrix]

        for name, proposals in mutations.items():
            source = _Source(
                lambda value, proposals=proposals: _envelope(value, proposals)
            )
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self._provider(source).select(request_json, timeout_s=3.0)

    def test_constructor_rejects_invalid_matching_configuration(self):
        source = _Source({})
        cases = (
            {"fingertip_depth_m": 0.0},
            {"maximum_position_error_m": 0.0},
            {"maximum_yaw_error_rad": 0.0},
            {"maximum_approach_tilt_rad": math.pi},
            {"expected_approach_axis_robot_base": (0.0, 0.0, 0.0)},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self._provider(source, **overrides)


if __name__ == "__main__":
    unittest.main()
