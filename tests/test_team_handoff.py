import json
from pathlib import Path
import unittest

from sorting_core import BinTarget, TipPoseBase, load_config
from sorting_core.grasping import (
    FeasibilityResult,
    GraspCandidate,
    GraspFeatures,
    GraspPlanningContext,
    GraspPlanningError,
    GraspPlanningPipeline,
    GraspScore,
    HighestSuccessSelector,
)
from sorting_core.perception import (
    CameraFrame,
    PerceptionRequest,
    scene_observation_from_dict,
    scene_observation_to_dict,
    validate_scene_for_request,
)
from sorting_core.task_perception import cube_scene_from_observation
from sorting_core.team_integrations import CUBE_TASK_CLASS_IDS


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "team_plugins" / "fixtures"
CALIBRATION_ID = "golden-table-calibration-v1"


def _raw_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _scene(name):
    return scene_observation_from_dict(_raw_fixture(name))


def _request(scene, *, capture_time_s=None):
    timestamp = scene.capture_time_s if capture_time_s is None else capture_time_s
    return PerceptionRequest(
        scene_id=scene.scene_id,
        capture_time_s=timestamp,
        frames=(
            CameraFrame(
                camera_id="table",
                capture_time_s=timestamp,
                calibration_id=CALIBRATION_ID,
                rgb=[[[0, 0, 0]]],
            ),
        ),
        metadata={"source": "golden-cross-team-fixture"},
    )


def _planning_context(config, generic_scene, unsupported_objects, colour):
    destination = BinTarget(colour, config.task.bin_centres_base_m[colour])
    return GraspPlanningContext(
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
        generic_scene=generic_scene,
        unsupported_objects=unsupported_objects,
    )


class _TwoFeasibleCandidates:
    def generate(self, scene, target, context):
        xyz = (
            target.position_base_m[0],
            target.position_base_m[1],
            context.pick_tip_z_m,
        )
        low_pick = TipPoseBase(xyz, -0.2)
        high_pick = TipPoseBase(xyz, 0.2)

        def candidate(candidate_id, pick):
            return GraspCandidate(
                candidate_id=candidate_id,
                scene_id=scene.scene_id,
                target_track_id=target.track_id,
                pick_tip_pose=pick,
                approach_tip_pose=TipPoseBase(
                    (xyz[0], xyz[1], xyz[2] + context.approach_clearance_m),
                    pick.yaw_rad,
                ),
                generator_id="golden-two-feasible-v1",
            )

        return (
            candidate("lower-probability", low_pick),
            candidate("higher-probability", high_pick),
        )


class _AcceptBothCandidates:
    def check(self, scene, candidate, context):
        return FeasibilityResult(
            True,
            checker_id="golden-accept-both-v1",
            metrics={"absolute_yaw_rad": abs(candidate.pick_tip_pose.yaw_rad)},
        )


class _CandidateIdentityFeatures:
    def extract(self, scene, candidate, context):
        return GraspFeatures(
            "golden-candidate-identity-v1",
            {"is_higher_probability_candidate": float(
                candidate.candidate_id == "higher-probability"
            )},
        )


class _UnequalSuccessScorer:
    def score(self, candidate, features):
        return GraspScore(
            0.85 if candidate.candidate_id == "higher-probability" else 0.2,
            scorer_id="golden-unequal-scorer-v1",
            model_version="hand-authored-test-scores-v1",
        )


class GoldenTeamHandoffFixtureTests(unittest.TestCase):
    def test_valid_cube_contract_round_trip_reaches_legacy_planner(self):
        raw = _raw_fixture("valid_cube_scene.v1.json")
        scene = scene_observation_from_dict(raw)
        self.assertEqual(scene_observation_to_dict(scene), raw)
        self.assertIs(validate_scene_for_request(_request(scene), scene), scene)

        projection = cube_scene_from_observation(
            scene,
            class_ids=CUBE_TASK_CLASS_IDS,
        )
        self.assertEqual(projection.unsupported_objects, ())
        self.assertEqual(
            {cube.colour for cube in projection.scene.cubes},
            {"red", "yellow", "blue"},
        )

        config = load_config()
        target = next(
            cube for cube in projection.scene.cubes if cube.colour == "red"
        )
        decision = GraspPlanningPipeline.legacy_default(config).decide_for_target(
            projection.scene,
            target,
            _planning_context(
                config,
                scene,
                projection.unsupported_objects,
                target.colour,
            ),
        )
        selection = decision.plan.grasp_selection
        self.assertIsNotNone(selection)
        self.assertEqual(selection.selector_id, "highest-success-v1")
        self.assertEqual(
            selection.selection_objective,
            "maximize_complete_grasp_success_probability",
        )
        self.assertEqual(selection.success_probability, 1.0)
        self.assertIsNone(selection.expected_total_episode_time_s)
        self.assertEqual(decision.plan.target.track_id, "cube-red-001")

    def test_two_feasible_candidates_select_higher_success_without_time(self):
        scene = _scene("valid_cube_scene.v1.json")
        validate_scene_for_request(_request(scene), scene)
        projection = cube_scene_from_observation(
            scene,
            class_ids=CUBE_TASK_CLASS_IDS,
        )
        target = next(
            cube for cube in projection.scene.cubes if cube.colour == "red"
        )
        config = load_config()
        pipeline = GraspPlanningPipeline(
            _TwoFeasibleCandidates(),
            _AcceptBothCandidates(),
            _CandidateIdentityFeatures(),
            _UnequalSuccessScorer(),
            HighestSuccessSelector(),
        )
        decision = pipeline.decide_for_target(
            projection.scene,
            target,
            _planning_context(
                config,
                scene,
                projection.unsupported_objects,
                target.colour,
            ),
        )

        self.assertEqual(len(decision.evaluations), 2)
        self.assertEqual(decision.rejections, ())
        self.assertEqual(
            {item.candidate.candidate_id for item in decision.evaluations},
            {"lower-probability", "higher-probability"},
        )
        selection = decision.plan.grasp_selection
        self.assertEqual(selection.candidate_id, "higher-probability")
        self.assertEqual(selection.selector_id, "highest-success-v1")
        self.assertEqual(selection.success_probability, 0.85)
        self.assertIsNone(selection.expected_total_episode_time_s)
        self.assertIsNone(selection.expected_execution_time_s)
        self.assertIsNone(selection.expected_recovery_time_s)

    def test_valid_screw_round_trips_but_cube_task_keeps_it_unsupported(self):
        raw = _raw_fixture("valid_screw_scene.v1.json")
        scene = scene_observation_from_dict(raw)
        self.assertEqual(scene_observation_to_dict(scene), raw)
        validate_scene_for_request(_request(scene), scene)

        projection = cube_scene_from_observation(
            scene,
            class_ids=CUBE_TASK_CLASS_IDS,
        )
        self.assertEqual(projection.scene.cubes, ())
        self.assertEqual(len(projection.unsupported_objects), 1)
        self.assertEqual(
            projection.unsupported_objects[0].class_id,
            "screw_iso10642_m6x30",
        )
        self.assertAlmostEqual(
            projection.unsupported_objects[0].pose.position_base_m[2],
            0.0014,
        )
        self.assertNotIn("screw_iso10642_m6x30", CUBE_TASK_CLASS_IDS)

    def test_mixed_cube_and_screw_directly_blocks_legacy_planner(self):
        raw = _raw_fixture("mixed_cube_screw_scene.v1.json")
        scene = scene_observation_from_dict(raw)
        self.assertEqual(scene_observation_to_dict(scene), raw)
        validate_scene_for_request(_request(scene), scene)
        projection = cube_scene_from_observation(
            scene,
            class_ids=CUBE_TASK_CLASS_IDS,
        )
        self.assertEqual(len(projection.scene.cubes), 1)
        self.assertEqual(len(projection.unsupported_objects), 1)
        self.assertEqual(
            projection.unsupported_objects[0].class_id,
            "screw_iso10642_m6x30",
        )

        config = load_config()
        target = next(
            cube for cube in projection.scene.cubes if cube.colour == "red"
        )
        with self.assertRaisesRegex(GraspPlanningError, "unsupported objects"):
            GraspPlanningPipeline.legacy_default(config).plan_for_target(
                projection.scene,
                target,
                _planning_context(
                    config,
                    scene,
                    projection.unsupported_objects,
                    target.colour,
                ),
            )

    def test_stale_scene_fails_request_validation(self):
        scene = _scene("stale_cube_scene.v1.json")
        with self.assertRaisesRegex(ValueError, "stale capture timestamp"):
            validate_scene_for_request(
                _request(scene, capture_time_s=scene.capture_time_s + 0.25),
                scene,
            )

    def test_occluded_unknown_remains_visible_and_blocks_legacy_grasp(self):
        raw = _raw_fixture("occluded_mixed_scene.v1.json")
        scene = scene_observation_from_dict(raw)
        self.assertEqual(scene_observation_to_dict(scene), raw)
        validate_scene_for_request(_request(scene), scene)

        projection = cube_scene_from_observation(
            scene,
            class_ids=CUBE_TASK_CLASS_IDS,
        )
        self.assertEqual(len(projection.scene.cubes), 1)
        self.assertEqual(len(projection.unsupported_objects), 1)
        unknown = projection.unsupported_objects[0]
        self.assertEqual(unknown.attributes["region_status"], "unknown_not_clear")
        self.assertLess(unknown.visibility, 0.2)

        config = load_config()
        target = next(
            cube for cube in projection.scene.cubes if cube.colour == "red"
        )
        # The class remains unsupported and therefore blocks the legacy
        # checker. Its visibility/uncertainty metadata is only round-tripped;
        # the legacy checker does not interpret it as geometric uncertainty.
        with self.assertRaisesRegex(GraspPlanningError, "unsupported objects"):
            GraspPlanningPipeline.legacy_default(config).plan_for_target(
                projection.scene,
                target,
                _planning_context(
                    config,
                    scene,
                    projection.unsupported_objects,
                    target.colour,
                ),
            )

    def test_bad_quaternion_fixture_fails_strict_parser(self):
        with self.assertRaisesRegex(ValueError, "unit quaternion"):
            scene_observation_from_dict(
                _raw_fixture("invalid_bad_quaternion_scene.v1.json")
            )


if __name__ == "__main__":
    unittest.main()
