import math
import unittest

from sorting_core import (
    BinTarget,
    CubeDetection,
    DetectedObject,
    Pose3D,
    SceneDetections,
    SceneObservation,
    TipPoseBase,
    load_config,
    semantic_phase_plan,
)
from sorting_core.grasping import (
    ConstantSuccessScorer,
    FeasibilityResult,
    GraspCandidate,
    GraspFeatures,
    GraspPlanningContext,
    GraspPlanningError,
    GraspPlanningPipeline,
    GraspScore,
    HighestSuccessSelector,
    LegacyFeatureExtractor,
    LegacyGeometricChecker,
    LegacyTopDownCubeGenerator,
    SuccessTimeSelector,
)


def _cube(colour, x, y, *, timestamp=10.0):
    return CubeDetection(
        track_id=f"{colour}-0",
        colour=colour,
        capture_time_s=timestamp,
        position_base_m=(x, y, -0.0001),
        yaw_rad=0.1,
        confidence=0.9,
    )


class _TwoCandidateGenerator:
    def generate(self, scene, target, context):
        first = TipPoseBase(
            (target.position_base_m[0], target.position_base_m[1], context.pick_tip_z_m),
            0.2,
        )
        second = TipPoseBase(first.xyz_m, -0.2)
        return (
            GraspCandidate("candidate-b", scene.scene_id, target.track_id, first, first),
            GraspCandidate("candidate-a", scene.scene_id, target.track_id, second, second),
        )


class _AcceptAll:
    def check(self, scene, candidate, context):
        return FeasibilityResult(True, checker_id="test")


class _RejectB:
    def check(self, scene, candidate, context):
        if candidate.candidate_id == "candidate-b":
            return FeasibilityResult(False, ("blocked",), checker_id="test")
        return FeasibilityResult(True, checker_id="test")


class _Features:
    def extract(self, scene, candidate, context):
        return GraspFeatures("test-v1", {"yaw": candidate.pick_tip_pose.yaw_rad})


class _RecordingScorer:
    def __init__(self, scores=None):
        self.seen = []
        self.scores = scores or {}

    def score(self, candidate, features):
        self.seen.append(candidate.candidate_id)
        return GraspScore(
            self.scores.get(candidate.candidate_id, 0.5), scorer_id="test"
        )


class GraspIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.scene = SceneDetections(
            "scene-1",
            10.0,
            (
                _cube("blue", 0.44, 0.19),
                _cube("red", 0.44, -0.19),
                _cube("yellow", 0.44, 0.0),
            ),
        )
        self.target = next(c for c in self.scene.cubes if c.colour == "red")
        destination = BinTarget(
            "red", self.config.task.bin_centres_base_m["red"]
        )
        self.context = GraspPlanningContext(
            destination=destination,
            release_tip_pose=TipPoseBase(
                (
                    destination.centre_base_m[0],
                    destination.centre_base_m[1],
                    self.config.bin_release_tip_base_z_m,
                ),
                0.0,
            ),
            pick_tip_z_m=self.config.pick_tip_base_z_m,
            approach_clearance_m=self.config.task.approach_clearance_m,
        )

    def _legacy_pipeline(self):
        return GraspPlanningPipeline(
            LegacyTopDownCubeGenerator(),
            LegacyGeometricChecker(
                maximum_radial_reach_m=self.config.hardware.maximum_radial_reach_m,
                minimum_centre_clearance_m=(
                    self.config.task.minimum_distinct_centres_m
                ),
            ),
            LegacyFeatureExtractor(),
            ConstantSuccessScorer(),
            HighestSuccessSelector(),
        )

    def test_feasibility_acceptance_requires_a_real_boolean(self):
        with self.assertRaisesRegex(ValueError, "accepted must be a boolean"):
            FeasibilityResult("false", checker_id="broken-plugin")

    def test_legacy_pipeline_preserves_plan_id_and_poses(self):
        actual = self._legacy_pipeline().plan_for_target(
            self.scene, self.target, self.context
        )
        bin_x, bin_y = self.config.task.bin_centres_base_m["red"]
        pick = TipPoseBase(
            (
                self.target.position_base_m[0],
                self.target.position_base_m[1],
                self.config.pick_tip_base_z_m,
            ),
            self.target.yaw_rad,
        )
        release = TipPoseBase(
            (bin_x, bin_y, self.config.bin_release_tip_base_z_m), 0.0
        )
        above_pick = TipPoseBase(
            (
                pick.xyz_m[0],
                pick.xyz_m[1],
                pick.xyz_m[2] + self.config.task.approach_clearance_m,
            ),
            pick.yaw_rad,
        )
        above_bin = TipPoseBase(
            (
                release.xyz_m[0],
                release.xyz_m[1],
                release.xyz_m[2] + self.config.task.approach_clearance_m,
            ),
            release.yaw_rad,
        )
        self.assertEqual(actual.plan_id, "scene-1:red-0:red")
        self.assertEqual(actual.pick_tip_pose, pick)
        self.assertEqual(actual.approach_tip_pose, above_pick)
        self.assertEqual(actual.release_tip_pose, release)
        self.assertEqual(actual.destination, BinTarget("red", (bin_x, bin_y)))

        open_gripper = self.config.gripper.open_command
        closed_gripper = self.config.gripper.closed_command
        expected_phases = (
            ("approach_pick", above_pick, open_gripper, 0.0, False),
            ("revalidate_before_descent", None, open_gripper, 0.0, True),
            ("descend_pick", pick, open_gripper, 0.0, False),
            ("close", pick, closed_gripper, 0.6, False),
            ("dwell", pick, closed_gripper, 0.5, False),
            ("lift", above_pick, closed_gripper, 0.0, False),
            ("carry", above_bin, closed_gripper, 0.0, False),
            ("lower", release, closed_gripper, 0.0, False),
            ("release", release, open_gripper, 0.6, False),
            ("settle", release, open_gripper, 0.6, False),
            ("retreat", above_bin, open_gripper, 0.0, False),
        )
        actual_phases = tuple(
            (
                phase.name,
                phase.target,
                phase.gripper,
                phase.dwell_s,
                phase.requires_reobservation_before,
            )
            for phase in semantic_phase_plan(actual, self.config)
        )
        self.assertEqual(actual_phases, expected_phases)

    def test_rejected_candidates_are_never_featured_or_scored(self):
        scorer = _RecordingScorer()
        pipeline = GraspPlanningPipeline(
            _TwoCandidateGenerator(),
            _RejectB(),
            _Features(),
            scorer,
            HighestSuccessSelector(),
        )
        decision = pipeline.decide_for_target(
            self.scene, self.target, self.context
        )
        plan = decision.plan
        self.assertEqual(scorer.seen, ["candidate-a"])
        self.assertAlmostEqual(plan.pick_tip_pose.yaw_rad, -0.2)
        self.assertEqual(
            [item.candidate.candidate_id for item in decision.rejections],
            ["candidate-b"],
        )
        self.assertEqual(
            decision.selected.candidate.candidate_id, "candidate-a"
        )
        self.assertEqual(plan.grasp_selection.candidate_id, "candidate-a")

    def test_highest_success_wins_among_multiple_candidates(self):
        scorer = _RecordingScorer({"candidate-b": 0.9, "candidate-a": 0.4})
        pipeline = GraspPlanningPipeline(
            _TwoCandidateGenerator(),
            _AcceptAll(),
            _Features(),
            scorer,
            HighestSuccessSelector(),
        )
        plan = pipeline.plan_for_target(self.scene, self.target, self.context)
        self.assertAlmostEqual(plan.pick_tip_pose.yaw_rad, 0.2)
        self.assertEqual(scorer.seen, ["candidate-b", "candidate-a"])

    def test_equal_scores_use_candidate_id_tie_break(self):
        pipeline = GraspPlanningPipeline(
            _TwoCandidateGenerator(),
            _AcceptAll(),
            _Features(),
            ConstantSuccessScorer(0.5),
            HighestSuccessSelector(),
        )
        plan = pipeline.plan_for_target(self.scene, self.target, self.context)
        self.assertAlmostEqual(plan.pick_tip_pose.yaw_rad, -0.2)
        self.assertEqual(
            plan.grasp_selection.selector_id, "highest-success-v1"
        )

    def test_success_time_selector_uses_explicit_total_episode_time(self):
        class TimedScorer:
            def score(self, candidate, features):
                values = {
                    "candidate-a": (0.60, 3.0),
                    "candidate-b": (0.90, 9.0),
                }
                probability, total = values[candidate.candidate_id]
                return GraspScore(
                    probability,
                    expected_total_episode_time_s=total,
                    scorer_id="timed-test",
                )

        pipeline = GraspPlanningPipeline(
            _TwoCandidateGenerator(),
            _AcceptAll(),
            _Features(),
            TimedScorer(),
            SuccessTimeSelector(),
        )
        plan = pipeline.plan_for_target(self.scene, self.target, self.context)
        self.assertEqual(plan.grasp_selection.candidate_id, "candidate-a")
        self.assertEqual(
            plan.grasp_selection.selector_id, "success-time-ratio-v1"
        )
        self.assertEqual(
            plan.grasp_selection.selection_objective,
            "minimize_expected_episode_seconds_per_success",
        )

    def test_nonfinite_custom_score_fails_closed(self):
        class BadScorer:
            def score(self, candidate, features):
                result = GraspScore(0.5)
                object.__setattr__(result, "success_probability", math.nan)
                return result

        pipeline = GraspPlanningPipeline(
            _TwoCandidateGenerator(),
            _AcceptAll(),
            _Features(),
            BadScorer(),
            HighestSuccessSelector(),
        )
        with self.assertRaisesRegex(GraspPlanningError, "invalid score"):
            pipeline.plan_for_target(self.scene, self.target, self.context)

    def test_zero_success_candidates_are_not_executable(self):
        pipeline = GraspPlanningPipeline(
            _TwoCandidateGenerator(),
            _AcceptAll(),
            _Features(),
            ConstantSuccessScorer(0.0),
            HighestSuccessSelector(),
        )
        with self.assertRaisesRegex(
            GraspPlanningError, "non-zero predicted success"
        ):
            pipeline.plan_for_target(self.scene, self.target, self.context)

    def test_model_runtime_error_is_wrapped_at_plugin_boundary(self):
        class ExplodingScorer:
            def score(self, candidate, features):
                raise RuntimeError("model runtime unavailable")

        pipeline = GraspPlanningPipeline(
            _TwoCandidateGenerator(),
            _AcceptAll(),
            _Features(),
            ExplodingScorer(),
            HighestSuccessSelector(),
        )
        with self.assertRaisesRegex(
            GraspPlanningError, "grasp scorer failed.*model runtime unavailable"
        ):
            pipeline.plan_for_target(self.scene, self.target, self.context)

    def test_legacy_features_disambiguate_missing_neighbour_and_opening(self):
        features = LegacyFeatureExtractor().extract(
            SceneDetections("isolated", 10.0, (self.target,)),
            LegacyTopDownCubeGenerator().generate(
                SceneDetections("isolated", 10.0, (self.target,)),
                self.target,
                self.context,
            )[0],
            self.context,
        )
        self.assertEqual(features.schema_id, "legacy-top-down-v2")
        self.assertEqual(features.values["has_other_detection"], 0.0)
        self.assertEqual(features.values["required_opening_known"], 0.0)

    def test_all_rejected_candidates_fail_with_reasons(self):
        class RejectAll:
            def check(self, scene, candidate, context):
                return FeasibilityResult(
                    False, ("no finger clearance",), checker_id="test"
                )

        pipeline = GraspPlanningPipeline(
            _TwoCandidateGenerator(),
            RejectAll(),
            _Features(),
            ConstantSuccessScorer(),
            HighestSuccessSelector(),
        )
        with self.assertRaisesRegex(
            GraspPlanningError, "no feasible grasp candidates"
        ):
            pipeline.plan_for_target(self.scene, self.target, self.context)

    def test_legacy_checker_blocks_generic_objects_it_does_not_model(self):
        generic_scene = SceneObservation(
            scene_id=self.scene.scene_id,
            capture_time_s=self.scene.capture_time_s,
            objects=tuple(
                DetectedObject(
                    track_id=cube.track_id,
                    class_id="cube",
                    sort_label=cube.colour,
                    capture_time_s=cube.capture_time_s,
                    pose=Pose3D.from_yaw(cube.position_base_m, cube.yaw_rad),
                    dimensions_m=(0.01, 0.01, 0.01),
                    source_camera_ids=("table",),
                )
                for cube in self.scene.cubes
            )
            + (
                DetectedObject(
                    track_id="screw-0",
                    class_id="screw",
                    sort_label="red",
                    capture_time_s=self.scene.capture_time_s,
                    pose=Pose3D.from_yaw((0.43, -0.18, 0.003), 0.0),
                    dimensions_m=(0.03, 0.006, 0.006),
                    source_camera_ids=("table",),
                ),
            ),
            provider_name="mixed-object-provider",
            provider_version="test",
            calibration_ids=("table-cal",),
        )
        context = GraspPlanningContext(
            destination=self.context.destination,
            release_tip_pose=self.context.release_tip_pose,
            pick_tip_z_m=self.context.pick_tip_z_m,
            approach_clearance_m=self.context.approach_clearance_m,
            generic_scene=generic_scene,
            unsupported_objects=(generic_scene.objects[-1],),
        )
        with self.assertRaisesRegex(
            GraspPlanningError, "unsupported objects"
        ):
            self._legacy_pipeline().plan_for_target(
                self.scene, self.target, context
            )

    def test_supported_non_target_generic_object_is_not_misclassified(self):
        supported_objects = tuple(
            DetectedObject(
                track_id=cube.track_id,
                class_id="cube",
                sort_label=cube.colour,
                capture_time_s=cube.capture_time_s,
                pose=Pose3D.from_yaw(cube.position_base_m, cube.yaw_rad),
                dimensions_m=(0.01, 0.01, 0.01),
                source_camera_ids=("table",),
            )
            for cube in self.scene.cubes
        ) + (
            DetectedObject(
                track_id="red-at-destination",
                class_id="cube",
                sort_label="red",
                capture_time_s=self.scene.capture_time_s,
                pose=Pose3D.from_yaw((0.24, -0.19, 0.003), 0.0),
                dimensions_m=(0.01, 0.01, 0.01),
                source_camera_ids=("table",),
            ),
        )
        generic_scene = SceneObservation(
            scene_id=self.scene.scene_id,
            capture_time_s=self.scene.capture_time_s,
            objects=supported_objects,
            provider_name="cube-provider",
            provider_version="test",
            calibration_ids=("table-cal",),
        )
        context = GraspPlanningContext(
            destination=self.context.destination,
            release_tip_pose=self.context.release_tip_pose,
            pick_tip_z_m=self.context.pick_tip_z_m,
            approach_clearance_m=self.context.approach_clearance_m,
            generic_scene=generic_scene,
            unsupported_objects=(),
        )
        plan = self._legacy_pipeline().plan_for_target(
            self.scene, self.target, context
        )
        self.assertEqual(plan.target.track_id, self.target.track_id)

    def test_stale_generic_scene_cannot_supply_obstacles_or_features(self):
        generic_scene = SceneObservation(
            scene_id="different-scene",
            capture_time_s=self.scene.capture_time_s,
            objects=(),
            provider_name="test",
            provider_version="1",
        )
        context = GraspPlanningContext(
            destination=self.context.destination,
            release_tip_pose=self.context.release_tip_pose,
            pick_tip_z_m=self.context.pick_tip_z_m,
            approach_clearance_m=self.context.approach_clearance_m,
            generic_scene=generic_scene,
            unsupported_objects=(),
        )
        with self.assertRaisesRegex(GraspPlanningError, "different scene IDs"):
            self._legacy_pipeline().plan_for_target(
                self.scene, self.target, context
            )

    def test_generic_scene_requires_explicit_unsupported_projection(self):
        generic_scene = SceneObservation(
            scene_id=self.scene.scene_id,
            capture_time_s=self.scene.capture_time_s,
            objects=(),
            provider_name="test",
            provider_version="1",
        )
        with self.assertRaisesRegex(
            ValueError, "explicit unsupported_objects tuple"
        ):
            GraspPlanningContext(
                destination=self.context.destination,
                release_tip_pose=self.context.release_tip_pose,
                pick_tip_z_m=self.context.pick_tip_z_m,
                approach_clearance_m=self.context.approach_clearance_m,
                generic_scene=generic_scene,
            )


if __name__ == "__main__":
    unittest.main()
