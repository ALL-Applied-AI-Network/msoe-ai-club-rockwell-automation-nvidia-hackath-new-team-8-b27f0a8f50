from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from sorting_core import (
    BinTarget,
    CubeDetection,
    DetectedObject,
    Pose3D,
    SceneDetections,
    SceneObservation,
    TipPoseBase,
    cube_scene_to_observation,
    load_config,
)
from sorting_core.grasping import (
    GraspCandidate,
    GraspFeatures,
    GraspPlanningContext,
    GraspPlanningError,
)
from sorting_core.integration_plugins import GraspFactoryContext

from team_plugins.grasping.multigrasp_pipeline import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_ID,
    JsonLogisticScorer,
    SCREW_CLASS_ID,
    _bounding_radius,
    build_offline_screw_planner,
    create,
    create_simulation,
)
from team_plugins.grasping.prepare_screw_trials import prepare
from team_plugins.grasping.train_success_model import train


class MultiGraspPipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.capture_time_s = 10.0

    def _factory_context(self, **options):
        defaults = {
            "scorer": "geometric_prior",
            "yaw_offsets_deg": [0.0, 45.0, 90.0],
            "required_clearance_m": 0.001,
            "minimum_confidence": 0.2,
            "minimum_pose_quality": 0.2,
            "simulation_only": True,
        }
        defaults.update(options)
        return GraspFactoryContext(
            config=self.config,
            checkpoint=None,
            options=defaults,
            profile_id="test-multigrasp",
            profile_path="test-profile.json",
            profile_sha256="0" * 64,
        )

    def _cube_inputs(
        self,
        *,
        target_confidence=0.9,
        target_yaw_rad=0.0,
        obstacle_xy=(0.50, 0.10),
        obstacle_dimensions=(0.01, 0.01, 0.01),
    ):
        target = CubeDetection(
            track_id="red-0",
            colour="red",
            capture_time_s=self.capture_time_s,
            position_base_m=(0.44, 0.0, -0.0001),
            yaw_rad=target_yaw_rad,
            confidence=target_confidence,
        )
        scene = SceneDetections(
            "cube-scene",
            self.capture_time_s,
            (target,),
        )
        target_object = DetectedObject(
            track_id=target.track_id,
            class_id="cube",
            sort_label="red",
            capture_time_s=self.capture_time_s,
            pose=Pose3D.from_yaw(target.position_base_m, target.yaw_rad),
            dimensions_m=(0.01, 0.01, 0.01),
            confidence=target_confidence,
            pose_quality=target_confidence,
            visibility=0.95,
            source_camera_ids=("table",),
        )
        obstacle = DetectedObject(
            track_id="obstacle-0",
            class_id="cube",
            sort_label="blue",
            capture_time_s=self.capture_time_s,
            pose=Pose3D.from_yaw(
                (obstacle_xy[0], obstacle_xy[1], -0.0001), 0.0
            ),
            dimensions_m=obstacle_dimensions,
            confidence=0.9,
            pose_quality=0.9,
            visibility=0.9,
            source_camera_ids=("table",),
        )
        observation = SceneObservation(
            scene_id=scene.scene_id,
            capture_time_s=scene.capture_time_s,
            objects=(target_object, obstacle),
            provider_name="test-provider",
            provider_version="1",
            calibration_ids=("table-cal",),
        )
        destination = BinTarget(
            "red", self.config.task.bin_centres_base_m["red"]
        )
        context = GraspPlanningContext(
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
            generic_scene=observation,
            unsupported_objects=(),
        )
        return scene, target, context

    def _screw_scene(self, obstacle=None):
        half = math.sqrt(0.5)
        screw = DetectedObject(
            track_id="screw-0",
            class_id=SCREW_CLASS_ID,
            sort_label="red",
            capture_time_s=self.capture_time_s,
            pose=Pose3D(
                position_base_m=(0.40, 0.0, 0.0014),
                orientation_xyzw=(0.0, -half, 0.0, half),
            ),
            dimensions_m=(0.012, 0.01199627, 0.030),
            confidence=0.95,
            pose_quality=0.9,
            visibility=0.9,
            source_camera_ids=("table",),
        )
        objects = (screw,) if obstacle is None else (screw, obstacle)
        return SceneObservation(
            scene_id="screw-scene",
            capture_time_s=self.capture_time_s,
            objects=objects,
            provider_name="test-provider",
            provider_version="1",
            calibration_ids=("table-cal",),
        )

    def test_factory_builds_three_scored_cube_candidates(self):
        pipeline = create_simulation(self._factory_context())
        scene, target, context = self._cube_inputs()
        decision = pipeline.decide_for_target(scene, target, context)
        self.assertEqual(len(decision.evaluations), 3)
        self.assertFalse(decision.rejections)
        self.assertEqual(
            decision.selected.candidate.candidate_id, "cube-yaw-000-0"
        )
        self.assertEqual(
            decision.selected.features.schema_id, FEATURE_SCHEMA_ID
        )
        self.assertEqual(
            tuple(decision.selected.features.values), FEATURE_NAMES
        )

    def test_standard_factory_rejects_the_uncalibrated_ranker(self):
        with self.assertRaisesRegex(
            ValueError, "unavailable through the executable integration factory"
        ):
            create(self._factory_context())

    def test_baseline_cube_adapter_uses_documented_confidence_compatibility(self):
        pipeline = create_simulation(self._factory_context())
        scene, target, context = self._cube_inputs()
        adapted = cube_scene_to_observation(
            scene,
            dimensions_m=self.config.cube_edge_m,
            source_camera_ids=("table",),
        )
        compatible = replace(
            context,
            generic_scene=adapted,
            unsupported_objects=(),
        )
        decision = pipeline.decide_for_target(scene, target, compatible)
        self.assertEqual(len(decision.evaluations), 3)
        features = decision.selected.features.values
        self.assertEqual(features["pose_quality"], target.confidence)
        self.assertEqual(features["visibility"], target.confidence)

    def test_cube_openings_include_edge_and_diagonal_widths(self):
        pipeline = create_simulation(self._factory_context())
        scene, target, context = self._cube_inputs()
        candidates = pipeline.generator.generate(scene, target, context)
        self.assertAlmostEqual(candidates[0].required_opening_m, 0.0103)
        self.assertAlmostEqual(
            candidates[1].required_opening_m,
            math.sqrt(2.0) * 0.01 + 0.0003,
        )
        self.assertAlmostEqual(candidates[2].required_opening_m, 0.0103)

    def test_cube_obstacle_rejects_only_intersecting_yaw_envelopes(self):
        pipeline = create_simulation(self._factory_context())
        scene, target, context = self._cube_inputs(obstacle_xy=(0.459, 0.0))
        decision = pipeline.decide_for_target(scene, target, context)
        accepted = {
            item.candidate.candidate_id for item in decision.evaluations
        }
        rejected = {
            item.candidate.candidate_id for item in decision.rejections
        }
        self.assertIn("cube-yaw-090-2", accepted)
        self.assertIn("cube-yaw-000-0", rejected)
        self.assertTrue(
            all(
                "top-down finger envelope is obstructed"
                in item.feasibility.rejection_reasons
                for item in decision.rejections
            )
        )

    def test_rotated_target_preserves_finger_and_closing_axes(self):
        pipeline = create_simulation(self._factory_context())
        target_yaw = math.radians(30.0)
        distance = 0.019
        scene, target, context = self._cube_inputs(
            target_yaw_rad=target_yaw,
            obstacle_xy=(
                0.44 + distance * math.cos(target_yaw),
                distance * math.sin(target_yaw),
            ),
        )
        decision = pipeline.decide_for_target(scene, target, context)
        accepted = {
            item.candidate.candidate_id for item in decision.evaluations
        }
        rejected = {
            item.candidate.candidate_id for item in decision.rejections
        }
        self.assertIn("cube-yaw-000-0", rejected)
        self.assertIn("cube-yaw-090-2", accepted)

    def test_finger_width_and_closing_extent_boundaries(self):
        required_clearance = 0.001
        pipeline = create_simulation(
            self._factory_context(
                yaw_offsets_deg=[0.0],
                required_clearance_m=required_clearance,
            )
        )
        tiny_dimensions = (1e-6, 1e-6, 1e-6)
        obstacle_radius = math.sqrt(3.0) * 0.5e-6
        half_extents = {
            "finger_length_axis": (
                0.01 / 2.0
                + self.config.gripper.finger_width_m / 2.0
            ),
            "closing_axis": (
                0.0103 / 2.0
                + self.config.gripper.finger_thickness_m / 2.0
            ),
        }
        epsilon = 0.0001
        for axis_name, half_extent in half_extents.items():
            for delta, obstructed in (
                (-epsilon, True),
                (epsilon, False),
            ):
                distance = (
                    half_extent
                    + obstacle_radius
                    + required_clearance
                    + delta
                )
                obstacle_xy = (
                    (0.44 + distance, 0.0)
                    if axis_name == "finger_length_axis"
                    else (0.44, distance)
                )
                scene, target, context = self._cube_inputs(
                    obstacle_xy=obstacle_xy,
                    obstacle_dimensions=tiny_dimensions,
                )
                candidate = pipeline.generator.generate(
                    scene,
                    target,
                    context,
                )[0]
                result = pipeline.checker.check(
                    scene,
                    candidate,
                    context,
                )
                with self.subTest(
                    axis=axis_name,
                    obstructed=obstructed,
                ):
                    self.assertEqual(
                        "top-down finger envelope is obstructed"
                        in result.rejection_reasons,
                        obstructed,
                    )
                    self.assertAlmostEqual(
                        result.metrics["minimum_footprint_clearance_m"],
                        required_clearance + delta,
                        places=9,
                    )

    def test_low_quality_cube_fails_before_scoring(self):
        pipeline = create_simulation(self._factory_context())
        scene, target, context = self._cube_inputs(target_confidence=0.1)
        with self.assertRaisesRegex(
            GraspPlanningError, "no feasible grasp candidates"
        ):
            pipeline.decide_for_target(scene, target, context)

    def test_unsupported_generic_object_fails_closed(self):
        pipeline = create_simulation(self._factory_context())
        scene, target, context = self._cube_inputs()
        unsupported = context.generic_scene.objects[1]
        guarded = replace(context, unsupported_objects=(unsupported,))
        with self.assertRaisesRegex(
            GraspPlanningError, "no feasible grasp candidates"
        ):
            pipeline.decide_for_target(scene, target, guarded)

    def test_missing_generic_target_fails_closed(self):
        pipeline = create_simulation(self._factory_context())
        scene, target, context = self._cube_inputs()
        incomplete_scene = replace(
            context.generic_scene,
            objects=(context.generic_scene.objects[1],),
        )
        guarded = replace(
            context,
            generic_scene=incomplete_scene,
            unsupported_objects=(),
        )
        with self.assertRaisesRegex(
            GraspPlanningError, "no feasible grasp candidates"
        ):
            pipeline.decide_for_target(scene, target, guarded)

    def test_generic_target_pose_and_class_mismatches_fail_closed(self):
        pipeline = create_simulation(self._factory_context())
        scene, target, context = self._cube_inputs()
        observed_target, obstacle = context.generic_scene.objects
        variants = (
            replace(
                observed_target,
                pose=Pose3D.from_yaw((0.45, 0.0, -0.0001), 0.0),
            ),
            replace(
                observed_target,
                pose=Pose3D.from_yaw(
                    observed_target.pose.position_base_m, 0.2
                ),
            ),
            replace(
                observed_target,
                dimensions_m=(0.016, 0.016, 0.016),
            ),
            replace(observed_target, class_id=SCREW_CLASS_ID),
        )
        for index, bad_target in enumerate(variants):
            with self.subTest(variant=index):
                inconsistent = replace(
                    context.generic_scene,
                    objects=(bad_target, obstacle),
                )
                guarded = replace(
                    context,
                    generic_scene=inconsistent,
                    unsupported_objects=(),
                )
                with self.assertRaisesRegex(
                    GraspPlanningError, "no feasible grasp candidates"
                ):
                    pipeline.decide_for_target(scene, target, guarded)

    def test_screw_without_destination_label_is_rejected(self):
        scene = self._screw_scene()
        unlabelled = replace(scene.objects[0], sort_label=None)
        planner = build_offline_screw_planner(self.config)
        with self.assertRaisesRegex(
            GraspPlanningError, "destination sort label"
        ):
            planner.decide(
                replace(scene, objects=(unlabelled,)),
                "screw-0",
                pick_tip_z_m=self.config.pick_tip_base_z_m,
                approach_clearance_m=self.config.task.approach_clearance_m,
                release_tip_pose=TipPoseBase((0.24, -0.19, 0.04), 0.0),
            )

    def test_object_radius_uses_the_orientation_independent_half_diagonal(self):
        screw = self._screw_scene().objects[0]
        expected = 0.5 * math.sqrt(
            sum(value * value for value in screw.dimensions_m)
        )
        self.assertAlmostEqual(_bounding_radius(screw), expected)
        self.assertGreater(expected, 0.017)

    def test_screw_catalog_dimension_mismatch_is_rejected(self):
        scene = self._screw_scene()
        bad_screw = replace(scene.objects[0], dimensions_m=(0.006, 0.006, 0.020))
        planner = build_offline_screw_planner(self.config)
        with self.assertRaisesRegex(ValueError, "catalog geometry"):
            planner.decide(
                replace(scene, objects=(bad_screw,)),
                "screw-0",
                pick_tip_z_m=self.config.pick_tip_base_z_m,
                approach_clearance_m=self.config.task.approach_clearance_m,
                release_tip_pose=TipPoseBase((0.24, -0.19, 0.04), 0.0),
            )

    def test_screw_candidates_are_measured_from_catalog_head(self):
        planner = build_offline_screw_planner(self.config)
        scene = self._screw_scene()
        decision = planner.decide(
            scene,
            "screw-0",
            pick_tip_z_m=self.config.pick_tip_base_z_m,
            approach_clearance_m=self.config.task.approach_clearance_m,
            release_tip_pose=TipPoseBase((0.24, -0.19, 0.04), 0.0),
        )
        self.assertFalse(decision.execution_supported)
        self.assertEqual(len(decision.evaluations), 3)
        positions = [
            item.candidate.pick_tip_pose.xyz_m[0]
            for item in decision.evaluations
        ]
        self.assertEqual(
            [round(value, 3) for value in positions],
            [0.412, 0.416, 0.420],
        )
        self.assertEqual(
            decision.selected.candidate.candidate_id,
            "screw-shaft-016mm-1",
        )

    def test_screw_tip_obstacle_rejects_tip_side_candidate(self):
        obstacle = DetectedObject(
            track_id="yellow-cube",
            class_id="cube",
            sort_label="yellow",
            capture_time_s=self.capture_time_s,
            pose=Pose3D.from_yaw((0.441, 0.0, 0.0), 0.0),
            dimensions_m=(0.01, 0.01, 0.01),
            confidence=0.9,
            pose_quality=0.9,
            visibility=0.9,
            source_camera_ids=("table",),
        )
        planner = build_offline_screw_planner(self.config)
        decision = planner.decide(
            self._screw_scene(obstacle),
            "screw-0",
            pick_tip_z_m=self.config.pick_tip_base_z_m,
            approach_clearance_m=self.config.task.approach_clearance_m,
            release_tip_pose=TipPoseBase((0.24, -0.19, 0.04), 0.0),
        )
        rejected = {
            item.candidate.candidate_id for item in decision.rejections
        }
        self.assertIn("screw-shaft-020mm-2", rejected)
        self.assertIn(
            "top-down finger envelope is obstructed",
            decision.rejections[0].feasibility.rejection_reasons,
        )

    def test_unknown_obstacle_geometry_fails_closed_for_screw(self):
        obstacle = DetectedObject(
            track_id="unknown-0",
            class_id="unknown",
            sort_label=None,
            capture_time_s=self.capture_time_s,
            pose=Pose3D.from_yaw((0.45, 0.0, 0.0), 0.0),
            dimensions_m=None,
            confidence=0.5,
            source_camera_ids=("table",),
        )
        planner = build_offline_screw_planner(self.config)
        with self.assertRaisesRegex(
            GraspPlanningError, "no feasible offline screw candidates"
        ):
            planner.decide(
                self._screw_scene(obstacle),
                "screw-0",
                pick_tip_z_m=self.config.pick_tip_base_z_m,
                approach_clearance_m=self.config.task.approach_clearance_m,
                release_tip_pose=TipPoseBase((0.24, -0.19, 0.04), 0.0),
            )

    def test_prepared_screw_trials_keep_the_same_landing_head_pose(self):
        from demos.scripted_cube_screw_bucket_sort.scenario import (
            build_sort_items,
            load_scenario_config,
        )

        with tempfile.TemporaryDirectory() as directory:
            manifest_path = prepare(
                Path(directory) / "trials", require_clean=False
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["offsets_mm"], [12, 16, 20])
            self.assertRegex(manifest["source_commit"], r"^[0-9a-f]{40}$")
            self.assertIsInstance(
                manifest["relevant_source_tree_clean"], bool
            )
            self.assertEqual(
                set(manifest["source_tree_sha256"]),
                {"config", "demos", "sorting_core"},
            )
            landing_heads = []
            grasp_distances = []
            for case in manifest["cases"]:
                config = load_scenario_config(case["config_path"])
                self.assertEqual(
                    case["sort_order"],
                    [
                        "cube-red",
                        "cube-yellow",
                        "cube-blue",
                        "screw-red-01",
                        "screw-red-02",
                    ],
                )
                self.assertEqual(
                    case["scenario_sha256"],
                    hashlib.sha256(
                        Path(case["config_path"]).read_bytes()
                    ).hexdigest(),
                )
                target = next(
                    item
                    for item in build_sort_items(config)
                    if item.item_id == "screw-red-01"
                )
                landing_heads.append(
                    tuple(
                        round(value, 9)
                        for value in target.placement_reference_base_m
                    )
                )
                grasp_distances.append(
                    round(
                        math.dist(
                            target.spawn_reference_base_m,
                            target.spawn_grasp_point_base_m,
                        )
                        * 1000
                    )
                )
            self.assertEqual(grasp_distances, [12, 16, 20])
            self.assertEqual(len(set(landing_heads)), 1)

    def test_trainer_produces_a_loadable_calibrated_json_model(self):
        rows = []
        for index in range(60):
            success = index % 2 == 0
            values = {name: 0.0 for name in FEATURE_NAMES}
            values["minimum_footprint_clearance_m"] = (
                0.02 if success else 0.0
            )
            values["confidence"] = 0.95 if success else 0.25
            values["pose_quality"] = 0.9 if success else 0.3
            values["visibility"] = 0.9
            values["required_opening_m"] = 0.01
            values["opening_fraction"] = 0.65
            rows.append(
                {
                    "episode_id": f"episode-{index:03d}",
                    "group_id": f"layout-{index // 2:03d}",
                    "candidate_id": f"candidate-{index % 2}",
                    "split": "train" if index // 2 < 24 else "test",
                    "features": values,
                    "success": success,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "episodes.jsonl"
            dataset.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            model = root / "model.json"
            metrics = root / "metrics.json"
            result = train(
                dataset,
                model,
                metrics,
                iterations=600,
                learning_rate=0.08,
                l2=0.01,
            )
            self.assertLess(result["test"]["brier_score"], 0.1)
            self.assertEqual(
                result["test"]["selection"]["top_candidate_ranking_accuracy"],
                1.0,
            )
            self.assertRegex(result["training_run_sha256"], r"^[0-9a-f]{64}$")
            alternate = train(
                dataset,
                root / "model-alternate.json",
                root / "metrics-alternate.json",
                iterations=600,
                learning_rate=0.08,
                l2=0.02,
            )
            self.assertNotEqual(
                result["model_version"], alternate["model_version"]
            )
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            scorer = JsonLogisticScorer(model, expected_sha256=digest)
            candidate = GraspCandidate(
                "candidate",
                "scene",
                "target",
                TipPoseBase((0.4, 0.0, 0.0), 0.0),
                TipPoseBase((0.4, 0.0, 0.1), 0.0),
            )
            success_features = GraspFeatures(
                FEATURE_SCHEMA_ID, rows[0]["features"]
            )
            failure_features = GraspFeatures(
                FEATURE_SCHEMA_ID, rows[1]["features"]
            )
            self.assertGreater(
                scorer.score(candidate, success_features).success_probability,
                scorer.score(candidate, failure_features).success_probability,
            )

    def test_trainer_rejects_group_leakage_and_path_aliases(self):
        values = {name: 0.0 for name in FEATURE_NAMES}
        rows = [
            {
                "episode_id": "train-0",
                "group_id": "same-layout",
                "candidate_id": "candidate-0",
                "split": "train",
                "features": values,
                "success": True,
            },
            {
                "episode_id": "test-0",
                "group_id": "same-layout",
                "candidate_id": "candidate-1",
                "split": "test",
                "features": values,
                "success": False,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "episodes.jsonl"
            dataset.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "crosses train and test"):
                train(dataset, root / "model.json", root / "metrics.json")
            with self.assertRaisesRegex(ValueError, "paths must be distinct"):
                train(dataset, dataset, root / "metrics-2.json")

    def test_json_logistic_checkpoint_is_hash_bound_and_non_executable(self):
        payload = {
            "schema_version": 1,
            "model_type": "logistic_regression",
            "feature_schema_id": FEATURE_SCHEMA_ID,
            "feature_order": list(FEATURE_NAMES),
            "means": [0.0] * len(FEATURE_NAMES),
            "scales": [1.0] * len(FEATURE_NAMES),
            "coefficients": [0.0] * len(FEATURE_NAMES),
            "intercept": 0.0,
            "model_version": "test-logistic-v1",
            "training_data_sha256": "1" * 64,
        }
        raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            scorer = JsonLogisticScorer(path, expected_sha256=digest)
            features = GraspFeatures(
                FEATURE_SCHEMA_ID, {name: 0.0 for name in FEATURE_NAMES}
            )
            candidate = GraspCandidate(
                "candidate",
                "scene",
                "target",
                TipPoseBase((0.4, 0.0, 0.0), 0.0),
                TipPoseBase((0.4, 0.0, 0.1), 0.0),
            )
            self.assertAlmostEqual(
                scorer.score(candidate, features).success_probability, 0.5
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                JsonLogisticScorer(path, expected_sha256="0" * 64)
            payload["model_version"] = ["not", "a", "string"]
            invalid = (json.dumps(payload, sort_keys=True) + "\n").encode()
            path.write_bytes(invalid)
            with self.assertRaisesRegex(ValueError, "must be a string"):
                JsonLogisticScorer(
                    path,
                    expected_sha256=hashlib.sha256(invalid).hexdigest(),
                )


if __name__ == "__main__":
    unittest.main()
