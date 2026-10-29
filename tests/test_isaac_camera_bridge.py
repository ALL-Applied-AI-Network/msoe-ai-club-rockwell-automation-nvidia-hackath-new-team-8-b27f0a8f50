from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from isaac_bridge import (  # noqa: E402
    CalibratedPinholeCamera,
    IsaacCameraSortAdapter,
    IsaacObservationError,
)
from demos.scripted_colour_bucket_sort.colour_bucket_demo import (  # noqa: E402
    _clipped_amodal_bbox,
    _phase_interpolates_spread,
    _should_stop_after_truth_evaluation,
)
from sorting_core import (  # noqa: E402
    CameraFrame,
    ConstantSuccessScorer,
    DetectedObject,
    PerceptionRequest,
    Pose3D,
    RobotState,
    SceneObservation,
    GraspPlanningPipeline,
    HighestSuccessSelector,
    LegacyFeatureExtractor,
    LegacyGeometricChecker,
    LegacyTopDownCubeGenerator,
    load_config,
)


class PinholeCameraTests(unittest.TestCase):
    def setUp(self):
        self.camera = CalibratedPinholeCamera.from_look_at(
            eye_base_m=(0.34, 0.0, 0.80),
            look_at_base_m=(0.34, 0.0, 0.0),
            up_base=(0.0, 1.0, 0.0),
            resolution_px=(1000, 800),
            focal_length_mm=20.0,
            horizontal_aperture_mm=20.0,
            default_plane_base_z_m=-0.0051,
        )

    def test_project_and_plane_intersection_round_trip(self):
        for point in (
            np.array([0.44, -0.19, -0.0001]),
            np.array([0.44, 0.00, -0.0001]),
            np.array([0.24, 0.19, 0.0499]),
        ):
            uv = self.camera.project(point)
            self.assertIsNotNone(uv)
            recovered = self.camera.to_plane(*uv, z=float(point[2]))
            np.testing.assert_allclose(recovered, point, atol=1e-10)

    def test_capture_metadata_schema_and_row_vector_guard(self):
        base_origin_world = np.array([1.2, -0.4, 0.9])
        camera_to_world = np.asarray(
            self.camera.camera_to_base_row_major, dtype=float
        ).copy()
        camera_to_world[3, :3] += base_origin_world
        metadata = {
            "resolution": list(self.camera.resolution_px),
            "focal_length_mm": self.camera.focal_length_mm,
            "horizontal_aperture_mm": self.camera.horizontal_aperture_mm,
            "vertical_aperture_mm": self.camera.vertical_aperture_mm,
            "camera_to_world": camera_to_world.tolist(),
            "table_top_z": float(
                base_origin_world[2] + self.camera.default_plane_base_z_m
            ),
            "robot_base_xy": base_origin_world[:2].tolist(),
        }
        with self.assertRaisesRegex(IsaacObservationError, "surface_base_z_m"):
            CalibratedPinholeCamera.from_capture_metadata(metadata)

        loaded = CalibratedPinholeCamera.from_capture_metadata(
            metadata,
            surface_base_z_m=self.camera.default_plane_base_z_m,
        )
        np.testing.assert_allclose(loaded.eye_base_m, self.camera.eye_base_m)
        self.assertAlmostEqual(
            loaded.default_plane_base_z_m,
            self.camera.default_plane_base_z_m,
        )
        test_point = np.array([0.44, -0.19, self.camera.default_plane_base_z_m])
        uv = loaded.project(test_point)
        np.testing.assert_allclose(
            loaded.to_plane(*uv), test_point, atol=1e-10
        )

        column_translation = np.eye(4)
        column_translation[:3, 3] = (0.2, 0.1, 0.8)
        with self.assertRaisesRegex(IsaacObservationError, "row-vector"):
            CalibratedPinholeCamera(
                (100, 100),
                10.0,
                20.0,
                20.0,
                tuple(tuple(row) for row in column_translation),
                0.0,
            )

    def test_amodal_projected_box_is_clipped_or_invalid(self):
        clipped, status = _clipped_amodal_bbox(
            np.array([[-5.0, 20.0], [50.0, 120.0], [80.0, 40.0]]),
            (100, 100),
        )
        self.assertEqual(status, "valid_clipped")
        self.assertEqual(clipped, [0.0, 20.0, 80.0, 100.0])

        outside, outside_status = _clipped_amodal_bbox(
            np.array([[110.0, 20.0], [130.0, 40.0]]),
            (100, 100),
        )
        self.assertIsNone(outside)
        self.assertEqual(outside_status, "invalid_outside_image")


class IsaacCameraSortAdapterTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.camera = CalibratedPinholeCamera.from_look_at(
            eye_base_m=(0.34, 0.0, 0.80),
            look_at_base_m=(0.34, 0.0, 0.0),
            up_base=(0.0, 1.0, 0.0),
            resolution_px=(1000, 800),
            focal_length_mm=20.0,
            horizontal_aperture_mm=20.0,
            default_plane_base_z_m=self.config.frames.surface_base_z_m,
        )
        self.adapter = IsaacCameraSortAdapter(self.camera, self.config)

    def _render_top_faces(self) -> np.ndarray:
        width, height = self.camera.resolution_px
        image = np.full((height, width, 3), 225, dtype=np.uint8)
        colours = {
            "red": (255, 0, 0),
            "yellow": (255, 255, 0),
            "blue": (0, 0, 255),
        }
        half = self.config.cube_edge_m / 2.0
        top_z = self.config.cube_top_base_z_m
        for colour in self.config.task.colour_order:
            x, y, yaw_deg = self.config.task.cube_spawns_base_xy_yaw_deg[colour]
            yaw = math.radians(yaw_deg)
            rotation = np.array(
                [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
            )
            local = np.array(
                [[-half, -half], [half, -half], [half, half], [-half, half]]
            )
            world_xy = local @ rotation.T + np.array([x, y])
            pixels = np.array(
                [self.camera.project((px, py, top_z)) for px, py in world_xy]
            )
            cv2.fillConvexPoly(
                image, np.round(pixels).astype(np.int32), colours[colour]
            )
        return image

    def test_injected_provider_bypasses_hsv_and_keeps_generic_scene(self):
        config = self.config

        class FakeProvider:
            request = None

            def observe(self, request):
                self.request = request
                objects = tuple(
                    DetectedObject(
                        track_id=f"model-{colour}",
                        class_id="cube",
                        sort_label=colour,
                        capture_time_s=request.capture_time_s,
                        pose=Pose3D.from_yaw((x, y, -0.0001), 0.1),
                        dimensions_m=(0.01, 0.01, 0.01),
                        confidence=0.95,
                        source_camera_ids=("table",),
                    )
                    for colour, (x, y, _) in (
                        (colour, config.task.cube_spawns_base_xy_yaw_deg[colour])
                        for colour in config.task.colour_order
                    )
                )
                return SceneObservation(
                    scene_id=request.scene_id,
                    capture_time_s=request.capture_time_s,
                    objects=objects,
                    provider_name="fake-team-model",
                    provider_version="test",
                    calibration_ids=(request.frame("table").calibration_id,),
                )

        provider = FakeProvider()
        adapter = IsaacCameraSortAdapter(
            self.camera, self.config, perception_provider=provider
        )
        width, height = self.camera.resolution_px
        result = adapter.observe_and_plan(
            np.zeros((height, width, 3), dtype=np.uint8),
            scene_id="external-model-scene",
            capture_time_s=8.0,
        )
        self.assertEqual(result.observation.provider_name, "fake-team-model")
        self.assertEqual(len(result.plans), 3)
        self.assertEqual(provider.request.scene_id, "external-model-scene")

    def test_observe_request_exposes_multi_camera_provider_seam(self):
        config = self.config

        class RecordingExtractor(LegacyFeatureExtractor):
            def __init__(self):
                self.robot_states = []

            def extract(self, scene, candidate, context):
                self.robot_states.append(context.robot_state)
                return super().extract(scene, candidate, context)

        class MultiCameraProvider:
            request = None

            def observe(self, request):
                self.request = request
                return SceneObservation(
                    scene_id=request.scene_id,
                    capture_time_s=request.capture_time_s,
                    objects=tuple(
                        DetectedObject(
                            track_id=f"fused-{colour}",
                            class_id="cube",
                            sort_label=colour,
                            capture_time_s=request.capture_time_s,
                            pose=Pose3D.from_yaw((x, y, -0.0001), 0.0),
                            dimensions_m=(0.01, 0.01, 0.01),
                            source_camera_ids=("table", "side"),
                        )
                        for colour, (x, y, _) in (
                            (
                                colour,
                                config.task.cube_spawns_base_xy_yaw_deg[colour],
                            )
                            for colour in config.task.colour_order
                        )
                    ),
                    provider_name="multi-camera-test",
                    provider_version="1",
                    calibration_ids=("table-cal", "side-cal"),
                )

        provider = MultiCameraProvider()
        extractor = RecordingExtractor()
        grasp_pipeline = GraspPlanningPipeline(
            LegacyTopDownCubeGenerator(),
            LegacyGeometricChecker(
                maximum_radial_reach_m=(
                    self.config.hardware.maximum_radial_reach_m
                ),
                minimum_centre_clearance_m=(
                    self.config.task.minimum_distinct_centres_m
                ),
            ),
            extractor,
            ConstantSuccessScorer(),
            HighestSuccessSelector(),
        )
        adapter = IsaacCameraSortAdapter(
            self.camera,
            self.config,
            perception_provider=provider,
            grasp_pipeline=grasp_pipeline,
        )
        request = PerceptionRequest(
            scene_id="multi-camera-scene",
            capture_time_s=9.0,
            frames=(
                CameraFrame("table", 8.99, "table-cal", rgb=object()),
                CameraFrame("side", 8.96, "side-cal", rgb=object()),
            ),
            robot_state=RobotState(
                arm_joints_rad=(0.0,) * 6,
                tcp_pose_base_m_wpr_rad=(0.4, 0.0, 0.1, -math.pi, 0.0, 0.0),
                gripper_state=0.0,
            ),
        )
        result = adapter.observe_request_and_plan(request)
        self.assertIs(provider.request, request)
        self.assertEqual(result.observation.provider_name, "multi-camera-test")
        self.assertEqual(len(result.plans), 3)
        self.assertEqual(extractor.robot_states, [request.robot_state] * 3)

    def test_provider_runtime_error_is_an_isaac_observation_error(self):
        class ExplodingProvider:
            def observe(self, request):
                raise RuntimeError("accelerator unavailable")

        adapter = IsaacCameraSortAdapter(
            self.camera, self.config, perception_provider=ExplodingProvider()
        )
        width, height = self.camera.resolution_px
        with self.assertRaisesRegex(
            IsaacObservationError, "accelerator unavailable"
        ):
            adapter.observe_generic(
                np.zeros((height, width, 3), dtype=np.uint8),
                scene_id="provider-failure",
                capture_time_s=9.5,
            )

    def test_rendered_rgb_produces_shared_detections_and_plans(self):
        image = self._render_top_faces()
        result = self.adapter.observe_and_plan(
            image, scene_id="isaac-camera-1", capture_time_s=4.5
        )

        self.assertEqual(
            [d.colour for d in result.scene.cubes], ["red", "yellow", "blue"]
        )
        self.assertEqual(
            [plan.target.colour for plan in result.plans],
            ["red", "yellow", "blue"],
        )
        for detection in result.scene.cubes:
            expected = self.config.task.cube_spawns_base_xy_yaw_deg[detection.colour]
            np.testing.assert_allclose(
                detection.position_base_m[:2], expected[:2], atol=0.002
            )
            self.assertEqual(detection.capture_time_s, 4.5)
            self.assertIsNotNone(detection.pixel_center_uv)

    def test_resolution_mismatch_fails_instead_of_rescaling_silently(self):
        image = self._render_top_faces()[::2, ::2]
        with self.assertRaisesRegex(IsaacObservationError, "resolution"):
            self.adapter.observe(
                image, scene_id="wrong-resolution", capture_time_s=5.0
            )

    def test_outside_pickup_colour_component_is_not_a_pick_target(self):
        image = self._render_top_faces()
        centre = self.camera.project((0.10, 0.0, self.config.cube_top_base_z_m))
        self.assertIsNotNone(centre)
        centre_px = np.round(centre).astype(np.int32)
        cv2.rectangle(
            image,
            tuple(centre_px - np.array([5, 5])),
            tuple(centre_px + np.array([5, 5])),
            (255, 0, 0),
            -1,
        )

        raw = self.adapter.observe_raw(
            image, scene_id="with-robot-colour-raw", capture_time_s=5.5
        )
        self.assertEqual(
            [d.colour for d in raw.cubes], ["red", "red", "yellow", "blue"]
        )
        scene = self.adapter.observe(
            image, scene_id="with-robot-colour-pickable", capture_time_s=5.5
        )
        self.assertEqual(
            [d.colour for d in scene.cubes], ["red", "yellow", "blue"]
        )

    def test_annotated_or_truth_pose_inputs_are_not_part_of_adapter_api(self):
        image = self._render_top_faces()
        with self.assertRaises(TypeError):
            self.adapter.observe_and_plan(
                image,
                scene_id="no-truth-channel",
                capture_time_s=6.0,
                simulator_truth={"red": (0.44, -0.19, 0.0)},
            )

    def test_evaluator_truth_cannot_stop_camera_or_training_actions(self):
        self.assertFalse(
            _should_stop_after_truth_evaluation(
                camera_perception=True, placement_verified=False
            )
        )
        self.assertTrue(
            _should_stop_after_truth_evaluation(
                camera_perception=False, placement_verified=False
            )
        )
        self.assertFalse(
            _should_stop_after_truth_evaluation(
                camera_perception=False, placement_verified=True
            )
        )

    def test_refreshed_descent_yaw_is_interpolated(self):
        self.assertTrue(_phase_interpolates_spread("approach_pick"))
        self.assertTrue(_phase_interpolates_spread("descend_pick"))
        self.assertTrue(_phase_interpolates_spread("carry"))
        self.assertFalse(_phase_interpolates_spread("close"))


if __name__ == "__main__":
    unittest.main()
