from __future__ import annotations

from dataclasses import replace
import math
import unittest

import cv2
import numpy as np

from sorting_core import load_config
from sorting_core.integration_plugins import VisionFactoryContext
from sorting_core.perception import CameraFrame, PerceptionRequest
from sorting_core.vision import VisionError
from team_plugins.computer_vision.hybrid_geometry_provider import (
    CubeScrewHybridVisionProvider,
    SCREW_CLASS_ID,
    ScrewDetectorConfig,
    create,
)


IMAGE_SIZE = (400, 400)
METERS_PER_PIXEL = 0.00025


def projector(uv, z_m):
    u, v = uv
    return np.array(
        (
            0.35 + float(u) * METERS_PER_PIXEL,
            (float(v) - 200.0) * METERS_PER_PIXEL,
            float(z_m),
        ),
        dtype=float,
    )


def boundary_projector(uv, z_m):
    u, v = uv
    return np.array(
        (
            0.278 + float(u) * METERS_PER_PIXEL,
            (float(v) - 200.0) * METERS_PER_PIXEL,
            float(z_m),
        ),
        dtype=float,
    )


def overlapping_boundary_projector(uv, z_m):
    projected = boundary_projector(uv, z_m)
    projected[0] -= 0.02
    return projected


def destination_projector(uv, z_m):
    u, v = uv
    return np.array(
        (
            0.20 + float(u) * METERS_PER_PIXEL,
            (float(v) - 200.0) * METERS_PER_PIXEL,
            float(z_m),
        ),
        dtype=float,
    )


def request(
    rgb: np.ndarray,
    *,
    scene_id: str,
    capture_time_s: float,
    frame_capture_time_s: float | None = None,
    calibration_id: str = "test-table-v1",
):
    if frame_capture_time_s is None:
        frame_capture_time_s = capture_time_s
    return PerceptionRequest(
        scene_id=scene_id,
        capture_time_s=capture_time_s,
        frames=(
            CameraFrame(
                camera_id="table",
                capture_time_s=frame_capture_time_s,
                calibration_id=calibration_id,
                rgb=rgb,
            ),
        ),
    )


def draw_screw(*, head_left: bool = True) -> np.ndarray:
    image = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 235, dtype=np.uint8)
    if head_left:
        cv2.circle(image, (110, 200), 24, (15, 15, 15), thickness=-1)
        cv2.rectangle(image, (110, 188), (194, 212), (15, 15, 15), thickness=-1)
        tip = np.array(((194, 188), (206, 200), (194, 212)), dtype=np.int32)
    else:
        cv2.circle(image, (290, 200), 24, (15, 15, 15), thickness=-1)
        cv2.rectangle(image, (206, 188), (290, 212), (15, 15, 15), thickness=-1)
        tip = np.array(((206, 188), (194, 200), (206, 212)), dtype=np.int32)
    cv2.fillConvexPoly(image, tip, (15, 15, 15))
    return image


def rotate_image(image: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    transform = cv2.getRotationMatrix2D((200.0, 200.0), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        transform,
        IMAGE_SIZE,
        flags=cv2.INTER_NEAREST,
        borderValue=(235, 235, 235),
    )
    return rotated, transform


def quaternion_rotation_matrix(quaternion):
    x, y, z, w = quaternion
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=float,
    )


class HybridGeometryProviderTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.screw_config = ScrewDetectorConfig(
            expected_image_size=IMAGE_SIZE,
            minimum_component_area_px=20.0,
            maximum_component_area_px=10000.0,
            minimum_track_observations=1,
        )

    def provider(self):
        return CubeScrewHybridVisionProvider(
            config=self.config,
            camera_projectors={"table": projector},
            calibration_ids={"table": "test-table-v1"},
            camera_id="table",
            expected_cube_edge_m=0.010,
            screw_config=self.screw_config,
        )

    def test_detects_head_origin_and_directed_shaft_axis(self):
        scene = self.provider().observe(
            request(draw_screw(), scene_id="scene-a", capture_time_s=1.25)
        )
        screws = [item for item in scene.objects if item.class_id == SCREW_CLASS_ID]
        self.assertEqual(len(screws), 1)
        screw = screws[0]
        head_x, head_y = screw.attributes["head_xy_base_m"]
        tip_x, tip_y = screw.attributes["tip_xy_base_m"]
        self.assertLess(head_x, tip_x)
        self.assertAlmostEqual(head_y, tip_y, delta=0.003)
        self.assertAlmostEqual(screw.pose.position_base_m[0], head_x)
        self.assertEqual(screw.sort_label, "red")
        self.assertEqual(screw.dimensions_m, (0.012, 0.01199627, 0.030))
        rotation = quaternion_rotation_matrix(screw.pose.orientation_xyzw)
        mapped_minus_z = rotation @ np.array((0.0, 0.0, -1.0))
        mapped_plus_x = rotation @ np.array((1.0, 0.0, 0.0))
        np.testing.assert_allclose(mapped_minus_z, (1.0, 0.0, 0.0), atol=0.08)
        np.testing.assert_allclose(mapped_plus_x, (0.0, 0.0, 1.0), atol=1e-6)
        self.assertFalse(screw.attributes["absolute_base_accuracy_verified"])

    def test_polarity_reverses_when_head_is_on_right(self):
        scene = self.provider().observe(
            request(draw_screw(head_left=False), scene_id="scene-b", capture_time_s=2.0)
        )
        screw = next(item for item in scene.objects if item.class_id == SCREW_CLASS_ID)
        head_x, _ = screw.attributes["head_xy_base_m"]
        tip_x, _ = screw.attributes["tip_xy_base_m"]
        self.assertGreater(head_x, tip_x)
        rotation = quaternion_rotation_matrix(screw.pose.orientation_xyzw)
        mapped_minus_z = rotation @ np.array((0.0, 0.0, -1.0))
        np.testing.assert_allclose(mapped_minus_z, (-1.0, 0.0, 0.0), atol=0.08)

    def test_track_id_is_stable_for_repeated_frame(self):
        provider = self.provider()
        first = provider.observe(
            request(draw_screw(), scene_id="scene-c1", capture_time_s=3.0)
        )
        second = provider.observe(
            request(draw_screw(), scene_id="scene-c2", capture_time_s=3.1)
        )
        first_screw = next(item for item in first.objects if item.class_id == SCREW_CLASS_ID)
        second_screw = next(item for item in second.objects if item.class_id == SCREW_CLASS_ID)
        self.assertEqual(first_screw.track_id, second_screw.track_id)
        self.assertEqual(second_screw.capture_time_s, second.capture_time_s)

    def test_screw_requires_configured_consecutive_confirmation(self):
        provider = CubeScrewHybridVisionProvider(
            config=self.config,
            camera_projectors={"table": projector},
            calibration_ids={"table": "test-table-v1"},
            camera_id="table",
            expected_cube_edge_m=0.010,
            screw_config=replace(
                self.screw_config,
                minimum_track_observations=3,
            ),
        )
        pending_ids = []
        for index, timestamp in enumerate((3.0, 3.1)):
            scene = provider.observe(
                request(
                    draw_screw(),
                    scene_id=f"scene-confirm-{index}",
                    capture_time_s=timestamp,
                )
            )
            self.assertFalse(
                any(item.class_id == SCREW_CLASS_ID for item in scene.objects)
            )
            pending = next(
                item for item in scene.objects if item.class_id == "unknown_obstacle"
            )
            self.assertEqual(
                pending.attributes["ambiguity_reason"],
                "temporal_confirmation_pending",
            )
            pending_ids.append(pending.track_id)
        confirmed = provider.observe(
            request(
                draw_screw(),
                scene_id="scene-confirm-2",
                capture_time_s=3.2,
            )
        )
        screw = next(
            item for item in confirmed.objects if item.class_id == SCREW_CLASS_ID
        )
        self.assertEqual(pending_ids, [screw.track_id, screw.track_id])
        self.assertEqual(screw.attributes["consecutive_track_observations"], 3)
        self.assertEqual(
            screw.attributes["confirmation_timestamp_source"],
            "camera_frame_capture_time_s",
        )

    def test_repeated_frozen_frame_does_not_confirm_screw(self):
        provider = CubeScrewHybridVisionProvider(
            config=self.config,
            camera_projectors={"table": projector},
            calibration_ids={"table": "test-table-v1"},
            camera_id="table",
            expected_cube_edge_m=0.010,
            screw_config=replace(
                self.screw_config,
                minimum_track_observations=3,
            ),
        )
        track_ids = []
        for index in range(4):
            scene = provider.observe(
                request(
                    draw_screw(),
                    scene_id=f"scene-frozen-{index}",
                    capture_time_s=3.0,
                )
            )
            self.assertFalse(
                any(item.class_id == SCREW_CLASS_ID for item in scene.objects)
            )
            pending = next(
                item for item in scene.objects if item.class_id == "unknown_obstacle"
            )
            self.assertEqual(
                pending.attributes["consecutive_track_observations"], 1
            )
            track_ids.append(pending.track_id)
        self.assertEqual(len(set(track_ids)), 1)

    def test_shape_change_resets_screw_confirmation(self):
        provider = CubeScrewHybridVisionProvider(
            config=self.config,
            camera_projectors={"table": projector},
            calibration_ids={"table": "test-table-v1"},
            camera_id="table",
            expected_cube_edge_m=0.010,
            screw_config=replace(
                self.screw_config,
                minimum_track_observations=3,
                maximum_track_axis_change_deg=10.0,
            ),
        )
        first = provider.observe(
            request(draw_screw(), scene_id="shape-a", capture_time_s=4.0)
        )
        first_pending = next(
            item for item in first.objects if item.class_id == "unknown_obstacle"
        )
        rotated, _ = rotate_image(draw_screw(), 25.0)
        second = provider.observe(
            request(rotated, scene_id="shape-b", capture_time_s=4.1)
        )
        second_pending = next(
            item for item in second.objects if item.class_id == "unknown_obstacle"
        )
        self.assertNotEqual(first_pending.track_id, second_pending.track_id)
        self.assertEqual(
            second_pending.attributes["consecutive_track_observations"], 1
        )

    def test_workspace_margin_must_be_nonnegative(self):
        with self.assertRaisesRegex(ValueError, "workspace_margin_m"):
            ScrewDetectorConfig(workspace_margin_m=-0.001).validate()

    def test_tracker_rejects_non_monotonic_scene_time(self):
        provider = self.provider()
        provider.observe(
            request(draw_screw(), scene_id="scene-time-first", capture_time_s=3.0)
        )
        with self.assertRaisesRegex(VisionError, "non-monotonic"):
            provider.observe(
                request(
                    draw_screw(),
                    scene_id="scene-time-backwards",
                    capture_time_s=2.9,
                )
            )

    def test_missing_track_is_not_fabricated_as_occlusion(self):
        provider = self.provider()
        provider.observe(
            request(draw_screw(), scene_id="scene-visible", capture_time_s=3.0)
        )
        blank = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 235, dtype=np.uint8)
        missing = provider.observe(
            request(blank, scene_id="scene-hidden", capture_time_s=3.1)
        )
        # Disappearance may mean a successful pickup, not occlusion. Without
        # image evidence or action feedback the provider must not fabricate a
        # blocking obstacle. The limitation is documented for dynamic scenes.
        self.assertEqual(missing.objects, ())

    def test_detects_rotated_screw_axis_across_table(self):
        source = draw_screw()
        for index, angle_deg in enumerate((25.0, 70.0, 125.0)):
            with self.subTest(angle_deg=angle_deg):
                rotated, transform = rotate_image(source, angle_deg)
                scene = self.provider().observe(
                    request(
                        rotated,
                        scene_id=f"scene-rotated-{index}",
                        capture_time_s=10.0 + index,
                    )
                )
                screw = next(
                    item for item in scene.objects if item.class_id == SCREW_CLASS_ID
                )
                expected_head = transform @ np.array((86.0, 200.0, 1.0))
                expected_tip = transform @ np.array((206.0, 200.0, 1.0))
                expected = expected_tip - expected_head
                expected /= np.linalg.norm(expected)
                observed_head = np.asarray(screw.attributes["head_xy_base_m"])
                observed_tip = np.asarray(screw.attributes["tip_xy_base_m"])
                observed = observed_tip - observed_head
                observed /= np.linalg.norm(observed)
                # The synthetic projector preserves image X/Y direction.
                self.assertGreater(float(observed @ expected), math.cos(math.radians(8.0)))

    def test_two_separated_screws_remain_two_instances(self):
        first = draw_screw()
        second = np.full_like(first, 235)
        cv2.circle(second, (290, 300), 24, (15, 15, 15), thickness=-1)
        cv2.rectangle(second, (206, 288), (290, 312), (15, 15, 15), thickness=-1)
        cv2.fillConvexPoly(
            second,
            np.array(((206, 288), (194, 300), (206, 312)), dtype=np.int32),
            (15, 15, 15),
        )
        combined = np.minimum(first, second)
        scene = self.provider().observe(
            request(combined, scene_id="scene-two", capture_time_s=14.0)
        )
        screws = [item for item in scene.objects if item.class_id == SCREW_CLASS_ID]
        self.assertEqual(len(screws), 2)
        self.assertEqual(len({item.track_id for item in screws}), 2)

    def test_large_midshaft_occlusion_does_not_publish_precise_screw(self):
        image = draw_screw()
        cv2.rectangle(image, (135, 165), (185, 235), (235, 235, 235), thickness=-1)
        scene = self.provider().observe(
            request(image, scene_id="scene-occluded", capture_time_s=15.0)
        )
        self.assertFalse(any(item.class_id == SCREW_CLASS_ID for item in scene.objects))

    def test_crossing_screws_are_rejected_as_merged_ambiguity(self):
        first = draw_screw()
        second, _ = rotate_image(draw_screw(head_left=False), 90.0)
        merged = np.minimum(first, second)
        scene = self.provider().observe(
            request(merged, scene_id="scene-touching", capture_time_s=16.0)
        )
        self.assertFalse(any(item.class_id == SCREW_CLASS_ID for item in scene.objects))

    def test_empty_and_dark_square_do_not_become_screws(self):
        for label, image in (
            ("empty", np.full((400, 400, 3), 235, dtype=np.uint8)),
            (
                "square",
                cv2.rectangle(
                    np.full((400, 400, 3), 235, dtype=np.uint8),
                    (150, 150),
                    (250, 250),
                    (15, 15, 15),
                    thickness=-1,
                ),
            ),
        ):
            with self.subTest(label=label):
                scene = self.provider().observe(
                    request(image, scene_id=f"scene-{label}", capture_time_s=17.0)
                )
                self.assertFalse(
                    any(item.class_id == SCREW_CLASS_ID for item in scene.objects)
                )

    def test_uniform_dark_bar_is_ambiguous_obstacle_not_screw(self):
        image = np.full((400, 400, 3), 235, dtype=np.uint8)
        cv2.rectangle(image, (100, 180), (220, 220), (15, 15, 15), thickness=-1)
        scene = self.provider().observe(
            request(image, scene_id="scene-d", capture_time_s=4.0)
        )
        self.assertFalse(any(item.class_id == SCREW_CLASS_ID for item in scene.objects))
        obstacles = [item for item in scene.objects if item.class_id == "unknown_obstacle"]
        self.assertEqual(len(obstacles), 1)
        self.assertTrue(obstacles[0].attributes["non_executable"])
        self.assertEqual(obstacles[0].pose_quality, 0.0)

    def test_screw_shaped_dark_detail_on_coloured_container_is_not_target(self):
        image = np.full((400, 400, 3), 235, dtype=np.uint8)
        cv2.rectangle(image, (45, 135), (260, 265), (20, 90, 210), thickness=-1)
        cv2.circle(image, (110, 200), 24, (15, 15, 15), thickness=-1)
        cv2.rectangle(image, (110, 188), (194, 212), (15, 15, 15), thickness=-1)
        cv2.fillConvexPoly(
            image,
            np.array(((194, 188), (206, 200), (194, 212)), dtype=np.int32),
            (15, 15, 15),
        )
        scene = self.provider().observe(
            request(image, scene_id="scene-container", capture_time_s=18.0)
        )
        self.assertFalse(any(item.class_id == SCREW_CLASS_ID for item in scene.objects))
        obstacle = next(
            item for item in scene.objects if item.class_id == "unknown_obstacle"
        )
        self.assertIn("not_isolated_on_white_table", obstacle.attributes["ambiguity_reason"])

    def test_complete_screw_footprint_must_be_inside_pickup_region(self):
        provider = CubeScrewHybridVisionProvider(
            config=self.config,
            camera_projectors={"table": boundary_projector},
            calibration_ids={"table": "test-table-v1"},
            camera_id="table",
            expected_cube_edge_m=0.010,
            screw_config=self.screw_config,
        )
        scene = provider.observe(
            request(draw_screw(), scene_id="scene-boundary", capture_time_s=19.0)
        )
        self.assertFalse(any(item.class_id == SCREW_CLASS_ID for item in scene.objects))
        obstacle = next(
            item for item in scene.objects if item.class_id == "unknown_obstacle"
        )
        self.assertIn(
            "footprint_outside_pickup_region",
            obstacle.attributes["ambiguity_reason"],
        )

    def test_boundary_overlap_is_obstacle_even_when_center_is_outside(self):
        provider = CubeScrewHybridVisionProvider(
            config=self.config,
            camera_projectors={"table": overlapping_boundary_projector},
            calibration_ids={"table": "test-table-v1"},
            camera_id="table",
            expected_cube_edge_m=0.010,
            screw_config=self.screw_config,
        )
        scene = provider.observe(
            request(draw_screw(), scene_id="scene-overlap", capture_time_s=19.5)
        )
        self.assertFalse(any(item.class_id == SCREW_CLASS_ID for item in scene.objects))
        obstacle = next(
            item for item in scene.objects if item.class_id == "unknown_obstacle"
        )
        self.assertIn(
            "footprint_outside_pickup_region",
            obstacle.attributes["ambiguity_reason"],
        )

    def test_coloured_cube_path_is_preserved(self):
        image = np.full((400, 400, 3), 235, dtype=np.uint8)
        # Forty pixels at this projector scale is the configured 10 mm edge.
        cv2.rectangle(image, (180, 180), (220, 220), (230, 20, 20), thickness=-1)
        scene = self.provider().observe(
            request(image, scene_id="scene-e", capture_time_s=5.0)
        )
        cubes = [item for item in scene.objects if item.class_id == "cube"]
        self.assertEqual(len(cubes), 1)
        self.assertEqual(cubes[0].sort_label, "red")
        self.assertFalse(any(item.class_id == SCREW_CLASS_ID for item in scene.objects))

    def test_outside_pickup_cube_is_preserved_for_destination_verification(self):
        image = np.full((400, 400, 3), 235, dtype=np.uint8)
        cv2.rectangle(image, (180, 180), (220, 220), (230, 20, 20), thickness=-1)
        provider = CubeScrewHybridVisionProvider(
            config=self.config,
            camera_projectors={"table": destination_projector},
            calibration_ids={"table": "test-table-v1"},
            camera_id="table",
            expected_cube_edge_m=0.010,
            screw_config=self.screw_config,
        )
        scene = provider.observe(
            request(image, scene_id="scene-destination", capture_time_s=5.5)
        )
        cube = next(item for item in scene.objects if item.class_id == "cube")
        self.assertFalse(cube.attributes["in_pickup_region"])
        self.assertLess(cube.pose.position_base_m[0], 0.30)

    def test_rejects_wrong_resolution_in_hardware_profile(self):
        image = np.full((200, 200, 3), 235, dtype=np.uint8)
        with self.assertRaisesRegex(VisionError, "uncropped native image"):
            self.provider().observe(
                request(image, scene_id="scene-f", capture_time_s=6.0)
            )

    def test_rejects_mismatched_calibration_before_pose_conversion(self):
        with self.assertRaisesRegex(VisionError, "calibration mismatch"):
            self.provider().observe(
                request(
                    draw_screw(),
                    scene_id="scene-calibration-mismatch",
                    capture_time_s=20.0,
                    calibration_id="different-table-calibration",
                )
            )

    def test_rejects_stale_or_future_source_frame(self):
        with self.assertRaisesRegex(VisionError, "stale by"):
            self.provider().observe(
                request(
                    draw_screw(),
                    scene_id="scene-stale",
                    capture_time_s=20.0,
                    frame_capture_time_s=19.8,
                )
            )
        with self.assertRaisesRegex(VisionError, "newer than"):
            self.provider().observe(
                request(
                    draw_screw(),
                    scene_id="scene-future",
                    capture_time_s=20.0,
                    frame_capture_time_s=20.01,
                )
            )

    def test_retains_source_time_when_frame_is_recent(self):
        scene = self.provider().observe(
            request(
                draw_screw(),
                scene_id="scene-recent",
                capture_time_s=20.0,
                frame_capture_time_s=19.9,
            )
        )
        screw = next(item for item in scene.objects if item.class_id == SCREW_CLASS_ID)
        self.assertEqual(screw.capture_time_s, 20.0)
        self.assertEqual(screw.attributes["source_frame_capture_time_s"], 19.9)
        self.assertAlmostEqual(screw.attributes["source_frame_age_s"], 0.1)
        self.assertEqual(
            screw.attributes["temporal_alignment"],
            "zero_order_hold_no_motion_compensation",
        )

    def test_options_reject_unknown_or_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "unknown screw detector options"):
            ScrewDetectorConfig.from_options({"invented": 1})
        with self.assertRaisesRegex(ValueError, "positive odd"):
            ScrewDetectorConfig.from_options({"morphology_close_px": 2})
        with self.assertRaisesRegex(ValueError, "boolean"):
            ScrewDetectorConfig.from_options({"require_expected_image_size": 1})

    def test_factory_context_options_and_profile_provenance(self):
        context = VisionFactoryContext(
            config=self.config,
            camera_projectors={"table": projector},
            calibration_ids={"table": "test-table-v1"},
            expected_cube_edge_m=0.010,
            checkpoint=None,
            detector=None,
            options={
                "camera_id": "table",
                "maximum_frame_age_s": 0.15,
                "screw": {
                    "expected_image_size": list(IMAGE_SIZE),
                    "minimum_component_area_px": 20.0,
                    "maximum_component_area_px": 10000.0,
                    "minimum_track_observations": 1,
                },
            },
            camera_ids=("table",),
            depth_camera_ids=(),
            declared_calibration_ids={"table": "test-table-v1"},
            declared_depth_calibration_ids={},
            profile_id="test-hybrid-profile",
            profile_path="team_plugins/computer_vision/test-profile.json",
            profile_sha256="a" * 64,
        )
        scene = create(context).observe(
            request(draw_screw(), scene_id="scene-factory", capture_time_s=30.0)
        )
        screw = next(item for item in scene.objects if item.class_id == SCREW_CLASS_ID)
        self.assertEqual(
            screw.attributes["integration_profile"]["profile_id"],
            "test-hybrid-profile",
        )
        self.assertEqual(
            screw.attributes["integration_profile"]["profile_sha256"],
            "a" * 64,
        )


if __name__ == "__main__":
    unittest.main()
