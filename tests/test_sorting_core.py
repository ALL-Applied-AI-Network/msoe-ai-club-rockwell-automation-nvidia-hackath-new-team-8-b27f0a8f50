from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sorting_core import (  # noqa: E402
    ActionValidationError,
    CubeDetection,
    PlanningError,
    SceneDetections,
    TipPoseBase,
    base_tip_pose_to_uf0_tcp,
    build_colour_sort_plans,
    build_trajectory_chunks,
    detect_coloured_cubes,
    load_config,
    semantic_phase_plan,
    uf0_tcp_to_base_tip,
    validate_tcp_chunk,
)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_checked_in_assumptions_and_demo_geometry(self):
        c = self.config
        self.assertEqual(c.task.colour_order, ("red", "yellow", "blue"))
        self.assertAlmostEqual(c.cube_edge_m, 0.010)
        self.assertIn("10 mm cubes", c.cube_size_note)
        self.assertEqual(
            dict(c.task.cube_spawns_base_xy_yaw_deg),
            {
                "red": (0.44, -0.19, 0.0),
                "yellow": (0.44, 0.0, 22.0),
                "blue": (0.44, 0.19, -28.0),
            },
        )
        self.assertEqual(
            dict(c.task.bin_centres_base_m),
            {
                "red": (0.24, -0.19),
                "yellow": (0.24, 0.0),
                "blue": (0.24, 0.19),
            },
        )
        self.assertAlmostEqual(c.bin.outer_m, 0.050)
        self.assertAlmostEqual(c.bin.pocket_m, 0.035)
        self.assertAlmostEqual(c.bin.height_m, 0.055)
        self.assertAlmostEqual(c.bin.floor_m, 0.030)
        self.assertAlmostEqual(c.bin.pocket_depth_m, 0.025)
        self.assertAlmostEqual(c.frames.surface_base_z_m, -0.0051)
        self.assertAlmostEqual(c.hardware.maximum_radial_reach_m, 0.55)

    def test_smc_and_tcp_conventions_are_explicit(self):
        c = self.config
        self.assertEqual(c.gripper.model, "SMC MHZL2-16D2")
        self.assertAlmostEqual(c.gripper.jaw_travel_per_side_m, 0.006)
        self.assertAlmostEqual(c.gripper.straight_open_gap_m, 0.012)
        self.assertAlmostEqual(c.gripper.hook_closed_gap_m, 0.00399)
        self.assertAlmostEqual(c.gripper.hook_open_gap_m, 0.01599)
        self.assertEqual(
            (
                c.gripper.finger_length_m,
                c.gripper.finger_width_m,
                c.gripper.finger_thickness_m,
            ),
            (0.076, 0.012, 0.006),
        )
        self.assertAlmostEqual(c.gripper.j6_to_fingertip_m, 0.293)
        self.assertAlmostEqual(c.frames.uf0_z_offset_mm, -330.0)
        self.assertAlmostEqual(c.frames.tcp_beyond_fingertips_mm, 0.0)
        self.assertIn("10 mm beyond", c.frames.tcp_conflict_note)


class FrameAndProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_base_tip_to_uf0_round_trip(self):
        pose = TipPoseBase((0.44, -0.19, self.config.pick_tip_base_z_m), math.radians(22))
        row = base_tip_pose_to_uf0_tcp(pose, 0.0, self.config)
        np.testing.assert_allclose(
            row,
            [440.0, -190.0, -332.6, -180.0, 0.0, 22.0, 0.0],
            atol=1e-9,
        )
        recovered = uf0_tcp_to_base_tip(row, self.config)
        np.testing.assert_allclose(recovered.xyz_m, pose.xyz_m, atol=1e-12)
        self.assertAlmostEqual(recovered.yaw_rad, pose.yaw_rad)

    def test_configurable_tcp_beyond_fingertips_offset(self):
        alternative = replace(
            self.config,
            frames=replace(self.config.frames, tcp_beyond_fingertips_mm=10.0),
        ).validate()
        pose = TipPoseBase((0.30, 0.0, 0.10), 0.0)
        default = base_tip_pose_to_uf0_tcp(pose, 0.0, self.config)
        shifted = base_tip_pose_to_uf0_tcp(pose, 0.0, alternative)
        self.assertAlmostEqual(shifted[2], default[2] - 10.0)
        np.testing.assert_allclose(
            uf0_tcp_to_base_tip(shifted, alternative).xyz_m, pose.xyz_m
        )

    def test_official_tcp_limits_and_mock_gripper_clipping(self):
        row = [300, 0, -90, -180, 0, 0, 2]
        checked = validate_tcp_chunk([row], 0.1, self.config)
        self.assertEqual(checked.shape, (1, 7))
        self.assertEqual(checked[0, 6], 1.0)
        with self.assertRaisesRegex(ActionValidationError, "tcp x outside"):
            validate_tcp_chunk([[149, 0, -90, -180, 0, 0, 0]], 0.1, self.config)
        with self.assertRaisesRegex(ActionValidationError, "dt must be"):
            validate_tcp_chunk([[300, 0, -90, -180, 0, 0, 0]], 0.01, self.config)
        with self.assertRaisesRegex(ActionValidationError, "shape"):
            validate_tcp_chunk(
                np.zeros((self.config.hardware.max_chunk_actions + 1, 7)),
                0.1,
                self.config,
            )


def _cube(colour, x, y, *, track_id=None, capture_time=10.0):
    return CubeDetection(
        track_id=track_id or f"{colour}-0",
        colour=colour,
        capture_time_s=capture_time,
        position_base_m=(x, y, -0.0001),
        yaw_rad=0.1,
        confidence=0.9,
    )


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.good = SceneDetections(
            "scene-1",
            10.0,
            (
                _cube("blue", 0.44, 0.19),
                _cube("red", 0.44, -0.19),
                _cube("yellow", 0.44, 0.0),
            ),
        )

    def test_selection_is_deterministic_and_matches_colours(self):
        plans = build_colour_sort_plans(self.good, self.config)
        self.assertEqual([p.target.colour for p in plans], ["red", "yellow", "blue"])
        self.assertEqual(
            [p.destination.centre_base_m for p in plans],
            [(0.24, -0.19), (0.24, 0.0), (0.24, 0.19)],
        )
        self.assertTrue(all(p.target.colour == p.destination.colour for p in plans))
        self.assertTrue(all(p.scene_id == "scene-1" for p in plans))

    def test_missing_duplicate_unknown_and_close_targets_are_rejected(self):
        cases = [
            (
                SceneDetections("s", 10.0, self.good.cubes[:2]),
                "missing cube colours",
            ),
            (
                SceneDetections(
                    "s",
                    10.0,
                    self.good.cubes + (_cube("red", 0.40, -0.19, track_id="red-1"),),
                ),
                "duplicate cube colours",
            ),
            (
                SceneDetections(
                    "s",
                    10.0,
                    self.good.cubes + (_cube("green", 0.30, 0.30),),
                ),
                "unknown cube colours",
            ),
            (
                SceneDetections(
                    "s",
                    10.0,
                    (
                        _cube("red", 0.40, 0.0),
                        _cube("yellow", 0.405, 0.0),
                        _cube("blue", 0.45, 0.10),
                    ),
                ),
                "too close",
            ),
        ]
        for scene, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                PlanningError, message
            ):
                build_colour_sort_plans(scene, self.config)

    def test_semantic_plan_stops_for_reobservation_before_descent(self):
        plan = build_colour_sort_plans(self.good, self.config)[0]
        phases = semantic_phase_plan(plan, self.config)
        self.assertEqual(
            [phase.name for phase in phases],
            [
                "approach_pick",
                "revalidate_before_descent",
                "descend_pick",
                "close",
                "dwell",
                "lift",
                "carry",
                "lower",
                "release",
                "settle",
                "retreat",
            ],
        )
        gate = phases[1]
        self.assertTrue(gate.requires_reobservation_before)
        self.assertIsNone(gate.target)

    def test_target_outside_radial_reach_is_rejected(self):
        scene = SceneDetections(
            "far",
            10.0,
            (
                _cube("red", 0.60, 0.0),
                _cube("yellow", 0.44, 0.0),
                _cube("blue", 0.44, 0.19),
            ),
        )
        with self.assertRaisesRegex(PlanningError, "exceeds radial reach"):
            build_colour_sort_plans(scene, self.config)

    def test_trajectory_rows_are_interpolated_valid_and_chunked(self):
        plan = build_colour_sort_plans(self.good, self.config)[0]
        phases = semantic_phase_plan(plan, self.config)
        slow = replace(
            self.config,
            hardware=replace(self.config.hardware, max_linear_step_mm=1.0),
        ).validate()
        chunks = build_trajectory_chunks(
            [300, 0, -90, -180, 0, 0, 0], phases, slow
        )
        self.assertTrue(any(chunk.requires_reobservation_before for chunk in chunks))
        self.assertTrue(any(chunk.part_index > 0 for chunk in chunks))
        for chunk in chunks:
            if chunk.requires_reobservation_before:
                self.assertEqual(chunk.actions, ())
            else:
                self.assertLessEqual(len(chunk.actions), slow.hardware.max_chunk_actions)
                validate_tcp_chunk(chunk.actions, chunk.dt_s, slow)
        release_rows = [
            row
            for chunk in chunks
            if chunk.phase_name == "release"
            for row in chunk.actions
        ]
        self.assertEqual(release_rows[-1][6], slow.gripper.open_command)


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.image = np.full((180, 320, 3), 235, dtype=np.uint8)
        self.centres = {"red": (60, 70), "yellow": (160, 80), "blue": (260, 90)}
        colours = {
            "red": (255, 0, 0),
            "yellow": (255, 255, 0),
            "blue": (0, 0, 255),
        }
        for index, colour in enumerate(self.config.task.colour_order):
            box = cv2.boxPoints((self.centres[colour], (30, 30), 8 + index * 11))
            cv2.fillConvexPoly(self.image, np.round(box).astype(np.int32), colours[colour])

    def test_native_rgb_detection_uses_top_face_projection_without_truth(self):
        z_values = []

        def project(uv, z):
            z_values.append(z)
            return np.array([0.35 + (uv[0] - 160) / 3000.0, (uv[1] - 90) / 3000.0, z])

        scene = detect_coloured_cubes(
            self.image,
            project,
            self.config,
            scene_id="rgb-1",
            capture_time_s=25.0,
        )
        self.assertEqual([d.colour for d in scene.cubes], ["red", "yellow", "blue"])
        self.assertTrue(all(abs(z - self.config.cube_top_base_z_m) < 1e-12 for z in z_values))
        self.assertTrue(
            all(
                abs(d.position_base_m[2] - self.config.cube_centre_base_z_m) < 1e-12
                for d in scene.cubes
            )
        )
        self.assertTrue(all(d.pixel_center_uv is not None for d in scene.cubes))
        build_colour_sort_plans(scene, self.config)

    def test_camera_model_to_plane_is_accepted(self):
        class Camera:
            def __init__(self):
                self.calls = 0

            def to_plane(self, u, v, z):
                self.calls += 1
                return np.array([0.35 + (u - 160) / 3000.0, (v - 90) / 3000.0, z])

        camera = Camera()
        scene = detect_coloured_cubes(
            self.image,
            camera,
            self.config,
            scene_id="rgb-2",
            capture_time_s=26.0,
        )
        self.assertEqual(len(scene.cubes), 3)
        self.assertGreater(camera.calls, 0)

    def test_larger_same_colour_bin_shape_is_not_a_cube(self):
        image = np.full((300, 500, 3), 235, dtype=np.uint8)
        # One valid 10 mm-scale square for each cube colour.
        for centre, colour in [
            ((80, 80), (255, 0, 0)),
            ((200, 80), (255, 255, 0)),
            ((320, 80), (0, 0, 255)),
        ]:
            cv2.rectangle(
                image,
                (centre[0] - 15, centre[1] - 15),
                (centre[0] + 15, centre[1] + 15),
                colour,
                -1,
            )
        # Same red HSV and a square aspect ratio, but 50 mm-scale like a bin.
        cv2.rectangle(image, (390, 170), (440, 220), (255, 0, 0), -1)

        def project(uv, z):
            return np.array([0.35 + (uv[0] - 250) / 3000.0, (uv[1] - 150) / 3000.0, z])

        scene = detect_coloured_cubes(
            image,
            project,
            self.config,
            scene_id="with-bins",
            capture_time_s=27.0,
        )
        self.assertEqual([d.colour for d in scene.cubes], ["red", "yellow", "blue"])


if __name__ == "__main__":
    unittest.main()
