from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from demos.scripted_cube_screw_bucket_sort import (
    SUPPLIED_SCREW_BOUNDS_MM,
    SUPPLIED_SCREW_SHA256,
    SUPPLIED_SCREW_TRIANGLES,
    ScenarioError,
    build_completion_report,
    build_sort_items,
    load_scenario_config,
    local_to_world_vertices,
    resolve_screw_model,
    screw_jaw_half_target,
    tool_tip_z_for_desired_grasp_height,
    validate_screw_model,
    verify_vertices_inside_bin,
)
from demos.scripted_cube_screw_bucket_sort import (
    cube_screw_bucket_demo as demo_module,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ASSETS = REPO_ROOT / "reference-assets"
SCREW_STL = (
    REFERENCE_ASSETS
    / "models"
    / "ISO10642-M6x30-countersunk-screw.stl"
)


class CubeScrewScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_scenario_config()
        cls.model = validate_screw_model(SCREW_STL, cls.config.screw)

    def test_runner_shutdown_mode_defers_app_until_artifacts_are_written(self):
        with patch.object(sys, "argv", ["demo", "--run", "fresh-run"]):
            standalone = demo_module.parse_args()
        with patch.object(
            sys,
            "argv",
            [
                "demo",
                "--run",
                "fresh-run",
                "--defer-shutdown",
            ],
        ):
            embedded = demo_module.parse_args()

        self.assertFalse(standalone.defer_shutdown)
        self.assertTrue(embedded.defer_shutdown)
        sentinel = object()
        self.assertIsNone(demo_module._take_deferred_simulation_app())
        demo_module._DEFERRED_SIMULATION_APP = sentinel
        self.assertIs(
            demo_module._take_deferred_simulation_app(),
            sentinel,
        )
        self.assertIsNone(demo_module._take_deferred_simulation_app())

    def test_embedded_failed_verification_remains_a_collectible_negative(self):
        demo_module._enforce_standalone_success(
            False,
            defer_shutdown=True,
        )
        demo_module._enforce_standalone_success(
            True,
            defer_shutdown=False,
        )
        with self.assertRaisesRegex(RuntimeError, "verification failed"):
            demo_module._enforce_standalone_success(
                False,
                defer_shutdown=False,
            )

    def test_scene_inherits_three_cubes_and_adds_two_unique_red_screws(self):
        items = build_sort_items(self.config)
        self.assertEqual(len(items), 5)
        self.assertEqual([item.item_id for item in items], list(self.config.sort_order))
        self.assertEqual(len({item.item_id for item in items}), 5)
        self.assertEqual(
            [(item.kind, item.destination_colour) for item in items],
            [
                ("cube", "red"),
                ("cube", "yellow"),
                ("cube", "blue"),
                ("screw", "red"),
                ("screw", "red"),
            ],
        )
        for item in items[-2:]:
            self.assertAlmostEqual(
                math.dist(item.spawn_reference_base_m, item.spawn_grasp_point_base_m),
                self.config.screw.grasp_offset_from_head_m,
                places=12,
            )
        red_bin_x, red_bin_y = self.config.cube_config.task.bin_centres_base_m[
            "red"
        ]
        first, second = items[-2:]
        self.assertAlmostEqual(
            first.placement_grasp_point_base_m[1] - red_bin_y, -0.004
        )
        self.assertAlmostEqual(
            second.placement_grasp_point_base_m[1] - red_bin_y, 0.004
        )
        self.assertAlmostEqual(first.target_axis_yaw_deg, 0.0)
        self.assertAlmostEqual(second.target_axis_yaw_deg, 180.0)
        self.assertAlmostEqual(
            first.placement_grasp_point_base_m[0] - red_bin_x, 0.001
        )
        self.assertAlmostEqual(
            second.placement_grasp_point_base_m[0] - red_bin_x, -0.001
        )

    def test_model_resolution_requires_explicit_path_or_reference_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                ScenarioError, "ROCKWELL_REFERENCE_ASSETS is unset"
            ):
                resolve_screw_model()
            self.assertEqual(resolve_screw_model(SCREW_STL), SCREW_STL.resolve())

        with patch.dict(
            os.environ,
            {"ROCKWELL_REFERENCE_ASSETS": str(REFERENCE_ASSETS)},
            clear=True,
        ):
            self.assertEqual(resolve_screw_model(), SCREW_STL.resolve())

    def test_exact_supplied_stl_is_validated_before_use(self):
        self.assertEqual(self.model.sha256, SUPPLIED_SCREW_SHA256)
        self.assertEqual(self.model.triangle_count, SUPPLIED_SCREW_TRIANGLES)
        for measured, expected in zip(
            self.model.bounds_mm[0] + self.model.bounds_mm[1],
            SUPPLIED_SCREW_BOUNDS_MM[0] + SUPPLIED_SCREW_BOUNDS_MM[1],
        ):
            self.assertAlmostEqual(measured, expected, places=5)
        self.assertEqual(len(self.model.vertices_m), 3 * SUPPLIED_SCREW_TRIANGLES)

    def test_modified_stl_cannot_be_silently_substituted(self):
        changed = bytearray(SCREW_STL.read_bytes())
        changed[0] ^= 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / SCREW_STL.name
            path.write_bytes(changed)
            with self.assertRaisesRegex(ScenarioError, "SHA-256 mismatch"):
                validate_screw_model(path, self.config.screw)

    def test_local_transform_matches_isaac_screw_axis_convention(self):
        head = (1.0, 2.0, 3.0)
        points = ((0.0, 0.0, -0.03), (0.006, 0.0, 0.0))
        yaw_zero = local_to_world_vertices(points, head_world_m=head, yaw_rad=0.0)
        self.assertEqual(yaw_zero[0], (1.03, 2.0, 3.0))
        self.assertEqual(yaw_zero[1], (1.0, 2.0, 3.006))

        yaw_ninety = local_to_world_vertices(
            (points[0],), head_world_m=head, yaw_rad=math.pi / 2.0
        )[0]
        self.assertAlmostEqual(yaw_ninety[0], 1.0, places=12)
        self.assertAlmostEqual(yaw_ninety[1], 2.03, places=12)
        self.assertAlmostEqual(yaw_ninety[2], 3.0, places=12)

    def test_screw_jaw_target_uses_shank_finger_geometry_and_margin(self):
        target = screw_jaw_half_target(
            shank_diameter_m=self.config.screw.shank_diameter_m,
            finger_inner_tip_m=0.002,
            seating_margin_m=self.config.screw.jaw_seating_margin_m,
            jaw_closed_m=0.0,
            jaw_open_m=0.006,
        )
        self.assertAlmostEqual(target, 0.0007, places=12)
        shallower_finger = screw_jaw_half_target(
            shank_diameter_m=self.config.screw.shank_diameter_m,
            finger_inner_tip_m=0.001,
            seating_margin_m=self.config.screw.jaw_seating_margin_m,
        )
        self.assertAlmostEqual(shallower_finger, 0.0017, places=12)
        self.assertGreater(shallower_finger, target)

    def test_screw_placement_height_preserves_pickup_grasp_to_tip_offset(self):
        tip_z, grasp_above_tip = tool_tip_z_for_desired_grasp_height(
            desired_grasp_base_z_m=0.0314,
            observed_grasp_world_z_m=0.899263,
            pickup_tip_base_z_m=-0.0026,
            table_top_world_z_m=0.9,
        )
        self.assertAlmostEqual(grasp_above_tip, 0.001863, places=9)
        self.assertAlmostEqual(tip_z, 0.029537, places=9)
        self.assertNotAlmostEqual(tip_z, 0.0649, places=6)

        with self.assertRaisesRegex(ScenarioError, "0-20 mm"):
            tool_tip_z_for_desired_grasp_height(
                desired_grasp_base_z_m=0.0314,
                observed_grasp_world_z_m=0.95,
                pickup_tip_base_z_m=-0.0026,
                table_top_world_z_m=0.9,
            )

    def test_all_cad_vertices_fit_each_planned_red_bin_placement(self):
        cube = self.config.cube_config
        red_bin = cube.task.bin_centres_base_m["red"]
        floor_z = cube.frames.surface_base_z_m + cube.bin.floor_m
        rim_z = cube.frames.surface_base_z_m + cube.bin.height_m
        screw_items = [item for item in build_sort_items(self.config) if item.kind == "screw"]
        self.assertEqual(len(screw_items), 2)
        for item in screw_items:
            with self.subTest(item_id=item.item_id):
                vertices = local_to_world_vertices(
                    self.model.vertices_m,
                    head_world_m=item.placement_reference_base_m,
                    yaw_rad=math.radians(item.target_axis_yaw_deg),
                )
                result = verify_vertices_inside_bin(
                    vertices,
                    bin_centre_xy_m=red_bin,
                    pocket_m=cube.bin.pocket_m,
                    floor_z_m=floor_z,
                    rim_z_m=rim_z,
                    corner_radius_m=0.004,
                )
                self.assertTrue(result.passed, result.as_dict())
                self.assertEqual(result.vertex_count, 3 * SUPPLIED_SCREW_TRIANGLES)
                self.assertGreater(result.minimum_wall_clearance_m, 0.002)
                self.assertGreater(result.floor_clearance_m, 0.0004)
                self.assertGreater(result.rim_clearance_m, 0.002)

    def test_bin_verification_checks_every_vertex_and_rim(self):
        result = verify_vertices_inside_bin(
            ((0.0, 0.0, 0.01), (0.03, 0.0, 0.06)),
            bin_centre_xy_m=(0.0, 0.0),
            pocket_m=0.035,
            floor_z_m=0.0,
            rim_z_m=0.05,
        )
        self.assertFalse(result.inside_pocket_xy)
        self.assertFalse(result.below_rim)
        self.assertFalse(result.passed)

    def test_bin_verification_rejects_rounded_corner_wall(self):
        point_inside_square_but_outside_corner = ((0.017, 0.017, 0.01),)
        square = verify_vertices_inside_bin(
            point_inside_square_but_outside_corner,
            bin_centre_xy_m=(0.0, 0.0),
            pocket_m=0.035,
            floor_z_m=0.0,
            rim_z_m=0.05,
        )
        rounded = verify_vertices_inside_bin(
            point_inside_square_but_outside_corner,
            bin_centre_xy_m=(0.0, 0.0),
            pocket_m=0.035,
            floor_z_m=0.0,
            rim_z_m=0.05,
            corner_radius_m=0.004,
        )
        self.assertTrue(square.inside_pocket_xy)
        self.assertFalse(rounded.inside_pocket_xy)

    def test_completion_report_derives_mixed_counts_from_items(self):
        items = build_sort_items(self.config)
        complete_events = [
            {
                "item_id": item.item_id,
                "pickup_verified": True,
                "placement_verified": True,
            }
            for item in items
        ]
        report = build_completion_report(self.config, complete_events)
        self.assertEqual(report["expected_items"], 5)
        self.assertEqual(report["expected_by_kind"], {"cube": 3, "screw": 2})
        self.assertEqual(
            report["expected_by_destination"], {"red": 3, "yellow": 1, "blue": 1}
        )
        self.assertEqual(report["verified_placements"], 5)
        self.assertTrue(report["runtime_sequence_complete"])
        self.assertTrue(report["task_complete"])
        self.assertEqual(report["completion_summary"], "pickups=5/5 placements=5/5")

        partial = build_completion_report(self.config, complete_events[:-1])
        self.assertEqual(partial["verified_placements"], 4)
        self.assertFalse(partial["runtime_sequence_complete"])
        self.assertFalse(partial["task_complete"])


if __name__ == "__main__":
    unittest.main()
