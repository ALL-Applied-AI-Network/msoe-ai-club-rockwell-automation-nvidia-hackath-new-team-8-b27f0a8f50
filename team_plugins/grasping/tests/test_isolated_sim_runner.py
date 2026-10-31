import argparse
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from demos.scripted_cube_screw_bucket_sort import scenario as scenario_module

from team_plugins.grasping.isolated_sim_runner import (
    AttemptIdentity,
    _finite_xyz,
    _resolved_outcome,
    _screw_retention_evidence,
    _screw_trainer_label,
    _validate_execution_trace,
    build_candidate,
    candidate_pick,
    run,
    same_parallel_axis,
    spread_from_tip_yaw,
    target_only_builder,
)


class IsolatedSimulationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.scenario = scenario_module.load_scenario_config()
        self.items = scenario_module.build_sort_items(self.scenario)

    def _identity(self, candidate_id):
        return AttemptIdentity(
            "layout-family-1",
            "group-1",
            candidate_id,
            "pilot",
        )

    def _item(self, item_id):
        return next(item for item in self.items if item.item_id == item_id)

    @staticmethod
    def _complete_trace():
        return {
            "pick_tip_xyz_m": [0.0, 0.0, 0.0],
            "pick_tip_yaw_rad": 0.0,
            "spread_axis": [0.0, 1.0, 0.0],
            "position_disagreement_m": 0.0,
            "yaw_disagreement_rad": 0.0,
            "held_phase_tip_end_xyz_m": {
                "lift": [0.0, 0.0, 0.1],
                "carry": [1.0, 0.0, 0.1],
                "lower": [1.0, 0.0, 0.02],
            },
        }

    @staticmethod
    def _complete_report(item_id, *, placement=True, phase_truth=True):
        source_truth = {
            "origin_world_m": [0.0, 0.0, 0.0],
            "grasp_point_world_m": [0.0, 0.0, 0.0],
        }
        event = {
            "item_id": item_id,
            "pickup_verified": True,
            "placement_verified": placement,
            "trajectory_ik_failures": 0,
            "pre_descent_revalidation": {"passed": True},
            "source_truth": source_truth,
        }
        if phase_truth:
            points = {
                "lift": [0.0, 0.0, 0.1],
                "carry": [1.0, 0.0, 0.1],
                "lower": [1.0, 0.0, 0.02],
                "release": [1.0, 0.0, 0.02],
                "settle": [1.0, 0.0, 0.0],
                "retreat": [1.0, 0.0, 0.0],
            }
            event["phase_truth"] = {
                name: {
                    "origin_world_m": list(point),
                    "grasp_point_world_m": list(point),
                }
                for name, point in points.items()
            }
        return {
            "events": [event],
            "final_scene_verification": {
                "items": {item_id: {"passed": placement}}
            },
            "task_complete": placement,
        }

    def test_tip_yaw_maps_to_jaw_closing_axis(self):
        self.assertEqual(spread_from_tip_yaw(0.0), (-0.0, 1.0, 0.0))
        axis = spread_from_tip_yaw(math.pi / 2.0)
        self.assertAlmostEqual(axis[0], -1.0)
        self.assertAlmostEqual(axis[1], 0.0)
        self.assertTrue(same_parallel_axis(axis, (1.0, 0.0, 0.0)))
        self.assertFalse(same_parallel_axis(axis, (0.0, 1.0, 0.0)))

    def test_finite_xyz_accepts_numpy_endpoints_and_rejects_bad_values(self):
        self.assertEqual(
            _finite_xyz(np.array([1.0, 2.0, 3.0]), "endpoint"),
            (1.0, 2.0, 3.0),
        )

        for value in (
            "123",
            b"123",
            object(),
            [1.0, 2.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, "bad"],
        ):
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(
                    ValueError, "must contain three coordinates"
                ):
                    _finite_xyz(value, "endpoint")

        for value in ([1.0, math.nan, 3.0], [math.inf, 2.0, 3.0]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "must contain finite coordinates"
                ):
                    _finite_xyz(value, "endpoint")

    def test_cube_candidate_is_regenerated_on_settled_pose(self):
        candidate = build_candidate(
            self.scenario,
            self._item("cube-yellow"),
            self._identity("cube-yaw-090-2"),
        )
        configured_yaw = math.radians(22.0)
        observed_yaw = math.radians(23.0)
        original_xy = [
            candidate.pick_tip_pose.xyz_m[0] + 0.001,
            candidate.pick_tip_pose.xyz_m[1] - 0.0005,
        ]
        original = {
            "xy": list(original_xy),
            "pick_z": candidate.pick_tip_pose.xyz_m[2],
            "approach_clearance_m": 9.0,
            "spread_dir": list(spread_from_tip_yaw(observed_yaw)),
            "place_xy": [0.24, 0.0],
        }
        patched = candidate_pick(
            original,
            candidate,
            configured_target_yaw_rad=configured_yaw,
        )
        self.assertEqual(original["xy"], original_xy)
        self.assertEqual(patched["xy"], original_xy)
        self.assertEqual(patched["pick_z"], original["pick_z"])
        self.assertAlmostEqual(
            math.degrees(patched["_team8_executed_tip_yaw_rad"]),
            113.0,
        )
        self.assertAlmostEqual(
            patched["_team8_position_disagreement_m"],
            math.hypot(0.001, 0.0005),
        )
        self.assertAlmostEqual(
            math.degrees(patched["_team8_yaw_disagreement_rad"]),
            1.0,
        )
        self.assertTrue(
            same_parallel_axis(
                patched["spread_dir"],
                spread_from_tip_yaw(math.radians(113.0)),
            )
        )

    def test_candidate_regeneration_rejects_stale_position_or_yaw(self):
        candidate = build_candidate(
            self.scenario,
            self._item("cube-yellow"),
            self._identity("cube-yaw-000-0"),
        )
        base = {
            "xy": list(candidate.pick_tip_pose.xyz_m[:2]),
            "pick_z": candidate.pick_tip_pose.xyz_m[2],
            "spread_dir": list(spread_from_tip_yaw(math.radians(22.0))),
        }
        stale_position = dict(base)
        stale_position["xy"] = [
            base["xy"][0] + 0.003,
            base["xy"][1],
        ]
        with self.assertRaisesRegex(ValueError, "position"):
            candidate_pick(
                stale_position,
                candidate,
                configured_target_yaw_rad=math.radians(22.0),
            )

        stale_yaw = dict(base)
        stale_yaw["spread_dir"] = list(
            spread_from_tip_yaw(math.radians(28.0))
        )
        with self.assertRaisesRegex(ValueError, "yaw"):
            candidate_pick(
                stale_yaw,
                candidate,
                configured_target_yaw_rad=math.radians(22.0),
            )

    def test_screw_candidate_must_match_scenario_offset(self):
        accepted = build_candidate(
            self.scenario,
            self._item("screw-red-01"),
            self._identity("screw-shaft-016mm-1"),
        )
        self.assertEqual(accepted.candidate_id, "screw-shaft-016mm-1")
        with self.assertRaisesRegex(
            ValueError, "scenario screw offset does not match"
        ):
            build_candidate(
                self.scenario,
                self._item("screw-red-01"),
                self._identity("screw-shaft-012mm-0"),
            )

    def test_target_filter_drives_one_item_completion_denominator(self):
        original = scenario_module.build_sort_items
        target_only = target_only_builder(original, "screw-red-01")
        try:
            scenario_module.build_sort_items = target_only
            completion = scenario_module.build_completion_report(
                self.scenario,
                [
                    {
                        "item_id": "screw-red-01",
                        "pickup_verified": True,
                        "placement_verified": True,
                    }
                ],
            )
        finally:
            scenario_module.build_sort_items = original
        self.assertEqual(completion["expected_items"], 1)
        self.assertEqual(
            completion["expected_item_ids"],
            ["screw-red-01"],
        )
        self.assertTrue(completion["task_complete"])

    def test_screw_label_requires_value_checked_retention_evidence(self):
        trace = self._complete_trace()
        complete = self._complete_report("screw-red-01")
        retention = _screw_retention_evidence(
            complete,
            "screw-red-01",
            trace,
        )
        self.assertEqual(
            _resolved_outcome(complete, "screw-red-01"),
            ("resolved_success", True),
        )
        self.assertEqual(retention["status"], "passed")
        self.assertIs(
            _screw_trainer_label(
                complete,
                "screw-red-01",
                retention,
            ),
            True,
        )

        incomplete = self._complete_report(
            "screw-red-01", phase_truth=False
        )
        unresolved_retention = _screw_retention_evidence(
            incomplete,
            "screw-red-01",
            trace,
        )
        self.assertEqual(unresolved_retention["status"], "unresolved")
        self.assertIsNone(
            _screw_trainer_label(
                incomplete,
                "screw-red-01",
                unresolved_retention,
            )
        )

        failed = self._complete_report(
            "screw-red-01", placement=False
        )
        failed_retention = _screw_retention_evidence(
            failed,
            "screw-red-01",
            trace,
        )
        self.assertIs(
            _screw_trainer_label(
                failed,
                "screw-red-01",
                failed_retention,
            ),
            False,
        )

    def test_screw_drop_during_lower_is_a_resolved_negative(self):
        report = self._complete_report("screw-red-01")
        report["events"][0]["phase_truth"]["lower"][
            "grasp_point_world_m"
        ] = [1.0, 0.0, 0.0]
        retention = _screw_retention_evidence(
            report,
            "screw-red-01",
            self._complete_trace(),
        )
        self.assertEqual(retention["status"], "failed")
        self.assertGreater(
            retention["phase_grasp_point_error_m"]["lower"],
            retention["maximum_grasp_point_error_m"],
        )
        self.assertIs(
            _screw_trainer_label(
                report,
                "screw-red-01",
                retention,
            ),
            False,
        )

    def test_execution_trace_requires_all_held_phase_endpoints(self):
        trace = self._complete_trace()
        del trace["held_phase_tip_end_xyz_m"]["lower"]
        with self.assertRaisesRegex(ValueError, "lower"):
            _validate_execution_trace(trace)

    def test_runner_refuses_preexisting_run_path_before_loading_demo(self):
        with tempfile.TemporaryDirectory() as raw_folder:
            folder = Path(raw_folder)
            scenario_path = folder / "scenario.json"
            scenario_path.write_text("{}\n", encoding="utf-8")
            run_path = folder / "existing-run"
            run_path.mkdir()
            with self.assertRaisesRegex(FileExistsError, "must not already exist"):
                run(
                    argparse.Namespace(
                        scenario=str(scenario_path),
                        run=str(run_path),
                    )
                )

    def test_cube_smoke_never_becomes_trainer_label(self):
        report = self._complete_report("cube-red")
        self.assertIsNone(_screw_trainer_label(report, "cube-red"))

    def test_missing_event_is_unresolved_not_negative(self):
        report = {
            "events": [],
            "final_scene_verification": {"items": {}},
            "task_complete": False,
        }
        self.assertEqual(
            _resolved_outcome(report, "screw-red-01"),
            ("unresolved", None),
        )


if __name__ == "__main__":
    unittest.main()
