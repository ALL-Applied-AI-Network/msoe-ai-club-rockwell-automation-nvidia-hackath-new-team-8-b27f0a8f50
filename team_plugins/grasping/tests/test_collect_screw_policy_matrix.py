import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from demos.scripted_cube_screw_bucket_sort.scenario import load_scenario_config

from team_plugins.grasping.collect_screw_policy_matrix import collect
from team_plugins.grasping.isolated_sim_runner import (
    RUNNER_ID,
    _screw_orientation_xyzw,
    _screw_retention_evidence,
)
from team_plugins.grasping.multigrasp_pipeline import SCREW_GENERATOR_ID
from team_plugins.grasping.prepare_screw_policy_matrix import (
    TARGET_ITEM_ID,
    prepare,
)
from team_plugins.grasping.run_screw_policy_matrix import RUN_MATRIX_ID


class ScrewPolicyMatrixCollectorTests(unittest.TestCase):
    @staticmethod
    def _write_json(path, value):
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _prepare(self, folder):
        manifest_path = prepare(folder / "pilot", require_clean=False)
        manifest = json.loads(manifest_path.read_text("utf-8"))
        manifest["relevant_source_tree_clean"] = True
        manifest["relevant_source_dirty_entries"] = []
        self._write_json(manifest_path, manifest)
        return manifest_path, manifest

    def _write_run(self, runs_root, case, source_commit, success):
        run_dir = runs_root / case["case_id"]
        run_dir.mkdir(parents=True)
        scenario_path = Path(case["scenario_path"])
        config = load_scenario_config(scenario_path)
        target = next(
            screw
            for screw in config.screws
            if screw.item_id == TARGET_ITEM_ID
        )
        yaw = math.radians(target.spawn_yaw_deg)
        quaternion_xyzw = _screw_orientation_xyzw(yaw)
        orientation_wxyz = [
            quaternion_xyzw[3],
            quaternion_xyzw[0],
            quaternion_xyzw[1],
            quaternion_xyzw[2],
        ]
        axis = [math.cos(yaw), math.sin(yaw), 0.0]
        grasp = [
            target.spawn_head_base_m[0]
            + config.screw.grasp_offset_from_head_m * axis[0],
            target.spawn_head_base_m[1]
            + config.screw.grasp_offset_from_head_m * axis[1],
            target.spawn_head_base_m[2],
        ]
        source_truth = {
            "origin_world_m": list(target.spawn_head_base_m),
            "orientation_wxyz": orientation_wxyz,
            "axis_world": axis,
            "grasp_point_world_m": grasp,
        }
        held_points = {
            "lift": [grasp[0], grasp[1], grasp[2] + 0.128],
            "carry": [0.24, -0.19, grasp[2] + 0.17],
            "lower": [0.24, -0.19, grasp[2] + 0.04],
        }
        phase_truth = {
            name: {
                "origin_world_m": list(point),
                "orientation_wxyz": orientation_wxyz,
                "axis_world": axis,
                "grasp_point_world_m": list(point),
            }
            for name, point in held_points.items()
        }
        for name, point in (
            ("release", held_points["lower"]),
            ("settle", [0.24, -0.19, grasp[2]]),
            ("retreat", [0.24, -0.19, grasp[2]]),
        ):
            phase_truth[name] = {
                "origin_world_m": list(point),
                "orientation_wxyz": orientation_wxyz,
                "axis_world": axis,
                "grasp_point_world_m": list(point),
            }
        event = {
            "item_id": TARGET_ITEM_ID,
            "pickup_verified": True,
            "placement_verified": success,
            "trajectory_ik_failures": 0,
            "pre_descent_revalidation": {"passed": True},
            "phase_truth": phase_truth,
            "source_truth": source_truth,
        }
        report = {
            "events": [event],
            "expected_items": 1,
            "processed_items": 1,
            "expected_item_ids": [TARGET_ITEM_ID],
            "final_scene_verification": {
                "items": {TARGET_ITEM_ID: {"passed": success}}
            },
            "task_complete": success,
        }
        trace = {
            "pick_tip_xyz_m": grasp,
            "pick_tip_yaw_rad": yaw,
            "spread_axis": [-math.sin(yaw), math.cos(yaw), 0.0],
            "position_disagreement_m": 0.0,
            "yaw_disagreement_rad": 0.0,
            "held_phase_tip_end_xyz_m": held_points,
        }
        retention = _screw_retention_evidence(
            report,
            TARGET_ITEM_ID,
            trace,
        )
        self.assertEqual(retention["status"], "passed")
        scenario_sha = hashlib.sha256(scenario_path.read_bytes()).hexdigest()
        status = "resolved_success" if success else "resolved_failure"
        execution = {
            "runner_id": RUNNER_ID,
            "simulation_only": True,
            "camera_perception": False,
            "hardware_integration": False,
            "simulator_truth_affects_runtime_control": True,
            "decision_source": (
                "configured candidate transformed onto settled simulator target truth"
            ),
            "candidate_generation_source": "configured scenario target-only truth",
            "candidate_generation_capture_time_s": 0.0,
            "runtime_transform_source": "settled simulator target truth",
            "simulator_obstacle_truth_used_for_candidate_generation": False,
            "other_configured_objects_remained_as_physics_obstacles": True,
            "fresh_process_required_per_candidate": True,
            "candidate_generator_id": SCREW_GENERATOR_ID,
            "candidate_scene_id": case["group_id"],
            "target_item_id": TARGET_ITEM_ID,
            "layout_family_id": case["layout_family_id"],
            "group_id": case["group_id"],
            "candidate_id": case["candidate_id"],
            "split": case["split"],
            "source_commit": source_commit,
            "scenario_sha256": scenario_sha,
            "relevant_source_tree_clean": True,
            "relevant_source_dirty_entries": [],
            "executed_pick_tip_xyz_m": trace["pick_tip_xyz_m"],
            "executed_pick_tip_yaw_rad": trace["pick_tip_yaw_rad"],
            "commanded_spread_axis": trace["spread_axis"],
            "configured_to_settled_position_disagreement_m": (
                trace["position_disagreement_m"]
            ),
            "configured_to_settled_yaw_disagreement_rad": (
                trace["yaw_disagreement_rad"]
            ),
            "outcome_status": status,
            "label_scope": "target_only",
            "target_only_complete_grasp_label": success,
            "target_only_label_eligible": True,
            "collision_safe_label_eligible": False,
            "non_target_obstacle_final_state_evaluated": False,
            "screw_retention_evidence": retention,
        }
        report["isolated_candidate_execution"] = execution
        identity = {
            "layout_family_id": case["layout_family_id"],
            "group_id": case["group_id"],
            "candidate_id": case["candidate_id"],
            "split": case["split"],
        }
        metadata = {
            "runner_id": RUNNER_ID,
            "identity": identity,
            "target_item_id": TARGET_ITEM_ID,
            "source_commit": source_commit,
            "relevant_source_tree_clean": True,
            "relevant_source_dirty_entries": [],
            "scenario_sha256": scenario_sha,
            "execution_trace": trace,
            "outcome_status": status,
            "execution_exception": None,
            "augmentation_exception": None,
            "exception": None,
        }
        self._write_json(run_dir / "execution_report.json", report)
        self._write_json(run_dir / "attempt_metadata.json", metadata)

    def _write_matrix_report(self, runs_root, manifest_path, manifest):
        attempts = []
        for case in manifest["cases"]:
            metadata_path = (
                runs_root / case["case_id"] / "attempt_metadata.json"
            )
            status = (
                json.loads(metadata_path.read_text("utf-8"))["outcome_status"]
                if metadata_path.is_file()
                else "unresolved"
            )
            attempts.append(
                {
                    "case_id": case["case_id"],
                    "layout_family_id": case["layout_family_id"],
                    "group_id": case["group_id"],
                    "candidate_id": case["candidate_id"],
                    "split": case["split"],
                    "status": status,
                    "return_code": (
                        0
                        if status in {"resolved_success", "resolved_failure"}
                        else 1
                    ),
                }
            )
        all_resolved = all(
            row["status"] in {"resolved_success", "resolved_failure"}
            for row in attempts
        )
        self._write_json(
            runs_root / "matrix_run_report.json",
            {
                "runner_id": RUN_MATRIX_ID,
                "source_commit": manifest["source_commit"],
                "manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
                "attempts": attempts,
                "all_attempts_resolved": all_resolved,
            },
        )

    def _collect(self, manifest_path, manifest, runs_root, output):
        with mock.patch(
            "team_plugins.grasping.collect_screw_policy_matrix._source_state",
            return_value=(manifest["source_commit"], True, ()),
        ):
            return collect(manifest_path, runs_root, output)

    def test_collector_emits_complete_pilot_rows_but_never_trainer_ready(self):
        with tempfile.TemporaryDirectory() as raw_folder:
            folder = Path(raw_folder)
            manifest_path, manifest = self._prepare(folder)
            runs_root = folder / "runs"
            runs_root.mkdir()
            for case in manifest["cases"]:
                success = case["candidate_id"] == "screw-shaft-016mm-1"
                self._write_run(
                    runs_root,
                    case,
                    manifest["source_commit"],
                    success,
                )
            self._write_matrix_report(runs_root, manifest_path, manifest)

            report_path = self._collect(
                manifest_path,
                manifest,
                runs_root,
                folder / "collection",
            )
            report = json.loads(report_path.read_text("utf-8"))
            self.assertFalse(report["trainer_ready"])
            self.assertTrue(report["pilot_only"])
            self.assertTrue(report["training_input_structurally_valid"])
            self.assertFalse(
                report["v1_raw_axial_offset_monotone_compatible"]
            )
            self.assertEqual(
                len(report["v1_raw_axial_offset_nonmonotone_groups"]),
                4,
            )
            self.assertEqual(report["dataset_rows"], 12)
            self.assertEqual(report["eligible_groups"], 4)
            self.assertEqual(report["train_successes"], 3)
            self.assertEqual(report["train_failures"], 6)
            self.assertEqual(report["test_successes"], 1)
            self.assertEqual(report["test_failures"], 2)
            self.assertFalse(report["incomplete_groups"])

            dataset_path = Path(report["dataset_path"])
            rows = [
                json.loads(line)
                for line in dataset_path.read_text("utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 12)
            self.assertEqual(
                len(
                    {
                        (row["group_id"], row["candidate_id"])
                        for row in rows
                    }
                ),
                12,
            )
            self.assertTrue(
                all(
                    tuple(row["features"])
                    == tuple(report["feature_order"])
                    for row in rows
                )
            )

    def test_collector_excludes_an_entire_incomplete_group(self):
        with tempfile.TemporaryDirectory() as raw_folder:
            folder = Path(raw_folder)
            manifest_path, manifest = self._prepare(folder)
            runs_root = folder / "runs"
            runs_root.mkdir()
            omitted_case = manifest["cases"][0]
            for case in manifest["cases"][1:]:
                success = case["candidate_id"] == "screw-shaft-016mm-1"
                self._write_run(
                    runs_root,
                    case,
                    manifest["source_commit"],
                    success,
                )
            self._write_matrix_report(runs_root, manifest_path, manifest)

            report_path = self._collect(
                manifest_path,
                manifest,
                runs_root,
                folder / "collection",
            )
            report = json.loads(report_path.read_text("utf-8"))
            self.assertFalse(report["trainer_ready"])
            self.assertFalse(report["all_matrix_attempts_resolved"])
            self.assertIn(
                omitted_case["group_id"],
                report["incomplete_groups"],
            )
            self.assertEqual(report["dataset_rows"], 9)

    def test_collector_rejects_checkout_commit_drift(self):
        with tempfile.TemporaryDirectory() as raw_folder:
            folder = Path(raw_folder)
            manifest_path, manifest = self._prepare(folder)
            runs_root = folder / "runs"
            runs_root.mkdir()
            with mock.patch(
                "team_plugins.grasping.collect_screw_policy_matrix._source_state",
                return_value=("0" * 40, True, ()),
            ):
                with self.assertRaisesRegex(ValueError, "collector source"):
                    collect(
                        manifest_path,
                        runs_root,
                        folder / "collection",
                    )

    def test_collector_rejects_recorded_execution_exception(self):
        with tempfile.TemporaryDirectory() as raw_folder:
            folder = Path(raw_folder)
            manifest_path, manifest = self._prepare(folder)
            runs_root = folder / "runs"
            runs_root.mkdir()
            for case in manifest["cases"]:
                self._write_run(
                    runs_root,
                    case,
                    manifest["source_commit"],
                    True,
                )
            self._write_matrix_report(runs_root, manifest_path, manifest)
            first = manifest["cases"][0]
            metadata_path = (
                runs_root / first["case_id"] / "attempt_metadata.json"
            )
            metadata = json.loads(metadata_path.read_text("utf-8"))
            metadata["execution_exception"] = {
                "type": "RuntimeError",
                "message": "late crash",
            }
            self._write_json(metadata_path, metadata)
            with self.assertRaisesRegex(
                ValueError,
                "records or omits execution_exception",
            ):
                self._collect(
                    manifest_path,
                    manifest,
                    runs_root,
                    folder / "collection",
                )

    def test_collector_rejects_resolved_nonzero_return_code(self):
        with tempfile.TemporaryDirectory() as raw_folder:
            folder = Path(raw_folder)
            manifest_path, manifest = self._prepare(folder)
            runs_root = folder / "runs"
            runs_root.mkdir()
            for case in manifest["cases"]:
                self._write_run(
                    runs_root,
                    case,
                    manifest["source_commit"],
                    True,
                )
            self._write_matrix_report(runs_root, manifest_path, manifest)
            matrix_path = runs_root / "matrix_run_report.json"
            matrix = json.loads(matrix_path.read_text("utf-8"))
            matrix["attempts"][0]["return_code"] = 7
            self._write_json(matrix_path, matrix)
            with self.assertRaisesRegex(ValueError, "nonzero return code"):
                self._collect(
                    manifest_path,
                    manifest,
                    runs_root,
                    folder / "collection",
                )

    def test_collector_rejects_matrix_metadata_status_disagreement(self):
        with tempfile.TemporaryDirectory() as raw_folder:
            folder = Path(raw_folder)
            manifest_path, manifest = self._prepare(folder)
            runs_root = folder / "runs"
            runs_root.mkdir()
            for case in manifest["cases"]:
                self._write_run(
                    runs_root,
                    case,
                    manifest["source_commit"],
                    True,
                )
            self._write_matrix_report(runs_root, manifest_path, manifest)
            first = manifest["cases"][0]
            metadata_path = (
                runs_root / first["case_id"] / "attempt_metadata.json"
            )
            metadata = json.loads(metadata_path.read_text("utf-8"))
            metadata["outcome_status"] = "resolved_failure"
            self._write_json(metadata_path, metadata)
            with self.assertRaisesRegex(ValueError, "statuses disagree"):
                self._collect(
                    manifest_path,
                    manifest,
                    runs_root,
                    folder / "collection",
                )


if __name__ == "__main__":
    unittest.main()
