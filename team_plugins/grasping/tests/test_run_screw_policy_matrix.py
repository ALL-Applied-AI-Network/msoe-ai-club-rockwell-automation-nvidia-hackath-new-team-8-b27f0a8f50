import json
from pathlib import Path
import tempfile
import unittest

from team_plugins.grasping.prepare_screw_policy_matrix import prepare
from team_plugins.grasping.run_screw_policy_matrix import (
    _archive_attempt,
    _case_command,
    _case_run_dir,
    _validate_manifest,
    _validate_prior_resolved_row,
)


class ScrewPolicyMatrixRunnerTests(unittest.TestCase):
    def test_case_command_is_one_target_one_candidate_physics_only(self):
        case = {
            "scenario_path": "/external/config/scenario.json",
            "layout_family_id": "family-a",
            "group_id": "group-a",
            "candidate_id": "screw-shaft-016mm-1",
            "split": "train",
        }
        command = _case_command(
            isaac_python=Path("/isaac-sim/python.sh"),
            case=case,
            run_dir=Path("/external/runs/case-a"),
        )
        self.assertEqual(command[0], "/isaac-sim/python.sh")
        self.assertIn("team_plugins.grasping.isolated_sim_runner", command)
        self.assertEqual(command.count("--target-item"), 1)
        self.assertEqual(command.count("--candidate-id"), 1)
        self.assertEqual(
            command[command.index("--candidate-id") + 1],
            "screw-shaft-016mm-1",
        )
        self.assertIn("--physics-only", command)

    def test_manifest_validator_accepts_prepared_catalog_ids(self):
        with tempfile.TemporaryDirectory() as raw_folder:
            manifest_path = prepare(
                Path(raw_folder) / "pilot",
                require_clean=False,
            )
            manifest = json.loads(manifest_path.read_text("utf-8"))
            manifest["relevant_source_tree_clean"] = True
            manifest["relevant_source_dirty_entries"] = []
            commit, cases = _validate_manifest(manifest)
            self.assertEqual(commit, manifest["source_commit"])
            self.assertEqual(len(cases), 12)
            self.assertEqual(
                {case["candidate_id"] for case in cases},
                {
                    "screw-shaft-012mm-0",
                    "screw-shaft-016mm-1",
                    "screw-shaft-020mm-2",
                },
            )

    def test_case_path_rejects_traversal_and_reserved_names(self):
        root = Path("/external/runs")
        for value in ("../escape", "/absolute", "logs", "_archived_incomplete"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _case_run_dir(root, value)

    def test_resume_rejects_prior_nonzero_or_missing_exit_code(self):
        _validate_prior_resolved_row(
            {"return_code": 0},
            "case-a",
        )
        for prior in (
            {"return_code": 7},
            {"return_code": None},
            {},
        ):
            with self.subTest(prior=prior):
                with self.assertRaisesRegex(ValueError, "zero exit"):
                    _validate_prior_resolved_row(prior, "case-a")

    def test_incomplete_attempt_is_preserved_in_archive(self):
        with tempfile.TemporaryDirectory() as raw_folder:
            runs_root = Path(raw_folder) / "runs"
            runs_root.mkdir()
            run_dir = runs_root / "case-a"
            run_dir.mkdir()
            (run_dir / "partial.txt").write_text("partial\n", encoding="utf-8")
            logs_root = runs_root / "logs"
            logs_root.mkdir()
            log_path = logs_root / "case-a.log"
            log_path.write_text("trace\n", encoding="utf-8")
            archive = _archive_attempt(
                runs_root=runs_root,
                run_dir=run_dir,
                log_path=log_path,
                case_id="case-a",
                reason="interrupted",
            )
            self.assertFalse(run_dir.exists())
            self.assertFalse(log_path.exists())
            self.assertEqual(
                (archive / "run_dir" / "partial.txt").read_text("utf-8"),
                "partial\n",
            )
            self.assertEqual(
                (archive / "runner.log").read_text("utf-8"),
                "trace\n",
            )
            record = json.loads(
                (archive / "archive_record.json").read_text("utf-8")
            )
            self.assertTrue(record["preserved_without_deletion"])


if __name__ == "__main__":
    unittest.main()
