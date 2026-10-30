from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.summarize_evaluation import (
    EvaluationValidationError,
    load_evaluation_result,
    summarize_evaluation_results,
    validate_evaluation_result,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "evaluation"
CUBE_RESULT = FIXTURES / "cube_evaluation.v1.json"
SCREW_RESULT = FIXTURES / "screw_observation_evaluation.v1.json"
SCHEMA = REPO_ROOT / "docs" / "schemas" / "evaluation-result.v1.schema.json"
SCRIPT = REPO_ROOT / "scripts" / "summarize_evaluation.py"


class EvaluationResultTests(unittest.TestCase):
    def test_schema_and_public_examples_are_valid_json(self):
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["$id"], "urn:rockwell-hackathon:evaluation-result:v1"
        )
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
        self.assertEqual(
            schema["properties"]["schema_version"]["type"], "integer"
        )
        self.assertEqual(
            schema["$defs"]["provenance"]["properties"]["source_run_ids"][
                "minItems"
            ],
            1,
        )

        cube = load_evaluation_result(CUBE_RESULT)
        screw = load_evaluation_result(SCREW_RESULT)
        self.assertEqual(cube["evaluation_id"], "fixture-cube-evaluation-v1")
        self.assertEqual(
            screw["evaluation_id"], "fixture-screw-observation-evaluation-v1"
        )

    def test_aggregation_preserves_missing_measurements(self):
        summary = summarize_evaluation_results(
            [
                load_evaluation_result(CUBE_RESULT),
                load_evaluation_result(SCREW_RESULT),
            ],
            allow_mixed=True,
        )

        self.assertEqual(summary["report_count"], 2)
        self.assertEqual(summary["configuration_count"], 2)
        self.assertTrue(summary["warnings"])
        self.assertEqual(summary["provenance"]["splits"], ["test"])
        self.assertEqual(summary["episodes"]["total"], 3)
        self.assertEqual(summary["episodes"]["unique_conditions"], 3)
        self.assertEqual(summary["episodes"]["executed"], 2)
        self.assertEqual(
            summary["episodes"]["complete_relocation_successes"], 1
        )
        self.assertEqual(
            summary["episodes"]["complete_relocation_success_rate"], 0.5
        )
        self.assertEqual(summary["episodes"]["contact_events"], 1)
        self.assertEqual(summary["episodes"]["retries"], 1)

        cube = summary["vision"]["per_class"]["cube"]
        self.assertEqual(cube["recall"], 1.0)
        self.assertEqual(cube["false_positives_per_frame"], 0.5)
        self.assertEqual(cube["position_error_mm"]["mean"], 2.0)
        self.assertEqual(
            cube["symmetry_aware_orientation_error_deg"]["mean"], 3.0
        )

        screw = summary["vision"]["per_class"]["screw_iso10642_m6x30"]
        self.assertEqual(screw["recall"], 0.0)
        self.assertEqual(screw["position_error_mm"]["sample_count"], 0)
        self.assertIsNone(screw["position_error_mm"]["mean"])

        grasp = summary["grasp"]
        self.assertEqual(grasp["attempts"], 2)
        self.assertEqual(grasp["generated_candidates"], 5)
        self.assertEqual(grasp["accepted_candidates"], 4)
        self.assertEqual(grasp["rejected_candidates"], 1)
        self.assertEqual(grasp["rejection_reasons"], {"approach_clearance": 1})
        self.assertAlmostEqual(
            grasp["selected_probability"]["brier_score"], 0.265
        )
        self.assertEqual(grasp["ranking"]["accuracy"], 0.5)

        self.assertEqual(summary["timing_s"]["perception_s"]["sample_count"], 3)
        self.assertEqual(summary["timing_s"]["execution_s"]["sample_count"], 2)
        self.assertEqual(summary["timing_s"]["execution_s"]["mean"], 5.5)

    def test_mixed_configuration_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(
            EvaluationValidationError, "heterogeneous evaluation configurations"
        ):
            summarize_evaluation_results(
                [
                    load_evaluation_result(CUBE_RESULT),
                    load_evaluation_result(SCREW_RESULT),
                ]
            )

    def test_schema_version_runtime_type_matches_schema(self):
        for invalid_version in (True, 1.0):
            with self.subTest(invalid_version=invalid_version):
                report = copy.deepcopy(load_evaluation_result(CUBE_RESULT))
                report["schema_version"] = invalid_version
                with self.assertRaisesRegex(
                    EvaluationValidationError, "schema_version: must equal 1"
                ):
                    validate_evaluation_result(report)

    def test_candidate_partition_must_be_exact(self):
        report = copy.deepcopy(load_evaluation_result(CUBE_RESULT))
        report["episodes"][0]["grasp_attempt"]["generated_candidate_ids"].append(
            "not-accounted-for"
        )
        with self.assertRaisesRegex(
            EvaluationValidationError, "must exactly partition"
        ):
            validate_evaluation_result(report)

    def test_duplicate_evaluation_ids_are_not_double_counted(self):
        report = load_evaluation_result(CUBE_RESULT)
        with self.assertRaisesRegex(
            EvaluationValidationError, "duplicate evaluation_id"
        ):
            summarize_evaluation_results([report, copy.deepcopy(report)])

    def test_overlapping_source_run_ids_are_rejected(self):
        first = load_evaluation_result(CUBE_RESULT)
        second = copy.deepcopy(first)
        second["evaluation_id"] = "fixture-cube-evaluation-v1-copy"
        for episode in second["episodes"]:
            episode["episode_id"] += "-copy"
        with self.assertRaisesRegex(
            EvaluationValidationError, "overlapping source_run_ids"
        ):
            summarize_evaluation_results([first, second])

    def test_duplicate_episode_ids_across_reports_are_rejected(self):
        first = load_evaluation_result(CUBE_RESULT)
        second = copy.deepcopy(first)
        second["evaluation_id"] = "fixture-cube-evaluation-v1-copy"
        second["provenance"]["source_run_ids"] = ["public-fixture-cube-run-copy"]
        with self.assertRaisesRegex(
            EvaluationValidationError, "duplicate episode_id across reports"
        ):
            summarize_evaluation_results([first, second])

    def test_condition_id_is_required(self):
        report = copy.deepcopy(load_evaluation_result(CUBE_RESULT))
        report["episodes"][0].pop("condition_id")
        with self.assertRaisesRegex(EvaluationValidationError, "condition_id"):
            validate_evaluation_result(report)

    def test_source_run_id_is_required(self):
        report = copy.deepcopy(load_evaluation_result(CUBE_RESULT))
        report["provenance"]["source_run_ids"] = []
        with self.assertRaisesRegex(
            EvaluationValidationError, "must contain at least one source run ID"
        ):
            validate_evaluation_result(report)

    def test_loader_rejects_duplicate_keys_and_nonstandard_numbers(self):
        invalid_documents = {
            "duplicate": '{"schema_version": 1, "schema_version": 1}',
            "NaN": '{"schema_version": NaN}',
        }
        patterns = {
            "duplicate": "duplicate JSON key: schema_version",
            "NaN": "non-standard JSON constant NaN",
        }
        with tempfile.TemporaryDirectory() as raw_directory:
            for label, document in invalid_documents.items():
                with self.subTest(label=label):
                    path = Path(raw_directory) / f"{label}.json"
                    path.write_text(document, encoding="utf-8")
                    with self.assertRaisesRegex(
                        EvaluationValidationError, patterns[label]
                    ):
                        load_evaluation_result(path)

    def test_vision_reference_accounting_must_be_exact(self):
        report = copy.deepcopy(load_evaluation_result(CUBE_RESULT))
        report["vision"]["per_class"][0]["dropouts"] = 1
        with self.assertRaisesRegex(
            EvaluationValidationError,
            "true_positives \+ dropouts must equal reference_instances",
        ):
            validate_evaluation_result(report)

    def test_zero_vision_frames_cannot_have_evidence(self):
        report = copy.deepcopy(load_evaluation_result(CUBE_RESULT))
        report["vision"]["frames_evaluated"] = 0
        report["vision"]["inference_latency_ms"] = []
        with self.assertRaisesRegex(
            EvaluationValidationError,
            "zero evaluated frames cannot have nonzero counts or pose samples",
        ):
            validate_evaluation_result(report)

    def test_latency_sample_count_cannot_exceed_frames(self):
        report = copy.deepcopy(load_evaluation_result(CUBE_RESULT))
        report["vision"]["frames_evaluated"] = 1
        with self.assertRaisesRegex(
            EvaluationValidationError, "sample count cannot exceed frames_evaluated"
        ):
            validate_evaluation_result(report)

    def test_outcome_observed_rejects_indeterminate_statuses(self):
        for status in ("cancelled", "unresolved", "not_executed"):
            with self.subTest(status=status):
                report = copy.deepcopy(load_evaluation_result(CUBE_RESULT))
                report["episodes"][0]["outcome"]["status"] = status
                with self.assertRaisesRegex(
                    EvaluationValidationError,
                    "selected_candidate_outcome_observed.*cannot be true",
                ):
                    validate_evaluation_result(report)

    def test_outcome_observed_requires_a_selected_candidate(self):
        report = copy.deepcopy(load_evaluation_result(CUBE_RESULT))
        attempt = report["episodes"][0]["grasp_attempt"]
        attempt["selected_candidate_id"] = None
        attempt["ranking_correct"] = None
        attempt["ranking_evidence_id"] = None
        report["episodes"][0]["outcome"]["status"] = "failed_lift"
        with self.assertRaisesRegex(
            EvaluationValidationError, "cannot be true when no candidate was selected"
        ):
            validate_evaluation_result(report)

    def test_grasp_label_presence_tracks_observation_flag(self):
        observed_without_label = copy.deepcopy(load_evaluation_result(CUBE_RESULT))
        observed_without_label["episodes"][0]["grasp_attempt"][
            "selected_candidate_complete_grasp_success"
        ] = None
        with self.assertRaisesRegex(
            EvaluationValidationError,
            "must be boolean when the selected candidate outcome was observed",
        ):
            validate_evaluation_result(observed_without_label)

        label_without_observation = copy.deepcopy(
            load_evaluation_result(CUBE_RESULT)
        )
        attempt = label_without_observation["episodes"][1]["grasp_attempt"]
        attempt["selected_candidate_outcome_observed"] = False
        with self.assertRaisesRegex(
            EvaluationValidationError,
            "must be null when the selected candidate outcome was not observed",
        ):
            validate_evaluation_result(label_without_observation)

    def test_complete_relocation_requires_identity_selection_and_verification(self):
        base = load_evaluation_result(CUBE_RESULT)
        mutations = {
            "target track": lambda episode: episode.update(target_track_id=None),
            "target class": lambda episode: episode.update(target_class_id=None),
            "selection": self._remove_selected_candidate,
            "verification": lambda episode: episode["outcome"].update(
                verification=[]
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                report = copy.deepcopy(base)
                mutate(report["episodes"][0])
                with self.assertRaisesRegex(
                    EvaluationValidationError, "complete_relocation_success requires"
                ):
                    validate_evaluation_result(report)

    def test_complete_relocation_cannot_be_declared_unobserved(self):
        report = copy.deepcopy(load_evaluation_result(CUBE_RESULT))
        attempt = report["episodes"][0]["grasp_attempt"]
        attempt["selected_candidate_outcome_observed"] = False
        attempt["selected_candidate_complete_grasp_success"] = None
        with self.assertRaisesRegex(
            EvaluationValidationError,
            "complete_relocation_success requires "
            "selected_candidate_outcome_observed=true",
        ):
            validate_evaluation_result(report)

    def test_terminal_outcomes_require_traceable_execution_evidence(self):
        def clear_observation(episode):
            attempt = episode["grasp_attempt"]
            attempt["selected_candidate_outcome_observed"] = False
            attempt["selected_candidate_complete_grasp_success"] = None

        mutations = {
            "target track": lambda episode: episode.update(target_track_id=None),
            "target class": lambda episode: episode.update(target_class_id=None),
            "selected candidate": self._remove_selected_candidate,
            "observed outcome": clear_observation,
            "verification": lambda episode: episode["outcome"].update(
                verification=[]
            ),
        }
        for status in (
            "complete_relocation_success",
            "failed_lift",
            "drop_or_slip",
            "wrong_destination",
        ):
            for label, mutate in mutations.items():
                with self.subTest(status=status, missing=label):
                    report = copy.deepcopy(load_evaluation_result(CUBE_RESULT))
                    episode = report["episodes"][1]
                    episode["outcome"]["status"] = status
                    episode["grasp_attempt"][
                        "selected_candidate_complete_grasp_success"
                    ] = status == "complete_relocation_success"
                    mutate(episode)
                    with self.assertRaisesRegex(
                        EvaluationValidationError, f"{status} requires"
                    ):
                        validate_evaluation_result(report)

    @staticmethod
    def _remove_selected_candidate(episode):
        attempt = episode["grasp_attempt"]
        attempt["selected_candidate_id"] = None
        attempt["selected_candidate_outcome_observed"] = False
        attempt["selected_candidate_complete_grasp_success"] = None
        attempt["ranking_correct"] = None
        attempt["ranking_evidence_id"] = None

    def test_ranking_requires_multiple_candidates_and_evidence(self):
        base = load_evaluation_result(CUBE_RESULT)

        no_evidence = copy.deepcopy(base)
        no_evidence["episodes"][0]["grasp_attempt"]["ranking_evidence_id"] = None
        with self.assertRaisesRegex(
            EvaluationValidationError,
            "ranking_evidence_id.*required when ranking_correct",
        ):
            validate_evaluation_result(no_evidence)

        one_candidate = copy.deepcopy(base)
        attempt = one_candidate["episodes"][0]["grasp_attempt"]
        attempt["generated_candidate_ids"] = ["cube-yaw-0"]
        attempt["accepted_candidates"] = [attempt["accepted_candidates"][0]]
        with self.assertRaisesRegex(
            EvaluationValidationError, "requires at least two accepted candidates"
        ):
            validate_evaluation_result(one_candidate)

        evidence_without_result = copy.deepcopy(base)
        evidence_without_result["episodes"][0]["grasp_attempt"][
            "ranking_correct"
        ] = None
        with self.assertRaisesRegex(
            EvaluationValidationError,
            "ranking_evidence_id.*must be null when ranking_correct is null",
        ):
            validate_evaluation_result(evidence_without_result)

    def test_not_executed_selection_is_not_labeled_as_failure(self):
        report = copy.deepcopy(load_evaluation_result(CUBE_RESULT))
        report["episodes"][1]["outcome"]["status"] = "not_executed"
        report["episodes"][1]["grasp_attempt"][
            "selected_candidate_outcome_observed"
        ] = False
        report["episodes"][1]["grasp_attempt"][
            "selected_candidate_complete_grasp_success"
        ] = None
        summary = summarize_evaluation_results([report])
        self.assertEqual(summary["episodes"]["executed"], 1)
        selected = summary["grasp"]["selected_probability"]
        self.assertEqual(selected["sample_count"], 1)
        self.assertAlmostEqual(selected["brier_score"], 0.04)

    def test_wrong_destination_does_not_imply_grasp_failure(self):
        report = copy.deepcopy(load_evaluation_result(CUBE_RESULT))
        episode = report["episodes"][1]
        episode["outcome"]["status"] = "wrong_destination"
        episode["grasp_attempt"][
            "selected_candidate_complete_grasp_success"
        ] = True
        summary = summarize_evaluation_results([report])
        self.assertEqual(summary["episodes"]["outcomes"]["wrong_destination"], 1)
        selected = summary["grasp"]["selected_probability"]
        self.assertEqual(selected["sample_count"], 2)
        self.assertEqual(selected["observed_success_rate"], 1.0)
        self.assertAlmostEqual(selected["brier_score"], 0.065)

    def test_mechanical_failure_statuses_require_negative_grasp_label(self):
        for status in ("failed_lift", "drop_or_slip"):
            with self.subTest(status=status):
                report = copy.deepcopy(load_evaluation_result(CUBE_RESULT))
                episode = report["episodes"][1]
                episode["outcome"]["status"] = status
                episode["grasp_attempt"][
                    "selected_candidate_complete_grasp_success"
                ] = True
                with self.assertRaisesRegex(
                    EvaluationValidationError,
                    f"{status} requires "
                    "selected_candidate_complete_grasp_success=false",
                ):
                    validate_evaluation_result(report)

    def test_cli_prints_text_and_writes_json(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            output = Path(raw_directory) / "summary.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(CUBE_RESULT),
                    str(SCREW_RESULT),
                    "--allow-mixed",
                    "--json-output",
                    str(output),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout.startswith("WARNING:"), result.stdout)
            self.assertIn("Evaluation summary", result.stdout)
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(written["warnings"])
            self.assertEqual(written["report_count"], 2)
            self.assertEqual(written["episodes"]["executed"], 2)

    def test_cli_refuses_to_overwrite_an_input(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            report_path = Path(raw_directory) / "report.json"
            original = CUBE_RESULT.read_text(encoding="utf-8")
            report_path.write_text(original, encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(report_path),
                    "--json-output",
                    str(report_path),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not overwrite an input report", result.stderr)
            self.assertEqual(report_path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
