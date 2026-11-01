from __future__ import annotations

import contextlib
from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import analyze_hardware_motion_log as analyzer
from scripts.analyze_hardware_motion_log import analyze_logs, main


class HardwareMotionLogAnalyzerTests(unittest.TestCase):
    def test_mixed_old_and_new_events_are_summarized_without_overclaim(self) -> None:
        lines = [
            "2026-11-01 INFO ordinary text\n",
            '2026-11-01 08:01:02,123 INFO {"event":"phase_action",'
            '"phase":"carry","action_kind":"motion",'
            '"action_rows":15,"geometric_rows_sent":13,"terminal_padding_rows":2,'
            '"motion_profile":"smooth","queued_at_observation":0,"motion_call":1}\n',
            '2026-11-01T08:01:02.223-05:00 {"event":"phase_action",'
            '"phase":"carry","action_kind":"motion",'
            '"action_rows":3,"geometric_rows_sent":1,"terminal_padding_rows":2,'
            '"motion_profile":"smooth","queued_at_observation":0,"motion_call":2}\n',
            '2026-11-01 08:01:02,323 INFO {"event":"phase_action",'
            '"phase":"carry","action_kind":"motion",'
            '"action_rows":2,"geometric_rows_sent":2,"terminal_padding_rows":0,'
            '"motion_profile":"smooth","queued_at_observation":0,"motion_call":3}\n',
            '{"event":"phase_action","phase":"dwell","action_kind":"dwell",'
            '"action_rows":4,"terminal_padding_rows":0,"queued_at_observation":2}\n',
            '{"event":"phase_action","phase":"legacy","action_rows":1}\n',
            '{"event":"pre_descent_visibility_retry","attempt":1}\n',
            '{"event":"post_place_visibility_retry","attempt":1}\n',
            '{"event":"policy_blocked","reason":"target remained ambiguous"}\n',
            "prefix {definitely not json\n",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hardware.log"
            path.write_text("".join(lines), encoding="utf-8")

            report = analyze_logs([path])

        self.assertEqual(report["input"]["lines_total"], len(lines))
        self.assertEqual(report["input"]["json_event_lines"], 8)
        self.assertEqual(report["input"]["malformed_json_candidate_lines"], 1)
        actions = report["phase_actions"]
        self.assertEqual(actions["calls"], 5)
        self.assertEqual(actions["motion_calls"], 3)
        self.assertEqual(actions["dwell_calls"], 1)
        self.assertEqual(actions["unknown_kind_calls"], 1)
        self.assertEqual(actions["terminal_padding_rows"], 4)
        self.assertEqual(actions["short_motion_calls"], 2)
        self.assertEqual(actions["motion_calls_observed_with_zero_queue"], 3)
        self.assertEqual(
            actions["queue_depths"],
            {
                "observations": 4,
                "minimum": 0,
                "maximum": 2,
                "mean": 0.5,
                "histogram": {"0": 3, "2": 1},
            },
        )
        self.assertEqual(
            actions["by_phase"]["carry"]["maximum_motion_call_index"], 3
        )
        self.assertEqual(actions["by_action_kind"]["motion"]["action_rows"], 20)
        self.assertEqual(actions["by_motion_profile"]["smooth"]["calls"], 3)
        self.assertTrue(report["indicators"]["possible_queue_starvation"])
        self.assertEqual(
            report["indicators"]["queue_starvation_evidence_phases"], ["carry"]
        )
        starvation_evidence = report["indicators"][
            "queue_starvation_evidence_occurrences"
        ]
        self.assertEqual(len(starvation_evidence), 1)
        self.assertEqual(starvation_evidence[0]["occurrence_id"], "source-001:occurrence-0001")
        self.assertEqual(starvation_evidence[0]["source"], str(path.resolve()))
        self.assertEqual(starvation_evidence[0]["line_start"], 2)
        self.assertEqual(starvation_evidence[0]["line_end"], 4)
        self.assertEqual(
            starvation_evidence[0]["timestamped_phase_action_lines"], 3
        )
        datetime.fromisoformat(
            starvation_evidence[0]["timestamp_prefix_start"]["iso8601"]
        )
        datetime.fromisoformat(
            starvation_evidence[0]["timestamp_prefix_end"]["iso8601"]
        )
        self.assertTrue(report["indicators"]["repeated_micro_corrections"])
        self.assertEqual(
            report["indicators"]["repeated_micro_correction_evidence_phases"],
            ["carry"],
        )
        self.assertIn("not proof", report["indicators"]["interpretation"])
        self.assertEqual(
            report["visibility_retries"],
            {
                "post_place_visibility_retry": 1,
                "pre_descent_visibility_retry": 1,
            },
        )
        self.assertEqual(
            report["blocked_reasons"],
            [{"reason": "target remained ambiguous", "count": 1}],
        )
        self.assertEqual(
            report["telemetry_coverage"]["action_kind"], {"events": 4, "of": 5}
        )
        self.assertTrue(report["sources"][0]["stable_during_analysis"])
        self.assertEqual(
            report["sources"][0]["sha256_before"],
            report["sources"][0]["sha256_after"],
        )

    def test_no_json_events_returns_an_explicit_empty_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plain.log"
            path.write_text("hello\nno events here\n", encoding="utf-8")
            report = analyze_logs([path])

        self.assertEqual(report["event_counts"], {})
        self.assertEqual(report["phase_actions"]["calls"], 0)
        self.assertEqual(
            report["phase_actions"]["queue_depths"],
            {
                "observations": 0,
                "minimum": None,
                "maximum": None,
                "mean": None,
                "histogram": {},
            },
        )
        self.assertFalse(report["indicators"]["possible_queue_starvation"])
        self.assertFalse(report["indicators"]["repeated_micro_corrections"])
        self.assertEqual(report["client_dispatches"]["events"], 0)
        self.assertEqual(
            report["client_dispatches"]["dispatch_pipeline_s"]["observations"],
            0,
        )
        self.assertEqual(
            report["client_dispatches"]["indicators"]["latency_budget"],
            {
                "preview_events": 0,
                "evaluated_events": 0,
                "unassessable_events": 0,
                "approached_events": 0,
                "exceeded_events": 0,
                "threshold_fraction": 0.8,
                "evidence": [],
            },
        )

    def test_dispatch_pipeline_and_queue_budget_are_measured_without_guessing(self) -> None:
        events = [
            {
                "event": "client_action_dispatch",
                "action_rows": 32,
                "action_interval_s": 0.1,
                "dispatch_pipeline_s": 0.2,
                "dispatch_pipeline_ewma_s": 0.2,
                "prefetch_lead_rows": 0,
                "preview_queue_depth": None,
                "full_observation_queue_depth": 0,
                "queue_preview_used": False,
                "replacement_used": False,
            },
            {
                "event": "client_action_dispatch",
                "action_rows": 32,
                "action_interval_s": 0.1,
                "dispatch_pipeline_s": 0.45,
                "dispatch_pipeline_ewma_s": 0.3,
                "prefetch_lead_rows": 5,
                "preview_queue_depth": 5,
                "full_observation_queue_depth": 4,
                "queue_preview_used": True,
                "replacement_used": True,
            },
            {
                "event": "client_action_dispatch",
                "action_rows": 32,
                "action_interval_s": 0.1,
                "dispatch_pipeline_s": 0.25,
                "dispatch_pipeline_ewma_s": 0.32,
                "prefetch_lead_rows": 0,
                "preview_queue_depth": 2,
                "full_observation_queue_depth": 1,
                "queue_preview_used": True,
                "replacement_used": False,
            },
            {
                "event": "client_action_dispatch",
                "action_rows": 16,
                "dispatch_pipeline_s": 0.3,
                "dispatch_pipeline_ewma_s": 0.3,
                "prefetch_lead_rows": 3,
                "preview_queue_depth": 3,
                "full_observation_queue_depth": 2,
                "queue_preview_used": True,
                "replacement_used": True,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dispatches.jsonl"
            path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )

            report = analyze_logs([path])

        dispatches = report["client_dispatches"]
        self.assertEqual(dispatches["events"], 4)
        self.assertEqual(dispatches["action_rows"]["mean"], 28.0)
        self.assertEqual(
            dispatches["dispatch_pipeline_s"],
            {
                "observations": 4,
                "minimum": 0.2,
                "maximum": 0.45,
                "mean": 0.3,
                "median": 0.275,
                "p95_nearest_rank": 0.45,
            },
        )
        self.assertEqual(dispatches["action_interval_s"]["observations"], 3)
        self.assertEqual(
            dispatches["prefetch_lead_rows"]["histogram"],
            {"0": 2, "3": 1, "5": 1},
        )
        self.assertEqual(
            dispatches["queue_preview_used"],
            {"true": 3, "false": 1, "missing_or_invalid": 0},
        )
        self.assertEqual(
            dispatches["replacement_used"],
            {"true": 2, "false": 2, "missing_or_invalid": 0},
        )
        self.assertEqual(
            dispatches["telemetry_coverage"]["action_interval_s"],
            {"events": 3, "of": 4},
        )
        indicators = dispatches["indicators"]
        self.assertEqual(indicators["zero_lead_long_dispatches"], 1)
        self.assertEqual(
            indicators["zero_lead_long_dispatch_minimum_action_rows"], 8
        )
        self.assertEqual(
            indicators["zero_lead_long_dispatch_evidence"][0]["line"], 3
        )
        budget = indicators["latency_budget"]
        self.assertEqual(budget["preview_events"], 3)
        self.assertEqual(budget["evaluated_events"], 2)
        self.assertEqual(budget["unassessable_events"], 1)
        self.assertEqual(budget["approached_events"], 1)
        self.assertEqual(budget["exceeded_events"], 1)
        self.assertEqual(
            [row["classification"] for row in budget["evidence"]],
            ["approached", "exceeded"],
        )
        self.assertEqual(
            budget["evidence"][0]["pipeline_fraction_of_available_time"],
            0.9,
        )
        self.assertEqual(
            budget["evidence"][1]["pipeline_fraction_of_available_time"],
            1.25,
        )
        self.assertIn("not proof", indicators["interpretation"])
        self.assertIn("without action_interval_s", indicators["interpretation"])

    def test_three_distinct_carry_occurrences_do_not_combine_into_a_flag(self) -> None:
        events = [
            {
                "event": "phase_action",
                "phase": "carry",
                "action_kind": "motion",
                "action_rows": 2,
                "geometric_rows_sent": 2,
                "queued_at_observation": 0,
                "motion_call": 1,
            }
            for _ in range(3)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "boundaries.jsonl"
            path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            report = analyze_logs([path])

        self.assertEqual(report["phase_actions"]["short_motion_calls"], 3)
        self.assertEqual(
            report["phase_actions"]["motion_calls_observed_with_zero_queue"], 3
        )
        self.assertFalse(report["indicators"]["possible_queue_starvation"])
        self.assertFalse(report["indicators"]["repeated_micro_corrections"])
        self.assertEqual(
            report["indicators"]["queue_starvation_evidence_occurrences"], []
        )

    def test_source_mutation_during_analysis_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "changing.log"
            source.write_text('{"event":"holding"}\n', encoding="utf-8")
            real_fingerprint = analyzer._fingerprint
            calls = 0

            def fingerprint_then_mutate(path: Path) -> dict[str, object]:
                nonlocal calls
                fingerprint = real_fingerprint(path)
                if calls == 0:
                    path.write_text(
                        path.read_text(encoding="utf-8")
                        + '{"event":"phase_action","phase":"carry"}\n',
                        encoding="utf-8",
                    )
                calls += 1
                return fingerprint

            with mock.patch.object(
                analyzer, "_fingerprint", side_effect=fingerprint_then_mutate
            ):
                with self.assertRaisesRegex(RuntimeError, "source changed"):
                    analyze_logs([source])

    def test_cli_writes_only_to_an_explicit_new_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.log"
            output = root / "report.json"
            source.write_text('{"event":"holding","reason":"done"}\n')

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main([str(source), "--output", str(output)]), 0)

            self.assertEqual(json.loads(stdout.getvalue()), json.loads(output.read_text()))
            original_output = output.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "new path"):
                    main([str(source), "--output", str(output)])
            self.assertEqual(output.read_bytes(), original_output)
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                main([str(source), "--output", str(source)])

    def test_duplicate_sources_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "input.log"
            source.write_text("nothing\n")
            with self.assertRaisesRegex(ValueError, "unique"):
                analyze_logs([source, source])


if __name__ == "__main__":
    unittest.main()
