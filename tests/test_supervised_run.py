from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
CELL = REPO / "hardware" / "fanuc_cell"
if str(CELL) not in sys.path:
    sys.path.insert(0, str(CELL))

import supervised_run  # noqa: E402


class SupervisedRunTests(unittest.TestCase):
    def _clean_environment(self):
        return mock.patch.dict(
            os.environ,
            {
                supervised_run.MOTION_ENV: "1",
                supervised_run.DESCENT_ENV: "1",
                supervised_run.DESTINATION_ENV: "stale-value",
                supervised_run.CUBE_EDGE_ENV: "999",
                supervised_run.GRASP_FIT_ENV: "1",
                supervised_run.MAX_ACTIONS_ENV: "8",
                supervised_run.STABLE_OBSERVATIONS_ENV: "1",
                supervised_run._POLICY_TUNING_ENVIRONMENT[0]: "999",
            },
            clear=False,
        )

    def test_plain_launcher_clears_inherited_arming_and_defaults_to_hold(self):
        calls: list[list[str]] = []
        with self._clean_environment():
            result = supervised_run.main([], delegate=lambda a: calls.append(list(a)))
            self.assertEqual(result, 0)
            self.assertNotIn(supervised_run.MOTION_ENV, os.environ)
            self.assertNotIn(supervised_run.DESCENT_ENV, os.environ)
            self.assertEqual(
                os.environ[supervised_run.DESTINATION_ENV], "virtual-table"
            )
            self.assertEqual(os.environ[supervised_run.CUBE_EDGE_ENV], "10")
            self.assertNotIn(supervised_run.GRASP_FIT_ENV, os.environ)
            self.assertEqual(os.environ[supervised_run.MAX_ACTIONS_ENV], "32")
            self.assertEqual(os.environ[supervised_run.STABLE_OBSERVATIONS_ENV], "3")
            for name in supervised_run._POLICY_TUNING_ENVIRONMENT:
                self.assertNotIn(name, os.environ)
        self.assertEqual(calls, [[]])

    def test_hover_requires_explicit_cube_profile_and_never_enables_descent(self):
        with self.assertRaises(SystemExit):
            supervised_run.main(["hover"], delegate=lambda _: None)

        calls: list[list[str]] = []
        with self._clean_environment():
            supervised_run.main(
                ["hover", "--cube-mm", "16"],
                delegate=lambda a: calls.append(list(a)),
            )
            self.assertEqual(os.environ[supervised_run.MOTION_ENV], "1")
            self.assertNotIn(supervised_run.DESCENT_ENV, os.environ)
            self.assertEqual(os.environ[supervised_run.CUBE_EDGE_ENV], "16")
        self.assertEqual(calls, [[]])

    def test_ten_mm_sort_uses_mode_and_profile_without_inert_acknowledgements(self):
        calls: list[list[str]] = []
        supervised_run.main(
            ["sort", "--cube-mm", "10", "--destination", "virtual-table"],
            delegate=lambda a: calls.append(list(a)),
        )
        self.assertEqual(os.environ[supervised_run.MOTION_ENV], "1")
        self.assertEqual(os.environ[supervised_run.DESCENT_ENV], "1")
        self.assertEqual(os.environ[supervised_run.DESTINATION_ENV], "virtual-table")
        self.assertNotIn(supervised_run.GRASP_FIT_ENV, os.environ)
        self.assertEqual(calls, [[]])

    def test_sixteen_mm_sort_requires_behavior_bearing_fit_override(self):
        with self.assertRaises(SystemExit):
            supervised_run.main(
                ["sort", "--cube-mm", "16"], delegate=lambda _: None
            )
        calls: list[list[str]] = []
        supervised_run.main(
            ["sort", "--cube-mm", "16", "--confirm-grasp-fit"],
            delegate=lambda a: calls.append(list(a)),
        )
        self.assertEqual(os.environ[supervised_run.GRASP_FIT_ENV], "1")
        self.assertEqual(calls, [[]])

    def test_dry_run_never_delegates_or_prints_forwarded_secret(self):
        called = False

        def delegate(_):
            nonlocal called
            called = True

        output = io.StringIO()
        with mock.patch.dict(os.environ, {supervised_run.MOTION_ENV: "sentinel"}):
            with contextlib.redirect_stdout(output):
                result = supervised_run.main(
                    [
                        "sort",
                        "--cube-mm",
                        "16",
                        "--destination",
                        "virtual-table",
                        "--token",
                        "do-not-print-this",
                        "--dry-run",
                    ],
                    delegate=delegate,
                )
            self.assertEqual(os.environ[supervised_run.MOTION_ENV], "sentinel")
        self.assertEqual(result, 0)
        self.assertFalse(called)
        self.assertNotIn("do-not-print-this", output.getvalue())
        summary = json.loads(output.getvalue())
        self.assertEqual(summary["cube_edge_mm"], 16)
        self.assertTrue(summary["descent_enabled"])
        self.assertIn("--confirm-grasp-fit", summary["missing_required_inputs"])

    def test_mock_arguments_are_forwarded_without_raw_environment_exports(self):
        calls: list[list[str]] = []
        supervised_run.main(
            [
                "hover",
                "--cube-mm",
                "16",
                "--mock",
                "--trials",
                "1",
                "--trial-s",
                "2",
            ],
            delegate=lambda a: calls.append(list(a)),
        )
        self.assertEqual(
            calls,
            [["--mock", "--trials", "1", "--trial-s", "2.0"]],
        )

    def test_only_allowlisted_arguments_are_forwarded(self):
        calls: list[list[str]] = []
        supervised_run.main(
            [
                "--url",
                "https://example.invalid",
                "--checkpoint",
                "model.bin",
                "--ca",
                "ca.pem",
            ],
            delegate=lambda a: calls.append(list(a)),
        )
        self.assertEqual(
            calls,
            [[
                "--url",
                "https://example.invalid",
                "--checkpoint",
                "model.bin",
                "--ca",
                "ca.pem",
            ]],
        )


if __name__ == "__main__":
    unittest.main()
