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
                supervised_run.MOTION_PROFILE_ENV: "conservative",
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
            self.assertEqual(
                os.environ[supervised_run.MOTION_PROFILE_ENV], "smooth"
            )
            self.assertEqual(os.environ[supervised_run.STABLE_OBSERVATIONS_ENV], "3")
            for name in supervised_run._POLICY_TUNING_ENVIRONMENT:
                self.assertNotIn(name, os.environ)
        self.assertEqual(calls, [[]])

    def test_conservative_motion_profile_remains_an_explicit_fallback(self):
        calls: list[list[str]] = []
        supervised_run.main(
            ["hover", "--cube-mm", "10", "--motion-profile", "conservative"],
            delegate=lambda a: calls.append(list(a)),
        )
        self.assertEqual(
            os.environ[supervised_run.MOTION_PROFILE_ENV], "conservative"
        )
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
        self.assertEqual(summary["motion_profile"], "smooth")
        self.assertEqual(summary["nominal_linear_speed_mm_s"], 100.0)
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

    def test_yolo11n_selection_sets_component_profile_before_delegating(self):
        observed: dict[str, str | None] = {}

        def delegate(arguments):
            observed["profile"] = os.environ.get(
                supervised_run.INTEGRATION_PROFILE_ENV
            )
            observed["checkpoint"] = os.environ.get(
                supervised_run.VISION_CHECKPOINT_ENV
            )
            observed["arguments"] = list(arguments)

        with mock.patch.dict(
            os.environ,
            {
                supervised_run.INTEGRATION_PROFILE_ENV: "stale-profile.json",
                supervised_run.VISION_CHECKPOINT_ENV: "stale-checkpoint.pt",
            },
            clear=False,
        ):
            supervised_run.main(
                [
                    "hover",
                    "--cube-mm",
                    "16",
                    "--vision",
                    "yolo11n",
                    "--vision-checkpoint",
                    "seed55-best.pt",
                ],
                delegate=delegate,
            )

        self.assertEqual(
            observed["profile"], str(supervised_run.YOLO11N_PROFILE_PATH)
        )
        self.assertEqual(observed["checkpoint"], "seed55-best.pt")
        self.assertEqual(observed["arguments"], [])

    def test_yolo11n_requires_its_component_checkpoint(self):
        with self.assertRaises(SystemExit):
            supervised_run.main(
                ["hover", "--cube-mm", "16", "--vision", "yolo11n"],
                delegate=lambda _: None,
            )
        with self.assertRaises(SystemExit):
            supervised_run.main(
                ["--vision-checkpoint", "seed55-best.pt"],
                delegate=lambda _: None,
            )

    def test_explicit_baseline_clears_inherited_vision_selection(self):
        calls: list[list[str]] = []
        with mock.patch.dict(
            os.environ,
            {
                supervised_run.INTEGRATION_PROFILE_ENV: "stale-profile.json",
                supervised_run.VISION_CHECKPOINT_ENV: "stale-checkpoint.pt",
            },
            clear=False,
        ):
            supervised_run.main(
                ["--vision", "baseline"],
                delegate=lambda args: calls.append(list(args)),
            )
            self.assertNotIn(supervised_run.INTEGRATION_PROFILE_ENV, os.environ)
            self.assertNotIn(supervised_run.VISION_CHECKPOINT_ENV, os.environ)
        self.assertEqual(calls, [[]])

    def test_no_vision_option_preserves_inherited_profile_workflow(self):
        with mock.patch.dict(
            os.environ,
            {
                supervised_run.INTEGRATION_PROFILE_ENV: "existing-profile.json",
                supervised_run.VISION_CHECKPOINT_ENV: "existing-checkpoint.pt",
            },
            clear=False,
        ):
            supervised_run.main([], delegate=lambda _: None)
            self.assertEqual(
                os.environ[supervised_run.INTEGRATION_PROFILE_ENV],
                "existing-profile.json",
            )
            self.assertEqual(
                os.environ[supervised_run.VISION_CHECKPOINT_ENV],
                "existing-checkpoint.pt",
            )

    def test_yolo11n_dry_run_does_not_mutate_environment_or_delegate(self):
        called = False

        def delegate(_):
            nonlocal called
            called = True

        output = io.StringIO()
        with mock.patch.dict(
            os.environ,
            {supervised_run.INTEGRATION_PROFILE_ENV: "sentinel"},
            clear=False,
        ):
            with contextlib.redirect_stdout(output):
                result = supervised_run.main(
                    [
                        "hover",
                        "--cube-mm",
                        "16",
                        "--vision",
                        "yolo11n",
                        "--vision-checkpoint",
                        "seed55-best.pt",
                        "--dry-run",
                    ],
                    delegate=delegate,
                )
            self.assertEqual(
                os.environ[supervised_run.INTEGRATION_PROFILE_ENV], "sentinel"
            )
        self.assertEqual(result, 0)
        self.assertFalse(called)
        summary = json.loads(output.getvalue())
        self.assertEqual(summary["vision"], "yolo11n")
        self.assertTrue(summary["vision_checkpoint_configured"])
        self.assertNotIn("seed55-best.pt", output.getvalue())


if __name__ == "__main__":
    unittest.main()
