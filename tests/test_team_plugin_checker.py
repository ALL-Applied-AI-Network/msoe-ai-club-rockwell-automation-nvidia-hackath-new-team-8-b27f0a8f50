from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_team_plugin.py"
SPEC = importlib.util.spec_from_file_location("check_team_plugin", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


def _git_result(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        ["git"], returncode, stdout=stdout, stderr=stderr
    )


class TeamPluginCheckerTests(unittest.TestCase):
    def test_default_profiles_follow_named_team_branches(self):
        self.assertEqual(
            CHECKER.default_profile_for_branch("computer-vision"),
            Path("team_plugins/computer_vision/profile.example.json"),
        )
        self.assertEqual(
            CHECKER.default_profile_for_branch("grasping"),
            Path("team_plugins/grasping/profile.example.json"),
        )
        self.assertEqual(
            CHECKER.default_profile_for_branch("main"),
            Path("team_plugins/profiles/baseline_plugins.json"),
        )
        self.assertEqual(
            CHECKER.default_profile_for_branch(None),
            Path("team_plugins/profiles/baseline_plugins.json"),
        )

    def test_team_branch_fails_when_origin_main_is_not_an_ancestor(self):
        with mock.patch.object(
            CHECKER, "current_branch", return_value="computer-vision"
        ), mock.patch.object(
            CHECKER,
            "_run_git",
            side_effect=(_git_result(0), _git_result(1)),
        ):
            result = CHECKER.inspect_branch_sync(REPO_ROOT)
        self.assertEqual(result.status, "fail")
        self.assertIn("behind origin/main", result.message)

    def test_missing_origin_main_warns_without_fetching(self):
        with mock.patch.object(
            CHECKER, "current_branch", return_value="grasping"
        ), mock.patch.object(
            CHECKER, "_run_git", return_value=_git_result(1)
        ):
            result = CHECKER.inspect_branch_sync(REPO_ROOT)
        self.assertEqual(result.status, "warn")
        self.assertIn("git fetch origin", result.message)

    def test_dirty_tree_fails_by_default_and_allow_dirty_is_explicit(self):
        with mock.patch.object(
            CHECKER,
            "_run_git",
            return_value=_git_result(0, " M team_plugins/example.py\n"),
        ):
            blocked = CHECKER.inspect_worktree(REPO_ROOT)
            development = CHECKER.inspect_worktree(
                REPO_ROOT, allow_dirty=True
            )
        self.assertEqual(blocked.status, "fail")
        self.assertIn("--allow-dirty", blocked.message)
        self.assertEqual(development.status, "warn")
        self.assertIn("development", development.message)
        self.assertTrue(
            CHECKER._parser().parse_args(["--allow-dirty"]).allow_dirty
        )

    def test_unpushed_commits_are_a_visible_warning_not_a_failure(self):
        with mock.patch.object(
            CHECKER, "current_branch", return_value="computer-vision"
        ), mock.patch.object(
            CHECKER,
            "_run_git",
            side_effect=(
                _git_result(0, "origin/computer-vision\n"),
                _git_result(0, "2\n"),
            ),
        ):
            result = CHECKER.inspect_unpushed_commits(REPO_ROOT)
        self.assertEqual(result.status, "warn")
        self.assertIn("2 commits", result.message)
        self.assertIn("push after checks", result.message)

    def test_test_subprocess_environment_clears_all_rockwell_variables(self):
        profile = REPO_ROOT / "team_plugins/profiles/baseline_plugins.json"
        base = {
            CHECKER.PROFILE_ENV: "stale-profile.json",
            "ROCKWELL_ENABLE_PHYSICAL_MOTION": "1",
            "ROCKWELL_ENABLE_PHYSICAL_DESCENT": "1",
            "ROCKWELL_CUSTOM_TEST_VALUE": "must-not-leak",
            "PYTHONPATH": "existing-path",
            "UNCHANGED": "yes",
        }
        active = CHECKER._test_environment(
            profile, profile_active=True, base=base
        )
        baseline = CHECKER._test_environment(
            profile, profile_active=False, base=base
        )
        self.assertEqual(active[CHECKER.PROFILE_ENV], str(profile))
        self.assertEqual(
            sorted(key for key in active if key.startswith("ROCKWELL_")),
            [CHECKER.PROFILE_ENV],
        )
        self.assertFalse(
            any(key.startswith("ROCKWELL_") for key in baseline)
        )
        self.assertEqual(active["UNCHANGED"], "yes")
        self.assertEqual(
            active["PYTHONPATH"].split(os.pathsep)[0], str(REPO_ROOT)
        )

    def test_test_plan_covers_both_components_and_profile_unset_baseline(self):
        profile = SimpleNamespace(vision=object(), grasp=object())
        phases = CHECKER.build_test_phases(profile)
        patterns = {phase.pattern for phase in phases}
        self.assertIn("test_integration_plugins.py", patterns)
        self.assertIn("test_team_handoff.py", patterns)
        self.assertIn("test_perception_integration.py", patterns)
        self.assertIn("test_grasping_integration.py", patterns)
        self.assertEqual(phases[-1].label, "complete profile-unset baseline")
        self.assertFalse(phases[-1].profile_active)

    def test_checked_in_combined_profile_and_factories_validate(self):
        path = CHECKER.resolve_profile(
            "team_plugins/profiles/baseline_plugins.json"
        )
        profile = CHECKER.validate_profile(path)
        self.assertEqual(profile.profile_id, "team-baseline-plugins-v1")
        self.assertIsNotNone(profile.vision)
        self.assertIsNotNone(profile.grasp)

    def test_every_checked_in_team_profile_and_factory_validates(self):
        profiles = sorted(
            path
            for path in (REPO_ROOT / "team_plugins").rglob("*.json")
            if path.parent.name == "profiles" or "profile" in path.name.lower()
        )
        self.assertGreaterEqual(len(profiles), 3)
        for path in profiles:
            with self.subTest(profile=path.relative_to(REPO_ROOT)):
                CHECKER.validate_profile(path)


if __name__ == "__main__":
    unittest.main()
