#!/usr/bin/env python3
"""Run the CPU-only checks required before a team-plugin handoff.

The checker deliberately does not fetch Git refs, start Isaac, contact the
physical-cell service, or run model inference on external data. Its branch
check uses the existing local ``origin/main`` ref, so teammates should run
``git fetch origin`` immediately before this command when network access is
available.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROFILE_ENV = "ROCKWELL_INTEGRATION_PROFILE"
TEAM_BRANCHES = frozenset({"computer-vision", "grasping"})
DEFAULT_PROFILES = {
    "computer-vision": Path("team_plugins/computer_vision/profile.example.json"),
    "grasping": Path("team_plugins/grasping/profile.example.json"),
    "main": Path("team_plugins/profiles/baseline_plugins.json"),
}


class CheckFailure(RuntimeError):
    """Raised when a handoff prerequisite or test phase fails."""


@dataclass(frozen=True)
class BranchSyncResult:
    branch: str | None
    status: str
    message: str

    def __post_init__(self) -> None:
        if self.status not in {"pass", "warn", "fail", "skip"}:
            raise ValueError(f"invalid branch-sync status {self.status!r}")


@dataclass(frozen=True)
class CheckResult:
    status: str
    message: str

    def __post_init__(self) -> None:
        if self.status not in {"pass", "warn", "fail", "skip"}:
            raise ValueError(f"invalid check status {self.status!r}")


@dataclass(frozen=True)
class TestPhase:
    label: str
    start_directory: Path
    pattern: str
    profile_active: bool
    optional: bool = False


def _run_git(
    arguments: Sequence[str],
    *,
    repository: Path = REPOSITORY_ROOT,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def current_branch(repository: Path = REPOSITORY_ROOT) -> str | None:
    """Return the checked-out branch, or ``None`` outside/on detached Git."""

    result = _run_git(("branch", "--show-current"), repository=repository)
    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    return branch or None


def inspect_worktree(
    repository: Path = REPOSITORY_ROOT,
    *,
    allow_dirty: bool = False,
) -> CheckResult:
    """Require a clean handoff while allowing an explicit development escape."""

    result = _run_git(
        ("status", "--porcelain=v1", "--untracked-files=normal"),
        repository=repository,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "git status failed"
        return CheckResult("fail", f"could not inspect working tree: {detail}")
    changes = tuple(line for line in result.stdout.splitlines() if line.strip())
    if not changes:
        return CheckResult("pass", "working tree is clean")
    count = len(changes)
    if allow_dirty:
        return CheckResult(
            "warn",
            (
                f"working tree has {count} changed/untracked entr"
                f"{'y' if count == 1 else 'ies'}; allowed only for development"
            ),
        )
    return CheckResult(
        "fail",
        (
            f"working tree has {count} changed/untracked entr"
            f"{'y' if count == 1 else 'ies'}; hand off only committed work "
            "(use --allow-dirty only for an in-progress development check)"
        ),
    )


def inspect_unpushed_commits(
    repository: Path = REPOSITORY_ROOT,
) -> CheckResult:
    """Warn, but do not fail, when HEAD is ahead of its configured upstream."""

    branch = current_branch(repository)
    if branch is None:
        return CheckResult(
            "skip", "branch is unavailable or detached; upstream was not compared"
        )
    upstream = _run_git(
        (
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ),
        repository=repository,
    )
    if upstream.returncode != 0:
        return CheckResult(
            "skip", f"branch {branch!r} has no locally available upstream"
        )
    upstream_name = upstream.stdout.strip() or "configured upstream"
    ahead = _run_git(
        ("rev-list", "--count", "@{upstream}..HEAD"),
        repository=repository,
    )
    if ahead.returncode != 0:
        detail = ahead.stderr.strip() or "git rev-list failed"
        return CheckResult("warn", f"could not compare upstream commits: {detail}")
    try:
        count = int(ahead.stdout.strip())
    except ValueError:
        return CheckResult(
            "warn", f"could not parse upstream commit count: {ahead.stdout.strip()!r}"
        )
    if count > 0:
        return CheckResult(
            "warn",
            (
                f"HEAD has {count} commit{'s' if count != 1 else ''} not on "
                f"{upstream_name}; this is expected before push, but push after checks"
            ),
        )
    return CheckResult("pass", f"HEAD has no unpushed commits relative to {upstream_name}")


def inspect_branch_sync(repository: Path = REPOSITORY_ROOT) -> BranchSyncResult:
    """Check a team branch against the already-fetched ``origin/main`` ref.

    Missing Git metadata or a missing remote ref is a warning because this
    checker intentionally performs no network operation. A known team branch
    that is provably behind ``origin/main`` is a failure.
    """

    branch = current_branch(repository)
    if branch is None:
        return BranchSyncResult(
            None,
            "warn",
            "Git branch is unavailable or detached; branch freshness was not checked.",
        )
    if branch not in TEAM_BRANCHES:
        return BranchSyncResult(
            branch,
            "skip",
            f"branch {branch!r} is not a team branch; no origin/main gate applies",
        )

    remote_ref = _run_git(
        ("show-ref", "--verify", "--quiet", "refs/remotes/origin/main"),
        repository=repository,
    )
    if remote_ref.returncode != 0:
        return BranchSyncResult(
            branch,
            "warn",
            "origin/main is unavailable locally; run 'git fetch origin' and rerun",
        )

    ancestor = _run_git(
        ("merge-base", "--is-ancestor", "origin/main", "HEAD"),
        repository=repository,
    )
    if ancestor.returncode == 0:
        return BranchSyncResult(
            branch,
            "pass",
            f"{branch} contains the locally fetched origin/main",
        )
    if ancestor.returncode == 1:
        return BranchSyncResult(
            branch,
            "fail",
            (
                f"{branch} is behind origin/main; merge origin/main into the clean "
                "team branch before handoff"
            ),
        )
    detail = ancestor.stderr.strip() or "git merge-base failed"
    return BranchSyncResult(branch, "warn", f"could not check origin/main: {detail}")


def default_profile_for_branch(branch: str | None) -> Path:
    """Choose the checked-in smoke-test profile for the current role."""

    return DEFAULT_PROFILES.get(
        branch or "",
        Path("team_plugins/profiles/baseline_plugins.json"),
    )


def resolve_profile(path_value: str | os.PathLike[str]) -> Path:
    path = Path(path_value).expanduser()
    resolved = path if path.is_absolute() else REPOSITORY_ROOT / path
    try:
        resolved = resolved.resolve(strict=True)
    except OSError as exc:
        raise CheckFailure(f"cannot resolve profile {path}: {exc}") from exc
    if not resolved.is_file():
        raise CheckFailure(f"profile is not a file: {resolved}")
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise CheckFailure(
            "profile must be a reviewed file inside the repository"
        ) from exc
    return resolved


def validate_profile(profile_path: Path):
    """Validate strict JSON and import every explicitly selected factory."""

    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from sorting_core.integration_plugins import (  # imported after path setup
        IntegrationLoadError,
        load_factory,
        load_integration_profile,
    )

    try:
        profile = load_integration_profile(profile_path)
        if profile.vision is not None:
            load_factory(profile.vision.factory, "vision")
        if profile.grasp is not None:
            load_factory(profile.grasp.factory, "grasp")
    except IntegrationLoadError as exc:
        raise CheckFailure(f"integration profile is invalid: {exc}") from exc
    return profile


def build_test_phases(profile) -> tuple[TestPhase, ...]:
    """Return profile-focused phases followed by the profile-unset baseline."""

    phases: list[TestPhase] = [
        TestPhase(
            "integration profile contracts",
            Path("tests"),
            "test_integration_plugins.py",
            True,
        ),
        TestPhase(
            "golden vision-to-grasp handoff fixtures",
            Path("tests"),
            "test_team_handoff.py",
            True,
        ),
    ]
    if profile.vision is not None:
        phases.extend(
            (
                TestPhase(
                    "shared perception contracts",
                    Path("tests"),
                    "test_perception_integration.py",
                    True,
                ),
                TestPhase(
                    "vision team-owned tests",
                    Path("team_plugins/computer_vision/tests"),
                    "test*.py",
                    True,
                    optional=True,
                ),
            )
        )
    if profile.grasp is not None:
        phases.extend(
            (
                TestPhase(
                    "shared grasp contracts",
                    Path("tests"),
                    "test_grasping_integration.py",
                    True,
                ),
                TestPhase(
                    "grasp team-owned tests",
                    Path("team_plugins/grasping/tests"),
                    "test*.py",
                    True,
                    optional=True,
                ),
            )
        )
    phases.append(
        TestPhase(
            "complete profile-unset baseline",
            Path("tests"),
            "test*.py",
            False,
        )
    )
    return tuple(phases)


def _test_environment(
    profile_path: Path,
    *,
    profile_active: bool,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    for key in tuple(environment):
        if key.startswith("ROCKWELL_"):
            environment.pop(key)
    if profile_active:
        environment[PROFILE_ENV] = str(profile_path)
    existing = environment.get("PYTHONPATH", "").strip()
    paths = [str(REPOSITORY_ROOT)]
    if existing:
        paths.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    return environment


def run_test_phase(phase: TestPhase, profile_path: Path) -> None:
    start_directory = REPOSITORY_ROOT / phase.start_directory
    if not start_directory.is_dir():
        if phase.optional:
            print(f"[skip] {phase.label}: {phase.start_directory} does not exist")
            return
        raise CheckFailure(
            f"required test directory is missing for {phase.label}: {start_directory}"
        )
    has_tests = any(start_directory.glob(phase.pattern))
    if not has_tests:
        if phase.optional:
            print(
                f"[skip] {phase.label}: no files match "
                f"{phase.start_directory}/{phase.pattern}"
            )
            return
        raise CheckFailure(
            f"required tests are missing for {phase.label}: "
            f"{phase.start_directory}/{phase.pattern}"
        )

    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(phase.start_directory),
        "-p",
        phase.pattern,
        "-v",
    ]
    state = "profile selected" if phase.profile_active else "profile unset"
    print(f"\n== {phase.label} ({state}) ==", flush=True)
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=_test_environment(
            profile_path,
            profile_active=phase.profile_active,
        ),
        check=False,
    )
    if result.returncode != 0:
        raise CheckFailure(
            f"{phase.label} failed with exit code {result.returncode}"
        )


def run_test_plan(phases: Iterable[TestPhase], profile_path: Path) -> None:
    for phase in phases:
        run_test_phase(phase, profile_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one team integration profile, run focused CPU contracts, "
            "then prove the profile-unset baseline."
        )
    )
    parser.add_argument(
        "--profile",
        help=(
            "profile JSON relative to the repository root; defaults to the "
            "current branch's checked-in example"
        ),
    )
    parser.add_argument(
        "--skip-branch-sync",
        action="store_true",
        help="skip the local origin/main ancestry gate (intended for detached CI)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "allow changed/untracked files for an in-progress development run; "
            "never use for team handoff"
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate profile/factory imports without running tests",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    branch = current_branch()
    profile_value = arguments.profile or default_profile_for_branch(branch)

    try:
        worktree_result = inspect_worktree(allow_dirty=arguments.allow_dirty)
        print(f"[{worktree_result.status}] worktree: {worktree_result.message}")
        if worktree_result.status == "fail":
            raise CheckFailure(worktree_result.message)

        if not arguments.skip_branch_sync:
            branch_result = inspect_branch_sync()
            print(f"[{branch_result.status}] branch: {branch_result.message}")
            if branch_result.status == "fail":
                raise CheckFailure(branch_result.message)
        else:
            print("[skip] branch: local origin/main ancestry check disabled")

        upstream_result = inspect_unpushed_commits()
        print(f"[{upstream_result.status}] upstream: {upstream_result.message}")

        profile_path = resolve_profile(profile_value)
        profile = validate_profile(profile_path)
        relative = profile_path.relative_to(REPOSITORY_ROOT)
        components = ", ".join(
            name
            for name, selected in (
                ("vision", profile.vision),
                ("grasp", profile.grasp),
            )
            if selected is not None
        )
        print(
            f"[pass] profile: {profile.profile_id} ({components}) "
            f"sha256={profile.source_sha256} path={relative}"
        )
        print("[pass] factories: every selected module:callable imported")
        if arguments.profile is None:
            print(
                "[note] using the branch's example profile; pass --profile "
                "with your versioned profile to test team work"
            )
        if arguments.validate_only:
            print("Team plugin profile validation passed.")
            return 0

        run_test_plan(build_test_phases(profile), profile_path)
    except (CheckFailure, ValueError) as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        return 1

    print("\nTeam plugin handoff checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
