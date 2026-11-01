"""Launch the real-cell client with explicit, process-local safety settings.

This wrapper exists because shell exports persist and multi-line shell commands
are easy to mistype.  It clears every policy arming variable in this process,
then sets only the values implied by the selected mode.  It does not replace
the cell operator, controller limits, or emergency-stop procedure.

Examples (run from ``hardware/fanuc_cell``)::

    uv run supervised_run.py --dry-run
    uv run supervised_run.py hover --cube-mm 10
    uv run supervised_run.py hover --cube-mm 16 --mock --trials 1

The ``sort`` mode enables physical descent. Do not use it until the hover-only
test and mechanical fit check have passed with the assigned operator.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Callable, Sequence


MOTION_ENV = "ROCKWELL_ENABLE_PHYSICAL_MOTION"
DESCENT_ENV = "ROCKWELL_ENABLE_PHYSICAL_DESCENT"
DESTINATION_ENV = "ROCKWELL_DESTINATION_MODE"
CUBE_EDGE_ENV = "ROCKWELL_VISION_CUBE_EDGE_MM"
GRASP_FIT_ENV = "ROCKWELL_CONFIRM_CUBE_GRASP_FIT"
MAX_ACTIONS_ENV = "ROCKWELL_MAX_ACTIONS_PER_ACT"
STABLE_OBSERVATIONS_ENV = "ROCKWELL_INITIAL_STABLE_OBSERVATIONS"
MOTION_PROFILE_ENV = "ROCKWELL_HARDWARE_MOTION_PROFILE"
INTEGRATION_PROFILE_ENV = "ROCKWELL_INTEGRATION_PROFILE"
VISION_CHECKPOINT_ENV = "ROCKWELL_VISION_CHECKPOINT"
GRASP_CHECKPOINT_ENV = "ROCKWELL_GRASP_CHECKPOINT"
BASELINE_PROFILE_PATH = (
    Path(__file__).resolve().parents[2]
    / "team_plugins"
    / "profiles"
    / "baseline_plugins.json"
)
YOLO11N_PROFILE_PATH = (
    Path(__file__).resolve().parents[2]
    / "team_plugins"
    / "profiles"
    / "yolo11n_baseline_grasp.v1.json"
)
_POLICY_TUNING_ENVIRONMENT = (
    "ROCKWELL_MAX_OBSERVATION_AGE_S",
    "ROCKWELL_MAX_TARGET_MOTION_M",
    "ROCKWELL_MAX_TARGET_YAW_MOTION_DEG",
    "ROCKWELL_MIN_HARDWARE_CENTRE_CLEARANCE_M",
    "ROCKWELL_MIN_DETECTION_CONFIDENCE",
)

_CONTROLLED_ENVIRONMENT = (
    MOTION_ENV,
    DESCENT_ENV,
    DESTINATION_ENV,
    CUBE_EDGE_ENV,
    GRASP_FIT_ENV,
    MAX_ACTIONS_ENV,
    STABLE_OBSERVATIONS_ENV,
    MOTION_PROFILE_ENV,
    *_POLICY_TUNING_ENVIRONMENT,
)


@dataclass(frozen=True)
class LaunchSettings:
    """Non-secret policy settings selected by this launcher."""

    mode: str
    destination: str
    cube_edge_mm: int
    max_actions_per_act: int
    motion_profile: str
    vision: str | None = None
    vision_checkpoint: str | None = None
    grasp: str | None = None
    grasp_checkpoint: str | None = None
    grasp_checkpoint_sha256: str | None = None

    @property
    def components_explicit(self) -> bool:
        return self.vision is not None or self.grasp is not None

    @property
    def effective_vision(self) -> str:
        return self.vision or "baseline"

    @property
    def effective_grasp(self) -> str:
        return self.grasp or "baseline"

    @property
    def motion_enabled(self) -> bool:
        return self.mode in {"hover", "sort"}

    @property
    def descent_enabled(self) -> bool:
        return self.mode == "sort"

    def environment(self, *, grasp_fit_confirmed: bool = False) -> dict[str, str]:
        values = {
            DESTINATION_ENV: self.destination,
            CUBE_EDGE_ENV: str(self.cube_edge_mm),
            MAX_ACTIONS_ENV: str(self.max_actions_per_act),
            MOTION_PROFILE_ENV: self.motion_profile,
            # Do not let an old debugging export weaken the live default.
            STABLE_OBSERVATIONS_ENV: "3",
        }
        if self.motion_enabled:
            values[MOTION_ENV] = "1"
        if self.descent_enabled:
            values[DESCENT_ENV] = "1"
        if grasp_fit_confirmed:
            values[GRASP_FIT_ENV] = "1"
        return values

    def public_summary(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "motion_enabled": self.motion_enabled,
            "descent_enabled": self.descent_enabled,
            "destination_mode": self.destination,
            "cube_edge_mm": self.cube_edge_mm,
            "max_actions_per_act": self.max_actions_per_act,
            "motion_profile": self.motion_profile,
            "nominal_linear_speed_mm_s": (
                100.0 if self.motion_profile == "smooth" else 50.0
            ),
            "vision": (
                self.effective_vision if self.components_explicit else "unchanged"
            ),
            "vision_checkpoint_configured": self.vision_checkpoint is not None,
            "grasp": (
                self.effective_grasp if self.components_explicit else "unchanged"
            ),
            "grasp_checkpoint_configured": self.grasp_checkpoint is not None,
            "combined_profile_selected": self.components_explicit,
            "scope": "process-local; inherited ROCKWELL_* arming flags cleared",
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("hold", "hover", "sort"),
        default="hold",
        help=(
            "hold (default): repeat measured pose only; hover: allow motion but "
            "stop before descent; sort: allow the complete pick/place sequence"
        ),
    )
    parser.add_argument(
        "--destination",
        choices=("bins", "virtual-table"),
        default="virtual-table",
        help=(
            "virtual-table (default) uses fixed XY zones and a near-table "
            "set-down height; bins must be selected explicitly and uses the "
            "configured physical-bin release height"
        ),
    )
    parser.add_argument(
        "--cube-mm",
        type=int,
        choices=(10, 16),
        help=(
            "operator-confirmed physical cube edge used by the hardware camera "
            "profile; required for hover/sort and does not change the preserved "
            "Isaac task geometry"
        ),
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        choices=range(1, 101),
        default=32,
        metavar="1..100",
        help=(
            "maximum bounded TCP rows returned by one Policy.act call "
            "(default: 32; the official protocol limit is 100)"
        ),
    )
    parser.add_argument(
        "--motion-profile",
        choices=("smooth", "conservative"),
        default="smooth",
        help=(
            "smooth (default): hardware-only 10 mm / 0.1 s rows and 8-degree "
            "angular rows; conservative: the prior 5 mm / 5-degree profile. "
            "Neither option changes Isaac trajectories"
        ),
    )
    parser.add_argument(
        "--vision",
        choices=("baseline", "yolo11n"),
        help=(
            "select the hardware vision provider for this process: baseline "
            "explicitly selects the existing HSV/geometry path; yolo11n uses "
            "the checked-in neural-proposal profile. With no option, preserve "
            "the existing ROCKWELL_INTEGRATION_PROFILE behavior"
        ),
    )
    parser.add_argument(
        "--vision-checkpoint",
        help=(
            "external YOLO11n .pt checkpoint; required with --vision yolo11n "
            "and never forwarded to the grasp provider"
        ),
    )
    parser.add_argument(
        "--grasp",
        choices=("baseline", "multigrasp"),
        help=(
            "select the grasp provider for this process: baseline preserves "
            "the existing single grasp; multigrasp generates feasible cube "
            "candidates and ranks them with a hash-pinned logistic model. "
            "If either component is selected, an omitted component defaults "
            "to baseline"
        ),
    )
    parser.add_argument(
        "--grasp-checkpoint",
        help=(
            "absolute path to an external strict-JSON grasp-success model; required with "
            "--grasp multigrasp and kept separate from the vision checkpoint"
        ),
    )
    parser.add_argument(
        "--grasp-checkpoint-sha256",
        help=(
            "independently recorded SHA-256 of --grasp-checkpoint; required "
            "with --grasp multigrasp"
        ),
    )
    parser.add_argument(
        "--confirm-grasp-fit",
        action="store_true",
        help=(
            "enable 16 mm descent after the operator has physically checked that "
            "the selected cube fits the installed hooks; not needed for hover"
        ),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="forward to the official localhost-only mock; never contacts the robot",
    )
    parser.add_argument("--trials", type=int, help="with --mock: rollout count")
    parser.add_argument("--trial-s", type=float, help="with --mock: seconds per rollout")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print non-secret effective settings and exit without contacting the cell",
    )
    parser.add_argument("--url", help="forward a non-default cell URL to run.py")
    parser.add_argument("--token", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", help="forward a policy checkpoint path to run.py")
    parser.add_argument("--ca", help="forward a CA certificate path to run.py")
    return parser


def _settings(args: argparse.Namespace) -> LaunchSettings:
    return LaunchSettings(
        mode=args.mode,
        destination=args.destination,
        cube_edge_mm=args.cube_mm if args.cube_mm is not None else 10,
        max_actions_per_act=args.max_actions,
        motion_profile=args.motion_profile,
        vision=args.vision,
        vision_checkpoint=args.vision_checkpoint,
        grasp=args.grasp,
        grasp_checkpoint=args.grasp_checkpoint,
        grasp_checkpoint_sha256=args.grasp_checkpoint_sha256,
    )


def _required_inputs(args: argparse.Namespace) -> list[str]:
    missing: list[str] = []
    if args.mode in {"hover", "sort"}:
        if args.cube_mm is None:
            missing.append("--cube-mm {10|16}")
    if args.mode == "sort" and args.cube_mm == 16 and not args.confirm_grasp_fit:
        missing.append("--confirm-grasp-fit")
    if args.mock and args.mode == "hold":
        missing.append("mode hover (the official mock has no initial TCP pose)")
    if not args.mock and (args.trials is not None or args.trial_s is not None):
        missing.append("--mock (required with --trials/--trial-s)")
    if args.vision == "yolo11n" and args.vision_checkpoint is None:
        missing.append("--vision-checkpoint PATH")
    if args.vision != "yolo11n" and args.vision_checkpoint is not None:
        missing.append("--vision yolo11n (required with --vision-checkpoint)")
    if args.grasp == "multigrasp":
        if args.cube_mm not in {None, 10}:
            missing.append("--cube-mm 10 (multigrasp v1 geometry is 10 mm)")
        if args.grasp_checkpoint is None:
            missing.append("--grasp-checkpoint PATH")
        digest = args.grasp_checkpoint_sha256
        if digest is None:
            missing.append("--grasp-checkpoint-sha256 HEX")
        elif len(digest.strip()) != 64 or any(
            character not in "0123456789abcdefABCDEF"
            for character in digest.strip()
        ):
            missing.append("a 64-character hexadecimal --grasp-checkpoint-sha256")
    elif args.grasp_checkpoint is not None or args.grasp_checkpoint_sha256 is not None:
        missing.append("--grasp multigrasp (required with its checkpoint options)")
    if args.checkpoint is not None and (
        args.vision == "yolo11n" or args.grasp == "multigrasp"
    ):
        missing.append(
            "remove ambiguous --checkpoint; use the component checkpoint options"
        )
    return missing


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_profile_path(settings: LaunchSettings) -> Path:
    return (
        YOLO11N_PROFILE_PATH
        if settings.effective_vision == "yolo11n"
        else BASELINE_PROFILE_PATH
    )


@contextmanager
def _resolved_profile(settings: LaunchSettings):
    """Yield a complete profile while keeping generated bytes alive for run.py."""

    if not settings.components_explicit:
        yield None
        return

    base_path = _base_profile_path(settings).resolve(strict=True)
    if settings.effective_grasp == "baseline":
        yield base_path
        return

    checkpoint = Path(str(settings.grasp_checkpoint)).expanduser()
    if not checkpoint.is_absolute():
        raise ValueError("--grasp-checkpoint must be an absolute path")
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError(f"grasp checkpoint is not a file: {checkpoint}")
    expected_sha256 = str(settings.grasp_checkpoint_sha256).strip().lower()
    actual_sha256 = _checkpoint_sha256(checkpoint)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "grasp checkpoint SHA-256 mismatch: expected "
            f"{expected_sha256}, received {actual_sha256}"
        )

    payload = json.loads(base_path.read_text(encoding="utf-8"))
    payload["profile_id"] = (
        f"team8-{settings.effective_vision}-vision-multigrasp-v1-"
        f"{expected_sha256[:12]}"
    )
    payload["grasp"] = {
        "factory": "team_plugins.grasping.multigrasp_pipeline:create",
        "options": {
            "scorer": "checkpoint_logistic",
            "checkpoint_sha256": expected_sha256,
        },
    }
    profile_bytes = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with tempfile.TemporaryDirectory(prefix="team8-integration-profile-") as directory:
        generated = Path(directory) / "combined_profile.json"
        generated.write_bytes(profile_bytes)
        yield generated


def _apply_environment(
    settings: LaunchSettings,
    *,
    grasp_fit_confirmed: bool = False,
    profile_path: Path | None = None,
) -> None:
    # Delete inherited values first.  In particular, a previous interactive
    # ``export ROCKWELL_ENABLE_PHYSICAL_MOTION=1`` must not arm a plain run.
    for name in _CONTROLLED_ENVIRONMENT:
        os.environ.pop(name, None)
    os.environ.update(
        settings.environment(grasp_fit_confirmed=grasp_fit_confirmed)
    )
    # Component selection is deliberately separate from physical-motion
    # interlocks. With no component option, retain the established profile-env
    # workflow byte-for-byte. Any explicit selection resolves a complete
    # vision+grasp profile so one choice cannot silently disable the other.
    if not settings.components_explicit:
        return
    if profile_path is None:
        raise ValueError("explicit component selection requires a profile")
    os.environ.pop(VISION_CHECKPOINT_ENV, None)
    os.environ.pop(GRASP_CHECKPOINT_ENV, None)
    os.environ[INTEGRATION_PROFILE_ENV] = str(profile_path)
    if settings.effective_vision == "yolo11n":
        if settings.vision_checkpoint is None:  # guarded by _required_inputs
            raise ValueError("YOLO11n vision requires a checkpoint")
        os.environ[VISION_CHECKPOINT_ENV] = str(
            Path(settings.vision_checkpoint).expanduser().resolve()
        )
    if settings.effective_grasp == "multigrasp":
        if settings.grasp_checkpoint is None:  # guarded by _required_inputs
            raise ValueError("multigrasp requires a checkpoint")
        os.environ[GRASP_CHECKPOINT_ENV] = str(
            Path(settings.grasp_checkpoint).expanduser().resolve()
        )


def _forwarded_arguments(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = []
    if args.mock:
        forwarded.append("--mock")
        if args.trials is not None:
            forwarded.extend(("--trials", str(args.trials)))
        if args.trial_s is not None:
            forwarded.extend(("--trial-s", str(args.trial_s)))
    for option, value in (
        ("--url", args.url),
        ("--token", args.token),
        ("--checkpoint", args.checkpoint),
        ("--ca", args.ca),
    ):
        if value is not None:
            forwarded.extend((option, value))
    return forwarded


def _delegate_to_official_client(arguments: Sequence[str]) -> None:
    # Import only after the controlled environment is installed: Policy is
    # imported by run.py and reads these values when instantiated/reset.
    import run as official_run

    previous = sys.argv
    try:
        sys.argv = [str(official_run.HERE / "run.py"), *arguments]
        official_run.main()
    finally:
        sys.argv = previous


def main(
    argv: Sequence[str] | None = None,
    *,
    delegate: Callable[[Sequence[str]], None] | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    settings = _settings(args)
    missing = _required_inputs(args)
    if args.dry_run:
        summary = settings.public_summary()
        summary["mock"] = args.mock
        summary["missing_required_inputs"] = missing
        print(json.dumps(summary, sort_keys=True))
        return 0
    if missing:
        parser.error(
            f"invalid launch configuration: "
            + ", ".join(missing)
        )

    with ExitStack() as stack:
        try:
            profile_path = stack.enter_context(_resolved_profile(settings))
            _apply_environment(
                settings,
                grasp_fit_confirmed=args.confirm_grasp_fit,
                profile_path=profile_path,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"invalid integration configuration: {exc}")
        print(
            json.dumps(
                {"event": "supervised_launch", **settings.public_summary()}
            )
        )
        (delegate or _delegate_to_official_client)(_forwarded_arguments(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
