"""Launch the real-cell client with explicit, process-local safety settings.

This wrapper exists because shell exports persist and multi-line shell commands
are easy to mistype.  It clears every policy arming variable in this process,
then sets only the values implied by the selected mode.  It does not replace
the cell operator, controller limits, or emergency-stop procedure.

Examples (run from ``hardware/fanuc_cell``)::

    uv run supervised_run.py --dry-run
    uv run supervised_run.py hover --operator-present --cube-mm 10

The ``sort`` mode additionally requires explicit acknowledgements printed by
``--help``.  Do not use it until the hover-only test and mechanical fit check
have passed under the assigned operator's supervision.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import sys
from typing import Callable, Sequence


MOTION_ENV = "ROCKWELL_ENABLE_PHYSICAL_MOTION"
DESCENT_ENV = "ROCKWELL_ENABLE_PHYSICAL_DESCENT"
DESTINATION_ENV = "ROCKWELL_DESTINATION_MODE"
CUBE_EDGE_ENV = "ROCKWELL_VISION_CUBE_EDGE_MM"
GRASP_FIT_ENV = "ROCKWELL_CONFIRM_CUBE_GRASP_FIT"
MAX_ACTIONS_ENV = "ROCKWELL_MAX_ACTIONS_PER_ACT"
STABLE_OBSERVATIONS_ENV = "ROCKWELL_INITIAL_STABLE_OBSERVATIONS"

_CONTROLLED_ENVIRONMENT = (
    MOTION_ENV,
    DESCENT_ENV,
    DESTINATION_ENV,
    CUBE_EDGE_ENV,
    GRASP_FIT_ENV,
    MAX_ACTIONS_ENV,
    STABLE_OBSERVATIONS_ENV,
)


@dataclass(frozen=True)
class LaunchSettings:
    """Non-secret policy settings selected by this launcher."""

    mode: str
    destination: str
    cube_edge_mm: int
    max_actions_per_act: int

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
        choices=range(1, 9),
        default=1,
        metavar="1..8",
        help="maximum bounded TCP rows returned by one Policy.act call (default: 1)",
    )
    parser.add_argument(
        "--operator-present",
        action="store_true",
        help="acknowledge that the assigned cell operator is present for physical motion",
    )
    parser.add_argument(
        "--confirm-standard-fingers",
        action="store_true",
        help="acknowledge that the installed 76 mm standard custom fingers were verified",
    )
    parser.add_argument(
        "--confirm-tcp-at-fingertips",
        action="store_true",
        help="acknowledge that active UTOOL1 was verified at the fingertips",
    )
    parser.add_argument(
        "--confirmed-cube-mm",
        type=int,
        choices=(10, 16),
        help="measured/organizer-confirmed cube edge; must match --cube-mm for sort",
    )
    parser.add_argument(
        "--confirm-grasp-fit",
        action="store_true",
        help=(
            "acknowledge that the selected cube was physically checked to fit the "
            "installed hooks; this is especially consequential for a 16 mm cube"
        ),
    )
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
    )


def _required_acknowledgements(args: argparse.Namespace) -> list[str]:
    missing: list[str] = []
    if args.mode in {"hover", "sort"}:
        if not args.operator_present:
            missing.append("--operator-present")
        if args.cube_mm is None:
            missing.append("--cube-mm {10|16}")
    if args.mode == "sort":
        if not args.confirm_standard_fingers:
            missing.append("--confirm-standard-fingers")
        if not args.confirm_tcp_at_fingertips:
            missing.append("--confirm-tcp-at-fingertips")
        if (
            args.cube_mm is not None
            and args.confirmed_cube_mm != args.cube_mm
        ):
            missing.append(f"--confirmed-cube-mm {args.cube_mm}")
        if not args.confirm_grasp_fit:
            missing.append("--confirm-grasp-fit")
    return missing


def _apply_environment(
    settings: LaunchSettings, *, grasp_fit_confirmed: bool = False
) -> None:
    # Delete inherited values first.  In particular, a previous interactive
    # ``export ROCKWELL_ENABLE_PHYSICAL_MOTION=1`` must not arm a plain run.
    for name in _CONTROLLED_ENVIRONMENT:
        os.environ.pop(name, None)
    os.environ.update(
        settings.environment(grasp_fit_confirmed=grasp_fit_confirmed)
    )


def _forwarded_arguments(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = []
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
    missing = _required_acknowledgements(args)
    if args.dry_run:
        summary = settings.public_summary()
        summary["missing_acknowledgements_for_live_run"] = missing
        print(json.dumps(summary, sort_keys=True))
        return 0
    if missing:
        parser.error(
            f"{args.mode} mode requires explicit acknowledgement(s): "
            + ", ".join(missing)
        )

    _apply_environment(settings, grasp_fit_confirmed=args.confirm_grasp_fit)
    print(json.dumps({"event": "supervised_launch", **settings.public_summary()}))
    (delegate or _delegate_to_official_client)(_forwarded_arguments(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
