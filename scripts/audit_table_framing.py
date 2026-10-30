#!/usr/bin/env python3
"""Audit the manually paired real/sim table-camera landmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sorting_core.table_framing import (  # noqa: E402
    TableFramingError,
    audit_table_framing,
    load_table_framing_evidence,
    verify_evidence_sources,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare hashed manual table-edge landmarks without changing camera "
            "calibration or table geometry."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "config" / "table_framing_landmarks.json",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=REPO.parent,
        help="root containing HardwareResults; files are read and hashed only",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-source-files",
        action="store_true",
        help="return nonzero unless every configured external source hash matches",
    )
    args = parser.parse_args()

    try:
        evidence = load_table_framing_evidence(args.config)
        report = audit_table_framing(evidence)
        report["source_verification"] = verify_evidence_sources(
            evidence, args.workspace_root
        )
    except TableFramingError as exc:
        parser.error(str(exc))

    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(args.output)

    if args.require_source_files and not report["source_verification"][
        "all_sources_verified"
    ]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
