#!/usr/bin/env python3
"""Convert one downloaded LeRobot episode into an Isaac-only shadow replay.

``pyarrow`` is deliberately an optional tool dependency rather than a core
runtime dependency.  For example:

    uv run --with pyarrow scripts/prepare_hardware_replay.py \
      --dataset /path/to/r20261029-88a87b --episode 0 \
      --split appearance_fit --sample-count 20 \
      --output /path/outside/repo/replay.json

The output contains measured state only.  Recorded action rows are audited for
the known reset sentinel and are never exported as robot commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sorting_core.hardware_replay import (  # noqa: E402
    HardwareReplayError,
    allowed_frame_ranges,
    build_replay_sample,
    frame_is_allowed,
    initial_joint_action_sentinel_length,
    validate_replay_samples,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--curation",
        type=Path,
        default=REPO_ROOT / "config" / "hardware_frame_curation.json",
        help="privacy curation manifest (checked-in manifest by default)",
    )
    parser.add_argument(
        "--split",
        default="appearance_fit",
        help="curation split containing the requested episode",
    )
    parser.add_argument(
        "--ignore-curation",
        action="store_true",
        help="permit reviewed-range filtering to be disabled explicitly",
    )
    parser.add_argument(
        "--frames",
        help="comma-separated frame indices; default is uniform reviewed sampling",
    )
    parser.add_argument("--sample-count", type=int, default=20)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_file(dataset: Path, episode: int, parquet) -> Path:
    matches = []
    for path in sorted((dataset / "data").rglob("*.parquet")):
        column = parquet.read_table(path, columns=["episode_index"])["episode_index"]
        if len(column) and int(column[0].as_py()) == episode:
            if any(int(value.as_py()) != episode for value in column):
                raise HardwareReplayError(f"mixed episode indices in {path}")
            matches.append(path)
    if len(matches) != 1:
        raise HardwareReplayError(
            f"expected one parquet file for episode {episode}, found {len(matches)}"
        )
    return matches[0]


def _uniform_indices(eligible: list[int], count: int) -> list[int]:
    if count < 1:
        raise HardwareReplayError("sample-count must be at least one")
    if not eligible:
        raise HardwareReplayError("no eligible frames remain after filtering")
    if count >= len(eligible):
        return eligible
    if count == 1:
        return [eligible[len(eligible) // 2]]
    selected = {
        eligible[round(index * (len(eligible) - 1) / (count - 1))]
        for index in range(count)
    }
    return sorted(selected)


def _parse_explicit_frames(value: str) -> list[int]:
    try:
        result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise HardwareReplayError("--frames must contain comma-separated integers") from exc
    if not result or result[0] < 0:
        raise HardwareReplayError("--frames must contain non-negative integers")
    return result


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = _arguments()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    try:
        output.relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        raise SystemExit("Replay outputs are generated data and must stay outside Git")
    if args.episode < 0:
        raise SystemExit("--episode must be non-negative")
    if not dataset.is_dir():
        raise SystemExit(f"dataset directory does not exist: {dataset}")

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required only for conversion; run with `uv run --with pyarrow`"
        ) from exc

    try:
        source = _episode_file(dataset, args.episode, parquet)
        table = parquet.read_table(
            source,
            columns=[
                "timestamp",
                "frame_index",
                "episode_index",
                "observation.state",
                "observation.tcp_pose",
                "action",
            ],
        )
        timestamps = table["timestamp"].to_pylist()
        frames = [int(value) for value in table["frame_index"].to_pylist()]
        states = table["observation.state"].to_pylist()
        tcp_poses = table["observation.tcp_pose"].to_pylist()
        actions = table["action"].to_pylist()
        if not (
            len(timestamps)
            == len(frames)
            == len(states)
            == len(tcp_poses)
            == len(actions)
        ):
            raise HardwareReplayError("episode columns have mismatched lengths")
        if frames != list(range(len(frames))):
            raise HardwareReplayError("episode frame_index is not a zero-based grid")

        sentinel_length = initial_joint_action_sentinel_length(states, actions)
        ranges: tuple[tuple[int, int], ...]
        if args.ignore_curation:
            ranges = ((0, len(frames)),)
            curation_record = None
        else:
            curation_path = args.curation.resolve()
            curation = json.loads(curation_path.read_text(encoding="utf-8"))
            ranges = allowed_frame_ranges(
                curation,
                dataset_id=dataset.name,
                episode_index=args.episode,
                split_name=args.split,
            )
            curation_record = {
                "manifest_sha256": _sha256(curation_path),
                "split": args.split,
                "allowed_ranges": [list(value) for value in ranges],
                "privacy_note": (
                    "Only reviewed ranges are exported. The coarse review still "
                    "requires a finer privacy scrub before model training/publication."
                ),
            }

        eligible = [
            frame
            for frame in frames
            if frame >= sentinel_length and frame_is_allowed(frame, ranges)
        ]
        if args.frames:
            selected = _parse_explicit_frames(args.frames)
            invalid = [frame for frame in selected if frame not in eligible]
            if invalid:
                raise HardwareReplayError(
                    f"requested frames are outside reviewed/non-sentinel ranges: {invalid}"
                )
        else:
            selected = _uniform_indices(eligible, args.sample_count)

        samples = [
            build_replay_sample(
                frame_index=frame,
                timestamp_s=timestamps[frame],
                observation_state=states[frame],
                tcp_pose=tcp_poses[frame],
            )
            for frame in selected
        ]
        validation = validate_replay_samples(samples)
    except (HardwareReplayError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    document = {
        "schema_version": 1,
        "replay_kind": "measured_state_shadow_replay",
        "physical_robot_commands_permitted": False,
        "source": {
            "dataset_id": dataset.name,
            "episode_index": args.episode,
            "episode_parquet_sha256": _sha256(source),
            "total_episode_rows": len(frames),
            "nominal_rate_hz": 10.0,
        },
        "curation": curation_record,
        "action_audit": {
            "exported": False,
            "initial_joint_like_action_sentinel_rows": sentinel_length,
            "excluded_frame_range": [0, sentinel_length],
            "warning": (
                "The action field changes semantics after the reset prefix; no "
                "recorded action is replayed."
            ),
        },
        "validation": validation,
        "samples": [sample.as_dict() for sample in samples],
        "limitations": [
            "State/video rows align at 10 Hz, but cameras have no independent capture timestamps.",
            "This shadow replay sets measured poses; it does not reproduce controller dynamics.",
            "Compressed 640x360 video cannot replace native camera calibration images.",
        ],
    }
    _atomic_json(output, document)
    print(
        f"wrote {len(samples)} measured-state samples to {output}; "
        f"sentinel rows excluded={sentinel_length}; "
        f"median FK residual={validation['median_fk_position_error_mm']:.4f} mm, "
        f"{validation['median_fk_orientation_error_deg']:.4f} deg"
    )


if __name__ == "__main__":
    main()
