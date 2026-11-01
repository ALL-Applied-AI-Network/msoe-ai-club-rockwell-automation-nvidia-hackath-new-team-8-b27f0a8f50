#!/usr/bin/env python3
"""Compare one authoritative replay with a candidate/shadow replay.

Both inputs must be clean, provenance-stable outputs from ``replay_camera``
over the exact same sampled RGB frames.  This tool never loads a model,
contacts the cell, or treats either provider as ground truth.  It measures
candidate availability and disagreement with the chosen baseline only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence

from sorting_core.perception import (
    DetectedObject,
    SceneObservation,
    scene_observation_from_dict,
)


SCHEMA_VERSION = 2
REPLAY_TOOL = "team_plugins.computer_vision.replay_camera"
COMPARISON_TOOL = "team_plugins.computer_vision.compare_replays"
COMPARISON_TOOL_PATH = Path(__file__).resolve()
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
MAX_ERROR_EXAMPLES_PER_CATEGORY = 3
MAX_FALLBACK_ERROR_PREFIX_CHARACTERS = 160
YOLO_GEOMETRY_DISAGREEMENT_PATTERN = re.compile(
    r"^YOLO11n and calibrated geometry disagree inside the inference crop:\s*"
    r"unmatched_yolo_boxes=(?P<unmatched_yolo_boxes>\d+),\s*"
    r"unmatched_geometry_detections="
    r"(?P<unmatched_geometry_detections>\d+),"
)

MATCHING_PROVENANCE_FIELDS = (
    "capture_sha256",
    "calibration_id",
    "calibration_sha256",
    "project_commit",
    "branch",
    "replay_tool_sha256",
    "source_size",
    "native_calibration_size",
    "qualitative_upscale",
    "cube_edge_mm_profile",
    "sample_every_n_frames",
    "decoded_source_fps",
    "start_frame",
    "maximum_frames",
    "processed_frame_indices",
    "processed_frames",
)


@dataclass(frozen=True)
class ReplayRecord:
    frame_index: int
    timestamp_s: float
    latency_ms: float
    input_rgb_sha256: str
    input_rgb_shape_hwc: tuple[int, int, int]
    input_rgb_dtype: str
    input_channel_order: str
    scene: SceneObservation | None
    vision_error: Mapping[str, str] | None


@dataclass(frozen=True)
class ReplayBundle:
    directory: Path
    report: Mapping[str, Any]
    records: tuple[ReplayRecord, ...]
    report_fingerprint: Mapping[str, Any]
    scenes_fingerprint: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = _sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"input changed while it was being hashed: {path}")
    return {
        "sha256": digest,
        "size_bytes": after.st_size,
        "modified_time_ns": after.st_mtime_ns,
    }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


def _sha256_value(value: Any, name: str) -> str:
    text = _nonempty_string(value, name)
    if SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _load_json_object(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {name}: {path}: {exc}") from exc
    return _mapping(value, name)


def _validate_report(report: Mapping[str, Any], *, name: str) -> None:
    if report.get("schema_version") != 1:
        raise ValueError(f"{name}.schema_version must be 1")
    if report.get("tool") != REPLAY_TOOL:
        raise ValueError(f"{name}.tool must identify replay_camera")
    for field in (
        "capture_sha256",
        "calibration_sha256",
        "provider_artifact_sha256",
        "replay_tool_sha256",
        "scenes_jsonl_sha256",
    ):
        _sha256_value(report.get(field), f"{name}.{field}")
    project_commit = _nonempty_string(
        report.get("project_commit"), f"{name}.project_commit"
    )
    if GIT_COMMIT_PATTERN.fullmatch(project_commit) is None:
        raise ValueError(f"{name}.project_commit must be a lowercase Git SHA")
    _nonempty_string(report.get("branch"), f"{name}.branch")
    _nonempty_string(report.get("calibration_id"), f"{name}.calibration_id")
    _nonempty_string(report.get("vision_factory"), f"{name}.vision_factory")
    _nonempty_string(
        report.get("provider_factory_module"),
        f"{name}.provider_factory_module",
    )
    profile = _mapping(report.get("integration_profile"), f"{name}.integration_profile")
    _nonempty_string(profile.get("profile_id"), f"{name}.integration_profile.profile_id")
    _sha256_value(
        profile.get("profile_sha256"),
        f"{name}.integration_profile.profile_sha256",
    )
    checkpoint = report.get("vision_checkpoint")
    if checkpoint is not None:
        checkpoint_mapping = _mapping(checkpoint, f"{name}.vision_checkpoint")
        _nonempty_string(
            checkpoint_mapping.get("file_name"),
            f"{name}.vision_checkpoint.file_name",
        )
        _sha256_value(
            checkpoint_mapping.get("sha256"),
            f"{name}.vision_checkpoint.sha256",
        )

    for field in (
        "worktree_dirty",
        "worktree_dirty_at_start",
        "worktree_dirty_at_end",
    ):
        if report.get(field) is not False:
            raise ValueError(f"{name}.{field} must be false for comparison evidence")
    if report.get("provenance_stable_during_replay") is not True:
        raise ValueError(f"{name} replay provenance was not stable")
    if report.get("provenance_fields_changed_during_replay") != []:
        raise ValueError(f"{name} reports changed replay provenance")

    indices = _sequence(
        report.get("processed_frame_indices"),
        f"{name}.processed_frame_indices",
    )
    parsed_indices = [
        _integer(value, f"{name}.processed_frame_indices[{index}]")
        for index, value in enumerate(indices)
    ]
    if parsed_indices != sorted(set(parsed_indices)):
        raise ValueError(f"{name}.processed_frame_indices must be unique and increasing")
    processed = _integer(report.get("processed_frames"), f"{name}.processed_frames", minimum=1)
    if processed != len(parsed_indices):
        raise ValueError(f"{name}.processed_frames does not match its frame indices")
    _integer(
        report.get("validated_scene_frames"),
        f"{name}.validated_scene_frames",
    )
    _integer(report.get("vision_error_frames"), f"{name}.vision_error_frames")
    _integer(
        report.get("sample_every_n_frames"),
        f"{name}.sample_every_n_frames",
        minimum=1,
    )
    _integer(report.get("start_frame"), f"{name}.start_frame")
    maximum_frames = report.get("maximum_frames")
    if maximum_frames is not None:
        _integer(maximum_frames, f"{name}.maximum_frames", minimum=1)
    _finite_number(
        report.get("decoded_source_fps"),
        f"{name}.decoded_source_fps",
        minimum=1e-12,
    )
    _finite_number(
        report.get("cube_edge_mm_profile"),
        f"{name}.cube_edge_mm_profile",
        minimum=1e-12,
    )
    for field in ("source_size", "native_calibration_size"):
        size = _sequence(report.get(field), f"{name}.{field}")
        if len(size) != 2:
            raise ValueError(f"{name}.{field} must contain width and height")
        for index, value in enumerate(size):
            _integer(value, f"{name}.{field}[{index}]", minimum=1)
    if type(report.get("qualitative_upscale")) is not bool:
        raise ValueError(f"{name}.qualitative_upscale must be a boolean")


def _parse_record(raw: Any, *, name: str, baseline: bool) -> ReplayRecord:
    record = _mapping(raw, name)
    frame_index = _integer(record.get("frame_index"), f"{name}.frame_index")
    timestamp_s = _finite_number(record.get("timestamp_s"), f"{name}.timestamp_s")
    latency_ms = _finite_number(
        record.get("latency_ms"), f"{name}.latency_ms", minimum=0.0
    )
    input_rgb_sha256 = _sha256_value(
        record.get("input_rgb_sha256"), f"{name}.input_rgb_sha256"
    )
    shape = _sequence(record.get("input_rgb_shape_hwc"), f"{name}.input_rgb_shape_hwc")
    if len(shape) != 3:
        raise ValueError(f"{name}.input_rgb_shape_hwc must contain H, W, C")
    parsed_shape = tuple(
        _integer(value, f"{name}.input_rgb_shape_hwc[{index}]", minimum=1)
        for index, value in enumerate(shape)
    )
    if parsed_shape[2] != 3:
        raise ValueError(f"{name}.input_rgb_shape_hwc channel count must be 3")
    input_rgb_dtype = _nonempty_string(
        record.get("input_rgb_dtype"), f"{name}.input_rgb_dtype"
    )
    if record.get("input_channel_order") != "RGB":
        raise ValueError(f"{name}.input_channel_order must be RGB")

    has_scene = "scene" in record
    has_error = "vision_error" in record
    if baseline and (not has_scene or has_error):
        raise ValueError(f"{name} baseline record must contain one scene and no error")
    if not baseline and has_scene == has_error:
        raise ValueError(
            f"{name} candidate record must contain exactly one scene or vision_error"
        )
    if has_scene and "explicit_abstention" in record:
        raise ValueError(f"{name} scene record must not claim an explicit abstention")
    scene: SceneObservation | None = None
    error: Mapping[str, str] | None = None
    if has_scene:
        try:
            scene = scene_observation_from_dict(record["scene"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}.scene violates the SceneObservation contract: {exc}") from exc
        if abs(scene.capture_time_s - timestamp_s) > 1e-6:
            raise ValueError(f"{name}.scene timestamp does not match the replay record")
    else:
        if record.get("explicit_abstention") is not True:
            raise ValueError(f"{name} error record must be an explicit abstention")
        raw_error = _mapping(record.get("vision_error"), f"{name}.vision_error")
        error = {
            "error_type": _nonempty_string(
                raw_error.get("error_type"), f"{name}.vision_error.error_type"
            ),
            "message": _nonempty_string(
                raw_error.get("message"), f"{name}.vision_error.message"
            ),
        }
    return ReplayRecord(
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        latency_ms=latency_ms,
        input_rgb_sha256=input_rgb_sha256,
        input_rgb_shape_hwc=parsed_shape,
        input_rgb_dtype=input_rgb_dtype,
        input_channel_order="RGB",
        scene=scene,
        vision_error=error,
    )


def _load_records(path: Path, *, name: str, baseline: bool) -> tuple[ReplayRecord, ...]:
    records: list[ReplayRecord] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError(f"{name} contains a blank line at {line_number}")
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{name} contains invalid JSON at line {line_number}: {exc}"
                    ) from exc
                records.append(
                    _parse_record(
                        raw,
                        name=f"{name}[{line_number}]",
                        baseline=baseline,
                    )
                )
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read {name}: {path}: {exc}") from exc
    if not records:
        raise ValueError(f"{name} must contain at least one replay record")
    return tuple(records)


def _object_counts(records: Iterable[ReplayRecord]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        if record.scene is not None:
            counts.update(item.class_id for item in record.scene.objects)
    return dict(sorted(counts.items()))


def _error_descriptor(error: Mapping[str, str]) -> dict[str, Any]:
    """Return a compact, deterministic description of one provider error.

    Full messages remain in the candidate replay's hash-pinned ``scenes.jsonl``.
    The comparison uses a category plus a message digest so variable track IDs,
    coordinates, and boxes do not duplicate a long message in every frame.
    """

    error_type = error["error_type"]
    message = error["message"]
    match = YOLO_GEOMETRY_DISAGREEMENT_PATTERN.match(message)
    if match is not None:
        category = "yolo_geometry_disagreement"
        attributes: dict[str, Any] = {
            "unmatched_yolo_boxes": int(match.group("unmatched_yolo_boxes")),
            "unmatched_geometry_detections": int(
                match.group("unmatched_geometry_detections")
            ),
        }
    else:
        prefix = " ".join(message.partition(":")[0].split())
        if not prefix:
            prefix = "unclassified_message"
        bounded_prefix = prefix[:MAX_FALLBACK_ERROR_PREFIX_CHARACTERS]
        category = "message_prefix"
        attributes = {
            "bounded_prefix_sha256": hashlib.sha256(
                bounded_prefix.encode("utf-8")
            ).hexdigest(),
            "bounded_prefix_character_count": len(bounded_prefix),
            "prefix_was_truncated": (
                len(prefix) > MAX_FALLBACK_ERROR_PREFIX_CHARACTERS
            ),
        }
    identity = {
        "error_type": error_type,
        "category": category,
        "attributes": attributes,
    }
    encoded_identity = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **identity,
        "category_id": hashlib.sha256(encoded_identity).hexdigest(),
        "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
    }


def _validate_bundle_consistency(
    report: Mapping[str, Any],
    records: tuple[ReplayRecord, ...],
    *,
    name: str,
    baseline: bool,
) -> None:
    native_width, native_height = report["native_calibration_size"]
    expected_shape = (native_height, native_width, 3)
    for record in records:
        if record.input_rgb_shape_hwc != expected_shape:
            raise ValueError(
                f"{name} frame {record.frame_index} RGB shape does not match "
                "native calibration size"
            )
        if record.input_rgb_dtype != "uint8" or record.input_channel_order != "RGB":
            raise ValueError(
                f"{name} frame {record.frame_index} must retain uint8 RGB input"
            )
    expected_indices = list(report["processed_frame_indices"])
    observed_indices = [record.frame_index for record in records]
    if observed_indices != expected_indices:
        raise ValueError(f"{name} record frame indices do not match report.json")
    scenes = sum(record.scene is not None for record in records)
    errors = sum(record.vision_error is not None for record in records)
    if report["processed_frames"] != len(records):
        raise ValueError(f"{name} record count does not match report.json")
    if report["validated_scene_frames"] != scenes:
        raise ValueError(f"{name} validated-scene count does not match report.json")
    if report["vision_error_frames"] != errors:
        raise ValueError(f"{name} vision-error count does not match report.json")
    if baseline and (scenes != len(records) or errors):
        raise ValueError(f"{name} baseline must contain a validated scene for every frame")
    reported_counts = _mapping(
        report.get("object_observations_by_class"),
        f"{name}.object_observations_by_class",
    )
    normalized_counts = {
        _nonempty_string(key, f"{name}.object_observations_by_class key"): _integer(
            value,
            f"{name}.object_observations_by_class[{key!r}]",
        )
        for key, value in reported_counts.items()
    }
    if normalized_counts != _object_counts(records):
        raise ValueError(f"{name} object counts do not match scenes.jsonl")

    identities = {
        (record.scene.provider_name, record.scene.provider_version)
        for record in records
        if record.scene is not None
    }
    if len(identities) > 1:
        raise ValueError(f"{name} provider identity changes across scenes")
    if identities:
        identity = next(iter(identities))
        if (report.get("provider_name"), report.get("provider_version")) != identity:
            raise ValueError(f"{name} report provider identity does not match scenes")
    elif report.get("provider_name") is not None or report.get("provider_version") is not None:
        raise ValueError(f"{name} all-error replay must not claim a provider identity")


def load_replay_bundle(directory: Path, *, name: str, baseline: bool) -> ReplayBundle:
    try:
        resolved = Path(directory).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot resolve {name} directory: {directory}: {exc}") from exc
    if not resolved.is_dir():
        raise ValueError(f"{name} must be a replay output directory: {resolved}")
    report_path = resolved / "report.json"
    scenes_path = resolved / "scenes.jsonl"
    if not report_path.is_file() or not scenes_path.is_file():
        raise ValueError(f"{name} must contain report.json and scenes.jsonl")

    before = {
        "report": _fingerprint(report_path),
        "scenes": _fingerprint(scenes_path),
    }
    report = _load_json_object(report_path, f"{name}.report.json")
    _validate_report(report, name=f"{name}.report.json")
    if report["scenes_jsonl_sha256"] != before["scenes"]["sha256"]:
        raise ValueError(f"{name} scenes.jsonl hash does not match report.json")
    records = _load_records(scenes_path, name=f"{name}.scenes.jsonl", baseline=baseline)
    _validate_bundle_consistency(report, records, name=name, baseline=baseline)
    after = {
        "report": _fingerprint(report_path),
        "scenes": _fingerprint(scenes_path),
    }
    if before != after:
        raise RuntimeError(f"{name} replay inputs changed during comparison")
    return ReplayBundle(
        directory=resolved,
        report=report,
        records=records,
        report_fingerprint=before["report"],
        scenes_fingerprint=before["scenes"],
    )


def _matching_provenance(
    baseline: ReplayBundle, candidate: ReplayBundle
) -> dict[str, Any]:
    mismatches = [
        field
        for field in MATCHING_PROVENANCE_FIELDS
        if baseline.report.get(field) != candidate.report.get(field)
    ]
    if mismatches:
        raise ValueError(
            "replays do not describe the same sampled evidence: "
            + ", ".join(mismatches)
        )
    for baseline_record, candidate_record in zip(
        baseline.records, candidate.records, strict=True
    ):
        if baseline_record.frame_index != candidate_record.frame_index:
            raise ValueError("replay frame indices differ")
        if abs(baseline_record.timestamp_s - candidate_record.timestamp_s) > 1e-9:
            raise ValueError(
                f"frame {baseline_record.frame_index} replay timestamps differ"
            )
        if baseline_record.input_rgb_sha256 != candidate_record.input_rgb_sha256:
            raise ValueError(
                f"frame {baseline_record.frame_index} input RGB hashes differ"
            )
        if (
            baseline_record.input_rgb_shape_hwc
            != candidate_record.input_rgb_shape_hwc
            or baseline_record.input_rgb_dtype != candidate_record.input_rgb_dtype
            or baseline_record.input_channel_order
            != candidate_record.input_channel_order
        ):
            raise ValueError(
                f"frame {baseline_record.frame_index} input RGB metadata differs"
            )
    return {field: baseline.report.get(field) for field in MATCHING_PROVENANCE_FIELDS}


def _semantic_key(item: DetectedObject) -> tuple[str, str | None]:
    return item.class_id, item.sort_label


def _distance_m(left: DetectedObject, right: DetectedObject) -> float:
    return math.sqrt(
        sum(
            (a - b) ** 2
            for a, b in zip(
                left.pose.position_base_m,
                right.pose.position_base_m,
                strict=True,
            )
        )
    )


def _orientation_delta_deg(left: DetectedObject, right: DetectedObject) -> float:
    dot = abs(
        sum(
            a * b
            for a, b in zip(
                left.pose.orientation_xyzw,
                right.pose.orientation_xyzw,
                strict=True,
            )
        )
    )
    dot = min(1.0, max(0.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def _match_objects(
    baseline: Sequence[DetectedObject],
    candidate: Sequence[DetectedObject],
) -> tuple[
    list[tuple[DetectedObject, DetectedObject, str]],
    list[DetectedObject],
    list[DetectedObject],
]:
    """Match exact semantic track IDs first, then nearest same-label objects.

    The nearest-neighbour step is deterministic and has no distance acceptance
    threshold.  Distances are disagreement measurements, not correctness or
    identity claims.
    """

    unmatched_baseline = set(range(len(baseline)))
    unmatched_candidate = set(range(len(candidate)))
    matches: list[tuple[DetectedObject, DetectedObject, str]] = []

    candidate_by_track = {
        item.track_id: index for index, item in enumerate(candidate)
    }
    for baseline_index, item in enumerate(baseline):
        candidate_index = candidate_by_track.get(item.track_id)
        if candidate_index is None or candidate_index not in unmatched_candidate:
            continue
        other = candidate[candidate_index]
        if _semantic_key(item) != _semantic_key(other):
            continue
        matches.append((item, other, "exact_track_and_semantic_label"))
        unmatched_baseline.remove(baseline_index)
        unmatched_candidate.remove(candidate_index)

    pairs: list[tuple[float, str, str, int, int]] = []
    for baseline_index in unmatched_baseline:
        left = baseline[baseline_index]
        for candidate_index in unmatched_candidate:
            right = candidate[candidate_index]
            if _semantic_key(left) != _semantic_key(right):
                continue
            pairs.append(
                (
                    _distance_m(left, right),
                    left.track_id,
                    right.track_id,
                    baseline_index,
                    candidate_index,
                )
            )
    for (
        unused_distance,
        unused_left_id,
        unused_right_id,
        baseline_index,
        candidate_index,
    ) in sorted(pairs):
        if baseline_index not in unmatched_baseline or candidate_index not in unmatched_candidate:
            continue
        matches.append(
            (
                baseline[baseline_index],
                candidate[candidate_index],
                "nearest_same_class_and_sort_label",
            )
        )
        unmatched_baseline.remove(baseline_index)
        unmatched_candidate.remove(candidate_index)

    return (
        matches,
        [baseline[index] for index in sorted(unmatched_baseline)],
        [candidate[index] for index in sorted(unmatched_candidate)],
    )


def _error_streaks(records: Sequence[ReplayRecord]) -> dict[str, Any]:
    streaks: list[dict[str, Any]] = []
    start: int | None = None
    reasons: Counter[str] = Counter()
    descriptors: dict[str, Mapping[str, Any]] = {}
    for ordinal, record in enumerate(records, start=1):
        if record.vision_error is not None:
            if start is None:
                start = ordinal
                reasons.clear()
                descriptors.clear()
            descriptor = _error_descriptor(record.vision_error)
            category_id = descriptor["category_id"]
            reasons[category_id] += 1
            descriptors[category_id] = descriptor
            continue
        if start is not None:
            end = ordinal - 1
            streaks.append(
                _streak_record(records, start, end, reasons, descriptors)
            )
            start = None
            reasons = Counter()
            descriptors = {}
    if start is not None:
        streaks.append(
            _streak_record(records, start, len(records), reasons, descriptors)
        )
    return {
        "unit": "consecutive processed replay frames (1-based processed ordinals)",
        "count": len(streaks),
        "maximum_length_processed_frames": max(
            (streak["length_processed_frames"] for streak in streaks),
            default=0,
        ),
        "streaks": streaks,
    }


def _streak_record(
    records: Sequence[ReplayRecord],
    start: int,
    end: int,
    reasons: Counter[str],
    descriptors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "start_processed_ordinal": start,
        "end_processed_ordinal": end,
        "length_processed_frames": end - start + 1,
        "start_source_frame_index": records[start - 1].frame_index,
        "end_source_frame_index": records[end - 1].frame_index,
        "reasons": [
            {
                "category_id": category_id,
                "error_type": descriptors[category_id]["error_type"],
                "category": descriptors[category_id]["category"],
                "attributes": descriptors[category_id]["attributes"],
                "processed_frames": count,
            }
            for category_id, count in sorted(reasons.items())
        ],
    }


def _error_categories(records: Sequence[ReplayRecord]) -> list[dict[str, Any]]:
    categories: dict[str, dict[str, Any]] = {}
    distinct_messages: dict[str, set[str]] = defaultdict(set)
    for processed_ordinal, record in enumerate(records, start=1):
        if record.vision_error is None:
            continue
        descriptor = _error_descriptor(record.vision_error)
        category_id = descriptor["category_id"]
        if category_id not in categories:
            categories[category_id] = {
                "category_id": category_id,
                "error_type": descriptor["error_type"],
                "category": descriptor["category"],
                "attributes": descriptor["attributes"],
                "frames": 0,
                "first_processed_ordinal": processed_ordinal,
                "first_source_frame_index": record.frame_index,
                "last_processed_ordinal": processed_ordinal,
                "last_source_frame_index": record.frame_index,
                "representative_examples": [],
            }
        category_row = categories[category_id]
        category_row["frames"] += 1
        category_row["last_processed_ordinal"] = processed_ordinal
        category_row["last_source_frame_index"] = record.frame_index
        message_sha256 = descriptor["message_sha256"]
        distinct_messages[category_id].add(message_sha256)
        examples = category_row["representative_examples"]
        if len(examples) < MAX_ERROR_EXAMPLES_PER_CATEGORY:
            examples.append(
                {
                    "processed_ordinal": processed_ordinal,
                    "source_frame_index": record.frame_index,
                    "message_sha256": message_sha256,
                }
            )
    result = []
    for category_id in sorted(categories):
        row = categories[category_id]
        row["distinct_message_sha256_count"] = len(
            distinct_messages[category_id]
        )
        result.append(row)
    return result


def _comparison_frame(
    baseline_record: ReplayRecord,
    candidate_record: ReplayRecord,
) -> tuple[dict[str, Any], list[float], int, int, int]:
    if baseline_record.scene is None:  # Enforced during bundle loading.
        raise RuntimeError("baseline scene is missing")
    common = {
        "frame_index": baseline_record.frame_index,
        "timestamp_s": baseline_record.timestamp_s,
        "input_rgb_sha256": baseline_record.input_rgb_sha256,
        "baseline_latency_ms": baseline_record.latency_ms,
        "candidate_latency_ms": candidate_record.latency_ms,
        "baseline_object_count": len(baseline_record.scene.objects),
    }
    if candidate_record.vision_error is not None:
        descriptor = _error_descriptor(candidate_record.vision_error)
        return (
            {
                **common,
                "candidate_status": "explicit_abstention",
                "candidate_object_count": None,
                "vision_error": {
                    "error_type": descriptor["error_type"],
                    "category_id": descriptor["category_id"],
                    "message_sha256": descriptor["message_sha256"],
                },
            },
            [],
            0,
            0,
            0,
        )
    candidate_scene = candidate_record.scene
    if candidate_scene is None:
        raise RuntimeError("candidate frame contains neither scene nor error")
    if baseline_record.scene.scene_id != candidate_scene.scene_id:
        raise ValueError(
            f"frame {baseline_record.frame_index} scene IDs differ between providers"
        )
    if baseline_record.scene.calibration_ids != candidate_scene.calibration_ids:
        raise ValueError(
            f"frame {baseline_record.frame_index} scene calibration IDs differ"
        )

    matches, baseline_only, candidate_only = _match_objects(
        baseline_record.scene.objects,
        candidate_scene.objects,
    )
    match_rows = []
    position_deltas_mm: list[float] = []
    orientation_disagreements = 0
    track_id_disagreements = 0
    for left, right, strategy in matches:
        delta_mm = _distance_m(left, right) * 1000.0
        orientation_delta = _orientation_delta_deg(left, right)
        position_deltas_mm.append(delta_mm)
        orientation_disagreements += int(orientation_delta > 1e-6)
        track_id_disagreements += int(left.track_id != right.track_id)
        match_rows.append(
            {
                "match_strategy": strategy,
                "class_id": left.class_id,
                "sort_label": left.sort_label,
                "baseline_track_id": left.track_id,
                "candidate_track_id": right.track_id,
                "track_id_equal": left.track_id == right.track_id,
                "position_delta_mm": delta_mm,
                "raw_frame_orientation_delta_deg_not_symmetry_reduced": (
                    orientation_delta
                ),
                "baseline_symmetry_id": left.attributes.get("symmetry_id"),
                "candidate_symmetry_id": right.attributes.get("symmetry_id"),
            }
        )

    baseline_by_track = {item.track_id: item for item in baseline_record.scene.objects}
    candidate_by_track = {item.track_id: item for item in candidate_scene.objects}
    label_changes = []
    for track_id in sorted(set(baseline_by_track) & set(candidate_by_track)):
        left_key = _semantic_key(baseline_by_track[track_id])
        right_key = _semantic_key(candidate_by_track[track_id])
        if left_key != right_key:
            label_changes.append(
                {
                    "track_id": track_id,
                    "baseline_class_id": left_key[0],
                    "baseline_sort_label": left_key[1],
                    "candidate_class_id": right_key[0],
                    "candidate_sort_label": right_key[1],
                }
            )
    row = {
        **common,
        "candidate_status": "validated_scene",
        "candidate_object_count": len(candidate_scene.objects),
        "object_count_delta_candidate_minus_baseline": (
            len(candidate_scene.objects) - len(baseline_record.scene.objects)
        ),
        "matches": match_rows,
        "baseline_only_objects": [
            {
                "track_id": item.track_id,
                "class_id": item.class_id,
                "sort_label": item.sort_label,
            }
            for item in baseline_only
        ],
        "candidate_only_objects": [
            {
                "track_id": item.track_id,
                "class_id": item.class_id,
                "sort_label": item.sort_label,
            }
            for item in candidate_only
        ],
        "same_track_label_changes": label_changes,
    }
    return (
        row,
        position_deltas_mm,
        len(baseline_only) + len(candidate_only),
        orientation_disagreements,
        track_id_disagreements,
    )


def compare_replays(baseline: ReplayBundle, candidate: ReplayBundle) -> dict[str, Any]:
    matching_provenance = _matching_provenance(baseline, candidate)
    frame_rows = []
    position_deltas_mm: list[float] = []
    unmatched_objects = 0
    orientation_disagreements = 0
    track_id_disagreements = 0
    accepted = 0
    object_count_difference_frames = 0
    label_difference_frames = 0
    baseline_latencies = []
    candidate_latencies = []
    for baseline_record, candidate_record in zip(
        baseline.records, candidate.records, strict=True
    ):
        row, deltas, unmatched, orientation_changed, track_changed = _comparison_frame(
            baseline_record,
            candidate_record,
        )
        frame_rows.append(row)
        baseline_latencies.append(baseline_record.latency_ms)
        candidate_latencies.append(candidate_record.latency_ms)
        position_deltas_mm.extend(deltas)
        unmatched_objects += unmatched
        orientation_disagreements += orientation_changed
        track_id_disagreements += track_changed
        if candidate_record.scene is not None:
            accepted += 1
            object_count_difference_frames += int(
                row["object_count_delta_candidate_minus_baseline"] != 0
            )
            label_difference_frames += int(bool(row["same_track_label_changes"]))
    total = len(baseline.records)
    abstained = total - accepted
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": COMPARISON_TOOL,
        "comparison_tool_sha256": _sha256(COMPARISON_TOOL_PATH),
        "baseline": _bundle_provenance(baseline),
        "candidate": _bundle_provenance(candidate),
        "matched_replay_provenance": matching_provenance,
        "availability": {
            "processed_frames": total,
            "candidate_validated_scene_frames": accepted,
            "candidate_explicit_abstention_frames": abstained,
            "candidate_acceptance_fraction": accepted / total,
            "candidate_abstention_fraction": abstained / total,
            "error_categories": _error_categories(candidate.records),
            "error_streaks": _error_streaks(candidate.records),
            "full_error_detail_source": {
                "candidate_bundle_name": candidate.directory.name,
                "relative_file": "scenes.jsonl",
                "sha256": candidate.scenes_fingerprint["sha256"],
                "note": (
                    "Full per-frame provider messages remain in the immutable "
                    "candidate replay; this comparison stores only their "
                    "SHA-256 digests and source-frame locations."
                ),
            },
        },
        "disagreement": {
            "accepted_frames_with_object_count_difference": (
                object_count_difference_frames
            ),
            "accepted_frames_with_same_track_label_difference": (
                label_difference_frames
            ),
            "unmatched_objects_across_accepted_frames": unmatched_objects,
            "matched_object_pairs": len(position_deltas_mm),
            "matched_pairs_with_track_id_difference": track_id_disagreements,
            "matched_pairs_with_raw_orientation_difference": (
                orientation_disagreements
            ),
            "position_delta_mm": _number_summary(position_deltas_mm),
            "orientation_interpretation": (
                "Raw quaternion-frame deltas account for q/-q equivalence but are "
                "not reduced by object symmetry. They are provider disagreement, "
                "not physical orientation error or grasp accuracy."
            ),
            "matching_strategy": (
                "Exact track_id+class_id+sort_label first; remaining objects are "
                "greedily paired by smallest robot-base Euclidean distance only "
                "within identical class_id+sort_label groups. No distance threshold "
                "or ground-truth identity is implied."
            ),
        },
        "latency_ms": {
            "baseline": _number_summary(baseline_latencies),
            "candidate": _number_summary(candidate_latencies),
        },
        "frames": frame_rows,
        "evidence_limit": (
            "This compares two providers on identical replay pixels. The baseline "
            "is not ground truth; agreement does not establish detection accuracy, "
            "millimetre pose accuracy, grasp success, or hardware readiness."
        ),
    }


def _number_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "mean": None, "maximum": None}
    return {
        "count": len(values),
        "minimum": min(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
    }


def _bundle_provenance(bundle: ReplayBundle) -> dict[str, Any]:
    report = bundle.report
    profile = report["integration_profile"]
    return {
        "bundle_name": bundle.directory.name,
        "report_json_sha256": bundle.report_fingerprint["sha256"],
        "scenes_jsonl_sha256": bundle.scenes_fingerprint["sha256"],
        "provider_name": report.get("provider_name"),
        "provider_version": report.get("provider_version"),
        "vision_factory": report["vision_factory"],
        "provider_artifact_sha256": report["provider_artifact_sha256"],
        "vision_checkpoint": report.get("vision_checkpoint"),
        "integration_profile": {
            "profile_id": profile["profile_id"],
            "profile_sha256": profile["profile_sha256"],
        },
    }


def _verify_unchanged(bundle: ReplayBundle, *, name: str) -> None:
    report = _fingerprint(bundle.directory / "report.json")
    scenes = _fingerprint(bundle.directory / "scenes.jsonl")
    if report != bundle.report_fingerprint or scenes != bundle.scenes_fingerprint:
        raise RuntimeError(f"{name} replay inputs changed during comparison")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="authoritative replay output directory")
    parser.add_argument("candidate", type=Path, help="candidate/shadow replay output directory")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "optional new JSON report path; with this option stdout contains "
            "only a compact JSON receipt"
        ),
    )
    return parser


def _output_receipt(result: Mapping[str, Any], output: Path) -> dict[str, Any]:
    availability = result["availability"]
    streaks = availability["error_streaks"]
    return {
        "schema_version": 1,
        "tool": f"{COMPARISON_TOOL}.output_receipt",
        "comparison_schema_version": result["schema_version"],
        "output_file": output.name,
        "output_sha256": _sha256(output),
        "processed_frames": availability["processed_frames"],
        "candidate_validated_scene_frames": availability[
            "candidate_validated_scene_frames"
        ],
        "candidate_explicit_abstention_frames": availability[
            "candidate_explicit_abstention_frames"
        ],
        "maximum_error_streak_processed_frames": streaks[
            "maximum_length_processed_frames"
        ],
        "error_category_count": len(availability["error_categories"]),
        "evidence_limit": result["evidence_limit"],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    baseline = load_replay_bundle(args.baseline, name="baseline", baseline=True)
    candidate = load_replay_bundle(args.candidate, name="candidate", baseline=False)
    output = args.output.expanduser().resolve() if args.output is not None else None
    if output is not None:
        inputs = {
            baseline.directory / "report.json",
            baseline.directory / "scenes.jsonl",
            candidate.directory / "report.json",
            candidate.directory / "scenes.jsonl",
        }
        if output in inputs:
            raise ValueError("--output must not overwrite a replay input")
        if output.exists():
            raise ValueError("--output must be a new path")
        if not output.parent.is_dir():
            raise ValueError("--output parent directory must already exist")
    result = compare_replays(baseline, candidate)
    _verify_unchanged(baseline, name="baseline")
    _verify_unchanged(candidate, name="candidate")
    if result["comparison_tool_sha256"] != _sha256(COMPARISON_TOOL_PATH):
        raise RuntimeError("comparison tool changed during comparison")
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(encoded, end="")
    else:
        try:
            with output.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
        except FileExistsError as exc:
            raise ValueError("--output must be a new path") from exc
        receipt = _output_receipt(result, output)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
