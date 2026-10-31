#!/usr/bin/env python3
"""Validate cube/screw annotations and emit evaluation-result.v1 JSON.

This standard-library-only tool intentionally separates what compressed RGB
can establish (class/count, image-axis and tracking evidence) from physical
pose claims. Robot-base pose samples are emitted only when the manifest carries
independently measured truth and the prediction carries a valid pose.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
EVALUATOR_VERSION = "1.1"
SUPPORTED_CLASSES = {"cube", "screw_iso10642_m6x30"}
SYMMETRY_IDS = {
    "cube": "upright_square_yaw_modulo_90_deg",
    "screw": "catalog_axial_symmetry_unresolved",
}
HEX40 = frozenset("0123456789abcdef")


class AnnotationValidationError(ValueError):
    """Raised when an annotation manifest is ambiguous or malformed."""


def _fail(path: str, message: str) -> None:
    raise AnnotationValidationError(f"{path}: {message}")


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    return value


def _array(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    return value


def _keys(
    value: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    path: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    extra = sorted(set(value) - allowed)
    if missing:
        _fail(path, f"missing required fields: {', '.join(missing)}")
    if extra:
        _fail(path, f"unknown fields: {', '.join(extra)}")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty string")
    if value != value.strip():
        _fail(path, "must not contain leading or trailing whitespace")
    return value


def _nullable_string(value: Any, path: str) -> str | None:
    return None if value is None else _string(value, path)


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "must be a boolean")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(path, "must be a finite number")
    return result


def _nonnegative(value: Any, path: str) -> float:
    result = _number(value, path)
    if result < 0.0:
        _fail(path, "must be non-negative")
    return result


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(path, f"must be an integer >= {minimum}")
    return value


def _vector(value: Any, size: int, path: str) -> tuple[float, ...]:
    values = tuple(
        _number(item, f"{path}[{index}]")
        for index, item in enumerate(_array(value, path))
    )
    if len(values) != size:
        _fail(path, f"must contain exactly {size} values")
    return values


def _sha(value: Any, size: int, path: str) -> str:
    result = _string(value, path)
    if result != result.lower():
        _fail(path, f"must be a lowercase {size}-character hexadecimal digest")
    if len(result) != size or any(character not in HEX40 for character in result):
        _fail(path, f"must be a lowercase {size}-character hexadecimal digest")
    return result


def _enum(value: Any, allowed: set[str], path: str) -> str:
    result = _string(value, path)
    if result not in allowed:
        _fail(path, f"must be one of {sorted(allowed)}")
    return result


def _utc_datetime(value: Any, path: str) -> str:
    result = _string(value, path)
    if not result.endswith("Z"):
        _fail(path, "must be an explicit UTC date-time ending in Z")
    try:
        parsed = datetime.fromisoformat(result[:-1] + "+00:00")
    except ValueError:
        _fail(path, "must be a valid UTC date-time")
    if parsed.tzinfo != timezone.utc:
        _fail(path, "must use UTC")
    return result


def _unique(values: Sequence[str], path: str) -> None:
    if len(set(values)) != len(values):
        _fail(path, "must not contain duplicates")


def _polygon(value: Any, width: int, height: int, path: str) -> tuple[tuple[float, float], ...]:
    result = tuple(
        _vector(point, 2, f"{path}[{index}]")
        for index, point in enumerate(_array(value, path))
    )
    if len(result) < 3:
        _fail(path, "must contain at least three points")
    for index, (u, v) in enumerate(result):
        if not (0.0 <= u < width and 0.0 <= v < height):
            _fail(f"{path}[{index}]", "must lie inside the image")
    area_twice = abs(
        sum(
            result[index][0] * result[(index + 1) % len(result)][1]
            - result[(index + 1) % len(result)][0] * result[index][1]
            for index in range(len(result))
        )
    )
    if area_twice <= 1e-9:
        _fail(path, "must have non-zero area")
    return result


def _point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    """Return whether a point is inside or on the boundary of a polygon."""

    x, y = float(point[0]), float(point[1])
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) <= 1e-9 and min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9:
            return True
        if (y1 > y) != (y2 > y):
            intersection_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < intersection_x:
                inside = not inside
        previous = current
    return inside


def _validate_pose_truth(
    value: Any,
    class_id: str,
    path: str,
) -> None:
    if value is None:
        return
    truth = _object(value, path)
    _keys(
        truth,
        required=(
            "position_base_m",
            "position_uncertainty_mm",
            "cube_yaw_rad",
            "screw_head_to_tip_unit_base",
            "symmetry_id",
            "measurement_method",
        ),
        path=path,
    )
    _vector(truth["position_base_m"], 3, f"{path}.position_base_m")
    _nonnegative(truth["position_uncertainty_mm"], f"{path}.position_uncertainty_mm")
    symmetry_id = _string(truth["symmetry_id"], f"{path}.symmetry_id")
    _string(truth["measurement_method"], f"{path}.measurement_method")
    if class_id == "cube":
        if symmetry_id != SYMMETRY_IDS["cube"]:
            _fail(
                f"{path}.symmetry_id",
                f"must be {SYMMETRY_IDS['cube']!r} for a cube",
            )
        if truth["cube_yaw_rad"] is None:
            _fail(f"{path}.cube_yaw_rad", "is required for cube pose truth")
        _number(truth["cube_yaw_rad"], f"{path}.cube_yaw_rad")
        if truth["screw_head_to_tip_unit_base"] is not None:
            _fail(f"{path}.screw_head_to_tip_unit_base", "must be null for a cube")
    else:
        expected_symmetry = SYMMETRY_IDS["screw"]
        if symmetry_id != expected_symmetry:
            _fail(
                f"{path}.symmetry_id",
                f"must be {expected_symmetry!r} for this screw-axis observability",
            )
        if truth["cube_yaw_rad"] is not None:
            _fail(f"{path}.cube_yaw_rad", "must be null for a screw")
        axis = _vector(
            truth["screw_head_to_tip_unit_base"],
            3,
            f"{path}.screw_head_to_tip_unit_base",
        )
        norm = math.sqrt(sum(component * component for component in axis))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            _fail(f"{path}.screw_head_to_tip_unit_base", "must be a unit vector")


def _validate_reference(
    value: Any,
    *,
    width: int,
    height: int,
    path: str,
) -> Mapping[str, Any]:
    reference = _object(value, path)
    _keys(
        reference,
        required=(
            "instance_id",
            "class_id",
            "sort_label",
            "polygon_px",
            "bbox_xywh_px",
            "center_uv",
            "head_uv",
            "tip_uv",
            "axis_observability",
            "occlusion",
            "truncated",
            "touching_group_id",
            "ignore_for_scoring",
            "pose_truth",
        ),
        path=path,
    )
    _string(reference["instance_id"], f"{path}.instance_id")
    class_id = _enum(reference["class_id"], SUPPORTED_CLASSES, f"{path}.class_id")
    sort_label = _nullable_string(reference["sort_label"], f"{path}.sort_label")
    if class_id == "cube" and sort_label not in {"red", "yellow", "blue"}:
        _fail(f"{path}.sort_label", "a cube requires red, yellow, or blue")
    if class_id != "cube" and sort_label not in {None, "red"}:
        _fail(f"{path}.sort_label", "a screw may use null or red")
    polygon = _polygon(reference["polygon_px"], width, height, f"{path}.polygon_px")
    bbox = _vector(reference["bbox_xywh_px"], 4, f"{path}.bbox_xywh_px")
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= 0 or bbox[3] <= 0:
        _fail(f"{path}.bbox_xywh_px", "must be non-negative x/y and positive width/height")
    if bbox[0] + bbox[2] > width or bbox[1] + bbox[3] > height:
        _fail(f"{path}.bbox_xywh_px", "must stay within the image")
    if any(
        not (
            bbox[0] - 1e-9 <= u <= bbox[0] + bbox[2] + 1e-9
            and bbox[1] - 1e-9 <= v <= bbox[1] + bbox[3] + 1e-9
        )
        for u, v in polygon
    ):
        _fail(f"{path}.bbox_xywh_px", "must enclose polygon_px")
    center = _vector(reference["center_uv"], 2, f"{path}.center_uv")
    if not _point_in_polygon(center, polygon):
        _fail(f"{path}.center_uv", "must lie inside polygon_px")
    observability = _enum(
        reference["axis_observability"],
        {"directed", "undirected", "unobservable"},
        f"{path}.axis_observability",
    )
    head = None if reference["head_uv"] is None else _vector(reference["head_uv"], 2, f"{path}.head_uv")
    tip = None if reference["tip_uv"] is None else _vector(reference["tip_uv"], 2, f"{path}.tip_uv")
    if class_id == "cube":
        if head is not None or tip is not None or observability != "unobservable":
            _fail(path, "cube head/tip must be null and axis_observability unobservable")
    elif observability == "unobservable":
        if head is not None or tip is not None:
            _fail(path, "unobservable screw axis requires null head/tip")
    elif head is None or tip is None or math.dist(head, tip) <= 1e-9:
        _fail(path, "observable screw axis requires distinct head_uv and tip_uv")
    elif not _point_in_polygon(head, polygon) or not _point_in_polygon(tip, polygon):
        _fail(path, "observable screw head_uv and tip_uv must lie inside polygon_px")
    occlusion = _object(reference["occlusion"], f"{path}.occlusion")
    _keys(occlusion, required=("category", "fraction", "causes"), path=f"{path}.occlusion")
    category = _enum(occlusion["category"], {"none", "partial", "heavy", "unknown"}, f"{path}.occlusion.category")
    fraction = occlusion["fraction"]
    if fraction is not None:
        fraction_value = _number(fraction, f"{path}.occlusion.fraction")
        if not 0.0 <= fraction_value <= 1.0:
            _fail(f"{path}.occlusion.fraction", "must be in [0, 1]")
    if category == "none" and fraction not in {None, 0, 0.0}:
        _fail(f"{path}.occlusion", "category none cannot have positive fraction")
    causes = [
        _enum(item, {"cable", "hand", "robot", "container", "supported_object", "unknown"}, f"{path}.occlusion.causes[{index}]")
        for index, item in enumerate(_array(occlusion["causes"], f"{path}.occlusion.causes"))
    ]
    _unique(causes, f"{path}.occlusion.causes")
    _bool(reference["truncated"], f"{path}.truncated")
    _nullable_string(reference["touching_group_id"], f"{path}.touching_group_id")
    _bool(reference["ignore_for_scoring"], f"{path}.ignore_for_scoring")
    _validate_pose_truth(
        reference["pose_truth"],
        class_id,
        f"{path}.pose_truth",
    )
    return reference


def _validate_prediction(
    value: Any,
    *,
    width: int,
    height: int,
    path: str,
) -> Mapping[str, Any]:
    prediction = _object(value, path)
    _keys(
        prediction,
        required=(
            "prediction_id",
            "track_id",
            "class_id",
            "sort_label",
            "center_uv",
            "confidence",
            "pose_quality",
            "head_uv",
            "tip_uv",
            "axis_observability",
            "pose",
        ),
        path=path,
    )
    _string(prediction["prediction_id"], f"{path}.prediction_id")
    _string(prediction["track_id"], f"{path}.track_id")
    _string(prediction["class_id"], f"{path}.class_id")
    _nullable_string(prediction["sort_label"], f"{path}.sort_label")
    u, v = _vector(prediction["center_uv"], 2, f"{path}.center_uv")
    if not 0.0 <= u < width or not 0.0 <= v < height:
        _fail(f"{path}.center_uv", "must lie inside the image")
    confidence = _number(prediction["confidence"], f"{path}.confidence")
    if not 0.0 <= confidence <= 1.0:
        _fail(f"{path}.confidence", "must be in [0, 1]")
    if prediction["pose_quality"] is not None:
        quality = _number(prediction["pose_quality"], f"{path}.pose_quality")
        if not 0.0 <= quality <= 1.0:
            _fail(f"{path}.pose_quality", "must be in [0, 1]")
    class_id = prediction["class_id"]
    observability = _enum(
        prediction["axis_observability"],
        {"directed", "undirected", "unobservable"},
        f"{path}.axis_observability",
    )
    head = (
        None
        if prediction["head_uv"] is None
        else _vector(prediction["head_uv"], 2, f"{path}.head_uv")
    )
    tip = (
        None
        if prediction["tip_uv"] is None
        else _vector(prediction["tip_uv"], 2, f"{path}.tip_uv")
    )
    for name, point in (("head_uv", head), ("tip_uv", tip)):
        if point is not None and not (0.0 <= point[0] < width and 0.0 <= point[1] < height):
            _fail(f"{path}.{name}", "must lie inside the image")
    if class_id == "screw_iso10642_m6x30":
        if observability == "unobservable":
            if head is not None or tip is not None:
                _fail(path, "an unobservable predicted screw axis requires null head/tip")
        elif head is None or tip is None or math.dist(head, tip) <= 1e-9:
            _fail(path, "an observable predicted screw axis requires distinct head/tip")
    elif head is not None or tip is not None or observability != "unobservable":
        _fail(path, "non-screw predictions require null head/tip and an unobservable axis")
    pose = prediction["pose"]
    if pose is not None:
        pose_object = _object(pose, f"{path}.pose")
        _keys(pose_object, required=("position_base_m", "orientation_xyzw"), path=f"{path}.pose")
        _vector(pose_object["position_base_m"], 3, f"{path}.pose.position_base_m")
        quaternion = _vector(pose_object["orientation_xyzw"], 4, f"{path}.pose.orientation_xyzw")
        norm = math.sqrt(sum(component * component for component in quaternion))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            _fail(f"{path}.pose.orientation_xyzw", "must be a unit quaternion")
    return prediction


def validate_manifest(value: Any, *, source: str = "manifest") -> Mapping[str, Any]:
    manifest = _object(value, source)
    _keys(
        manifest,
        required=(
            "schema_version",
            "annotation_id",
            "created_at_utc",
            "split",
            "backend",
            "provenance",
            "source",
            "frames",
        ),
        path=source,
    )
    if manifest["schema_version"] != SCHEMA_VERSION or isinstance(manifest["schema_version"], bool):
        _fail(f"{source}.schema_version", "must be the integer 1")
    _string(manifest["annotation_id"], f"{source}.annotation_id")
    _utc_datetime(manifest["created_at_utc"], f"{source}.created_at_utc")
    split = _enum(manifest["split"], {"pilot", "train", "validation", "test"}, f"{source}.split")
    _enum(manifest["backend"], {"offline_replay", "isaac", "hardware"}, f"{source}.backend")

    provenance = _object(manifest["provenance"], f"{source}.provenance")
    _keys(
        provenance,
        required=(
            "project_commit",
            "branch",
            "integration_profile",
            "provider",
            "object_catalog_version",
            "calibration_ids",
            "review",
        ),
        path=f"{source}.provenance",
    )
    _sha(provenance["project_commit"], 40, f"{source}.provenance.project_commit")
    _string(provenance["branch"], f"{source}.provenance.branch")
    profile = provenance["integration_profile"]
    if profile is not None:
        profile_object = _object(profile, f"{source}.provenance.integration_profile")
        _keys(profile_object, required=("profile_id", "profile_sha256"), path=f"{source}.provenance.integration_profile")
        _string(profile_object["profile_id"], f"{source}.provenance.integration_profile.profile_id")
        _sha(profile_object["profile_sha256"], 64, f"{source}.provenance.integration_profile.profile_sha256")
    provider = _object(provenance["provider"], f"{source}.provenance.provider")
    _keys(provider, required=("name", "version", "artifact_sha256"), path=f"{source}.provenance.provider")
    _string(provider["name"], f"{source}.provenance.provider.name")
    _string(provider["version"], f"{source}.provenance.provider.version")
    if provider["artifact_sha256"] is not None:
        _sha(provider["artifact_sha256"], 64, f"{source}.provenance.provider.artifact_sha256")
    _string(provenance["object_catalog_version"], f"{source}.provenance.object_catalog_version")
    calibration_ids = [
        _string(item, f"{source}.provenance.calibration_ids[{index}]")
        for index, item in enumerate(_array(provenance["calibration_ids"], f"{source}.provenance.calibration_ids"))
    ]
    _unique(calibration_ids, f"{source}.provenance.calibration_ids")
    review = _object(provenance["review"], f"{source}.provenance.review")
    _keys(review, required=("annotator_id", "review_status", "reviewer_id"), path=f"{source}.provenance.review")
    _string(review["annotator_id"], f"{source}.provenance.review.annotator_id")
    status = _enum(review["review_status"], {"unreviewed", "reviewed", "adjudicated"}, f"{source}.provenance.review.review_status")
    reviewer = _nullable_string(review["reviewer_id"], f"{source}.provenance.review.reviewer_id")
    if status != "unreviewed" and reviewer is None:
        _fail(f"{source}.provenance.review.reviewer_id", "is required after review")
    if status == "unreviewed" and reviewer is not None:
        _fail(
            f"{source}.provenance.review.reviewer_id",
            "must be null while review_status is unreviewed",
        )
    if split in {"validation", "test"} and status == "unreviewed":
        _fail(
            f"{source}.provenance.review.review_status",
            f"{split} evidence must be reviewed or adjudicated",
        )

    source_object = _object(manifest["source"], f"{source}.source")
    _keys(
        source_object,
        required=(
            "dataset_id",
            "source_run_id",
            "capture_sha256",
            "camera_id",
            "image_width_px",
            "image_height_px",
            "timestamp_source",
        ),
        path=f"{source}.source",
    )
    _string(source_object["dataset_id"], f"{source}.source.dataset_id")
    _string(source_object["source_run_id"], f"{source}.source.source_run_id")
    _sha(source_object["capture_sha256"], 64, f"{source}.source.capture_sha256")
    _string(source_object["camera_id"], f"{source}.source.camera_id")
    width = _integer(source_object["image_width_px"], f"{source}.source.image_width_px", minimum=2)
    height = _integer(source_object["image_height_px"], f"{source}.source.image_height_px", minimum=2)
    _enum(source_object["timestamp_source"], {"measured_exposure", "backend_observation", "frame_index_over_fps", "unknown"}, f"{source}.source.timestamp_source")

    frames = _array(manifest["frames"], f"{source}.frames")
    if not frames:
        _fail(f"{source}.frames", "must contain at least one frame")
    frame_ids: list[str] = []
    frame_indices: list[int] = []
    identities: dict[tuple[str, str], tuple[str, str | None]] = {}
    has_base_pose_claim = False
    for frame_index, frame_value in enumerate(frames):
        frame_path = f"{source}.frames[{frame_index}]"
        frame = _object(frame_value, frame_path)
        _keys(
            frame,
            required=(
                "frame_id",
                "layout_id",
                "frame_index",
                "capture_time_s",
                "image_sha256",
                "inference_latency_ms",
                "references",
                "predictions",
                "regions",
            ),
            path=frame_path,
        )
        frame_ids.append(_string(frame["frame_id"], f"{frame_path}.frame_id"))
        _string(frame["layout_id"], f"{frame_path}.layout_id")
        frame_indices.append(_integer(frame["frame_index"], f"{frame_path}.frame_index"))
        if frame["capture_time_s"] is not None:
            _number(frame["capture_time_s"], f"{frame_path}.capture_time_s")
        _sha(frame["image_sha256"], 64, f"{frame_path}.image_sha256")
        if frame["inference_latency_ms"] is not None:
            _nonnegative(frame["inference_latency_ms"], f"{frame_path}.inference_latency_ms")
        references = [
            _validate_reference(item, width=width, height=height, path=f"{frame_path}.references[{index}]")
            for index, item in enumerate(_array(frame["references"], f"{frame_path}.references"))
        ]
        has_base_pose_claim = has_base_pose_claim or any(
            item["pose_truth"] is not None for item in references
        )
        _unique([item["instance_id"] for item in references], f"{frame_path}.references.instance_id")
        for reference in references:
            identity_key = (frame["layout_id"], reference["instance_id"])
            identity_value = (reference["class_id"], reference["sort_label"])
            previous_identity = identities.get(identity_key)
            if previous_identity is not None and previous_identity != identity_value:
                _fail(
                    f"{frame_path}.references",
                    "one layout/instance_id must keep the same class_id and sort_label",
                )
            identities[identity_key] = identity_value
        predictions = [
            _validate_prediction(item, width=width, height=height, path=f"{frame_path}.predictions[{index}]")
            for index, item in enumerate(_array(frame["predictions"], f"{frame_path}.predictions"))
        ]
        has_base_pose_claim = has_base_pose_claim or any(
            item["pose"] is not None for item in predictions
        )
        _unique([item["prediction_id"] for item in predictions], f"{frame_path}.predictions.prediction_id")
        _unique([item["track_id"] for item in predictions], f"{frame_path}.predictions.track_id")
        region_ids: list[str] = []
        for region_index, region_value in enumerate(_array(frame["regions"], f"{frame_path}.regions")):
            region_path = f"{frame_path}.regions[{region_index}]"
            region = _object(region_value, region_path)
            _keys(region, required=("region_id", "kind", "polygon_px"), path=region_path)
            region_ids.append(_string(region["region_id"], f"{region_path}.region_id"))
            _enum(region["kind"], {"ignore", "unknown", "cable", "hand", "robot", "container", "unsupported_object"}, f"{region_path}.kind")
            _polygon(region["polygon_px"], width, height, f"{region_path}.polygon_px")
        _unique(region_ids, f"{frame_path}.regions.region_id")
    _unique(frame_ids, f"{source}.frames.frame_id")
    _unique([str(index) for index in frame_indices], f"{source}.frames.frame_index")
    ordered_frames = sorted(frames, key=lambda item: item["frame_index"])
    frame_times = [item["capture_time_s"] for item in ordered_frames]
    known_times = [float(value) for value in frame_times if value is not None]
    if any(
        current < previous - 1e-9
        for previous, current in zip(known_times, known_times[1:])
    ):
        _fail(f"{source}.frames", "capture_time_s must not run backwards")
    timestamp_source = source_object["timestamp_source"]
    if timestamp_source in {"measured_exposure", "backend_observation", "frame_index_over_fps"} and any(
        value is None for value in frame_times
    ):
        _fail(
            f"{source}.frames.capture_time_s",
            f"cannot be null when timestamp_source is {timestamp_source}",
        )
    if has_base_pose_claim and not calibration_ids:
        _fail(
            f"{source}.provenance.calibration_ids",
            "must identify calibration provenance for robot-base pose claims",
        )
    return manifest


def _reject_constant(value: str) -> None:
    raise AnnotationValidationError(f"non-standard JSON number {value!r} is not allowed")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnnotationValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnnotationValidationError(f"cannot read strict JSON {path}: {exc}") from exc
    return validate_manifest(raw, source=str(path))


def _quaternion_matrix(quaternion: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    x, y, z, w = (float(component) for component in quaternion)
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _cube_orientation_error_deg(
    quaternion: Sequence[float], reference_yaw_rad: float
) -> float:
    """Full 3D rotation error minimized over upright square yaw symmetry."""

    predicted = _quaternion_matrix(quaternion)
    errors = []
    for quarter_turn in range(4):
        yaw = float(reference_yaw_rad) + quarter_turn * math.pi / 2.0
        expected = (
            (math.cos(yaw), -math.sin(yaw), 0.0),
            (math.sin(yaw), math.cos(yaw), 0.0),
            (0.0, 0.0, 1.0),
        )
        relative_trace = sum(
            predicted[row][column] * expected[row][column]
            for row in range(3)
            for column in range(3)
        )
        cosine = max(-1.0, min(1.0, (relative_trace - 1.0) / 2.0))
        errors.append(math.degrees(math.acos(cosine)))
    return min(errors)


def _screw_axis(quaternion: Sequence[float]) -> tuple[float, float, float]:
    rotation = _quaternion_matrix(quaternion)
    return (-rotation[0][2], -rotation[1][2], -rotation[2][2])


def _axis_error_deg(predicted: Sequence[float], reference: Sequence[float], *, directed: bool) -> float:
    dot = max(-1.0, min(1.0, sum(float(a) * float(b) for a, b in zip(predicted, reference))))
    angle = math.degrees(math.acos(dot))
    return angle if directed else min(angle, 180.0 - angle)


def _unit_image_axis(head: Sequence[float], tip: Sequence[float]) -> tuple[float, float]:
    vector = (float(tip[0]) - float(head[0]), float(tip[1]) - float(head[1]))
    norm = math.hypot(*vector)
    if norm <= 1e-12:  # validation rejects this; retain a fail-closed guard
        raise AnnotationValidationError("image screw axis has zero length")
    return (vector[0] / norm, vector[1] / norm)


def _confidence_stratum(confidence: float) -> str:
    """Fixed reporting bins, not runtime acceptance thresholds."""

    if confidence < 0.5:
        return "[0.0,0.5)"
    if confidence < 0.8:
        return "[0.5,0.8)"
    return "[0.8,1.0]"


def _empty_reference_outcomes() -> dict[str, int]:
    return {"reference_instances": 0, "true_positives": 0, "dropouts": 0}


def _prediction_ignored(prediction: Mapping[str, Any], regions: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        region["kind"] in {"ignore", "unknown"}
        and _point_in_polygon(prediction["center_uv"], region["polygon_px"])
        for region in regions
    )


def _minimum_cost_maximum_matching(
    references: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int]]:
    """Return maximum-cardinality matches, then minimize summed center distance.

    Successive shortest augmenting paths on the residual graph avoid the
    cardinality failure of a greedy nearest-pair assignment. Inputs and edges
    are sorted by stable IDs, so exact-cost ties resolve reproducibly.
    """

    reference_order = sorted(
        range(len(references)), key=lambda index: references[index]["instance_id"]
    )
    prediction_order = sorted(
        range(len(predictions)), key=lambda index: predictions[index]["prediction_id"]
    )
    reference_count = len(reference_order)
    prediction_count = len(prediction_order)
    source = 0
    first_reference = 1
    first_prediction = first_reference + reference_count
    sink = first_prediction + prediction_count
    graph: list[list[list[float | int]]] = [[] for _ in range(sink + 1)]

    def add_edge(start: int, end: int, capacity: int, cost: float) -> list[float | int]:
        forward: list[float | int] = [end, len(graph[end]), capacity, cost]
        reverse: list[float | int] = [start, len(graph[start]), 0, -cost]
        graph[start].append(forward)
        graph[end].append(reverse)
        return forward

    for ordered_reference_index in range(reference_count):
        add_edge(source, first_reference + ordered_reference_index, 1, 0.0)
    for ordered_prediction_index in range(prediction_count):
        add_edge(first_prediction + ordered_prediction_index, sink, 1, 0.0)

    candidate_edges: dict[tuple[int, int], list[float | int]] = {}
    for ordered_reference_index, reference_index in enumerate(reference_order):
        reference = references[reference_index]
        for ordered_prediction_index, prediction_index in enumerate(prediction_order):
            prediction = predictions[prediction_index]
            if prediction["class_id"] != reference["class_id"]:
                continue
            if prediction["sort_label"] != reference["sort_label"]:
                continue
            if not _point_in_polygon(prediction["center_uv"], reference["polygon_px"]):
                continue
            distance = math.dist(prediction["center_uv"], reference["center_uv"])
            candidate_edges[(reference_index, prediction_index)] = add_edge(
                first_reference + ordered_reference_index,
                first_prediction + ordered_prediction_index,
                1,
                distance,
            )

    node_count = len(graph)
    while True:
        distances = [math.inf] * node_count
        previous: list[tuple[int, int] | None] = [None] * node_count
        distances[source] = 0.0
        for _ in range(node_count - 1):
            changed = False
            for start in range(node_count):
                if not math.isfinite(distances[start]):
                    continue
                for edge_index, edge in enumerate(graph[start]):
                    end, _, capacity, cost = edge
                    if int(capacity) <= 0:
                        continue
                    candidate_distance = distances[start] + float(cost)
                    if candidate_distance < distances[int(end)] - 1e-12:
                        distances[int(end)] = candidate_distance
                        previous[int(end)] = (start, edge_index)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            break
        node = sink
        while node != source:
            step = previous[node]
            if step is None:  # pragma: no cover - guarded by the path search
                raise RuntimeError("incomplete residual path")
            start, edge_index = step
            edge = graph[start][edge_index]
            reverse_index = int(edge[1])
            edge[2] = int(edge[2]) - 1
            graph[node][reverse_index][2] = int(graph[node][reverse_index][2]) + 1
            node = start

    return sorted(
        (
            (reference_index, prediction_index)
            for (reference_index, prediction_index), edge in candidate_edges.items()
            if int(edge[2]) == 0
        ),
        key=lambda pair: (
            references[pair[0]]["instance_id"],
            predictions[pair[1]]["prediction_id"],
        ),
    )


def _match_frame(
    frame: Mapping[str, Any],
) -> tuple[
    list[tuple[Mapping[str, Any], Mapping[str, Any]]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
]:
    references = [
        item for item in frame["references"] if not item["ignore_for_scoring"]
    ]
    ignored_reference_polygons = [
        item["polygon_px"] for item in frame["references"] if item["ignore_for_scoring"]
    ]
    abstentions = [
        item for item in frame["predictions"] if item["class_id"] not in SUPPORTED_CLASSES
    ]
    supported_predictions = [
        item for item in frame["predictions"] if item["class_id"] in SUPPORTED_CLASSES
    ]
    predictions: list[Mapping[str, Any]] = []
    ignored_predictions: list[Mapping[str, Any]] = []
    for prediction in supported_predictions:
        overlaps_scored_reference = any(
            _point_in_polygon(prediction["center_uv"], reference["polygon_px"])
            for reference in references
        )
        suppressed = _prediction_ignored(prediction, frame["regions"]) or any(
            _point_in_polygon(prediction["center_uv"], polygon)
            for polygon in ignored_reference_polygons
        )
        # A broad ignore/unknown region must never erase an explicitly scored
        # object. A prediction inside a scored mask remains accountable.
        if suppressed and not overlaps_scored_reference:
            ignored_predictions.append(prediction)
        else:
            predictions.append(prediction)

    matched_pairs = _minimum_cost_maximum_matching(references, predictions)
    matched_reference = {reference_index for reference_index, _ in matched_pairs}
    matched_prediction = {prediction_index for _, prediction_index in matched_pairs}
    matches = [
        (references[reference_index], predictions[prediction_index])
        for reference_index, prediction_index in matched_pairs
    ]
    unmatched_references = [item for index, item in enumerate(references) if index not in matched_reference]
    unmatched_predictions = [item for index, item in enumerate(predictions) if index not in matched_prediction]
    return (
        matches,
        unmatched_references,
        unmatched_predictions,
        ignored_predictions,
        abstentions,
    )


def evaluate_manifest(manifest_value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = validate_manifest(manifest_value)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"reference_instances": 0, "true_positives": 0, "false_positives": 0, "dropouts": 0})
    pose_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    track_history: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    residual_history: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    latencies: list[float] = []
    ignored_predictions = 0
    abstentions_by_class: dict[str, int] = defaultdict(int)
    confidence_strata = {
        name: {
            "matched_supported": 0,
            "false_positive_supported": 0,
            "ignored_supported": 0,
            "abstention": 0,
        }
        for name in ("[0.0,0.5)", "[0.5,0.8)", "[0.8,1.0]")
    }
    reference_strata: dict[str, dict[str, dict[str, int]]] = {
        "occlusion": defaultdict(_empty_reference_outcomes),
        "truncation": defaultdict(_empty_reference_outcomes),
        "touching": defaultdict(_empty_reference_outcomes),
    }
    image_axis_samples: dict[str, list[dict[str, Any]]] = {
        "directed": [],
        "undirected": [],
    }
    image_axis_unavailable: dict[str, int] = defaultdict(int)
    matches_total = 0
    matched_without_pose_pair = 0
    pose_truth_uncertainties_mm: list[float] = []
    layout_latency: dict[str, float] = defaultdict(float)
    layout_ids: set[str] = set()

    for frame in sorted(manifest["frames"], key=lambda item: item["frame_index"]):
        layout_id = frame["layout_id"]
        layout_ids.add(layout_id)
        if frame["inference_latency_ms"] is not None:
            latency = float(frame["inference_latency_ms"])
            latencies.append(latency)
            layout_latency[layout_id] += latency
        (
            matches,
            misses,
            false_predictions,
            ignored_items,
            abstentions,
        ) = _match_frame(frame)
        ignored_predictions += len(ignored_items)
        for prediction in ignored_items:
            confidence_strata[_confidence_stratum(float(prediction["confidence"]))][
                "ignored_supported"
            ] += 1
        for prediction in abstentions:
            abstentions_by_class[prediction["class_id"]] += 1
            confidence_strata[_confidence_stratum(float(prediction["confidence"]))][
                "abstention"
            ] += 1
        for reference in [item for item in frame["references"] if not item["ignore_for_scoring"]]:
            counts[reference["class_id"]]["reference_instances"] += 1
        for reference, prediction in matches:
            class_id = reference["class_id"]
            counts[class_id]["true_positives"] += 1
            confidence_strata[_confidence_stratum(float(prediction["confidence"]))][
                "matched_supported"
            ] += 1
            matches_total += 1
            for dimension, stratum in (
                ("occlusion", reference["occlusion"]["category"]),
                ("truncation", "truncated" if reference["truncated"] else "not_truncated"),
                ("touching", "touching" if reference["touching_group_id"] is not None else "isolated"),
            ):
                reference_strata[dimension][stratum]["reference_instances"] += 1
                reference_strata[dimension][stratum]["true_positives"] += 1
            track_history[(layout_id, reference["instance_id"])].append((frame["frame_index"], prediction["track_id"]))
            if class_id == "screw_iso10642_m6x30":
                reference_axis_mode = reference["axis_observability"]
                prediction_axis_mode = prediction["axis_observability"]
                if reference_axis_mode == "unobservable":
                    image_axis_unavailable["reference_unobservable"] += 1
                elif prediction_axis_mode == "unobservable":
                    image_axis_unavailable["prediction_unobservable"] += 1
                elif reference_axis_mode == "directed" and prediction_axis_mode != "directed":
                    image_axis_unavailable["prediction_direction_unobservable"] += 1
                else:
                    reference_axis = _unit_image_axis(reference["head_uv"], reference["tip_uv"])
                    prediction_axis = _unit_image_axis(prediction["head_uv"], prediction["tip_uv"])
                    mode = "directed" if reference_axis_mode == "directed" else "undirected"
                    image_axis_samples[mode].append(
                        {
                            "frame_id": frame["frame_id"],
                            "instance_id": reference["instance_id"],
                            "prediction_id": prediction["prediction_id"],
                            "error_deg": _axis_error_deg(
                                prediction_axis,
                                reference_axis,
                                directed=mode == "directed",
                            ),
                        }
                    )
            truth = reference["pose_truth"]
            pose = prediction["pose"]
            if truth is None or pose is None:
                matched_without_pose_pair += 1
                continue
            pose_truth_uncertainties_mm.append(
                float(truth["position_uncertainty_mm"])
            )
            reference_position = tuple(float(value) for value in truth["position_base_m"])
            predicted_position = tuple(float(value) for value in pose["position_base_m"])
            residual = tuple(predicted - expected for predicted, expected in zip(predicted_position, reference_position))
            residual_history[(layout_id, reference["instance_id"])].append(residual)
            position_error_mm = 1000.0 * math.sqrt(sum(value * value for value in residual))
            quaternion = pose["orientation_xyzw"]
            if class_id == "cube":
                orientation_error = _cube_orientation_error_deg(
                    quaternion, float(truth["cube_yaw_rad"])
                )
            elif reference["axis_observability"] == "unobservable":
                orientation_error = None
            else:
                orientation_error = _axis_error_deg(
                    _screw_axis(quaternion),
                    truth["screw_head_to_tip_unit_base"],
                    directed=reference["axis_observability"] == "directed",
                )
            pose_samples[class_id].append(
                {
                    "position_error_mm": position_error_mm,
                    "symmetry_aware_orientation_error_deg": orientation_error,
                    "symmetry_id": truth["symmetry_id"],
                }
            )
        for reference in misses:
            counts[reference["class_id"]]["dropouts"] += 1
            for dimension, stratum in (
                ("occlusion", reference["occlusion"]["category"]),
                ("truncation", "truncated" if reference["truncated"] else "not_truncated"),
                ("touching", "touching" if reference["touching_group_id"] is not None else "isolated"),
            ):
                reference_strata[dimension][stratum]["reference_instances"] += 1
                reference_strata[dimension][stratum]["dropouts"] += 1
        for prediction in false_predictions:
            counts[prediction["class_id"]]["false_positives"] += 1
            confidence_strata[_confidence_stratum(float(prediction["confidence"]))][
                "false_positive_supported"
            ] += 1

    per_class = []
    for class_id in sorted(counts):
        result = {"class_id": class_id, **counts[class_id], "pose_samples": pose_samples[class_id]}
        per_class.append(result)

    id_switches_by_instance: dict[str, int] = {}
    for (layout_id, instance_id), entries in sorted(track_history.items()):
        ordered = [track_id for _, track_id in sorted(entries)]
        switches = sum(previous != current for previous, current in zip(ordered, ordered[1:]))
        id_switches_by_instance[f"{layout_id}:{instance_id}"] = switches

    residual_jitter: dict[str, dict[str, float | int]] = {}
    for (layout_id, instance_id), residuals in sorted(residual_history.items()):
        if len(residuals) < 2:
            continue
        mean = tuple(sum(vector[axis] for vector in residuals) / len(residuals) for axis in range(3))
        deviations_mm = [1000.0 * math.dist(vector, mean) for vector in residuals]
        residual_jitter[f"{layout_id}:{instance_id}"] = {
            "samples": len(residuals),
            "rms_residual_jitter_mm": math.sqrt(sum(value * value for value in deviations_mm) / len(deviations_mm)),
            "maximum_residual_jitter_mm": max(deviations_mm),
        }

    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    manifest_sha256 = hashlib.sha256(canonical).hexdigest()
    evaluator_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    evaluated_at_utc = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    evaluation_identity = hashlib.sha256(
        f"{manifest_sha256}:{EVALUATOR_VERSION}:{evaluator_sha256}".encode("ascii")
    ).hexdigest()
    profile = manifest["provenance"]["integration_profile"]
    provider = manifest["provenance"]["provider"]
    episodes = []
    for layout_id in sorted(layout_ids):
        timing = {}
        if layout_id in layout_latency:
            timing["perception_s"] = layout_latency[layout_id] / 1000.0
        episodes.append(
            {
                "episode_id": f"{manifest['annotation_id']}:{layout_id}",
                "condition_id": layout_id,
                "scene_id": f"{manifest['source']['source_run_id']}:{layout_id}",
                "target_track_id": None,
                "target_class_id": None,
                "grasp_attempt": None,
                "outcome": {
                    "status": "not_executed",
                    "contact_count": 0,
                    "operator_intervention_count": 0,
                    "retry_count": 0,
                    "verification": [],
                },
                "timing_s": timing,
            }
        )
    evaluation = {
        "schema_version": 1,
        "evaluation_id": f"annotation-{evaluation_identity[:16]}",
        "created_at_utc": evaluated_at_utc,
        "split": manifest["split"],
        "backend": manifest["backend"],
        "provenance": {
            "project_commit": manifest["provenance"]["project_commit"],
            "branch": manifest["provenance"]["branch"],
            "integration_profile": profile,
            "object_catalog_version": manifest["provenance"]["object_catalog_version"],
            "calibration_ids": manifest["provenance"]["calibration_ids"],
            "components": [
                {
                    "component": "vision",
                    "name": provider["name"],
                    "version": provider["version"],
                    "artifact_sha256": provider["artifact_sha256"],
                }
            ],
            "source_run_ids": [manifest["source"]["source_run_id"]],
        },
        "vision": {
            "frames_evaluated": len(manifest["frames"]),
            "inference_latency_ms": latencies,
            "per_class": per_class,
        },
        "episodes": episodes,
    }
    diagnostics = {
        "schema_version": 1,
        "annotation_id": manifest["annotation_id"],
        "manifest_sha256": manifest_sha256,
        "frames_evaluated": len(manifest["frames"]),
        "matches": matches_total,
        "ignored_predictions": ignored_predictions,
        "abstentions_total": sum(abstentions_by_class.values()),
        "abstentions_by_class": dict(sorted(abstentions_by_class.items())),
        "matched_without_pose_pair": matched_without_pose_pair,
        "pose_truth_uncertainty_mm": {
            "samples": len(pose_truth_uncertainties_mm),
            "minimum": (
                min(pose_truth_uncertainties_mm)
                if pose_truth_uncertainties_mm
                else None
            ),
            "maximum": (
                max(pose_truth_uncertainties_mm)
                if pose_truth_uncertainties_mm
                else None
            ),
        },
        "id_switches_total": sum(id_switches_by_instance.values()),
        "id_switches_by_instance": id_switches_by_instance,
        "position_residual_jitter": residual_jitter,
        "image_axis_error_deg": {
            "directed": image_axis_samples["directed"],
            "undirected": image_axis_samples["undirected"],
            "unavailable": dict(sorted(image_axis_unavailable.items())),
        },
        "confidence_strata": confidence_strata,
        "reference_strata": {
            dimension: {
                stratum: outcomes
                for stratum, outcomes in sorted(strata.items())
            }
            for dimension, strata in reference_strata.items()
        },
        "review_provenance": dict(manifest["provenance"]["review"]),
        "evaluation_provenance": {
            "evaluator": "team_plugins.computer_vision.evaluate_annotations",
            "evaluator_version": EVALUATOR_VERSION,
            "evaluator_artifact_sha256": evaluator_sha256,
            "annotation_created_at_utc": manifest["created_at_utc"],
            "evaluated_at_utc": evaluated_at_utc,
        },
        "evidence_limit": (
            "Robot-base pose errors exist only for independently measured pose_truth; "
            "image annotations alone cannot establish millimetre accuracy."
        ),
    }
    return evaluation, diagnostics


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--evaluation-out", type=Path, required=True)
    parser.add_argument("--diagnostics-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    evaluation_path = args.evaluation_out.expanduser().resolve()
    diagnostics_path = args.diagnostics_out.expanduser().resolve()
    if evaluation_path == manifest_path or diagnostics_path == manifest_path:
        raise AnnotationValidationError("output paths must not overwrite the annotation manifest")
    if evaluation_path == diagnostics_path:
        raise AnnotationValidationError("evaluation and diagnostics outputs must differ")
    manifest = load_manifest(manifest_path)
    evaluation, diagnostics = evaluate_manifest(manifest)
    _write_json(evaluation_path, evaluation)
    _write_json(diagnostics_path, diagnostics)
    print(
        json.dumps(
            {
                "evaluation_out": str(evaluation_path),
                "diagnostics_out": str(diagnostics_path),
                "frames_evaluated": diagnostics["frames_evaluated"],
                "matches": diagnostics["matches"],
                "id_switches_total": diagnostics["id_switches_total"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
