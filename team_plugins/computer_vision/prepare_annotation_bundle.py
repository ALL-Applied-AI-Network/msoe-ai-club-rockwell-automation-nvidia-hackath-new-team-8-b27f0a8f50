"""Prepare a lossless, reviewable pilot annotation bundle from a clean replay.

This is an offline evidence tool.  It never imports the robot client or emits
robot actions.  The input frame-index file contains one non-negative integer
per line, in strictly increasing order; blank lines and ``#`` comments are
allowed.  Every requested index must already exist in the clean replay's
``scenes.jsonl``.

The generated manifest intentionally has empty ``references`` and ``regions``.
It is schema-valid, but it is not ground truth until a human annotates those
fields and records an independent base-frame measurement where appropriate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from hardware.fanuc_cell.calibration.geometry import load as load_calibration
from hardware.fanuc_cell.calibration.geometry import project
from sorting_core import scene_observation_from_dict
from team_plugins.computer_vision.evaluate_annotations import validate_manifest
from team_plugins.computer_vision.replay_camera import _draw_overlay, _frames


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION = (
    REPO_ROOT / "hardware" / "fanuc_cell" / "calibration" / "cameras.json"
)
DEFAULT_PROFILE = Path(__file__).with_name("profile.hybrid-geometry.v1.json")
DEFAULT_PROVIDER = Path(__file__).with_name("hybrid_geometry_provider.py")
DEFAULT_REPLAY_TOOL = Path(__file__).with_name("replay_camera.py")
DEFAULT_CATALOG = REPO_ROOT / "docs" / "OBJECT_CATALOG.md"


class AnnotationBundleError(ValueError):
    """Raised when replay evidence cannot produce an attributable bundle."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_constant(value: str) -> None:
    raise AnnotationBundleError(f"non-standard JSON number {value!r} is not allowed")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnnotationBundleError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnnotationBundleError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnnotationBundleError(f"JSON root must be an object: {path}")
    return value


def _required_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise AnnotationBundleError(f"{path} must be an object")
    return value


def _required_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise AnnotationBundleError(f"{path} must be an array")
    return value


def _required_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AnnotationBundleError(f"{path} must be a non-empty trimmed string")
    return value


def read_frame_indices(path: Path) -> tuple[int, ...]:
    """Read a strictly increasing one-index-per-line selection file."""

    indices: list[int] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        token = raw.split("#", 1)[0].strip()
        if not token:
            continue
        if not token.isdecimal():
            raise AnnotationBundleError(
                f"{path}:{line_number}: expected one non-negative integer"
            )
        value = int(token)
        if indices and value <= indices[-1]:
            raise AnnotationBundleError(
                f"{path}:{line_number}: indices must be unique and strictly increasing"
            )
        indices.append(value)
    if not indices:
        raise AnnotationBundleError("frame-index file did not select any frames")
    return tuple(indices)


def _load_scenes(path: Path) -> dict[int, Mapping[str, Any]]:
    records: dict[int, Mapping[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AnnotationBundleError(f"cannot read replay scenes {path}: {exc}") from exc
    if not lines:
        raise AnnotationBundleError("replay scenes file is empty")
    for line_number, raw in enumerate(lines, 1):
        try:
            record = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_constant,
            )
        except json.JSONDecodeError as exc:
            raise AnnotationBundleError(
                f"{path}:{line_number}: invalid JSON: {exc}"
            ) from exc
        record = _required_mapping(record, f"{path}:{line_number}")
        frame_index = record.get("frame_index")
        if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
            raise AnnotationBundleError(
                f"{path}:{line_number}.frame_index must be a non-negative integer"
            )
        if frame_index in records:
            raise AnnotationBundleError(f"duplicate replay frame index {frame_index}")
        scene_value = _required_mapping(
            record.get("scene"), f"{path}:{line_number}.scene"
        )
        scene = scene_observation_from_dict(scene_value)
        timestamp = record.get("timestamp_s")
        if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
            raise AnnotationBundleError(
                f"{path}:{line_number}.timestamp_s must be a finite number"
            )
        if not np.isfinite(float(timestamp)) or not np.isclose(
            float(timestamp), scene.capture_time_s, rtol=0.0, atol=1e-9
        ):
            raise AnnotationBundleError(
                f"replay frame {frame_index} timestamp does not match its scene"
            )
        latency = record.get("latency_ms")
        if (
            not isinstance(latency, (int, float))
            or isinstance(latency, bool)
            or not np.isfinite(float(latency))
            or float(latency) < 0.0
        ):
            raise AnnotationBundleError(
                f"replay frame {frame_index} has invalid latency_ms"
            )
        input_rgb_sha256 = record.get("input_rgb_sha256")
        if (
            not isinstance(input_rgb_sha256, str)
            or len(input_rgb_sha256) != 64
            or any(character not in "0123456789abcdef" for character in input_rgb_sha256)
        ):
            raise AnnotationBundleError(
                f"replay frame {frame_index} lacks a valid input_rgb_sha256"
            )
        input_shape = record.get("input_rgb_shape_hwc")
        if (
            not isinstance(input_shape, list)
            or len(input_shape) != 3
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in input_shape
            )
            or input_shape[2] != 3
        ):
            raise AnnotationBundleError(
                f"replay frame {frame_index} has invalid input_rgb_shape_hwc"
            )
        if record.get("input_rgb_dtype") != "uint8":
            raise AnnotationBundleError(
                f"replay frame {frame_index} input_rgb_dtype is not uint8"
            )
        if record.get("input_channel_order") != "RGB":
            raise AnnotationBundleError(
                f"replay frame {frame_index} input_channel_order is not RGB"
            )
        records[frame_index] = record
    return records


def _git_worktree_dirty() -> bool:
    result = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=normal"),
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise AnnotationBundleError(
            "cannot verify repository cleanliness: "
            + (result.stderr.strip() or "git status failed")
        )
    return bool(result.stdout.strip())


def _git_value(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    value = result.stdout.strip()
    if result.returncode != 0 or not value:
        raise AnnotationBundleError(
            "cannot record repository provenance: "
            + (result.stderr.strip() or "git command returned no value")
        )
    return value


def _inside_repository(path: Path) -> bool:
    try:
        path.relative_to(REPO_ROOT)
        return True
    except ValueError:
        return False


def _verify_replay(
    *,
    report: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    scenes: Mapping[int, Mapping[str, Any]],
    selected_indices: Sequence[int],
    capture_sha256: str,
    calibration_sha256: str,
    scenes_jsonl_sha256: str,
    calibration_path: Path,
    profile_path: Path,
    provider_path: Path,
    replay_tool_path: Path,
    catalog_path: Path,
    verify_repository: bool,
    verify_local_artifacts: bool,
) -> None:
    if report.get("tool") != "team_plugins.computer_vision.replay_camera":
        raise AnnotationBundleError("report was not produced by the attributed replay tool")
    if any(
        report.get(name) is not False
        for name in ("worktree_dirty", "worktree_dirty_at_start", "worktree_dirty_at_end")
    ):
        raise AnnotationBundleError("replay report is not from a clean worktree")
    if report.get("provenance_stable_during_replay") is not True:
        raise AnnotationBundleError("replay provenance was not stable")
    if report.get("provenance_fields_changed_during_replay") != []:
        raise AnnotationBundleError("replay reports changed provenance fields")
    if report.get("evaluation_result_skipped_reason") is not None:
        raise AnnotationBundleError("replay did not emit a clean standardized result")
    if report.get("capture_sha256") != capture_sha256:
        raise AnnotationBundleError("capture hash does not match replay report")
    if report.get("calibration_sha256") != calibration_sha256:
        raise AnnotationBundleError("calibration hash does not match replay report")
    reported_scenes_hash = report.get("scenes_jsonl_sha256")
    if (
        not isinstance(reported_scenes_hash, str)
        or len(reported_scenes_hash) != 64
        or any(
            character not in "0123456789abcdef"
            for character in reported_scenes_hash
        )
    ):
        raise AnnotationBundleError(
            "replay report lacks a valid scenes.jsonl SHA-256"
        )
    if reported_scenes_hash != scenes_jsonl_sha256:
        raise AnnotationBundleError(
            "replay scenes.jsonl hash does not match replay report"
        )

    report_indices = report.get("processed_frame_indices")
    if (
        not isinstance(report_indices, list)
        or any(not isinstance(value, int) or isinstance(value, bool) for value in report_indices)
        or len(report_indices) != len(set(report_indices))
    ):
        raise AnnotationBundleError("report processed_frame_indices are invalid")
    scene_indices = list(scenes)
    if report_indices != scene_indices:
        raise AnnotationBundleError(
            "report processed_frame_indices do not exactly match scenes.jsonl"
        )
    if report.get("processed_frames") != len(scene_indices):
        raise AnnotationBundleError("report processed_frames does not match scenes.jsonl")
    missing = [value for value in selected_indices if value not in scenes]
    if missing:
        raise AnnotationBundleError(
            f"requested frame indices are absent from the replay: {missing}"
        )

    counts: Counter[str] = Counter()
    provider_pairs: set[tuple[str, str]] = set()
    for record in scenes.values():
        scene = scene_observation_from_dict(record["scene"])
        counts.update(item.class_id for item in scene.objects)
        provider_pairs.add((scene.provider_name, scene.provider_version))
    expected_counts = report.get("object_observations_by_class")
    if dict(sorted(counts.items())) != expected_counts:
        raise AnnotationBundleError("scene object counts do not match replay report")
    expected_provider = (report.get("provider_name"), report.get("provider_version"))
    if provider_pairs != {expected_provider}:
        raise AnnotationBundleError("scene provider identity does not match replay report")

    evaluation_provenance = _required_mapping(
        evaluation.get("provenance"), "evaluation.provenance"
    )
    if evaluation_provenance.get("project_commit") != report.get("project_commit"):
        raise AnnotationBundleError("evaluation commit does not match replay report")
    if evaluation_provenance.get("branch") != report.get("branch"):
        raise AnnotationBundleError("evaluation branch does not match replay report")
    # The standardized result intentionally omits the machine-specific path.
    evaluation_profile = _required_mapping(
        evaluation_provenance.get("integration_profile"),
        "evaluation.provenance.integration_profile",
    )
    report_profile = _required_mapping(
        report.get("integration_profile"), "report.integration_profile"
    )
    if evaluation_profile != {
        "profile_id": report_profile.get("profile_id"),
        "profile_sha256": report_profile.get("profile_sha256"),
    }:
        raise AnnotationBundleError("evaluation profile does not match replay report")
    if evaluation_provenance.get("calibration_ids") != [report.get("calibration_id")]:
        raise AnnotationBundleError("evaluation calibration does not match replay report")
    components = _required_list(
        evaluation_provenance.get("components"), "evaluation.provenance.components"
    )
    expected_component = {
        "component": "vision",
        "name": report.get("provider_name"),
        "version": report.get("provider_version"),
        "artifact_sha256": report.get("provider_artifact_sha256"),
    }
    if components != [expected_component]:
        raise AnnotationBundleError("evaluation provider does not match replay report")
    if evaluation_provenance.get("source_run_ids") != [
        f"capture-sha256:{capture_sha256}"
    ]:
        raise AnnotationBundleError("evaluation capture does not match replay report")

    if verify_local_artifacts:
        report_profile = _required_mapping(
            report.get("integration_profile"), "report.integration_profile"
        )
        object_catalog_version = _required_string(
            evaluation_provenance.get("object_catalog_version"),
            "evaluation.provenance.object_catalog_version",
        )
        expected_catalog = f"sha256:{_sha256(catalog_path)}"
        artifact_checks = {
            "profile": (_sha256(profile_path), report_profile.get("profile_sha256")),
            "provider": (_sha256(provider_path), report.get("provider_artifact_sha256")),
            "replay tool": (_sha256(replay_tool_path), report.get("replay_tool_sha256")),
            "object catalog": (expected_catalog, object_catalog_version),
        }
        for name, (actual, expected) in artifact_checks.items():
            if actual != expected:
                raise AnnotationBundleError(f"current {name} does not match replay provenance")
        if calibration_path.name + ":table" != report.get("calibration_id"):
            raise AnnotationBundleError("calibration ID does not identify the supplied file")

    if verify_repository:
        if _git_worktree_dirty():
            raise AnnotationBundleError("repository is dirty; commit or preserve changes first")
        replay_commit = report.get("project_commit")
        ancestry = subprocess.run(
            ("git", "merge-base", "--is-ancestor", str(replay_commit), "HEAD"),
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if ancestry.returncode != 0:
            raise AnnotationBundleError(
                "replay commit is not an ancestor of the current clean checkout"
            )


def _encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(
        ".png", image, (cv2.IMWRITE_PNG_COMPRESSION, 3)
    )
    if not ok:
        raise AnnotationBundleError("OpenCV failed to encode a lossless PNG")
    return bytes(encoded)


def _point(values: Sequence[Any], path: str, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise AnnotationBundleError(f"{path} must contain {length} finite numbers")
    return array


def _prediction_from_object(
    *,
    frame_index: int,
    detected: Any,
    camera: Mapping[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    position = _point(
        detected.pose.position_base_m,
        f"frame {frame_index} {detected.track_id} position",
        3,
    )
    orientation = _point(
        detected.pose.orientation_xyzw,
        f"frame {frame_index} {detected.track_id} orientation",
        4,
    )
    head_uv: np.ndarray | None = None
    tip_uv: np.ndarray | None = None
    axis_observability = "unobservable"
    pixel_center = detected.attributes.get("pixel_center_uv")
    if detected.class_id in {"cube", "screw_iso10642_m6x30"}:
        if pixel_center is None:
            raise AnnotationBundleError(
                f"frame {frame_index} {detected.track_id} lacks detector pixel_center_uv"
            )
        # Preserve the detector's actual mask/silhouette centroid. Reprojecting
        # the stored 3D object center is not equivalent: cube pose Z is at the
        # object center while the image measurement came from the top face.
        center_uv = _point(pixel_center, "pixel_center_uv", 2)
    else:
        center_uv = (
            project(camera, (position,))[0]
            if pixel_center is None
            else _point(pixel_center, "pixel_center_uv", 2)
        )
    if detected.class_id == "screw_iso10642_m6x30":
        head_xy = detected.attributes.get("head_xy_base_m")
        tip_xy = detected.attributes.get("tip_xy_base_m")
        if head_xy is None or tip_xy is None:
            raise AnnotationBundleError(
                f"frame {frame_index} screw {detected.track_id} lacks head/tip geometry"
            )
        head = np.asarray((*_point(head_xy, "screw head_xy_base_m", 2), position[2]))
        tip = np.asarray((*_point(tip_xy, "screw tip_xy_base_m", 2), position[2]))
        head_uv, tip_uv = project(camera, (head, tip))
        axis_observability = "directed"
    for name, point_value in (
        ("center_uv", center_uv),
        ("head_uv", head_uv),
        ("tip_uv", tip_uv),
    ):
        if point_value is None:
            continue
        if not (0.0 <= point_value[0] < width and 0.0 <= point_value[1] < height):
            raise AnnotationBundleError(
                f"frame {frame_index} {detected.track_id} {name} lies outside native image"
            )
    return {
        "prediction_id": f"frame-{frame_index:06d}:{detected.track_id}",
        "track_id": detected.track_id,
        "class_id": detected.class_id,
        "sort_label": detected.sort_label,
        "center_uv": [float(value) for value in center_uv],
        "confidence": float(detected.confidence),
        "pose_quality": (
            None if detected.pose_quality is None else float(detected.pose_quality)
        ),
        "head_uv": None if head_uv is None else [float(value) for value in head_uv],
        "tip_uv": None if tip_uv is None else [float(value) for value in tip_uv],
        "axis_observability": axis_observability,
        "pose": (
            {
                "position_base_m": [float(value) for value in position],
                "orientation_xyzw": [float(value) for value in orientation],
            }
            if detected.class_id in {"cube", "screw_iso10642_m6x30"}
            else None
        ),
    }


def prepare_bundle(
    *,
    capture: Path,
    replay_dir: Path,
    frame_indices_file: Path,
    output_dir: Path,
    annotation_id: str,
    dataset_id: str,
    source_run_id: str,
    layout_id: str,
    annotator_id: str,
    calibration_path: Path = DEFAULT_CALIBRATION,
    profile_path: Path = DEFAULT_PROFILE,
    provider_path: Path = DEFAULT_PROVIDER,
    replay_tool_path: Path = DEFAULT_REPLAY_TOOL,
    catalog_path: Path = DEFAULT_CATALOG,
    verify_repository: bool = True,
    verify_local_artifacts: bool = True,
) -> dict[str, Any]:
    """Build an external annotation bundle and return its provenance report."""

    capture = capture.expanduser().resolve(strict=True)
    replay_dir = replay_dir.expanduser().resolve(strict=True)
    frame_indices_file = frame_indices_file.expanduser().resolve(strict=True)
    calibration_path = calibration_path.expanduser().resolve(strict=True)
    profile_path = profile_path.expanduser().resolve(strict=True)
    provider_path = provider_path.expanduser().resolve(strict=True)
    replay_tool_path = replay_tool_path.expanduser().resolve(strict=True)
    catalog_path = catalog_path.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve()
    if _inside_repository(output_dir):
        raise AnnotationBundleError("output directory must be outside the repository")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise AnnotationBundleError("output path exists and is not a directory")
        if any(output_dir.iterdir()):
            raise AnnotationBundleError("output directory must be new or empty")

    annotation_id = _required_string(annotation_id, "annotation_id")
    dataset_id = _required_string(dataset_id, "dataset_id")
    source_run_id = _required_string(source_run_id, "source_run_id")
    layout_id = _required_string(layout_id, "layout_id")
    annotator_id = _required_string(annotator_id, "annotator_id")
    selected_indices = read_frame_indices(frame_indices_file)
    report_path = replay_dir / "report.json"
    scenes_path = replay_dir / "scenes.jsonl"
    report = _load_json(report_path)
    scenes = _load_scenes(scenes_path)
    evaluation_name = report.get("evaluation_result")
    if not isinstance(evaluation_name, str) or Path(evaluation_name).name != evaluation_name:
        raise AnnotationBundleError("replay report lacks a safe evaluation_result filename")
    evaluation_path = replay_dir / evaluation_name
    evaluation = _load_json(evaluation_path)
    capture_sha256 = _sha256(capture)
    calibration_sha256 = _sha256(calibration_path)
    scenes_jsonl_sha256 = _sha256(scenes_path)
    _verify_replay(
        report=report,
        evaluation=evaluation,
        scenes=scenes,
        selected_indices=selected_indices,
        capture_sha256=capture_sha256,
        calibration_sha256=calibration_sha256,
        scenes_jsonl_sha256=scenes_jsonl_sha256,
        calibration_path=calibration_path,
        profile_path=profile_path,
        provider_path=provider_path,
        replay_tool_path=replay_tool_path,
        catalog_path=catalog_path,
        verify_repository=verify_repository,
        verify_local_artifacts=verify_local_artifacts,
    )

    cameras = load_calibration(calibration_path)
    if "table" not in cameras:
        raise AnnotationBundleError("calibration does not contain a table camera")
    table = cameras["table"]
    native_size = tuple(int(value) for value in table["size"])
    if list(native_size) != report.get("native_calibration_size"):
        raise AnnotationBundleError("native table size does not match replay report")
    report_source_size = tuple(report.get("source_size", ()))
    if len(report_source_size) != 2:
        raise AnnotationBundleError("replay report has invalid source_size")

    selected_set = set(selected_indices)
    decoded: dict[int, tuple[bytes, bytes, np.ndarray]] = {}
    decoded_fps: float | None = None
    for frame_index, timestamp, bgr, fps in _frames(capture):
        if frame_index not in selected_set:
            if frame_index > selected_indices[-1]:
                break
            continue
        source_size = (int(bgr.shape[1]), int(bgr.shape[0]))
        if source_size != report_source_size:
            raise AnnotationBundleError(
                f"decoded frame {frame_index} size differs from replay report"
            )
        if not np.isclose(
            float(timestamp),
            float(scenes[frame_index]["timestamp_s"]),
            rtol=0.0,
            atol=1e-9,
        ):
            raise AnnotationBundleError(
                f"decoded frame {frame_index} timestamp differs from replay scene"
            )
        if decoded_fps is None:
            decoded_fps = float(fps)
        elif not np.isclose(decoded_fps, float(fps), rtol=0.0, atol=1e-9):
            raise AnnotationBundleError("decoded frame rate changed within the capture")
        if source_size != native_size:
            if source_size[0] * native_size[1] != source_size[1] * native_size[0]:
                raise AnnotationBundleError("source aspect ratio differs from table calibration")
            native_bgr = cv2.resize(bgr, native_size, interpolation=cv2.INTER_NEAREST)
        else:
            native_bgr = bgr
        replay_rgb = cv2.cvtColor(native_bgr, cv2.COLOR_BGR2RGB)
        expected_shape = scenes[frame_index]["input_rgb_shape_hwc"]
        if [int(value) for value in replay_rgb.shape] != expected_shape:
            raise AnnotationBundleError(
                f"re-decoded frame {frame_index} shape differs from replay input"
            )
        replay_rgb_sha256 = _sha256_bytes(replay_rgb.tobytes(order="C"))
        if replay_rgb_sha256 != scenes[frame_index]["input_rgb_sha256"]:
            raise AnnotationBundleError(
                f"re-decoded frame {frame_index} pixels differ from replay input_rgb_sha256"
            )
        decoded[frame_index] = (
            _encode_png(bgr),
            _encode_png(native_bgr),
            native_bgr,
        )
    if tuple(decoded) != selected_indices:
        missing = [value for value in selected_indices if value not in decoded]
        raise AnnotationBundleError(f"capture did not decode every requested frame: {missing}")
    if decoded_fps is None or not np.isclose(
        decoded_fps,
        float(report.get("decoded_source_fps")),
        rtol=0.0,
        atol=1e-9,
    ):
        raise AnnotationBundleError("decoded source frame rate differs from replay report")

    evaluation_provenance = _required_mapping(
        evaluation["provenance"], "evaluation.provenance"
    )
    profile = _required_mapping(
        report.get("integration_profile"), "report.integration_profile"
    )
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest_frames: list[dict[str, Any]] = []
    bundle_frames: list[dict[str, Any]] = []
    preview_pngs: dict[int, bytes] = {}
    for frame_index in selected_indices:
        record = scenes[frame_index]
        scene = scene_observation_from_dict(record["scene"])
        predictions = [
            _prediction_from_object(
                frame_index=frame_index,
                detected=detected,
                camera=table,
                width=native_size[0],
                height=native_size[1],
            )
            for detected in scene.objects
        ]
        source_name = f"source/frame-{frame_index:06d}.png"
        annotation_name = f"annotation/frame-{frame_index:06d}.png"
        preview_name = f"preview/frame-{frame_index:06d}.png"
        source_png, annotation_png, native_bgr = decoded[frame_index]
        # Keep annotation pixels untouched. This overlay is a non-authoritative
        # review aid and is never used for image_sha256 or ground truth.
        preview_png = _encode_png(_draw_overlay(native_bgr, scene, table))
        preview_pngs[frame_index] = preview_png
        annotation_hash = _sha256_bytes(annotation_png)
        manifest_frames.append(
            {
                "frame_id": f"{source_run_id}:table:{frame_index:06d}",
                "layout_id": layout_id,
                "frame_index": frame_index,
                "capture_time_s": float(record["timestamp_s"]),
                "image_sha256": annotation_hash,
                "inference_latency_ms": float(record["latency_ms"]),
                "references": [],
                "predictions": predictions,
                "regions": [],
            }
        )
        bundle_frames.append(
            {
                "frame_index": frame_index,
                "source_png": source_name,
                "source_png_sha256": _sha256_bytes(source_png),
                "source_size": list(report_source_size),
                "annotation_png": annotation_name,
                "annotation_png_sha256": annotation_hash,
                "annotation_size": list(native_size),
                "annotation_transform": (
                    "identity"
                    if report_source_size == native_size
                    else "nearest-neighbour upscale verified against replay RGB hash"
                ),
                "replay_input_rgb_sha256": record["input_rgb_sha256"],
                "review_overlay_png": preview_name,
                "review_overlay_png_sha256": _sha256_bytes(preview_png),
                "review_overlay_is_ground_truth": False,
            }
        )
    manifest = {
        "schema_version": 1,
        "annotation_id": annotation_id,
        "created_at_utc": created_at,
        "split": "pilot",
        "backend": "offline_replay",
        "provenance": {
            "project_commit": report["project_commit"],
            "branch": report["branch"],
            "integration_profile": {
                "profile_id": profile["profile_id"],
                "profile_sha256": profile["profile_sha256"],
            },
            "provider": {
                "name": report["provider_name"],
                "version": report["provider_version"],
                "artifact_sha256": report["provider_artifact_sha256"],
            },
            "object_catalog_version": evaluation_provenance[
                "object_catalog_version"
            ],
            "calibration_ids": [report["calibration_id"]],
            "review": {
                "annotator_id": annotator_id,
                "review_status": "unreviewed",
                "reviewer_id": None,
            },
        },
        "source": {
            "dataset_id": dataset_id,
            "source_run_id": source_run_id,
            "capture_sha256": capture_sha256,
            "camera_id": "table",
            "image_width_px": native_size[0],
            "image_height_px": native_size[1],
            "timestamp_source": "frame_index_over_fps",
        },
        "frames": manifest_frames,
    }
    validate_manifest(manifest, source="generated annotation manifest")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source").mkdir()
    (output_dir / "annotation").mkdir()
    (output_dir / "preview").mkdir()
    for frame_record in bundle_frames:
        frame_index = frame_record["frame_index"]
        source_png, annotation_png, _native_bgr = decoded[frame_index]
        preview_png = preview_pngs[frame_index]
        (output_dir / frame_record["source_png"]).write_bytes(source_png)
        (output_dir / frame_record["annotation_png"]).write_bytes(annotation_png)
        (output_dir / frame_record["review_overlay_png"]).write_bytes(preview_png)
    selection_copy = "".join(f"{value}\n" for value in selected_indices).encode("ascii")
    (output_dir / "selected-frame-indices.txt").write_bytes(selection_copy)
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    manifest_name = "annotation-manifest.pilot.json"
    (output_dir / manifest_name).write_bytes(manifest_payload)
    bundle_report = {
        "schema_version": 1,
        "tool": "team_plugins.computer_vision.prepare_annotation_bundle",
        "bundle_tool_sha256": _sha256(Path(__file__).resolve()),
        "bundle_tool_project_commit": _git_value("rev-parse", "HEAD"),
        "bundle_tool_branch": _git_value("branch", "--show-current"),
        "bundle_tool_worktree_dirty": _git_worktree_dirty(),
        "created_at_utc": created_at,
        "capture_sha256": capture_sha256,
        "replay_report_sha256": _sha256(report_path),
        "replay_scenes_sha256": scenes_jsonl_sha256,
        "replay_evaluation_sha256": _sha256(evaluation_path),
        "calibration_sha256": calibration_sha256,
        "frame_indices_file_sha256": _sha256(frame_indices_file),
        "normalized_selection_sha256": _sha256_bytes(selection_copy),
        "selected_frame_indices": list(selected_indices),
        "annotation_manifest": manifest_name,
        "annotation_manifest_sha256": _sha256_bytes(manifest_payload),
        "frames": bundle_frames,
        "ready_for_scoring": False,
        "next_action": (
            "annotate references and unknown/ignore regions, independently measure any "
            "pose_truth, then set review provenance only after human review"
        ),
        "evidence_limit": (
            "lossless PNG encoding preserves decoded pixels, but compressed/upscaled "
            "recordings still cannot establish native sensor detail or robot-base accuracy"
        ),
    }
    report_payload = (
        json.dumps(bundle_report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    (output_dir / "bundle-report.json").write_bytes(report_payload)
    return bundle_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--frame-indices-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotation-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--layout-id", required=True)
    parser.add_argument("--annotator-id", required=True)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = prepare_bundle(
        capture=args.capture,
        replay_dir=args.replay_dir,
        frame_indices_file=args.frame_indices_file,
        output_dir=args.output_dir,
        annotation_id=args.annotation_id,
        dataset_id=args.dataset_id,
        source_run_id=args.source_run_id,
        layout_id=args.layout_id,
        annotator_id=args.annotator_id,
        calibration_path=args.calibration,
        profile_path=args.profile,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
