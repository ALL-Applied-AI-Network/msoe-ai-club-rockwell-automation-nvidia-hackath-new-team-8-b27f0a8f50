"""Offline, no-motion replay for the hybrid cube/screw provider.

The official downloaded recordings are resized/compressed 640x360 video.  This
tool may upscale those frames to the native calibrated table size for a
qualitative regression test, but its report marks the evidence as incapable of
proving millimetre pose accuracy.  Use native uncropped captures with measured
object poses for the final calibration test.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
from typing import Iterator

import cv2
import numpy as np

from hardware.fanuc_cell.calibration.geometry import load as load_calibration
from hardware.fanuc_cell.calibration.geometry import pixel_to_base, project
from sorting_core import load_config, scene_observation_to_dict
from sorting_core.integration_plugins import (
    VisionFactoryContext,
    load_factory,
    load_integration_profile,
)
from sorting_core.perception import (
    CameraFrame,
    PerceptionRequest,
    validate_scene_for_request,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION = REPO_ROOT / "hardware" / "fanuc_cell" / "calibration" / "cameras.json"
DEFAULT_PROFILE = Path(__file__).with_name("profile.hybrid-geometry.v1.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str | None:
    result = subprocess.run(
        ("git", *arguments),
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


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
        detail = result.stderr.strip() or "git status failed"
        raise RuntimeError(f"unable to record worktree state: {detail}")
    return bool(result.stdout.strip())


def _pyav_frames(path: Path) -> Iterator[tuple[int, float, np.ndarray, float]]:
    try:
        import av  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError(
            "AV1 replay requires optional PyAV "
            "(run with: uv run --project hardware/fanuc_cell --with av python -m ...)"
        ) from exc
    with av.open(str(path)) as container:
        streams = tuple(container.streams.video)
        if not streams:
            raise ValueError(f"capture has no video stream: {path}")
        stream = streams[0]
        pyav_fps = float(stream.average_rate) if stream.average_rate else 10.0
        if not np.isfinite(pyav_fps) or pyav_fps <= 0:
            pyav_fps = 10.0
        for pyav_index, decoded in enumerate(container.decode(stream)):
            yield (
                pyav_index,
                pyav_index / pyav_fps,
                decoded.to_ndarray(format="bgr24"),
                pyav_fps,
            )


def _frames(path: Path) -> Iterator[tuple[int, float, np.ndarray, float]]:
    image = (
        cv2.imread(str(path), cv2.IMREAD_COLOR)
        if path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
        else None
    )
    if image is not None:
        yield 0, 0.0, image, 1.0
        return
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"unable to open image/video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 10.0
    expected_frame_count = int(max(0.0, capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    if fourcc.strip("\x00 ").lower() in {"av01", "av1"}:
        capture.release()
        yield from _pyav_frames(path)
        return
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        yield index, index / fps, frame, fps
        index += 1
    capture.release()
    if index:
        if expected_frame_count and index + 1 < expected_frame_count:
            raise ValueError(
                f"OpenCV decoded only {index}/{expected_frame_count} advertised frames; "
                "refusing a partial replay"
            )
        return
    # The downloaded LeRobot archives use AV1. Some OpenCV builds open those
    # files but decode zero frames, so use an optional PyAV fallback rather
    # than silently reporting an empty evaluation.
    yield from _pyav_frames(path)


_OVERLAY_CLASS_STYLES = {
    "cube": ("C", (0, 220, 255)),
    "screw_iso10642_m6x30": ("S", (40, 220, 40)),
    "unknown_obstacle": ("U", (40, 40, 230)),
}
_OVERLAY_INELIGIBLE_COLOUR = (165, 165, 165)


def _overlay_status(detected) -> tuple[str, bool]:
    """Return a compact status and whether to highlight the geometry."""

    attributes = detected.attributes
    if detected.class_id == "cube":
        in_pickup = attributes.get("in_pickup_region")
        if in_pickup is True:
            return "in-region", True
        if in_pickup is False:
            return "out", False
        return "region?", False
    if detected.class_id == "screw_iso10642_m6x30":
        if attributes.get("non_executable") is True:
            return "noexec", False
        in_pickup = attributes.get("footprint_inside_pickup_region")
        if in_pickup is True:
            # The current executor is cube-only. Passing the vision geometry
            # gate makes this a grasp-team candidate, never a pick command.
            return "candidate", True
        if in_pickup is False:
            return "out", False
        return "region?", False
    if detected.class_id == "unknown_obstacle":
        return "noexec", False
    return "unsupported", False


def _overlay_axis_pixels(detected, camera: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Project an explicitly reported head/tip axis, including abstentions."""

    head_xy = detected.attributes.get("head_xy_base_m")
    tip_xy = detected.attributes.get("tip_xy_base_m")
    if head_xy is None or tip_xy is None:
        return None
    z_m = detected.pose.position_base_m[2]
    head = np.asarray((head_xy[0], head_xy[1], z_m), dtype=float)
    tip = np.asarray((tip_xy[0], tip_xy[1], z_m), dtype=float)
    head_uv, tip_uv = project(camera, (head, tip)).round().astype(int)
    return head_uv, tip_uv


def _overlay_anchor(detected, camera: dict) -> tuple[int, int]:
    axis = _overlay_axis_pixels(detected, camera)
    if axis is not None:
        return tuple(int(value) for value in axis[0])
    uv = detected.attributes.get("pixel_center_uv")
    if uv is None:
        xyz = np.asarray(detected.pose.position_base_m, dtype=float)
        uv = tuple(project(camera, (xyz,))[0])
    return tuple(int(value) for value in np.rint(uv))


def _overlay_row(index: int, detected, status: str) -> str:
    token, _ = _OVERLAY_CLASS_STYLES.get(
        detected.class_id, ("?", (255, 255, 255))
    )
    label = detected.sort_label or "-"
    quality = "-" if detected.pose_quality is None else f"{detected.pose_quality:.2f}"
    track = detected.track_id[-8:]
    candidate = detected.attributes.get("candidate_class")
    candidate_suffix = " cand:S" if candidate == "screw_iso10642_m6x30" else ""
    return (
        f"{index:02d} {token}:{label} c{detected.confidence:.2f} "
        f"q{quality} {status} t:{track}{candidate_suffix}"
    )


def _overlay_colour(detected, highlighted: bool) -> tuple[int, int, int]:
    """Keep abstentions visibly red while dimming merely out-of-region targets."""

    _, class_colour = _OVERLAY_CLASS_STYLES.get(
        detected.class_id, ("?", (255, 255, 255))
    )
    if detected.class_id == "unknown_obstacle":
        return class_colour
    return class_colour if highlighted else _OVERLAY_INELIGIBLE_COLOUR


def _overlay_panel_rectangle(
    image_shape: tuple[int, ...],
    anchors: list[tuple[int, int]],
    rows: list[str],
    header: str,
    font_scale: float,
    row_height: int,
) -> tuple[tuple[int, int, int, int], int, int]:
    """Choose the least-obstructive corner and a bounded column layout."""

    height, width = image_shape[:2]
    preferred_margin = max(6, int(round(min(width, height) * 0.008)))
    margin = min(preferred_margin, max(0, (min(width, height) - 2) // 4))
    maximum_rows = max(4, (height - 2 * margin) // row_height - 2)
    columns = max(1, (len(rows) + maximum_rows - 1) // maximum_rows)
    rows_per_column = max(1, (len(rows) + columns - 1) // columns)
    column_widths: list[int] = []
    for column in range(columns):
        column_rows = rows[column * rows_per_column : (column + 1) * rows_per_column]
        maximum_text_width = max(
            (
                cv2.getTextSize(
                    row,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    1,
                )[0][0]
                for row in column_rows
            ),
            default=0,
        )
        if column == 0:
            maximum_text_width = max(
                maximum_text_width,
                cv2.getTextSize(
                    header,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    1,
                )[0][0],
            )
        column_widths.append(maximum_text_width + 2 * margin)
    panel_width = min(width - 2 * margin, max(1, sum(column_widths) + margin))
    panel_height = min(
        height - 2 * margin,
        max(1, (rows_per_column + 2) * row_height + margin),
    )
    candidates = (
        (margin, margin),
        (width - margin - panel_width, margin),
        (margin, height - margin - panel_height),
        (width - margin - panel_width, height - margin - panel_height),
    )

    def overlap_score(origin: tuple[int, int]) -> tuple[int, int]:
        x0, y0 = origin
        x1, y1 = x0 + panel_width, y0 + panel_height
        inside = sum(x0 <= x <= x1 and y0 <= y <= y1 for x, y in anchors)
        # Candidate order is the deterministic tie break.
        return inside, candidates.index(origin)

    x0, y0 = min(candidates, key=overlap_score)
    return (x0, y0, x0 + panel_width, y0 + panel_height), columns, rows_per_column


def _draw_overlay(
    bgr: np.ndarray,
    scene,
    camera: dict,
) -> np.ndarray:
    """Draw numbered geometry plus a collision-free review legend.

    The provider result and source frame remain untouched. Long labels live in
    one deterministic corner panel instead of covering the small parts they
    describe. Markers distinguish cube pickup eligibility from screw geometry
    candidates. Cube region membership alone is not a complete pickability
    decision; the current executor remains cube-only and this function never
    changes which objects are executable.
    """

    overlay = bgr.copy()
    detected_with_anchors = [
        (detected, _overlay_anchor(detected, camera)) for detected in scene.objects
    ]
    detected_with_anchors.sort(
        key=lambda item: (
            {"cube": 0, "screw_iso10642_m6x30": 1, "unknown_obstacle": 2}.get(
                item[0].class_id, 3
            ),
            item[1][1],
            item[1][0],
            item[0].track_id,
        )
    )
    rows: list[str] = []
    statuses: Counter[str] = Counter()
    anchors: list[tuple[int, int]] = []
    for index, (detected, anchor) in enumerate(detected_with_anchors, start=1):
        status, highlighted = _overlay_status(detected)
        statuses[f"{detected.class_id}:{status}"] += 1
        anchors.append(anchor)
        colour = _overlay_colour(detected, highlighted)
        axis = _overlay_axis_pixels(detected, camera)
        if axis is not None:
            head_uv, tip_uv = axis
            if highlighted and detected.class_id == "screw_iso10642_m6x30":
                cv2.arrowedLine(
                    overlay,
                    tuple(head_uv),
                    tuple(tip_uv),
                    colour,
                    4,
                    tipLength=0.2,
                )
            else:
                cv2.line(overlay, tuple(head_uv), tuple(tip_uv), colour, 2)
        else:
            cv2.circle(overlay, anchor, 10, colour, 3 if highlighted else 1)
        if not highlighted:
            cv2.drawMarker(
                overlay,
                anchor,
                colour,
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=15,
                thickness=2,
            )
        marker = f"{index:02d}"
        marker_origin = (anchor[0] + 7, anchor[1] - 7)
        cv2.putText(
            overlay,
            marker,
            marker_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            marker,
            marker_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )
        rows.append(_overlay_row(index, detected, status))

    if not rows:
        return overlay
    height, width = overlay.shape[:2]
    font_scale = max(0.32, min(0.62, 0.62 * width / 1920.0))
    row_height = max(14, int(round(27 * width / 1920.0)))
    class_counts = Counter(item[0].class_id for item in detected_with_anchors)
    summary = (
        f"C:{class_counts['cube']} "
        f"S:{class_counts['screw_iso10642_m6x30']} "
        f"U:{class_counts['unknown_obstacle']} "
        f"cube-region:{statuses['cube:in-region']} "
        f"screw-cand:{statuses['screw_iso10642_m6x30:candidate']} "
        f"other:{sum(statuses.values()) - statuses['cube:in-region'] - statuses['screw_iso10642_m6x30:candidate']}"
    )
    panel, columns, rows_per_column = _overlay_panel_rectangle(
        overlay.shape, anchors, rows, summary, font_scale, row_height
    )
    x0, y0, x1, y1 = panel
    panel_pixels = overlay[y0:y1, x0:x1]
    overlay[y0:y1, x0:x1] = cv2.addWeighted(
        panel_pixels,
        0.22,
        np.zeros_like(panel_pixels),
        0.78,
        0.0,
    )
    cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (235, 235, 235), 1)
    text_x = x0 + max(5, row_height // 3)
    baseline_y = y0 + row_height
    cv2.putText(
        overlay,
        summary,
        (text_x, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    usable_width = max(1, x1 - x0 - 2 * max(5, row_height // 3))
    column_width = usable_width // columns
    for row_index, row in enumerate(rows):
        column = min(columns - 1, row_index // rows_per_column)
        within_column = row_index % rows_per_column
        detected = detected_with_anchors[row_index][0]
        _, highlighted = _overlay_status(detected)
        colour = _overlay_colour(detected, highlighted)
        cv2.putText(
            overlay,
            row,
            (
                text_x + column * column_width,
                baseline_y + (within_column + 1) * row_height,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            colour,
            1,
            cv2.LINE_AA,
        )
    return overlay


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--cube-edge-mm", type=float, choices=(10.0, 16.0), default=16.0)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--maximum-frames", type=int)
    parser.add_argument(
        "--allow-upscale-compressed",
        action="store_true",
        help="upscale same-aspect recordings to calibrated size; never proves pose accuracy",
    )
    parser.add_argument(
        "--save-source-frames",
        action="store_true",
        help="save the first 24 processed source frames next to review overlays",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.every <= 0:
        raise ValueError("--every must be positive")
    args.capture = args.capture.expanduser().resolve(strict=True)
    args.calibration = args.calibration.expanduser().resolve(strict=True)
    args.profile = args.profile.expanduser().resolve(strict=True)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists():
        if not args.output_dir.is_dir():
            raise ValueError("--output-dir must be a directory")
        if any(args.output_dir.iterdir()):
            raise ValueError(
                "--output-dir must be new or empty so stale evidence cannot survive a rerun"
            )
    else:
        args.output_dir.mkdir(parents=True)
    provider_path = Path(__file__).with_name("hybrid_geometry_provider.py")
    catalog_path = REPO_ROOT / "docs" / "OBJECT_CATALOG.md"
    project_commit = _git_value("rev-parse", "HEAD")
    branch = _git_value("branch", "--show-current")
    if project_commit is None or len(project_commit) != 40:
        raise RuntimeError("unable to record the repository commit for replay provenance")
    if branch is None:
        raise RuntimeError("unable to record the repository branch for replay provenance")
    worktree_dirty_at_start = _git_worktree_dirty()
    provenance_at_start = {
        "capture_sha256": _sha256(args.capture),
        "calibration_sha256": _sha256(args.calibration),
        "profile_sha256": _sha256(args.profile),
        "provider_artifact_sha256": _sha256(provider_path),
        "replay_tool_sha256": _sha256(Path(__file__)),
        "object_catalog_sha256": _sha256(catalog_path),
        "project_commit": project_commit,
        "branch": branch,
    }
    calibration = load_calibration(args.calibration)
    table = calibration["table"]
    native_size = tuple(int(value) for value in table["size"])
    projector = lambda uv, z: pixel_to_base(table, uv, z=z)
    profile = load_integration_profile(args.profile)
    if profile.source_sha256 != provenance_at_start["profile_sha256"]:
        raise RuntimeError("profile changed while it was being loaded")
    if profile.vision is None:
        raise ValueError("replay profile must select a vision provider")
    expected_factory = (
        "team_plugins.computer_vision.hybrid_geometry_provider:create"
    )
    if profile.vision.factory != expected_factory:
        raise ValueError(
            "this replay tool can attribute only the hybrid provider; expected "
            f"vision factory {expected_factory!r}, got {profile.vision.factory!r}"
        )
    calibration_id = f"{args.calibration.name}:table"
    effective_calibrations = {"table": calibration_id}
    if profile.vision.camera_ids != ("table",) or profile.vision.depth_camera_ids:
        raise ValueError("this table replay requires an RGB-only table-camera profile")
    factory = load_factory(profile.vision.factory, "vision")
    provider = factory(
        VisionFactoryContext(
            config=load_config(),
            camera_projectors={"table": projector},
            calibration_ids=effective_calibrations,
            expected_cube_edge_m=args.cube_edge_mm / 1000.0,
            checkpoint=None,
            detector=None,
            options=profile.vision.options,
            camera_ids=profile.vision.camera_ids,
            depth_camera_ids=profile.vision.depth_camera_ids,
            declared_calibration_ids=profile.vision.calibration_ids,
            declared_depth_calibration_ids=profile.vision.depth_calibration_ids,
            profile_id=profile.profile_id,
            profile_path=profile.source_path,
            profile_sha256=profile.source_sha256,
        )
    )
    records = []
    latencies_ms = []
    counts: Counter[str] = Counter()
    writer = None
    processed = 0
    source_size = None
    processed_frame_indices: list[int] = []
    source_fps = None
    for frame_index, timestamp_s, bgr, decoded_fps in _frames(args.capture):
        source_fps = decoded_fps
        if frame_index < args.start_frame or (frame_index - args.start_frame) % args.every:
            continue
        if args.maximum_frames is not None and processed >= args.maximum_frames:
            break
        start = time.perf_counter()
        source_size = (int(bgr.shape[1]), int(bgr.shape[0]))
        if source_size != native_size:
            if not args.allow_upscale_compressed:
                raise ValueError(
                    f"capture is {source_size}; calibrated native size is {native_size}; "
                    "pass --allow-upscale-compressed only for qualitative replay"
                )
            if source_size[0] * native_size[1] != source_size[1] * native_size[0]:
                raise ValueError("capture aspect ratio differs from native calibration")
            native_bgr = cv2.resize(bgr, native_size, interpolation=cv2.INTER_NEAREST)
        else:
            native_bgr = bgr
        rgb = cv2.cvtColor(native_bgr, cv2.COLOR_BGR2RGB)
        request = PerceptionRequest(
            scene_id=f"{args.capture.stem}:{frame_index}",
            capture_time_s=float(timestamp_s),
            frames=(
                CameraFrame(
                    camera_id="table",
                    capture_time_s=float(timestamp_s),
                    calibration_id=calibration_id,
                    rgb=rgb,
                ),
            ),
            metadata={
                "source_frame_index": frame_index,
                "source_size": source_size,
                "qualitative_upscale": source_size != native_size,
            },
        )
        scene = validate_scene_for_request(request, provider.observe(request))
        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(latency_ms)
        counts.update(item.class_id for item in scene.objects)
        records.append(
            {
                "frame_index": frame_index,
                "timestamp_s": timestamp_s,
                "latency_ms": latency_ms,
                "scene": scene_observation_to_dict(scene),
            }
        )
        overlay = _draw_overlay(native_bgr, scene, table)
        preview = cv2.resize(overlay, (960, 540), interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = cv2.VideoWriter(
                str(args.output_dir / "overlay.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                decoded_fps / args.every,
                (960, 540),
            )
            if not writer.isOpened():
                raise RuntimeError("unable to create overlay video")
        writer.write(preview)
        if processed < 24:
            cv2.imwrite(str(args.output_dir / f"frame-{frame_index:06d}.jpg"), preview)
            if args.save_source_frames:
                cv2.imwrite(
                    str(args.output_dir / f"source-{frame_index:06d}.jpg"),
                    bgr,
                )
        processed += 1
        processed_frame_indices.append(frame_index)
    if writer is not None:
        writer.release()
    if not records:
        raise ValueError("no frames were processed")
    (args.output_dir / "scenes.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    ordered = sorted(latencies_ms)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    capture_sha256 = provenance_at_start["capture_sha256"]
    calibration_sha256 = provenance_at_start["calibration_sha256"]
    provider_sha256 = provenance_at_start["provider_artifact_sha256"]
    catalog_sha256 = provenance_at_start["object_catalog_sha256"]
    worktree_dirty_at_end = _git_worktree_dirty()
    provenance_at_start["worktree_dirty"] = worktree_dirty_at_start
    provenance_at_end = {
        "capture_sha256": _sha256(args.capture),
        "calibration_sha256": _sha256(args.calibration),
        "profile_sha256": _sha256(args.profile),
        "provider_artifact_sha256": _sha256(provider_path),
        "replay_tool_sha256": _sha256(Path(__file__)),
        "object_catalog_sha256": _sha256(catalog_path),
        "project_commit": _git_value("rev-parse", "HEAD"),
        "branch": _git_value("branch", "--show-current"),
        "worktree_dirty": worktree_dirty_at_end,
    }
    provenance_changes = sorted(
        key
        for key, start_value in provenance_at_start.items()
        if provenance_at_end.get(key) != start_value
    )
    provenance_stable = not provenance_changes
    worktree_dirty = worktree_dirty_at_start or worktree_dirty_at_end
    created_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    evaluation_path = args.output_dir / "evaluation-result.v1.json"
    experiment_identity = {
        "capture_sha256": capture_sha256,
        "project_commit": project_commit,
        "profile_sha256": profile.source_sha256,
        "provider_artifact_sha256": provider_sha256,
        "replay_tool_sha256": provenance_at_start["replay_tool_sha256"],
        "cube_edge_mm_profile": args.cube_edge_mm,
        "sample_every_n_frames": args.every,
        "start_frame": args.start_frame,
        "maximum_frames": args.maximum_frames,
        "processed_frame_indices": processed_frame_indices,
    }
    experiment_sha256 = hashlib.sha256(
        json.dumps(experiment_identity, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    evaluation = {
        "schema_version": 1,
        "evaluation_id": f"hybrid-replay-{experiment_sha256[:16]}",
        "created_at_utc": created_at_utc,
        "split": "pilot",
        "backend": "offline_replay",
        "provenance": {
            "project_commit": project_commit,
            "branch": branch,
            "integration_profile": {
                "profile_id": profile.profile_id,
                "profile_sha256": profile.source_sha256,
            },
            "object_catalog_version": f"sha256:{catalog_sha256}",
            "calibration_ids": [calibration_id],
            "components": [
                {
                    "component": "vision",
                    "name": provider.PROVIDER_NAME,
                    "version": provider.PROVIDER_VERSION,
                    "artifact_sha256": provider_sha256,
                }
            ],
            "source_run_ids": [f"capture-sha256:{capture_sha256}"],
        },
        "vision": {
            "frames_evaluated": processed,
            "inference_latency_ms": latencies_ms,
            # No reference instances or poses exist in this compressed video.
            # An empty list is honest; output counts are retained in report.json.
            "per_class": [],
        },
        "episodes": [
            {
                "episode_id": f"offline-replay-{experiment_sha256[:16]}",
                "condition_id": (
                    "compressed-upscaled-unannotated"
                    if source_size != native_size
                    else "native-unannotated"
                ),
                "scene_id": f"{args.capture.stem}:sampled-set",
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
                "timing_s": {
                    "perception_s": sum(latencies_ms) / 1000.0,
                },
            }
        ],
    }
    evaluation_emitted = not worktree_dirty and provenance_stable
    if evaluation_emitted:
        evaluation_path.write_text(
            json.dumps(evaluation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    report = {
        "schema_version": 1,
        "tool": "team_plugins.computer_vision.replay_camera",
        "capture_name": args.capture.name,
        "capture_bytes": args.capture.stat().st_size,
        "capture_sha256": capture_sha256,
        "calibration_id": calibration_id,
        "calibration_sha256": calibration_sha256,
        "provider_name": provider.PROVIDER_NAME,
        "provider_version": provider.PROVIDER_VERSION,
        "provider_artifact_sha256": provider_sha256,
        "replay_tool_sha256": provenance_at_start["replay_tool_sha256"],
        "project_commit": project_commit,
        "branch": branch,
        "worktree_dirty": worktree_dirty,
        "worktree_dirty_at_start": worktree_dirty_at_start,
        "worktree_dirty_at_end": worktree_dirty_at_end,
        "provenance_stable_during_replay": provenance_stable,
        "provenance_fields_changed_during_replay": provenance_changes,
        "integration_profile": {
            "profile_id": profile.profile_id,
            "profile_path": profile.source_path,
            "profile_sha256": profile.source_sha256,
        },
        "experiment_sha256": experiment_sha256,
        "processed_frame_indices": processed_frame_indices,
        "source_size": source_size,
        "native_calibration_size": native_size,
        "qualitative_upscale": source_size != native_size,
        "cube_edge_mm_profile": args.cube_edge_mm,
        "sample_every_n_frames": args.every,
        "decoded_source_fps": source_fps,
        "start_frame": args.start_frame,
        "maximum_frames": args.maximum_frames,
        "processed_frames": processed,
        "object_observations_by_class": dict(sorted(counts.items())),
        "latency_ms": {
            "mean": statistics.fmean(latencies_ms),
            "median": statistics.median(latencies_ms),
            "p95": p95,
            "maximum": max(latencies_ms),
        },
        "evidence_limit": (
            "Compressed/resized video has no object-pose ground truth and cannot establish "
            "robot-base millimetre accuracy, native mask quality, or physical grasp success."
        ),
        "timestamp_limit": (
            "Replay timestamps are decoded frame_index/fps values, not measured "
            "hardware exposure timestamps."
        ),
        "evaluation_result": evaluation_path.name if evaluation_emitted else None,
        "evaluation_result_skipped_reason": (
            "repository/input provenance changed during replay"
            if not provenance_stable
            else (
                "standardized artifact requires a clean committed worktree"
                if worktree_dirty
                else None
            )
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if provenance_stable else 2


if __name__ == "__main__":
    raise SystemExit(main())
