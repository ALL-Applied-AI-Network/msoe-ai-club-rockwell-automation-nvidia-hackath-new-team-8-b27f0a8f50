"""Offline audit of table-camera images or recorded rollout videos.

This program never connects to the SIDC Lab service or the robot.  It replays a
saved image/video through the checked-in HSV and geometry gates and explains
why coloured components would or would not become cube candidates.

Recorded LeRobot/result videos are normally 640x360 while the calibrated table
camera is 1920x1080.  They are whole-frame resizes, so this tool maps contour
pixels back to native coordinates before applying the calibration.  That is
useful for diagnosis only: resizing, AV1/YUV compression, and missing native
pixels can change a tiny cube's contour.  A recorded video therefore cannot
certify millimetre grasp geometry or replace a native 1920x1080 PNG replay.

Examples (from ``hardware/fanuc_cell``)::

    uv run audit_table_capture.py path/to/table.mp4 --every 10
    uv run audit_table_capture.py table.png --overlay-dir audit-overlay \
        --json audit.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
for path in (HERE, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calibration.geometry import load as load_calibration  # noqa: E402
from calibration.geometry import pixel_to_base  # noqa: E402
from sorting_core import (  # noqa: E402
    detect_coloured_cubes,
    detection_in_pickup_region,
    load_config,
)
from sorting_core.contracts import CubeDetection  # noqa: E402


WARNING = (
    "OFFLINE DIAGNOSTIC ONLY: resized/compressed recordings can identify likely "
    "HSV or size-gate mismatches, but cannot certify millimetre pose accuracy, "
    "grasp fit, collision clearance, or safe robot motion. Confirm conclusions "
    "with native 1920x1080 PNG observations and physical measurements."
)


def _colour_mask(hsv: np.ndarray, ranges: Iterable[Iterable[int]]) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for h_min, h_max, s_min, v_min in ranges:
        component = cv2.inRange(
            hsv,
            np.array([h_min, s_min, v_min], dtype=np.uint8),
            np.array([h_max, 255, 255], dtype=np.uint8),
        )
        mask = cv2.bitwise_or(mask, component)
    return mask


def scale_uv_to_native(
    uv: tuple[float, float],
    capture_size: tuple[int, int],
    native_size: tuple[int, int],
) -> tuple[float, float]:
    """Map a whole-frame resized pixel centre to native camera coordinates."""

    capture_w, capture_h = capture_size
    native_w, native_h = native_size
    if min(capture_w, capture_h, native_w, native_h) <= 0:
        raise ValueError("capture and native sizes must be positive")
    # Centre-aware resampling is less biased than multiplying a pixel index.
    return (
        (float(uv[0]) + 0.5) * native_w / capture_w - 0.5,
        (float(uv[1]) + 0.5) * native_h / capture_h - 0.5,
    )


def _percentiles(values: np.ndarray) -> dict[str, list[float]]:
    if values.size == 0:
        return {}
    return {
        "p10": [round(float(v), 2) for v in np.percentile(values, 10, axis=0)],
        "median": [round(float(v), 2) for v in np.median(values, axis=0)],
        "p90": [round(float(v), 2) for v in np.percentile(values, 90, axis=0)],
    }


def _component_record(
    contour: np.ndarray,
    colour_mask: np.ndarray,
    hsv: np.ndarray,
    colour: str,
    capture_size: tuple[int, int],
    native_size: tuple[int, int],
    camera: dict[str, Any],
    config,
    expected_cube_edge_m: float,
    frame_index: int,
) -> dict[str, Any]:
    capture_w, capture_h = capture_size
    native_w, native_h = native_size
    area_capture = float(cv2.contourArea(contour))
    area_native_equivalent = area_capture * native_w / capture_w * native_h / capture_h
    rect = cv2.minAreaRect(contour)
    rect_width, rect_height = (float(v) for v in rect[1])
    aspect = (
        math.inf
        if min(rect_width, rect_height) <= 0.0
        else max(rect_width, rect_height) / min(rect_width, rect_height)
    )
    rectangular_area = rect_width * rect_height
    fill = min(1.0, area_capture / max(rectangular_area, 1e-9))
    moments = cv2.moments(contour)
    centre_capture = (
        (float(rect[0][0]), float(rect[0][1]))
        if moments["m00"] <= 0
        else (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )
    )
    centre_native = scale_uv_to_native(centre_capture, capture_size, native_size)
    top_z = config.frames.surface_base_z_m + expected_cube_edge_m
    centre_base = pixel_to_base(camera, centre_native, z=top_z)

    box_capture = cv2.boxPoints(rect).astype(float)
    box_native = np.array(
        [scale_uv_to_native(tuple(point), capture_size, native_size) for point in box_capture],
        dtype=float,
    )
    projected_edges: list[tuple[float, np.ndarray]] = []
    for index in range(4):
        start = pixel_to_base(camera, tuple(box_native[index]), z=top_z)
        end = pixel_to_base(camera, tuple(box_native[(index + 1) % 4]), z=top_z)
        edge = end[:2] - start[:2]
        projected_edges.append((float(np.linalg.norm(edge)), edge))
    edge_length, world_edge = min(
        projected_edges,
        key=lambda item: abs(item[0] - expected_cube_edge_m),
    )
    shortest_edge = min(length for length, _ in projected_edges)
    yaw_rad = math.atan2(float(world_edge[1]), float(world_edge[0]))
    shape_score = max(
        0.0,
        1.0
        - (aspect - 1.0)
        / max(config.vision.maximum_component_aspect_ratio - 1.0, 1e-9),
    )
    confidence = float(np.clip(fill * shape_score, 0.0, 1.0))

    contour_pixels = np.zeros(colour_mask.shape, dtype=np.uint8)
    cv2.drawContours(contour_pixels, [contour], -1, 255, thickness=cv2.FILLED)
    component_pixels = hsv[(contour_pixels > 0) & (colour_mask > 0)]

    minimum_area = config.vision.minimum_component_area_px
    maximum_area = native_w * native_h * config.vision.maximum_component_area_fraction
    minimum_edge = expected_cube_edge_m * config.vision.cube_edge_minimum_factor
    maximum_edge = expected_cube_edge_m * config.vision.cube_edge_maximum_factor
    reasons: list[str] = []
    if not minimum_area <= area_native_equivalent <= maximum_area:
        reasons.append("component_area")
    if not math.isfinite(aspect) or aspect >= config.vision.maximum_component_aspect_ratio:
        reasons.append("aspect_ratio")
    if fill < config.vision.minimum_component_fill_ratio:
        reasons.append("fill_ratio")
    if shortest_edge < minimum_edge:
        reasons.append("physical_edge_too_small")
    if not minimum_edge <= edge_length <= maximum_edge:
        reasons.append("physical_edge_size")
    if confidence < config.vision.minimum_detection_confidence:
        reasons.append("confidence")

    synthetic_detection = CubeDetection(
        track_id=f"audit:{frame_index}:{colour}",
        colour=colour,
        capture_time_s=float(frame_index),
        position_base_m=(
            float(centre_base[0]),
            float(centre_base[1]),
            config.frames.surface_base_z_m + expected_cube_edge_m / 2.0,
        ),
        yaw_rad=yaw_rad,
        confidence=confidence,
        pixel_center_uv=centre_native,
        mask_area_px=area_native_equivalent,
    )
    in_pickup_region = detection_in_pickup_region(synthetic_detection, config)
    if not in_pickup_region:
        reasons.append("outside_pickup_region")

    return {
        "colour": colour,
        "frame_index": frame_index,
        "centre_capture_uv": [round(v, 2) for v in centre_capture],
        "centre_native_uv": [round(v, 2) for v in centre_native],
        "centre_base_xy_m": [round(float(v), 6) for v in centre_base[:2]],
        "area_capture_px": round(area_capture, 2),
        "area_native_equivalent_px": round(area_native_equivalent, 2),
        "rect_capture_px": [round(rect_width, 2), round(rect_height, 2)],
        "aspect_ratio": round(aspect, 3),
        "fill_ratio": round(fill, 3),
        "confidence": round(confidence, 3),
        "projected_edges_mm": [round(length * 1000.0, 3) for length, _ in projected_edges],
        "selected_edge_mm": round(edge_length * 1000.0, 3),
        "shortest_edge_mm": round(shortest_edge * 1000.0, 3),
        "hsv_selected_pixels": _percentiles(component_pixels),
        "in_pickup_region": in_pickup_region,
        "passes_current_detector_and_pickup_gates": not reasons,
        "rejection_reasons": reasons,
        "_contour": contour,
    }


def analyse_frame(
    bgr: np.ndarray,
    *,
    frame_index: int = 0,
    config=None,
    camera: dict[str, Any] | None = None,
    expected_cube_edge_m: float | None = None,
) -> dict[str, Any]:
    """Return an explainable audit for one saved BGR camera frame."""

    if config is None:
        config = load_config()
    if camera is None:
        camera = load_calibration()["table"]
    cube_edge_m = (
        config.cube_edge_m
        if expected_cube_edge_m is None
        else float(expected_cube_edge_m)
    )
    if not math.isfinite(cube_edge_m) or cube_edge_m <= 0.0:
        raise ValueError("expected cube edge must be positive and finite")
    image = np.asarray(bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("frame must be an HxWx3 uint8 BGR image")
    capture_h, capture_w = image.shape[:2]
    native_size = tuple(int(v) for v in camera["size"])
    native_w, native_h = native_size
    aspect_error = abs(capture_w / capture_h - native_w / native_h)
    if aspect_error > 1e-6:
        raise ValueError(
            "capture aspect ratio differs from the calibrated full table frame; "
            "cropped/letterboxed images cannot be projected safely"
        )

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    components: list[dict[str, Any]] = []
    native_area_scale = native_w / capture_w * native_h / capture_h
    for colour in config.task.colour_order:
        mask = _colour_mask(hsv, config.vision.hsv_ranges_opencv[colour])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Reject zero/tiny codec speckles before constructing a validated
            # CubeDetection for the pickup-region helper.
            if (
                float(cv2.contourArea(contour)) * native_area_scale
                < config.vision.minimum_component_area_px
            ):
                continue
            record = _component_record(
                contour,
                mask,
                hsv,
                colour,
                (capture_w, capture_h),
                native_size,
                camera,
                config,
                cube_edge_m,
                frame_index,
            )
            components.append(record)

    # Re-run the actual shared detector after restoring the calibrated frame
    # dimensions. This is closer to policy behaviour than the explainable raw
    # component records above, while remaining explicitly non-certifying: the
    # recording has already discarded native pixels and introduced codec loss.
    native_bgr = (
        image
        if (capture_w, capture_h) == (native_w, native_h)
        else cv2.resize(image, (native_w, native_h), interpolation=cv2.INTER_LINEAR)
    )
    replayed = detect_coloured_cubes(
        cv2.cvtColor(native_bgr, cv2.COLOR_BGR2RGB),
        lambda uv, z: pixel_to_base(camera, uv, z=z),
        config,
        scene_id=f"offline-audit:{frame_index}",
        capture_time_s=float(frame_index),
        expected_cube_edge_m=cube_edge_m,
    )
    replayed_pickable_counts = Counter(
        cube.colour
        for cube in replayed.cubes
        if detection_in_pickup_region(cube, config)
    )

    return {
        "frame_index": frame_index,
        "capture_size": [capture_w, capture_h],
        "native_calibration_size": [native_w, native_h],
        "resized_for_recording": (capture_w, capture_h) != (native_w, native_h),
        "expected_cube_edge_mm": cube_edge_m * 1000.0,
        "replayed_detector_pickable_counts": {
            colour: replayed_pickable_counts.get(colour, 0)
            for colour in config.task.colour_order
        },
        "components": components,
    }


def draw_overlay(bgr: np.ndarray, frame_audit: dict[str, Any]) -> np.ndarray:
    """Draw accepted/rejected components without changing the source image."""

    overlay = np.asarray(bgr).copy()
    for component in frame_audit["components"]:
        accepted = component["passes_current_detector_and_pickup_gates"]
        colour = (40, 220, 40) if accepted else (30, 30, 230)
        contour = component["_contour"]
        cv2.drawContours(overlay, [contour], -1, colour, 1)
        u, v = (int(round(x)) for x in component["centre_capture_uv"])
        reasons = component["rejection_reasons"]
        label = component["colour"] + (" OK" if accepted else " " + ",".join(reasons[:2]))
        cv2.putText(
            overlay,
            label,
            (max(0, u + 3), max(12, v - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            colour,
            1,
            cv2.LINE_AA,
        )
    return overlay


def _json_safe_frame(frame: dict[str, Any]) -> dict[str, Any]:
    clean = dict(frame)
    clean["components"] = [
        {key: value for key, value in component.items() if not key.startswith("_")}
        for component in frame["components"]
    ]
    return clean


def summarise_frames(frames: list[dict[str, Any]], input_path: Path) -> dict[str, Any]:
    all_components = [component for frame in frames for component in frame["components"]]
    by_colour: dict[str, dict[str, Any]] = {}
    for colour in ("red", "yellow", "blue"):
        components = [c for c in all_components if c["colour"] == colour]
        pickable = [c for c in components if c["passes_current_detector_and_pickup_gates"]]
        in_region = [c for c in components if c["in_pickup_region"]]
        reason_counts = Counter(
            reason for component in in_region for reason in component["rejection_reasons"]
        )
        sizes = [c["selected_edge_mm"] for c in in_region]
        by_colour[colour] = {
            "components": len(components),
            "components_in_pickup_region": len(in_region),
            "passing_components": len(pickable),
            "pickup_rejection_reasons": dict(sorted(reason_counts.items())),
            "selected_edge_mm_in_pickup_region": (
                {
                    "median": round(float(np.median(sizes)), 3),
                    "p10": round(float(np.percentile(sizes, 10)), 3),
                    "p90": round(float(np.percentile(sizes, 90)), 3),
                }
                if sizes
                else None
            ),
        }
    first = frames[0] if frames else None
    replay_complete = [
        all(
            frame["replayed_detector_pickable_counts"].get(colour, 0) == 1
            for colour in ("red", "yellow", "blue")
        )
        for frame in frames
    ]
    longest_complete_run = 0
    current_complete_run = 0
    for complete in replay_complete:
        current_complete_run = current_complete_run + 1 if complete else 0
        longest_complete_run = max(longest_complete_run, current_complete_run)
    return {
        "tool": "offline_table_capture_audit",
        "input": str(input_path),
        "warning": WARNING,
        "frames_analysed": len(frames),
        "capture_size": first["capture_size"] if first else None,
        "native_calibration_size": first["native_calibration_size"] if first else None,
        "resized_or_compressed_evidence": bool(
            first and first["resized_for_recording"]
        ),
        "configured_cube_edge_mm": (
            first["expected_cube_edge_mm"] if first else None
        ),
        "replayed_detector": {
            "frames_with_exactly_one_pickable_cube_of_each_colour": sum(
                replay_complete
            ),
            "maximum_consecutive_sampled_complete_frames": longest_complete_run,
            "sample_interval_note": (
                "Consecutive means consecutive analysed samples, not native video frames."
            ),
        },
        "by_colour": by_colour,
    }


def _iter_capture(path: Path, every: int, max_frames: int | None):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        yield 0, image
        return

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot decode image or video: {path}")
    frame_index = 0
    analysed = 0
    try:
        while max_frames is None or analysed < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % every == 0:
                yield frame_index, frame
                analysed += 1
            frame_index += 1
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="saved full-frame table image or video")
    parser.add_argument("--every", type=int, default=10, help="analyse every Nth video frame")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument(
        "--cube-mm",
        type=int,
        choices=(10, 16),
        default=10,
        help="hardware perception profile to audit (default: 10)",
    )
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--max-overlays", type=int, default=20)
    parser.add_argument("--json", type=Path, dest="json_path")
    parser.add_argument(
        "--details-limit",
        type=int,
        default=0,
        help="include up to this many per-frame audits in JSON (0 = summary only)",
    )
    args = parser.parse_args()
    if args.every < 1 or args.max_frames < 1 or args.max_overlays < 0:
        parser.error("--every/--max-frames must be positive and --max-overlays non-negative")
    if args.details_limit < 0:
        parser.error("--details-limit must be non-negative")
    if not args.capture.is_file():
        parser.error(f"capture does not exist: {args.capture}")

    if args.overlay_dir is not None:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, Any]] = []
    overlays_written = 0
    for frame_index, bgr in _iter_capture(args.capture, args.every, args.max_frames):
        audit = analyse_frame(
            bgr,
            frame_index=frame_index,
            expected_cube_edge_m=args.cube_mm / 1000.0,
        )
        frames.append(audit)
        if args.overlay_dir is not None and overlays_written < args.max_overlays:
            overlay = draw_overlay(bgr, audit)
            destination = args.overlay_dir / f"frame-{frame_index:06d}.png"
            if not cv2.imwrite(str(destination), overlay):
                raise OSError(f"could not write overlay: {destination}")
            overlays_written += 1

    if not frames:
        raise SystemExit("capture decoded but contained no frames")
    report = summarise_frames(frames, args.capture)
    if args.details_limit:
        report["frame_details"] = [
            _json_safe_frame(frame) for frame in frames[: args.details_limit]
        ]
    report["overlays_written"] = overlays_written
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
