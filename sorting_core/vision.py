"""Deterministic HSV cube detection from a native RGB camera observation.

This initial detector is intentionally narrow: known red, yellow, and blue
10 mm cubes on the calibrated table.  It consumes image pixels and an injected
camera projection only—never simulator object poses or evaluation ground truth.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .config import SortConfig
from .contracts import CubeDetection, SceneDetections, normalise_square_yaw_rad


class VisionError(ValueError):
    """Raised for malformed images or unusable camera projections."""


def _pixel_to_base(projector: Any, uv: tuple[float, float], z_m: float) -> np.ndarray:
    """Call either ``fn(uv, z)`` or a supported camera-model method."""

    if hasattr(projector, "pixel_to_base"):
        value = projector.pixel_to_base(uv, z_m)
    elif hasattr(projector, "to_plane"):
        value = projector.to_plane(uv[0], uv[1], z=z_m)
    elif callable(projector):
        value = projector(uv, z_m)
    else:
        raise VisionError(
            "projector must be callable or expose pixel_to_base()/to_plane()"
        )
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise VisionError("camera projection must return three finite base coordinates")
    return point


def _colour_mask(hsv: np.ndarray, ranges) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for h_min, h_max, s_min, v_min in ranges:
        component = cv2.inRange(
            hsv,
            np.array([h_min, s_min, v_min], dtype=np.uint8),
            np.array([h_max, 255, 255], dtype=np.uint8),
        )
        mask = cv2.bitwise_or(mask, component)
    return mask


def detect_coloured_cubes(
    rgb: np.ndarray,
    projector: Any,
    config: SortConfig,
    *,
    scene_id: str,
    capture_time_s: float,
) -> SceneDetections:
    """Detect configured cube colours and estimate base XY/yaw from RGB.

    The input must be the uncropped camera frame in RGB channel order.  For the
    official calibration, inject
    ``lambda uv, z: calibration.geometry.pixel_to_base(camera, uv, z=z)``.
    The supplied simulator ``CameraModel`` can be passed directly because it
    exposes ``to_plane(u, v, z=...)``.

    Pixel centres and square edges are projected at the cube's *top face*, not
    the table plane; the returned detection position is the cube centre.  This
    avoids a systematic parallax error from treating visible top pixels as if
    they lay on the table.
    """

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise VisionError("rgb must be an HxWx3 uint8 native RGB image")
    height, width = image.shape[:2]
    if height < 2 or width < 2:
        raise VisionError("rgb image is too small")
    if not str(scene_id).strip() or not math.isfinite(float(capture_time_s)):
        raise VisionError("scene id and capture timestamp are required")

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    kernel_size = config.vision.morphology_kernel_px
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    top_z = config.cube_top_base_z_m
    maximum_area = width * height * config.vision.maximum_component_area_fraction
    pending: dict[str, list[dict[str, Any]]] = {
        colour: [] for colour in config.task.colour_order
    }

    for colour in config.task.colour_order:
        mask = _colour_mask(hsv, config.vision.hsv_ranges_opencv[colour])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not config.vision.minimum_component_area_px <= area <= maximum_area:
                continue
            rect = cv2.minAreaRect(contour)
            rect_width, rect_height = (float(v) for v in rect[1])
            if min(rect_width, rect_height) <= 0:
                continue
            aspect = max(rect_width, rect_height) / min(rect_width, rect_height)
            if aspect >= config.vision.maximum_component_aspect_ratio:
                # A stretched component commonly means two touching cubes or a
                # distractor. Rejecting the boundary too prevents constructing
                # a zero-shape-confidence detection in the first place.
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            centre_uv = (
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            )
            centre_base = _pixel_to_base(projector, centre_uv, top_z)
            box = cv2.boxPoints(rect).astype(float)
            projected_edges: list[tuple[float, np.ndarray]] = []
            for edge_index in range(len(box)):
                start = _pixel_to_base(projector, tuple(box[edge_index]), top_z)
                end = _pixel_to_base(
                    projector,
                    tuple(box[(edge_index + 1) % len(box)]),
                    top_z,
                )
                world_edge = end[:2] - start[:2]
                projected_edges.append(
                    (float(np.linalg.norm(world_edge)), world_edge)
                )

            # A colour mask contains the visible sides as well as the top face.
            # In an oblique camera view that makes one minimum-area-rectangle
            # dimension longer than the known cube footprint.  Selecting the
            # longest image edge therefore rejected a real blue cube in the
            # Isaac table-camera view.  Select the projected edge that best
            # agrees with the configured physical cube size instead.  This is
            # still camera geometry plus a declared part dimension; it never
            # consumes a simulator pose or other evaluator-only truth.
            edge_length, world_edge = min(
                projected_edges,
                key=lambda item: abs(item[0] - config.cube_edge_m),
            )
            if edge_length <= 1e-9:
                continue
            minimum_edge = config.cube_edge_m * config.vision.cube_edge_minimum_factor
            maximum_edge = config.cube_edge_m * config.vision.cube_edge_maximum_factor
            if not minimum_edge <= edge_length <= maximum_edge:
                # The matching bins use the same colours as the cubes.  Their
                # 35/50 mm edges must never become duplicate 10 mm targets.
                continue
            yaw = normalise_square_yaw_rad(
                math.atan2(float(world_edge[1]), float(world_edge[0]))
            )
            rectangular_area = rect_width * rect_height
            fill = min(1.0, area / max(rectangular_area, 1e-9))
            shape_score = max(
                0.0,
                1.0
                - (aspect - 1.0)
                / max(config.vision.maximum_component_aspect_ratio - 1.0, 1e-9),
            )
            pending[colour].append(
                {
                    "centre_uv": centre_uv,
                    "position": (
                        float(centre_base[0]),
                        float(centre_base[1]),
                        config.cube_centre_base_z_m,
                    ),
                    "yaw": yaw,
                    "confidence": float(np.clip(fill * shape_score, 0.0, 1.0)),
                    "area": area,
                }
            )

    detections: list[CubeDetection] = []
    for colour in config.task.colour_order:
        candidates = sorted(
            pending[colour], key=lambda item: (item["centre_uv"][1], item["centre_uv"][0])
        )
        for index, candidate in enumerate(candidates):
            detections.append(
                CubeDetection(
                    track_id=f"{scene_id}:{colour}:{index}",
                    colour=colour,
                    capture_time_s=capture_time_s,
                    position_base_m=candidate["position"],
                    yaw_rad=candidate["yaw"],
                    confidence=candidate["confidence"],
                    pixel_center_uv=candidate["centre_uv"],
                    mask_area_px=candidate["area"],
                )
            )
    return SceneDetections(
        scene_id=scene_id,
        capture_time_s=capture_time_s,
        cubes=tuple(detections),
    )


def detection_in_pickup_region(
    detection: CubeDetection, config: SortConfig
) -> bool:
    """Return whether a camera detection is eligible to become a pick target.

    Raw colour detections deliberately remain unfiltered so adapters can first
    classify destination-bin observations and report out-of-workspace coloured
    components.  This helper is task-space semantics, not object ground truth.
    """

    x_m, y_m = detection.position_base_m[:2]
    x_bounds = config.task.pickup_region_base_m["x"]
    y_bounds = config.task.pickup_region_base_m["y"]
    return (
        x_bounds[0] <= x_m <= x_bounds[1]
        and y_bounds[0] <= y_m <= y_bounds[1]
    )


def filter_scene_to_pickup_region(
    scene: SceneDetections, config: SortConfig
) -> SceneDetections:
    """Create the pick-eligible view of a broader raw camera scene."""

    return SceneDetections(
        scene_id=scene.scene_id,
        capture_time_s=scene.capture_time_s,
        cubes=tuple(
            detection
            for detection in scene.cubes
            if detection_in_pickup_region(detection, config)
        ),
    )
