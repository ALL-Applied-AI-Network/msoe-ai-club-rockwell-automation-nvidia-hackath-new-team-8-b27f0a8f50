"""Camera-derived cube and lying-screw perception.

This provider deliberately combines two narrow methods instead of pretending
one model solves every part:

* the unchanged, measured HSV/geometry cube baseline; and
* a conservative dark-silhouette detector for the catalogued ISO 10642 M6x30
  screw when it is lying on the table.

The screw result identifies the head-face origin and head-to-tip shaft axis.
Rotation about the nearly symmetric shaft is not observable at the current
camera scale, so local +X is canonically aligned with robot-base +Z.  The
ambiguity is retained in attributes for the grasp team.

This module consumes only RGB pixels and an injected calibrated projector.  It
never consumes simulator poses or configured spawn locations.  It is selected
only through an explicit integration profile; the profile-unset cube behavior
is therefore unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import cv2
import numpy as np

from sorting_core.perception import (
    DetectedObject,
    PerceptionRequest,
    Pose3D,
    SceneObservation,
)
from sorting_core.vision import ColourCubeVisionProvider, VisionError, _pixel_to_base

if TYPE_CHECKING:
    from sorting_core.config import SortConfig
    from sorting_core.integration_plugins import VisionFactoryContext
    from sorting_core.perception import VisionProvider


PLUGIN_API_VERSION = 1
SCREW_CLASS_ID = "screw_iso10642_m6x30"
SCREW_DIMENSIONS_M = (0.012, 0.01199627, 0.030)
SCREW_CAD_SHA256 = "6a842f1de0923083b0eb2a3a209eacae87cc83d4b62e15b768a0a8cd277cec2a"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result != value:
        raise ValueError(f"{name} must be an integer")
    return result


def _pair(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain width and height")
    try:
        result = tuple(_integer(item, name) for item in value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain width and height") from exc
    if len(result) != 2 or min(result) <= 1:
        raise ValueError(f"{name} must contain positive width and height")
    return result


@dataclass(frozen=True)
class ScrewDetectorConfig:
    """Measured-geometry gates for one lying M6x30 screw silhouette."""

    expected_image_size: tuple[int, int] = (1920, 1080)
    require_expected_image_size: bool = True
    maximum_gray: int = 100
    maximum_saturation: int = 150
    context_margin_px: int = 36
    minimum_white_context_fraction: float = 0.70
    context_minimum_gray: int = 135
    context_maximum_saturation: int = 90
    morphology_close_px: int = 3
    minimum_component_area_px: float = 35.0
    maximum_component_area_px: float = 6000.0
    minimum_length_m: float = 0.026
    maximum_length_m: float = 0.034
    minimum_width_m: float = 0.009
    maximum_width_m: float = 0.014
    minimum_aspect_ratio: float = 1.8
    terminal_fraction: float = 0.28
    minimum_head_width_ratio: float = 1.18
    origin_height_above_surface_m: float = 0.0065
    workspace_margin_m: float = 0.0
    minimum_detection_confidence: float = 0.30
    track_match_distance_m: float = 0.025
    track_ttl_s: float = 2.0
    minimum_track_observations: int = 3
    maximum_track_length_change_m: float = 0.004
    maximum_track_width_change_m: float = 0.003
    maximum_track_axis_change_deg: float = 15.0
    maximum_track_head_ratio_change: float = 0.5
    report_ambiguous_components: bool = True

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> "ScrewDetectorConfig":
        raw = dict(options)
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown screw detector options: {unknown}")
        if "expected_image_size" in raw:
            raw["expected_image_size"] = _pair(
                raw["expected_image_size"], "expected_image_size"
            )
        integer_fields = {
            "maximum_gray",
            "maximum_saturation",
            "context_margin_px",
            "context_minimum_gray",
            "context_maximum_saturation",
            "morphology_close_px",
            "minimum_track_observations",
        }
        boolean_fields = {
            "require_expected_image_size",
            "report_ambiguous_components",
        }
        for name in integer_fields & set(raw):
            raw[name] = _integer(raw[name], name)
        for name in boolean_fields & set(raw):
            if not isinstance(raw[name], bool):
                raise ValueError(f"{name} must be a boolean")
        for name in set(raw) - integer_fields - boolean_fields - {"expected_image_size"}:
            raw[name] = _finite(raw[name], name)
        result = cls(**raw)
        result.validate()
        return result

    def validate(self) -> None:
        if not 0 <= self.maximum_gray <= 255:
            raise ValueError("maximum_gray must be in [0, 255]")
        if not 0 <= self.maximum_saturation <= 255:
            raise ValueError("maximum_saturation must be in [0, 255]")
        if self.context_margin_px < 1:
            raise ValueError("context_margin_px must be positive")
        if not 0 <= self.context_minimum_gray <= 255:
            raise ValueError("context_minimum_gray must be in [0, 255]")
        if not 0 <= self.context_maximum_saturation <= 255:
            raise ValueError("context_maximum_saturation must be in [0, 255]")
        if not 0 <= self.minimum_white_context_fraction <= 1:
            raise ValueError("minimum_white_context_fraction must be in [0, 1]")
        if self.morphology_close_px < 1 or self.morphology_close_px % 2 == 0:
            raise ValueError("morphology_close_px must be a positive odd integer")
        if not 0 < self.minimum_component_area_px < self.maximum_component_area_px:
            raise ValueError("component-area bounds are invalid")
        if not 0 < self.minimum_length_m < self.maximum_length_m:
            raise ValueError("length bounds are invalid")
        if not 0 < self.minimum_width_m < self.maximum_width_m:
            raise ValueError("width bounds are invalid")
        if self.minimum_aspect_ratio <= 1:
            raise ValueError("minimum_aspect_ratio must exceed one")
        if not 0.1 <= self.terminal_fraction <= 0.4:
            raise ValueError("terminal_fraction must be in [0.1, 0.4]")
        if self.minimum_head_width_ratio <= 1:
            raise ValueError("minimum_head_width_ratio must exceed one")
        if self.origin_height_above_surface_m <= 0:
            raise ValueError("origin_height_above_surface_m must be positive")
        if not 0 <= self.minimum_detection_confidence <= 1:
            raise ValueError("minimum_detection_confidence must be in [0, 1]")
        if self.track_match_distance_m <= 0 or self.track_ttl_s <= 0:
            raise ValueError("track distance and TTL must be positive")
        if self.minimum_track_observations < 1:
            raise ValueError("minimum_track_observations must be positive")
        if self.workspace_margin_m < 0:
            raise ValueError("workspace_margin_m must be non-negative")
        if (
            self.maximum_track_length_change_m <= 0
            or self.maximum_track_width_change_m <= 0
            or not 0 < self.maximum_track_axis_change_deg <= 180
            or self.maximum_track_head_ratio_change <= 0
        ):
            raise ValueError("track-consistency bounds are invalid")


def _matrix_to_quaternion_xyzw(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper 3x3 rotation matrix to a canonical unit quaternion."""

    value = np.asarray(matrix, dtype=float)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise VisionError("rotation matrix must be finite and 3x3")
    trace = float(np.trace(value))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (value[2, 1] - value[1, 2]) / scale
        y = (value[0, 2] - value[2, 0]) / scale
        z = (value[1, 0] - value[0, 1]) / scale
    else:
        diagonal = np.diag(value)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + value[0, 0] - value[1, 1] - value[2, 2]) * 2.0
            w = (value[2, 1] - value[1, 2]) / scale
            x = 0.25 * scale
            y = (value[0, 1] + value[1, 0]) / scale
            z = (value[0, 2] + value[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + value[1, 1] - value[0, 0] - value[2, 2]) * 2.0
            w = (value[0, 2] - value[2, 0]) / scale
            x = (value[0, 1] + value[1, 0]) / scale
            y = 0.25 * scale
            z = (value[1, 2] + value[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + value[2, 2] - value[0, 0] - value[1, 1]) * 2.0
            w = (value[1, 0] - value[0, 1]) / scale
            x = (value[0, 2] + value[2, 0]) / scale
            y = (value[1, 2] + value[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray((x, y, z, w), dtype=float)
    quaternion /= float(np.linalg.norm(quaternion))
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return tuple(float(component) for component in quaternion)


def _lying_screw_quaternion(head_to_tip_xy: np.ndarray) -> tuple[float, float, float, float]:
    """Return the catalog lying-pose frame from a directed horizontal axis.

    Catalog local -Z points head-to-tip.  Axial roll is not observable, so the
    documented canonical choice maps local +X to robot-base +Z.
    """

    axis = np.asarray(head_to_tip_xy, dtype=float)
    if axis.shape != (2,) or not np.all(np.isfinite(axis)):
        raise VisionError("head-to-tip axis must contain two finite values")
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        raise VisionError("head-to-tip axis is degenerate")
    direction = axis / norm
    local_x_base = np.array((0.0, 0.0, 1.0), dtype=float)
    local_z_base = np.array((-direction[0], -direction[1], 0.0), dtype=float)
    local_y_base = np.cross(local_z_base, local_x_base)
    rotation = np.column_stack((local_x_base, local_y_base, local_z_base))
    return _matrix_to_quaternion_xyzw(rotation)


@dataclass(frozen=True)
class _ComponentMeasurement:
    contour: np.ndarray
    mask_area_px: float
    pixel_center_uv: tuple[float, float]
    centre_xy_m: tuple[float, float]
    head_xy_m: tuple[float, float]
    tip_xy_m: tuple[float, float]
    length_m: float
    width_m: float
    head_width_ratio: float
    fit_residual_m: float
    touches_image_border: bool
    bounding_box_xywh: tuple[int, int, int, int]
    footprint_bounds_xy_m: tuple[float, float, float, float]


def _measure_component(
    contour: np.ndarray,
    component_mask: np.ndarray,
    projector: Any,
    *,
    plane_z_m: float,
    terminal_fraction: float,
) -> _ComponentMeasurement | None:
    rows, cols = np.nonzero(component_mask)
    if len(rows) < 4:
        return None
    pixels = np.column_stack((cols.astype(float), rows.astype(float)))
    base_points = np.asarray(
        [_pixel_to_base(projector, tuple(uv), plane_z_m)[:2] for uv in pixels],
        dtype=float,
    )
    centre = np.mean(base_points, axis=0)
    centered = base_points - centre
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    perpendicular = np.array((-axis[1], axis[0]), dtype=float)
    longitudinal = centered @ axis
    transverse = centered @ perpendicular
    low, high = np.quantile(longitudinal, (0.01, 0.99))
    length = float(high - low)
    if length <= 1e-9:
        return None
    width = float(np.quantile(transverse, 0.99) - np.quantile(transverse, 0.01))
    cutoff = length * terminal_fraction
    low_terminal = longitudinal <= low + cutoff
    high_terminal = longitudinal >= high - cutoff
    if int(np.count_nonzero(low_terminal)) < 2 or int(np.count_nonzero(high_terminal)) < 2:
        return None

    def terminal_width(selector: np.ndarray) -> float:
        values = transverse[selector]
        return float(np.quantile(values, 0.95) - np.quantile(values, 0.05))

    low_width = terminal_width(low_terminal)
    high_width = terminal_width(high_terminal)
    head_is_low = low_width >= high_width
    head_selector = low_terminal if head_is_low else high_terminal
    head_longitudinal = low if head_is_low else high
    tip_longitudinal = high if head_is_low else low
    head_transverse = float(np.median(transverse[head_selector]))
    opposite_width = high_width if head_is_low else low_width
    head_width = low_width if head_is_low else high_width
    head_ratio = head_width / max(opposite_width, 1e-9)
    head = centre + axis * head_longitudinal + perpendicular * head_transverse
    tip = centre + axis * tip_longitudinal
    predicted_transverse = np.median(transverse)
    fit_residual = float(np.median(np.abs(transverse - predicted_transverse)))
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return None
    pixel_center = (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )
    height, width_px = component_mask.shape
    x, y, w, h = cv2.boundingRect(contour)
    touches_border = x <= 0 or y <= 0 or x + w >= width_px or y + h >= height
    return _ComponentMeasurement(
        contour=contour,
        mask_area_px=float(cv2.contourArea(contour)),
        pixel_center_uv=pixel_center,
        centre_xy_m=(float(centre[0]), float(centre[1])),
        head_xy_m=(float(head[0]), float(head[1])),
        tip_xy_m=(float(tip[0]), float(tip[1])),
        length_m=length,
        width_m=width,
        head_width_ratio=float(head_ratio),
        fit_residual_m=fit_residual,
        touches_image_border=touches_border,
        bounding_box_xywh=(int(x), int(y), int(w), int(h)),
        footprint_bounds_xy_m=(
            float(np.min(base_points[:, 0])),
            float(np.max(base_points[:, 0])),
            float(np.min(base_points[:, 1])),
            float(np.max(base_points[:, 1])),
        ),
    )


def _score_closeness(value: float, target: float, tolerance: float) -> float:
    return float(np.clip(1.0 - abs(value - target) / max(tolerance, 1e-9), 0.0, 1.0))


def detect_lying_screws(
    rgb: np.ndarray,
    projector: Any,
    config: "SortConfig",
    detector_config: ScrewDetectorConfig,
    *,
    scene_id: str,
    capture_time_s: float,
    camera_id: str,
) -> tuple[DetectedObject, ...]:
    """Detect isolated dark catalog screws and conservative ambiguous obstacles."""

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise VisionError("rgb must be an HxWx3 uint8 native RGB image")
    height, width = image.shape[:2]
    if detector_config.require_expected_image_size and (
        width,
        height,
    ) != detector_config.expected_image_size:
        raise VisionError(
            "screw profile requires an uncropped native image of size "
            f"{detector_config.expected_image_size}, got {(width, height)}"
        )
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = np.where(
        (gray <= detector_config.maximum_gray)
        & (hsv[:, :, 1] <= detector_config.maximum_saturation),
        255,
        0,
    ).astype(np.uint8)
    kernel = np.ones(
        (detector_config.morphology_close_px,) * 2,
        dtype=np.uint8,
    )
    if detector_config.morphology_close_px > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    origin_z = config.frames.surface_base_z_m + detector_config.origin_height_above_surface_m
    x_bounds = config.task.pickup_region_base_m["x"]
    y_bounds = config.task.pickup_region_base_m["y"]
    objects: list[DetectedObject] = []
    candidate_index = 0
    ambiguous_index = 0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not (
            detector_config.minimum_component_area_px
            <= area
            <= detector_config.maximum_component_area_px
        ):
            continue
        component_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], -1, 255, thickness=-1)
        measurement = _measure_component(
            contour,
            component_mask,
            projector,
            plane_z_m=origin_z,
            terminal_fraction=detector_config.terminal_fraction,
        )
        if measurement is None:
            continue
        centre_x, centre_y = measurement.centre_xy_m
        footprint_min_x, footprint_max_x, footprint_min_y, footprint_max_y = (
            measurement.footprint_bounds_xy_m
        )
        margin = detector_config.workspace_margin_m
        footprint_overlaps_workspace = (
            footprint_max_x >= x_bounds[0] - margin
            and footprint_min_x <= x_bounds[1] + margin
            and footprint_max_y >= y_bounds[0] - margin
            and footprint_min_y <= y_bounds[1] + margin
        )
        if not footprint_overlaps_workspace:
            continue
        aspect = measurement.length_m / max(measurement.width_m, 1e-9)
        box_x, box_y, box_w, box_h = measurement.bounding_box_xywh
        context_x0 = max(0, box_x - detector_config.context_margin_px)
        context_y0 = max(0, box_y - detector_config.context_margin_px)
        context_x1 = min(width, box_x + box_w + detector_config.context_margin_px)
        context_y1 = min(height, box_y + box_h + detector_config.context_margin_px)
        context_gray = gray[context_y0:context_y1, context_x0:context_x1]
        context_saturation = hsv[
            context_y0:context_y1,
            context_x0:context_x1,
            1,
        ]
        white_context_fraction = float(
            np.mean(
                (context_gray >= detector_config.context_minimum_gray)
                & (
                    context_saturation
                    <= detector_config.context_maximum_saturation
                )
            )
        )
        context_is_table_like = (
            white_context_fraction
            >= detector_config.minimum_white_context_fraction
        )
        geometry_in_range = (
            detector_config.minimum_length_m
            <= measurement.length_m
            <= detector_config.maximum_length_m
            and detector_config.minimum_width_m
            <= measurement.width_m
            <= detector_config.maximum_width_m
            and aspect >= detector_config.minimum_aspect_ratio
        )
        polarity_resolved = (
            measurement.head_width_ratio >= detector_config.minimum_head_width_ratio
        )
        fully_visible = not measurement.touches_image_border
        footprint_in_workspace = (
            x_bounds[0] + margin <= footprint_min_x
            and footprint_max_x <= x_bounds[1] - margin
            and y_bounds[0] + margin <= footprint_min_y
            and footprint_max_y <= y_bounds[1] - margin
        )
        head = np.asarray(measurement.head_xy_m, dtype=float)
        tip = np.asarray(measurement.tip_xy_m, dtype=float)
        axis = tip - head
        length_score = _score_closeness(measurement.length_m, 0.030, 0.012)
        width_score = _score_closeness(measurement.width_m, 0.012, 0.009)
        polarity_score = float(
            np.clip(
                (measurement.head_width_ratio - 1.0) / 0.8,
                0.0,
                1.0,
            )
        )
        confidence = float(
            np.clip(0.45 * length_score + 0.20 * width_score + 0.35 * polarity_score, 0.0, 1.0)
        )
        accepted = (
            geometry_in_range
            and polarity_resolved
            and fully_visible
            and footprint_in_workspace
            and context_is_table_like
            and confidence >= detector_config.minimum_detection_confidence
        )
        common_attributes = {
            "catalog_id": SCREW_CLASS_ID,
            "detector_family": "dark_silhouette_width_profile_v1",
            "pixel_center_uv": measurement.pixel_center_uv,
            "mask_area_px": measurement.mask_area_px,
            "projected_length_m": measurement.length_m,
            "projected_width_m": measurement.width_m,
            "head_width_ratio": measurement.head_width_ratio,
            "line_fit_residual_m": measurement.fit_residual_m,
            "white_table_context_fraction": white_context_fraction,
            "footprint_bounds_xy_base_m": measurement.footprint_bounds_xy_m,
            "footprint_inside_pickup_region": footprint_in_workspace,
            "head_xy_base_m": measurement.head_xy_m,
            "tip_xy_base_m": measurement.tip_xy_m,
            "pose_constraint": "lying_on_table",
            "orientation_observable": "head_to_tip_axis_only",
            "axial_roll": "unobservable; canonical local +X maps to base +Z",
            "symmetry_id": "catalog_axial_symmetry_unresolved",
            "orientation_interpretation": (
                "restricted lying-pose canonical representative; not a "
                "catalog-approved full 6D orientation"
            ),
            "occlusion_status": (
                "image_boundary_occlusion"
                if measurement.touches_image_border
                else "not_estimated_geometry_gate_only"
            ),
            "uncertainty_scope": "mask-fit only; absolute calibration error unverified",
            "absolute_base_accuracy_verified": False,
        }
        if accepted:
            quaternion = _lying_screw_quaternion(axis)
            objects.append(
                DetectedObject(
                    track_id=f"{scene_id}:screw:{candidate_index}",
                    class_id=SCREW_CLASS_ID,
                    sort_label="red",
                    capture_time_s=capture_time_s,
                    pose=Pose3D(
                        position_base_m=(
                            float(head[0]),
                            float(head[1]),
                            float(origin_z),
                        ),
                        orientation_xyzw=quaternion,
                    ),
                    dimensions_m=SCREW_DIMENSIONS_M,
                    confidence=confidence,
                    pose_quality=float(
                        np.clip(0.65 * length_score + 0.35 * polarity_score, 0.0, 1.0)
                    ),
                    visibility=None,
                    source_camera_ids=(camera_id,),
                    attributes=common_attributes,
                )
            )
            candidate_index += 1
        elif detector_config.report_ambiguous_components and geometry_in_range:
            reason_parts = []
            if not polarity_resolved:
                reason_parts.append("head_tip_polarity_ambiguous")
            if not fully_visible:
                reason_parts.append("image_boundary_occlusion")
            if not footprint_in_workspace:
                reason_parts.append("footprint_outside_pickup_region")
            if not context_is_table_like:
                reason_parts.append("not_isolated_on_white_table")
            if confidence < detector_config.minimum_detection_confidence:
                reason_parts.append("low_shape_confidence")
            reason = ",".join(reason_parts) or "geometry_not_executable"
            attributes = dict(common_attributes)
            attributes.update(
                {
                    "non_executable": True,
                    "ambiguity_reason": reason,
                    "candidate_class": SCREW_CLASS_ID,
                }
            )
            objects.append(
                DetectedObject(
                    track_id=f"{scene_id}:unknown-dark:{ambiguous_index}",
                    class_id="unknown_obstacle",
                    sort_label=None,
                    capture_time_s=capture_time_s,
                    pose=Pose3D(
                        position_base_m=(
                            centre_x,
                            centre_y,
                            float(origin_z),
                        ),
                        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                    ),
                    dimensions_m=(
                        max(measurement.length_m, 0.001),
                        max(measurement.length_m, 0.001),
                        max(2.0 * detector_config.origin_height_above_surface_m, 0.001),
                    ),
                    confidence=min(confidence, 0.49),
                    pose_quality=0.0,
                    visibility=None,
                    source_camera_ids=(camera_id,),
                    attributes={
                        **attributes,
                        "collision_proxy": (
                            "conservative_axis_aligned_length_square"
                        ),
                    },
                )
            )
            ambiguous_index += 1
    return tuple(objects)


class _NearestNeighbourTracker:
    """Best-effort IDs for separated slow objects, not motion prediction."""

    def __init__(
        self,
        *,
        match_distance_m: float,
        ttl_s: float,
        minimum_screw_observations: int,
        maximum_screw_length_change_m: float,
        maximum_screw_width_change_m: float,
        maximum_screw_axis_change_deg: float,
        maximum_screw_head_ratio_change: float,
    ) -> None:
        self._match_distance_m = match_distance_m
        self._ttl_s = ttl_s
        self._minimum_screw_observations = minimum_screw_observations
        self._maximum_screw_length_change_m = maximum_screw_length_change_m
        self._maximum_screw_width_change_m = maximum_screw_width_change_m
        self._minimum_screw_axis_dot = math.cos(
            math.radians(maximum_screw_axis_change_deg)
        )
        self._maximum_screw_head_ratio_change = maximum_screw_head_ratio_change
        self._next_id = 0
        self._tracks: dict[str, tuple[DetectedObject, np.ndarray, float, int]] = {}
        self._last_capture_time_s: float | None = None

    def assign(
        self, objects: Sequence[DetectedObject], *, capture_time_s: float
    ) -> tuple[DetectedObject, ...]:
        if (
            self._last_capture_time_s is not None
            and capture_time_s < self._last_capture_time_s - 1e-6
        ):
            raise VisionError("tracker received a non-monotonic capture timestamp")
        self._last_capture_time_s = capture_time_s
        self._tracks = {
            key: value
            for key, value in self._tracks.items()
            if capture_time_s - value[2] <= self._ttl_s
        }
        unmatched_tracks = set(self._tracks)
        assigned: list[DetectedObject] = []
        for detected in sorted(
            objects,
            key=lambda item: (
                item.class_id,
                item.sort_label or "",
                item.pose.position_base_m[0],
                item.pose.position_base_m[1],
            ),
        ):
            position = np.asarray(detected.pose.position_base_m, dtype=float)
            compatible = []
            for track_id in unmatched_tracks:
                previous_object, previous, _, _ = self._tracks[track_id]
                if (
                    previous_object.class_id != detected.class_id
                    or previous_object.sort_label != detected.sort_label
                ):
                    continue
                if detected.class_id == SCREW_CLASS_ID:
                    previous_attributes = previous_object.attributes
                    detected_attributes = detected.attributes
                    if (
                        abs(
                            float(previous_attributes["projected_length_m"])
                            - float(detected_attributes["projected_length_m"])
                        )
                        > self._maximum_screw_length_change_m
                        or abs(
                            float(previous_attributes["projected_width_m"])
                            - float(detected_attributes["projected_width_m"])
                        )
                        > self._maximum_screw_width_change_m
                        or abs(
                            float(previous_attributes["head_width_ratio"])
                            - float(detected_attributes["head_width_ratio"])
                        )
                        > self._maximum_screw_head_ratio_change
                    ):
                        continue
                    previous_axis = np.asarray(
                        previous_attributes["tip_xy_base_m"], dtype=float
                    ) - np.asarray(
                        previous_attributes["head_xy_base_m"], dtype=float
                    )
                    detected_axis = np.asarray(
                        detected_attributes["tip_xy_base_m"], dtype=float
                    ) - np.asarray(
                        detected_attributes["head_xy_base_m"], dtype=float
                    )
                    previous_axis /= max(float(np.linalg.norm(previous_axis)), 1e-12)
                    detected_axis /= max(float(np.linalg.norm(detected_axis)), 1e-12)
                    if (
                        float(np.dot(previous_axis, detected_axis))
                        < self._minimum_screw_axis_dot
                    ):
                        continue
                distance = float(np.linalg.norm(position[:2] - previous[:2]))
                if distance <= self._match_distance_m:
                    compatible.append((distance, track_id))
            if compatible:
                _, track_id = min(compatible)
                unmatched_tracks.remove(track_id)
                _, _, previous_capture_time_s, previous_hits = self._tracks[track_id]
                # A policy may call observe() more than once for one frozen
                # camera frame.  Replaying that frame must not satisfy a
                # multi-frame confirmation gate.  Only a strictly newer
                # camera-frame timestamp contributes another observation.
                is_new_frame = capture_time_s > previous_capture_time_s + 1e-9
            else:
                track_id = f"vision-track-{self._next_id:04d}"
                self._next_id += 1
                previous_hits = 0
                is_new_frame = True
            hits = previous_hits + int(is_new_frame)
            attributes = dict(detected.attributes)
            attributes["tracking_method"] = "nearest_neighbour_no_motion_prediction_v1"
            attributes["consecutive_track_observations"] = hits
            attributes["confirmation_timestamp_source"] = (
                "camera_frame_capture_time_s"
            )
            rebound = replace(detected, track_id=track_id, attributes=attributes)
            self._tracks[track_id] = (rebound, position, capture_time_s, hits)
            if (
                detected.class_id == SCREW_CLASS_ID
                and hits < self._minimum_screw_observations
            ):
                pending_attributes = dict(attributes)
                pending_attributes.update(
                    {
                        "non_executable": True,
                        "candidate_class": SCREW_CLASS_ID,
                        "ambiguity_reason": "temporal_confirmation_pending",
                        "required_track_observations": (
                            self._minimum_screw_observations
                        ),
                    }
                )
                assigned.append(
                    replace(
                        rebound,
                        class_id="unknown_obstacle",
                        sort_label=None,
                        confidence=min(rebound.confidence, 0.49),
                        pose_quality=0.0,
                        visibility=None,
                        attributes=pending_attributes,
                    )
                )
            else:
                assigned.append(rebound)
        for track_id in unmatched_tracks:
            previous_object, previous, previous_time_s, _ = self._tracks[track_id]
            self._tracks[track_id] = (
                previous_object,
                previous,
                previous_time_s,
                0,
            )
        return tuple(assigned)


class CubeScrewHybridVisionProvider:
    """Merge the unchanged cube baseline with conservative screw geometry."""

    PROVIDER_NAME = "hybrid-hsv-cubes-screw-silhouette"
    PROVIDER_VERSION = "1"

    def __init__(
        self,
        *,
        config: "SortConfig",
        camera_projectors: Mapping[str, Any],
        calibration_ids: Mapping[str, str],
        camera_id: str = "table",
        expected_cube_edge_m: float | None = None,
        detector: Any | None = None,
        screw_config: ScrewDetectorConfig | None = None,
        maximum_frame_age_s: float = 0.15,
        profile_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config.validate()
        self.camera_id = str(camera_id).strip()
        if self.camera_id not in camera_projectors:
            raise ValueError(f"missing projector for camera {self.camera_id}")
        if self.camera_id not in calibration_ids:
            raise ValueError(f"missing calibration id for camera {self.camera_id}")
        self._projector = camera_projectors[self.camera_id]
        self._calibration_id = str(calibration_ids[self.camera_id]).strip()
        self._screw_config = screw_config or ScrewDetectorConfig()
        self._screw_config.validate()
        self._maximum_frame_age_s = _finite(
            maximum_frame_age_s, "maximum_frame_age_s"
        )
        if self._maximum_frame_age_s < 0.0:
            raise ValueError("maximum_frame_age_s must be non-negative")
        self._profile_provenance = dict(
            profile_provenance
            or {
                "profile_id": "direct-constructor",
                "profile_path": None,
                "profile_sha256": None,
            }
        )
        self._cube_provider = ColourCubeVisionProvider(
            config=self.config,
            camera_projectors=camera_projectors,
            calibration_ids=calibration_ids,
            camera_id=self.camera_id,
            expected_cube_edge_m=expected_cube_edge_m,
            detector=detector,
        )
        self._tracker = _NearestNeighbourTracker(
            match_distance_m=self._screw_config.track_match_distance_m,
            ttl_s=self._screw_config.track_ttl_s,
            minimum_screw_observations=(
                self._screw_config.minimum_track_observations
            ),
            maximum_screw_length_change_m=(
                self._screw_config.maximum_track_length_change_m
            ),
            maximum_screw_width_change_m=(
                self._screw_config.maximum_track_width_change_m
            ),
            maximum_screw_axis_change_deg=(
                self._screw_config.maximum_track_axis_change_deg
            ),
            maximum_screw_head_ratio_change=(
                self._screw_config.maximum_track_head_ratio_change
            ),
        )

    def observe(self, request: PerceptionRequest) -> SceneObservation:
        frame = request.frame(self.camera_id)
        if frame.rgb is None:
            raise VisionError(f"camera {self.camera_id} has no RGB frame")
        if frame.calibration_id != self._calibration_id:
            raise VisionError(
                f"camera {self.camera_id} calibration mismatch: expected "
                f"{self._calibration_id!r}, received {frame.calibration_id!r}"
            )
        frame_age_s = request.capture_time_s - frame.capture_time_s
        if frame_age_s < -1e-6:
            raise VisionError(
                f"camera {self.camera_id} frame is newer than the scene reference"
            )
        if frame_age_s > self._maximum_frame_age_s:
            raise VisionError(
                f"camera {self.camera_id} frame is stale by {frame_age_s:.6f} s; "
                f"maximum is {self._maximum_frame_age_s:.6f} s"
            )
        cube_scene = self._cube_provider.observe(request)
        # Preserve the legacy detector's broad raw view. The hardware adapter
        # needs outside-pickup colour detections to verify virtual-table/bin
        # placements; eligibility belongs downstream, not in this provider.
        x_bounds = self.config.task.pickup_region_base_m["x"]
        y_bounds = self.config.task.pickup_region_base_m["y"]
        cubes: list[DetectedObject] = []
        for cube in cube_scene.objects:
            x_m, y_m = cube.pose.position_base_m[:2]
            in_pickup_region = (
                x_bounds[0] <= x_m <= x_bounds[1]
                and y_bounds[0] <= y_m <= y_bounds[1]
            )
            attributes = dict(cube.attributes)
            attributes.update(
                {
                    "symmetry_id": "upright_square_yaw_modulo_90_deg",
                    "detector_family": "legacy_hsv_known_size_geometry_v1",
                    "object_catalog_entry": "docs/OBJECT_CATALOG.md#cube",
                    "integration_profile": self._profile_provenance,
                    "pose_constraint": "upright_on_table",
                    "occlusion_status": "not_estimated",
                    "uncertainty_scope": (
                        "colour-mask shape only; absolute calibration error unverified"
                    ),
                    "absolute_base_accuracy_verified": False,
                    "in_pickup_region": in_pickup_region,
                    "source_frame_capture_time_s": frame.capture_time_s,
                    "source_frame_age_s": max(0.0, frame_age_s),
                    "temporal_alignment": (
                        "zero_order_hold_no_motion_compensation"
                        if frame_age_s > 1e-6
                        else "scene_reference_timestamp"
                    ),
                }
            )
            cubes.append(
                replace(
                    cube,
                    pose_quality=None,
                    attributes=attributes,
                )
            )
        screws = detect_lying_screws(
            np.asarray(frame.rgb),
            self._projector,
            self.config,
            self._screw_config,
            scene_id=request.scene_id,
            capture_time_s=request.capture_time_s,
            camera_id=self.camera_id,
        )
        timestamped_screws: list[DetectedObject] = []
        for screw in screws:
            attributes = dict(screw.attributes)
            attributes.update(
                {
                    "object_catalog_entry": (
                        "docs/OBJECT_CATALOG.md#screw_iso10642_m6x30-provisional"
                    ),
                    "catalog_geometry_sha256": SCREW_CAD_SHA256,
                    "integration_profile": self._profile_provenance,
                    "source_frame_capture_time_s": frame.capture_time_s,
                    "source_frame_age_s": max(0.0, frame_age_s),
                    "temporal_alignment": (
                        "zero_order_hold_no_motion_compensation"
                        if frame_age_s > 1e-6
                        else "scene_reference_timestamp"
                    ),
                }
            )
            timestamped_screws.append(replace(screw, attributes=attributes))
        tracked = self._tracker.assign(
            (*cubes, *timestamped_screws),
            # Confirmation must count distinct source frames, not repeated
            # observe() calls or scene-reference timestamps.
            capture_time_s=frame.capture_time_s,
        )
        return SceneObservation(
            scene_id=request.scene_id,
            capture_time_s=request.capture_time_s,
            objects=tracked,
            provider_name=self.PROVIDER_NAME,
            provider_version=self.PROVIDER_VERSION,
            calibration_ids=(self._calibration_id,),
        )


def create(context: "VisionFactoryContext") -> "VisionProvider":
    """Build the hybrid provider from an explicit team integration profile."""

    options = dict(context.options)
    camera_id = str(options.pop("camera_id", "table"))
    maximum_frame_age_s = options.pop("maximum_frame_age_s", 0.15)
    screw_raw = options.pop("screw", {})
    if options:
        raise ValueError(f"unknown hybrid provider options: {sorted(options)}")
    if not isinstance(screw_raw, Mapping):
        raise ValueError("vision.options.screw must be an object")
    return CubeScrewHybridVisionProvider(
        config=context.config,
        camera_projectors=context.camera_projectors,
        calibration_ids=context.calibration_ids,
        camera_id=camera_id,
        expected_cube_edge_m=context.expected_cube_edge_m,
        detector=context.detector,
        screw_config=ScrewDetectorConfig.from_options(screw_raw),
        maximum_frame_age_s=maximum_frame_age_s,
        profile_provenance={
            "profile_id": context.profile_id,
            "profile_path": context.profile_path,
            "profile_sha256": context.profile_sha256,
        },
    )


__all__ = [
    "CubeScrewHybridVisionProvider",
    "PLUGIN_API_VERSION",
    "SCREW_CLASS_ID",
    "SCREW_CAD_SHA256",
    "SCREW_DIMENSIONS_M",
    "ScrewDetectorConfig",
    "create",
    "detect_lying_screws",
]
