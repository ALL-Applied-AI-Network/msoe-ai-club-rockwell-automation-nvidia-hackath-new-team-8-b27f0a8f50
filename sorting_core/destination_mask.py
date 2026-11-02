"""Camera-space suppression for configured physical destination bins.

The three colour-sort bins have fixed robot-base positions and known outer
dimensions.  This module projects their complete 3-D cuboids into one fixed
camera and replaces only those pixels with nearby image texture before object
inference.  It deliberately does not infer bin locations from simulator truth
or from an image classifier.

The caller decides *when* the suppression is valid.  In particular, a
virtual-table destination has no physical bin to remove, and post-placement
verification may need the unmodified image.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .config import SortConfig


DEFAULT_BUCKET_MASK_MARGIN_PX = 8
DEFAULT_BUCKET_INPAINT_RADIUS_PX = 5.0


class DestinationMaskError(ValueError):
    """Raised when a configured destination cannot be projected safely."""


@dataclass(frozen=True)
class DestinationMaskResult:
    """One masked RGB copy plus auditable projected-mask evidence."""

    rgb: np.ndarray
    mask: np.ndarray
    polygons_uv: Mapping[str, tuple[tuple[int, int], ...]]
    masked_pixel_count: int
    margin_px: int

    def __post_init__(self) -> None:
        image = np.asarray(self.rgb)
        mask = np.asarray(self.mask)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise DestinationMaskError("masked rgb must be HxWx3 uint8")
        if mask.shape != image.shape[:2] or mask.dtype != np.uint8:
            raise DestinationMaskError("destination mask must be HxW uint8")
        count = int(np.count_nonzero(mask))
        if count != int(self.masked_pixel_count):
            raise DestinationMaskError("masked pixel count does not match mask")
        if isinstance(self.margin_px, bool) or int(self.margin_px) != self.margin_px:
            raise DestinationMaskError("mask margin must be an integer")
        if self.margin_px < 0:
            raise DestinationMaskError("mask margin must be non-negative")
        polygons = {
            str(colour): tuple((int(u), int(v)) for u, v in polygon)
            for colour, polygon in self.polygons_uv.items()
        }
        object.__setattr__(self, "polygons_uv", MappingProxyType(polygons))

    def metadata(self) -> dict[str, Any]:
        """Return JSON-compatible evidence for runtime logs."""

        return {
            "applied": True,
            "method": "configured_3d_bin_cuboid_projection_inpaint_v1",
            "masked_pixel_count": self.masked_pixel_count,
            "margin_px": self.margin_px,
            "polygons_uv": {
                colour: [list(point) for point in polygon]
                for colour, polygon in self.polygons_uv.items()
            },
        }


def _project_point(
    projector: Any,
    point_base_m: Sequence[float],
) -> np.ndarray:
    """Project one base-frame point using a camera object or callable."""

    if hasattr(projector, "project"):
        value = projector.project(point_base_m)
    elif callable(projector):
        value = projector(point_base_m)
    else:
        raise DestinationMaskError(
            "base-to-pixel projector must be callable or expose project()"
        )
    if value is None:
        raise DestinationMaskError("configured bin corner is behind the camera")
    uv = np.asarray(value, dtype=float)
    if uv.shape != (2,) or not np.all(np.isfinite(uv)):
        raise DestinationMaskError(
            "base-to-pixel projection must return two finite coordinates"
        )
    return uv


def _bin_corners_base_m(
    centre_xy_m: Sequence[float],
    config: SortConfig,
) -> tuple[tuple[float, float, float], ...]:
    try:
        centre_x, centre_y = (float(value) for value in centre_xy_m)
    except (TypeError, ValueError) as exc:
        raise DestinationMaskError(
            "configured bin centre must contain two numbers"
        ) from exc
    if not math.isfinite(centre_x) or not math.isfinite(centre_y):
        raise DestinationMaskError("configured bin centre must be finite")
    half_outer = float(config.bin.outer_m) / 2.0
    lower_z = float(config.frames.surface_base_z_m)
    upper_z = lower_z + float(config.bin.height_m)
    return tuple(
        (centre_x + dx, centre_y + dy, z_m)
        for z_m in (lower_z, upper_z)
        for dx in (-half_outer, half_outer)
        for dy in (-half_outer, half_outer)
    )


def build_configured_bucket_pixel_mask(
    image_shape: Sequence[int],
    projector: Any,
    config: SortConfig,
    *,
    margin_px: int = DEFAULT_BUCKET_MASK_MARGIN_PX,
) -> tuple[np.ndarray, Mapping[str, tuple[tuple[int, int], ...]]]:
    """Project every configured bin cuboid into a binary image mask.

    ``projector`` consumes one ``(x, y, z)`` point in robot-base metres and
    returns continuous ``(u, v)`` image coordinates.  The convex hull of all
    eight cuboid corners covers the visible rim, walls, and pocket for any
    fixed view.  A small pixel margin absorbs raster antialiasing and ordinary
    placement tolerance without hard-coding camera-specific rectangles.
    """

    try:
        height, width = (int(image_shape[0]), int(image_shape[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise DestinationMaskError(
            "image shape must contain positive height and width"
        ) from exc
    if height < 2 or width < 2:
        raise DestinationMaskError("image shape must contain positive dimensions")
    if isinstance(margin_px, bool) or int(margin_px) != margin_px or margin_px < 0:
        raise DestinationMaskError("margin_px must be a non-negative integer")

    checked_config = config.validate()
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons: dict[str, tuple[tuple[int, int], ...]] = {}
    for colour in checked_config.task.colour_order:
        corners = _bin_corners_base_m(
            checked_config.task.bin_centres_base_m[colour], checked_config
        )
        projected = np.asarray(
            [_project_point(projector, point) for point in corners],
            dtype=np.float32,
        )
        hull = cv2.convexHull(projected).reshape(-1, 2)
        if len(hull) < 3:
            raise DestinationMaskError(
                f"configured {colour} bin projection is degenerate"
            )
        integer_hull = np.rint(hull).astype(np.int32)
        polygons[colour] = tuple(
            (int(point[0]), int(point[1])) for point in integer_hull
        )
        cv2.fillConvexPoly(mask, integer_hull, 255)

    if margin_px:
        diameter = 2 * int(margin_px) + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (diameter, diameter)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask, MappingProxyType(polygons)


def ignore_configured_bucket_pixels(
    rgb: np.ndarray,
    projector: Any,
    config: SortConfig,
    *,
    margin_px: int = DEFAULT_BUCKET_MASK_MARGIN_PX,
    inpaint_radius_px: float = DEFAULT_BUCKET_INPAINT_RADIUS_PX,
) -> DestinationMaskResult:
    """Return an RGB copy in which configured physical bins are neutralized.

    OpenCV inpainting uses only pixels outside the projected/dilated bin mask,
    so the detector cannot consume the bucket colours, dark pocket, labels, or
    rim details.  Pixels outside the mask are restored byte-for-byte after
    inpainting, and the caller's source array is never mutated.
    """

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise DestinationMaskError("rgb must be an HxWx3 uint8 image")
    radius = float(inpaint_radius_px)
    if not math.isfinite(radius) or radius <= 0.0:
        raise DestinationMaskError("inpaint_radius_px must be positive and finite")
    mask, polygons = build_configured_bucket_pixel_mask(
        image.shape[:2], projector, config, margin_px=margin_px
    )
    if not np.any(mask):
        raise DestinationMaskError(
            "configured bin projections do not intersect the camera image"
        )
    # Inpaint each small destination region independently instead of running
    # the algorithm over the full 1920x1080 frame.  This keeps runtime bounded
    # by the three projected bins and leaves every other pixel byte-identical.
    masked = image.copy()
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    padding = int(math.ceil(radius)) + 2
    height, width = mask.shape
    for contour in contours:
        x, y, region_width, region_height = cv2.boundingRect(contour)
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(width, x + region_width + padding)
        y1 = min(height, y + region_height + padding)
        region_mask = mask[y0:y1, x0:x1]
        repaired = cv2.inpaint(
            image[y0:y1, x0:x1],
            region_mask,
            radius,
            cv2.INPAINT_TELEA,
        )
        selected = region_mask != 0
        masked_region = masked[y0:y1, x0:x1]
        masked_region[selected] = repaired[selected]
    return DestinationMaskResult(
        rgb=masked,
        mask=mask,
        polygons_uv=polygons,
        masked_pixel_count=int(np.count_nonzero(mask)),
        margin_px=int(margin_px),
    )


__all__ = [
    "DEFAULT_BUCKET_INPAINT_RADIUS_PX",
    "DEFAULT_BUCKET_MASK_MARGIN_PX",
    "DestinationMaskError",
    "DestinationMaskResult",
    "build_configured_bucket_pixel_mask",
    "ignore_configured_bucket_pixels",
]
