"""Model-independent utilities for the bounded FastSAM mask experiment.

This module intentionally does not import Ultralytics.  Tiling, mask-to-polygon
conversion, overlap suppression, and proposal-coverage scoring therefore run in
the lightweight repository environment.  The output remains image-space
evidence; it is not a ``VisionProvider`` and never fabricates robot coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class Tile:
    """One half-open crop in full-image pixel coordinates."""

    tile_id: str
    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        values = tuple(_integer(getattr(self, name), name) for name in ("x0", "y0", "x1", "y1"))
        if values[0] < 0 or values[1] < 0 or values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError("tile must be a non-empty positive rectangle")
        if not isinstance(self.tile_id, str) or not self.tile_id.strip():
            raise ValueError("tile_id must be non-empty")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(frozen=True)
class MaskProposal:
    """One class-agnostic instance proposal mapped into the full image."""

    proposal_id: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    polygon_xy: tuple[tuple[float, float], ...]
    centroid_uv: tuple[float, float]
    mask_area_px: int
    tile_id: str

    def __post_init__(self) -> None:
        confidence = _finite(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        bbox = tuple(_finite(value, "bbox_xyxy") for value in self.bbox_xyxy)
        if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("bbox_xyxy must be non-empty")
        polygon = tuple(tuple(_finite(value, "polygon_xy") for value in point) for point in self.polygon_xy)
        if len(polygon) < 3 or any(len(point) != 2 for point in polygon):
            raise ValueError("polygon_xy must contain at least three XY points")
        centroid = tuple(_finite(value, "centroid_uv") for value in self.centroid_uv)
        if len(centroid) != 2:
            raise ValueError("centroid_uv must contain U,V")
        area = _integer(self.mask_area_px, "mask_area_px")
        if area <= 0:
            raise ValueError("mask_area_px must be positive")
        if not isinstance(self.proposal_id, str) or not self.proposal_id.strip():
            raise ValueError("proposal_id must be non-empty")
        if not isinstance(self.tile_id, str) or not self.tile_id.strip():
            raise ValueError("tile_id must be non-empty")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "bbox_xyxy", bbox)
        object.__setattr__(self, "polygon_xy", polygon)
        object.__setattr__(self, "centroid_uv", centroid)
        object.__setattr__(self, "mask_area_px", area)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
            "polygon_xy": [list(point) for point in self.polygon_xy],
            "centroid_uv": list(self.centroid_uv),
            "mask_area_px": self.mask_area_px,
            "tile_id": self.tile_id,
        }


def _starts(lower: int, upper: int, window: int, overlap: float) -> tuple[int, ...]:
    extent = upper - lower
    if extent <= window:
        return (lower,)
    stride = max(1, int(round(window * (1.0 - overlap))))
    starts = list(range(lower, upper - window + 1, stride))
    final = upper - window
    if starts[-1] != final:
        starts.append(final)
    return tuple(starts)


def generate_tiles(
    *,
    image_size: Sequence[int],
    roi_xyxy: Sequence[int],
    tile_size: Sequence[int],
    overlap_fraction: float,
) -> tuple[Tile, ...]:
    """Cover an ROI completely while retaining native pixel coordinates."""

    image_width, image_height = tuple(_integer(value, "image_size") for value in image_size)
    x0, y0, x1, y1 = tuple(_integer(value, "roi_xyxy") for value in roi_xyxy)
    tile_width, tile_height = tuple(_integer(value, "tile_size") for value in tile_size)
    overlap = _finite(overlap_fraction, "overlap_fraction")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_size must be positive")
    if not (0 <= x0 < x1 <= image_width and 0 <= y0 < y1 <= image_height):
        raise ValueError("roi_xyxy must lie inside the image")
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("tile_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap_fraction must be in [0, 1)")
    tile_width = min(tile_width, x1 - x0)
    tile_height = min(tile_height, y1 - y0)
    return tuple(
        Tile(
            tile_id=f"tile-r{row:02d}-c{column:02d}",
            x0=tile_x,
            y0=tile_y,
            x1=tile_x + tile_width,
            y1=tile_y + tile_height,
        )
        for row, tile_y in enumerate(_starts(y0, y1, tile_height, overlap))
        for column, tile_x in enumerate(_starts(x0, x1, tile_width, overlap))
    )


def extract_proposals(
    *,
    masks: np.ndarray,
    confidences: Sequence[float],
    tile: Tile,
    minimum_mask_area_px: int,
) -> tuple[MaskProposal, ...]:
    """Convert binary model masks into compact full-frame polygons."""

    mask_stack = np.asarray(masks)
    if mask_stack.ndim != 3 or mask_stack.shape[1:] != (tile.height, tile.width):
        raise ValueError("masks must be NxHxW in the tile's original resolution")
    scores = tuple(_finite(value, "confidence") for value in confidences)
    if len(scores) != len(mask_stack):
        raise ValueError("masks and confidences must have equal length")
    minimum_area = _integer(minimum_mask_area_px, "minimum_mask_area_px")
    if minimum_area <= 0:
        raise ValueError("minimum_mask_area_px must be positive")
    proposals = []
    for index, (raw_mask, confidence) in enumerate(zip(mask_stack, scores)):
        binary = (raw_mask > 0.5).astype(np.uint8)
        if int(np.count_nonzero(binary)) < minimum_area:
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            continue
        # Keep the serialized polygon, area, centroid, and bounding box about
        # the same connected component.  FastSAM can occasionally leave tiny
        # disconnected islands in one raw mask; mixing those islands into the
        # box while drawing only the largest contour would be misleading.
        component = np.zeros_like(binary)
        cv2.drawContours(component, [contour], -1, color=1, thickness=cv2.FILLED)
        area = int(np.count_nonzero(component))
        if area < minimum_area:
            continue
        epsilon = max(0.5, 0.005 * cv2.arcLength(contour, closed=True))
        polygon = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2).astype(float)
        if len(polygon) < 3:
            continue
        rows, columns = np.nonzero(component)
        x0, x1 = float(columns.min()), float(columns.max() + 1)
        y0, y1 = float(rows.min()), float(rows.max() + 1)
        offset = np.array((tile.x0, tile.y0), dtype=float)
        full_polygon = polygon + offset
        proposals.append(
            MaskProposal(
                proposal_id=f"{tile.tile_id}:mask-{index:04d}",
                confidence=confidence,
                bbox_xyxy=(x0 + tile.x0, y0 + tile.y0, x1 + tile.x0, y1 + tile.y0),
                polygon_xy=tuple(tuple(float(value) for value in point) for point in full_polygon),
                centroid_uv=(float(columns.mean() + tile.x0), float(rows.mean() + tile.y0)),
                mask_area_px=area,
                tile_id=tile.tile_id,
            )
        )
    return tuple(proposals)


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    a = tuple(_finite(value, "first") for value in first)
    b = tuple(_finite(value, "second") for value in second)
    if len(a) != 4 or len(b) != 4:
        raise ValueError("boxes must contain four values")
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = ((a[2] - a[0]) * (a[3] - a[1])) + ((b[2] - b[0]) * (b[3] - b[1])) - intersection
    return 0.0 if union <= 0.0 else intersection / union


def merge_proposals(
    proposals: Iterable[MaskProposal], *, iou_threshold: float
) -> tuple[MaskProposal, ...]:
    """Suppress duplicate instances created by overlapping native-image tiles."""

    threshold = _finite(iou_threshold, "iou_threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    ordered = sorted(
        tuple(proposals),
        key=lambda item: (-item.confidence, -item.mask_area_px, item.proposal_id),
    )
    kept: list[MaskProposal] = []
    for candidate in ordered:
        if any(
            previous.tile_id != candidate.tile_id
            and box_iou(previous.bbox_xyxy, candidate.bbox_xyxy) > threshold
            for previous in kept
        ):
            continue
        kept.append(candidate)
    return tuple(sorted(kept, key=lambda item: item.proposal_id))


def evaluate_proposal_coverage(
    references: Sequence[Mapping[str, Any]],
    proposals: Sequence[MaskProposal],
    *,
    minimum_bbox_iou: float,
) -> dict[str, Any]:
    """Measure strict box coverage without calling unmatched proposals false positives.

    A nonzero IoU is intentionally insufficient: a scene-sized mask would then
    appear to cover every tiny part.  The explicit threshold makes the
    exploratory metric auditable until pixel masks are manually annotated.
    """

    threshold = _finite(minimum_bbox_iou, "minimum_bbox_iou")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("minimum_bbox_iou must be in (0, 1]")

    edges: dict[int, list[tuple[float, str, int]]] = {}
    for reference_index, reference in enumerate(references):
        reference_box = tuple(float(value) for value in reference["bbox_xyxy"])
        for proposal_index, proposal in enumerate(proposals):
            overlap = box_iou(reference_box, proposal.bbox_xyxy)
            if overlap >= threshold:
                edges.setdefault(reference_index, []).append(
                    (-overlap, proposal.proposal_id, proposal_index)
                )
    for options in edges.values():
        options.sort()

    # Deterministic augmenting-path matching maximizes covered references.
    # Preference order favors higher IoU, but coverage is never sacrificed to
    # one locally attractive pair in an overlapping/touching scene.
    proposal_owner: dict[int, int] = {}

    def assign(reference_index: int, visited: set[int]) -> bool:
        for _, _, proposal_index in edges.get(reference_index, ()):
            if proposal_index in visited:
                continue
            visited.add(proposal_index)
            previous_owner = proposal_owner.get(proposal_index)
            if previous_owner is None or assign(previous_owner, visited):
                proposal_owner[proposal_index] = reference_index
                return True
        return False

    reference_order = sorted(
        range(len(references)),
        key=lambda index: (len(edges.get(index, ())), str(references[index]["instance_id"])),
    )
    for reference_index in reference_order:
        assign(reference_index, set())
    matches = []
    for proposal_index, reference_index in sorted(
        proposal_owner.items(), key=lambda item: str(references[item[1]]["instance_id"])
    ):
        overlap = box_iou(
            references[reference_index]["bbox_xyxy"], proposals[proposal_index].bbox_xyxy
        )
        matches.append(
            {
                "instance_id": str(references[reference_index]["instance_id"]),
                "class_id": str(references[reference_index].get("class_id", "unlabeled")),
                "proposal_id": proposals[proposal_index].proposal_id,
                "bbox_iou": overlap,
            }
        )
    matched_references = set(proposal_owner.values())
    matched_proposals = set(proposal_owner)
    by_class: dict[str, dict[str, int]] = {}
    for index, reference in enumerate(references):
        class_id = str(reference.get("class_id", "unlabeled"))
        summary = by_class.setdefault(
            class_id, {"reference_instances": 0, "covered_references": 0, "dropouts": 0}
        )
        summary["reference_instances"] += 1
        if index in matched_references:
            summary["covered_references"] += 1
        else:
            summary["dropouts"] += 1
    return {
        "reference_instances": len(references),
        "covered_references": len(matched_references),
        "dropouts": len(references) - len(matched_references),
        "proposals": len(proposals),
        "unmatched_proposals": len(proposals) - len(matched_proposals),
        "minimum_bbox_iou": threshold,
        "matches": matches,
        "by_class": dict(sorted(by_class.items())),
        "interpretation": (
            "Class-agnostic unmatched proposals are not false positives; they may be "
            "unsupported objects or duplicate/over-segmented regions."
        ),
    }
