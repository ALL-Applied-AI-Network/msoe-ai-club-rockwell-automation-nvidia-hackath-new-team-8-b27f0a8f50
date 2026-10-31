"""Model-independent utilities for the bounded YOLOE segmentation experiment.

This module intentionally contains no Ultralytics import.  Coordinate tiling,
prediction extraction, overlap merging, and annotation scoring can therefore be
unit-tested in the lightweight repository environment.  The heavyweight model
is loaded lazily by :mod:`yoloe_visual_prompt_experiment`.

The output of this experiment is image-space evidence.  It is *not* a
``VisionProvider`` and does not fabricate robot-base position or orientation.
Those require the reviewed calibration/geometry stage and separate pose tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class ExperimentError(RuntimeError):
    """Raised when an experiment input or model result is unsafe to compare."""


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
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result != value:
        raise ValueError(f"{name} must be an integer")
    return result


@dataclass(frozen=True)
class Tile:
    """One half-open native-image crop in full-frame coordinates."""

    tile_id: str
    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        for name in ("x0", "y0", "x1", "y1"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.x0 < 0 or self.y0 < 0 or self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("tile must be a non-empty positive half-open rectangle")
        if not isinstance(self.tile_id, str) or not self.tile_id.strip():
            raise ValueError("tile_id must be a non-empty string")

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
class ImagePrediction:
    """One model instance mapped back into the uncropped source frame."""

    class_id: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    polygon_xy: tuple[tuple[float, float], ...]
    tile_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.class_id, str) or not self.class_id.strip():
            raise ValueError("class_id must be a non-empty string")
        confidence = _finite(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        bbox = tuple(_finite(value, "bbox_xyxy") for value in self.bbox_xyxy)
        if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("bbox_xyxy must be a non-empty (x0, y0, x1, y1)")
        polygon = tuple(
            tuple(_finite(value, "polygon_xy") for value in point)
            for point in self.polygon_xy
        )
        if len(polygon) < 3 or any(len(point) != 2 for point in polygon):
            raise ValueError("polygon_xy must contain at least three XY points")
        if not isinstance(self.tile_id, str) or not self.tile_id.strip():
            raise ValueError("tile_id must be a non-empty string")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "bbox_xyxy", bbox)
        object.__setattr__(self, "polygon_xy", polygon)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
            "polygon_xy": [list(point) for point in self.polygon_xy],
            "tile_id": self.tile_id,
        }


def _starts(lower: int, upper: int, window: int, overlap_fraction: float) -> tuple[int, ...]:
    extent = upper - lower
    if extent <= window:
        return (lower,)
    stride = max(1, int(round(window * (1.0 - overlap_fraction))))
    values = list(range(lower, upper - window + 1, stride))
    final = upper - window
    if values[-1] != final:
        values.append(final)
    return tuple(values)


def generate_tiles(
    *,
    image_size: Sequence[int],
    roi_xyxy: Sequence[int],
    tile_size: Sequence[int],
    overlap_fraction: float,
) -> tuple[Tile, ...]:
    """Cover one native-resolution ROI without resizing or dropping its edges."""

    image_width, image_height = tuple(
        _integer(value, "image_size") for value in image_size
    )
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_size must be positive")
    x0, y0, x1, y1 = tuple(_integer(value, "roi_xyxy") for value in roi_xyxy)
    if not (0 <= x0 < x1 <= image_width and 0 <= y0 < y1 <= image_height):
        raise ValueError("roi_xyxy must be a non-empty rectangle inside the image")
    tile_width, tile_height = tuple(
        _integer(value, "tile_size") for value in tile_size
    )
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("tile_size must be positive")
    overlap = _finite(overlap_fraction, "overlap_fraction")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap_fraction must be in [0, 1)")
    effective_width = min(tile_width, x1 - x0)
    effective_height = min(tile_height, y1 - y0)
    x_starts = _starts(x0, x1, effective_width, overlap)
    y_starts = _starts(y0, y1, effective_height, overlap)
    return tuple(
        Tile(
            tile_id=f"tile-r{row:02d}-c{column:02d}",
            x0=tile_x,
            y0=tile_y,
            x1=tile_x + effective_width,
            y1=tile_y + effective_height,
        )
        for row, tile_y in enumerate(y_starts)
        for column, tile_x in enumerate(x_starts)
    )


def box_iou(
    first: Sequence[float], second: Sequence[float]
) -> float:
    a = tuple(_finite(value, "first") for value in first)
    b = tuple(_finite(value, "second") for value in second)
    if len(a) != 4 or len(b) != 4:
        raise ValueError("boxes must have four coordinates")
    intersection_width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    intersection_height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def merge_tile_predictions(
    predictions: Iterable[ImagePrediction], *, iou_threshold: float
) -> tuple[ImagePrediction, ...]:
    """Class-aware deterministic box NMS for duplicate overlap-tile results."""

    threshold = _finite(iou_threshold, "iou_threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    ordered = sorted(
        tuple(predictions),
        key=lambda item: (
            -item.confidence,
            item.class_id,
            item.bbox_xyxy,
            item.tile_id,
        ),
    )
    kept: list[ImagePrediction] = []
    for candidate in ordered:
        if any(
            previous.class_id == candidate.class_id
            and previous.tile_id != candidate.tile_id
            and box_iou(previous.bbox_xyxy, candidate.bbox_xyxy) > threshold
            for previous in kept
        ):
            continue
        kept.append(candidate)
    return tuple(
        sorted(
            kept,
            key=lambda item: (
                item.class_id,
                item.bbox_xyxy[1],
                item.bbox_xyxy[0],
                -item.confidence,
            ),
        )
    )


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def extract_tile_predictions(
    result: Any,
    *,
    tile: Tile,
    class_ids: Sequence[str],
) -> tuple[ImagePrediction, ...]:
    """Validate one Ultralytics-like result and map it to full-frame pixels."""

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return ()
    masks = getattr(result, "masks", None)
    polygons = None if masks is None else getattr(masks, "xy", None)
    if polygons is None or len(polygons) != len(boxes):
        raise ExperimentError(
            "segmentation experiment returned boxes without one polygon per instance"
        )
    xyxy = _to_numpy(boxes.xyxy)
    confidence = _to_numpy(boxes.conf)
    class_index = _to_numpy(boxes.cls)
    if xyxy.shape != (len(boxes), 4):
        raise ExperimentError("model boxes.xyxy has an unexpected shape")
    if confidence.shape != (len(boxes),) or class_index.shape != (len(boxes),):
        raise ExperimentError("model confidence/class arrays have unexpected shapes")
    output: list[ImagePrediction] = []
    for index in range(len(boxes)):
        model_class = int(class_index[index])
        if model_class < 0 or model_class >= len(class_ids):
            raise ExperimentError(f"model returned unknown prompt class {model_class}")
        local_box = np.asarray(xyxy[index], dtype=float)
        if not np.all(np.isfinite(local_box)):
            raise ExperimentError("model returned a non-finite box")
        if (
            local_box[0] < -1e-3
            or local_box[1] < -1e-3
            or local_box[2] > tile.width + 1e-3
            or local_box[3] > tile.height + 1e-3
        ):
            raise ExperimentError("model returned a box outside its source tile")
        polygon = np.asarray(polygons[index], dtype=float)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ExperimentError("model returned an invalid segmentation polygon")
        if not np.all(np.isfinite(polygon)):
            raise ExperimentError("model returned a non-finite segmentation polygon")
        if (
            np.min(polygon[:, 0]) < -1e-3
            or np.min(polygon[:, 1]) < -1e-3
            or np.max(polygon[:, 0]) > tile.width + 1e-3
            or np.max(polygon[:, 1]) > tile.height + 1e-3
        ):
            raise ExperimentError("model returned a polygon outside its source tile")
        offset = np.array((tile.x0, tile.y0), dtype=float)
        full_box = local_box + np.array(
            (tile.x0, tile.y0, tile.x0, tile.y0), dtype=float
        )
        full_polygon = polygon + offset
        output.append(
            ImagePrediction(
                class_id=class_ids[model_class],
                confidence=float(confidence[index]),
                bbox_xyxy=tuple(float(value) for value in full_box),
                polygon_xy=tuple(
                    (float(point[0]), float(point[1])) for point in full_polygon
                ),
                tile_id=tile.tile_id,
            )
        )
    return tuple(output)


def evaluate_boxes(
    predictions: Sequence[ImagePrediction],
    references: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float,
) -> dict[str, Any]:
    """Greedily score reviewed boxes; never infer absent reference instances."""

    threshold = _finite(iou_threshold, "iou_threshold")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    parsed_references: list[tuple[str, tuple[float, float, float, float]]] = []
    for reference in references:
        class_id = reference.get("class_id")
        if not isinstance(class_id, str) or not class_id.strip():
            raise ValueError("reference class_id must be a non-empty string")
        box = tuple(_finite(value, "reference bbox") for value in reference.get("bbox_xyxy", ()))
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("reference bbox_xyxy must be a non-empty box")
        parsed_references.append((class_id, box))
    classes = sorted(
        {prediction.class_id for prediction in predictions}
        | {item[0] for item in parsed_references}
    )
    rows = []
    for class_id in classes:
        class_predictions = sorted(
            (item for item in predictions if item.class_id == class_id),
            key=lambda item: (-item.confidence, item.bbox_xyxy),
        )
        class_references = [box for label, box in parsed_references if label == class_id]
        unmatched = set(range(len(class_references)))
        true_positives = 0
        false_positives = 0
        matched_ious: list[float] = []
        for prediction in class_predictions:
            candidates = sorted(
                (
                    (box_iou(prediction.bbox_xyxy, class_references[index]), index)
                    for index in unmatched
                ),
                reverse=True,
            )
            if candidates and candidates[0][0] >= threshold:
                overlap, reference_index = candidates[0]
                unmatched.remove(reference_index)
                true_positives += 1
                matched_ious.append(overlap)
            else:
                false_positives += 1
        dropouts = len(unmatched)
        precision = (
            None
            if true_positives + false_positives == 0
            else true_positives / (true_positives + false_positives)
        )
        recall = (
            None
            if len(class_references) == 0
            else true_positives / len(class_references)
        )
        rows.append(
            {
                "class_id": class_id,
                "reference_instances": len(class_references),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "dropouts": dropouts,
                "precision": precision,
                "recall": recall,
                "matched_box_ious": matched_ious,
            }
        )
    return {
        "metric": "class_aware_box_iou_greedy_v1",
        "iou_threshold": threshold,
        "per_class": rows,
    }


__all__ = [
    "ExperimentError",
    "ImagePrediction",
    "Tile",
    "box_iou",
    "evaluate_boxes",
    "extract_tile_predictions",
    "generate_tiles",
    "merge_tile_predictions",
]
