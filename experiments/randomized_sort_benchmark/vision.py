"""Fair vision adapters for the randomized mixed-object benchmark.

The production YOLO provider is deliberately cube-only because its screw
performance was not validated on physical data.  This experiment module does
not weaken that boundary.  Instead, it offers an explicitly experimental
``yolo_assisted_hybrid`` condition in which YOLO boxes must agree one-to-one
with the calibrated hybrid geometry provider for both supported classes.
Metric position and orientation always come from calibrated geometry; a box
alone is never promoted into an executable pose.

Both benchmark modes consume the exact same :class:`PerceptionRequest`.  The
SHA-256 digest of the native RGB bytes is recorded in every report so the
orchestrator can prove matched input rather than relying on filenames.
Simulator truth is neither accepted nor read by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from sorting_core.perception import (
    DetectedObject,
    PerceptionRequest,
    SceneObservation,
    VisionProvider,
    validate_scene_for_request,
)
from sorting_core.vision import VisionError


DETERMINISTIC_HYBRID = "deterministic_hybrid"
YOLO_ASSISTED_HYBRID = "yolo_assisted_hybrid"
VISION_MODES = frozenset({DETERMINISTIC_HYBRID, YOLO_ASSISTED_HYBRID})

CUBE_CLASS_ID = "cube"
SCREW_CLASS_ID = "screw_iso10642_m6x30"
YOLO_CLASS_NAMES = {0: CUBE_CLASS_ID, 1: SCREW_CLASS_ID}
CLASS_TO_YOLO_ID = {value: key for key, value in YOLO_CLASS_NAMES.items()}

# Seed-55 is the current reviewed local artifact.  Its held-out physical set
# contained cubes, not screws; therefore screw results remain experimental and
# must be reported separately even when this gate accepts them.
PINNED_YOLO11N_SHA256 = (
    "010bb71b1b07fc77490a2bc2bd774102205640bdf9ba07d6fa8b3bd758788fbb"
)


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


def _integer_tuple(value: Any, length: int, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain {length} integers")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain {length} integers") from exc
    if len(values) != length or any(type(item) is not int for item in values):
        raise ValueError(f"{name} must contain {length} integers")
    return tuple(int(item) for item in values)


def _normalise_model_names(value: Any) -> dict[int, str]:
    if isinstance(value, Mapping):
        result: dict[int, str] = {}
        for raw_key, raw_name in value.items():
            if isinstance(raw_key, bool):
                raise ValueError("YOLO class-map keys must be integers")
            try:
                key = int(raw_key)
            except (TypeError, ValueError) as exc:
                raise ValueError("YOLO class-map keys must be integers") from exc
            result[key] = str(raw_name).strip()
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {index: str(name).strip() for index, name in enumerate(value)}
    raise ValueError("YOLO model must expose a class-name mapping")


def _as_numpy(value: Any) -> np.ndarray:
    detached = value.detach() if hasattr(value, "detach") else value
    host = detached.cpu() if hasattr(detached, "cpu") else detached
    return np.asarray(host)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class BenchmarkVisionConfig:
    """Versioned settings for one benchmark vision condition."""

    mode: str
    camera_id: str = "table"
    expected_image_size: tuple[int, int] = (1920, 1080)
    crop_xyxy: tuple[int, int, int, int] = (500, 400, 1500, 1050)
    inference_image_size: int = 960
    confidence_threshold: float = 0.6596596596596597
    iou_threshold: float = 0.7
    roi_margin_px: int = 4
    checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode).strip()
        if mode not in VISION_MODES:
            raise ValueError(f"mode must be one of {sorted(VISION_MODES)}")
        camera_id = str(self.camera_id).strip()
        if not camera_id:
            raise ValueError("camera_id must not be empty")
        expected_size = _integer_tuple(
            self.expected_image_size, 2, "expected_image_size"
        )
        if any(value <= 0 for value in expected_size):
            raise ValueError("expected_image_size values must be positive")
        crop = _integer_tuple(self.crop_xyxy, 4, "crop_xyxy")
        left, top, right, bottom = crop
        if not (0 <= left < right <= expected_size[0]):
            raise ValueError("crop_xyxy horizontal bounds are invalid")
        if not (0 <= top < bottom <= expected_size[1]):
            raise ValueError("crop_xyxy vertical bounds are invalid")
        if type(self.inference_image_size) is not int or self.inference_image_size <= 0:
            raise ValueError("inference_image_size must be a positive integer")
        confidence = _finite(self.confidence_threshold, "confidence_threshold")
        iou = _finite(self.iou_threshold, "iou_threshold")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if type(self.roi_margin_px) is not int or not 0 <= self.roi_margin_px <= 64:
            raise ValueError("roi_margin_px must be an integer in [0, 64]")
        digest = self.checkpoint_sha256
        if mode == YOLO_ASSISTED_HYBRID:
            if digest is None:
                raise ValueError("YOLO-assisted mode requires checkpoint_sha256")
            digest = str(digest).strip().lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        elif digest is not None:
            raise ValueError("deterministic mode must not specify a YOLO checkpoint")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(self, "expected_image_size", expected_size)
        object.__setattr__(self, "crop_xyxy", crop)
        object.__setattr__(self, "confidence_threshold", confidence)
        object.__setattr__(self, "iou_threshold", iou)
        object.__setattr__(self, "checkpoint_sha256", digest)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BenchmarkVisionConfig":
        """Build from a JSON profile while rejecting misspelled options."""

        if not isinstance(value, Mapping):
            raise ValueError("vision profile must be an object")
        options = dict(value)
        allowed = {
            "mode",
            "camera_id",
            "expected_image_size",
            "crop_xyxy",
            "inference_image_size",
            "confidence_threshold",
            "iou_threshold",
            "roi_margin_px",
            "checkpoint_sha256",
        }
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ValueError(f"unknown benchmark vision options: {unknown}")
        return cls(**options)


@dataclass(frozen=True)
class BenchmarkVisionProfile:
    """Small, strict envelope used by the checked-in JSON profiles."""

    profile_id: str
    vision: BenchmarkVisionConfig
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("benchmark vision profile schema_version must be 1")
        profile_id = str(self.profile_id).strip()
        if not profile_id:
            raise ValueError("profile_id must not be empty")
        if not isinstance(self.vision, BenchmarkVisionConfig):
            raise TypeError("vision must be a BenchmarkVisionConfig")
        object.__setattr__(self, "profile_id", profile_id)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BenchmarkVisionProfile":
        if not isinstance(value, Mapping):
            raise ValueError("benchmark vision profile must be an object")
        unknown = sorted(set(value) - {"schema_version", "profile_id", "vision"})
        if unknown:
            raise ValueError(f"unknown benchmark vision profile fields: {unknown}")
        return cls(
            schema_version=value.get("schema_version"),
            profile_id=value.get("profile_id", ""),
            vision=BenchmarkVisionConfig.from_mapping(value.get("vision", {})),
        )


def load_benchmark_vision_profile(path: Path | str) -> BenchmarkVisionProfile:
    """Load one strict JSON profile without resolving any model weights."""

    profile_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON benchmark vision profile: {profile_path}") from exc
    return BenchmarkVisionProfile.from_mapping(raw)


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    class_name: str
    confidence: float
    xyxy_full: tuple[float, float, float, float]


@dataclass(frozen=True)
class VisionGateReport:
    """Serializable evidence for one deterministic or YOLO-assisted call."""

    scene_id: str
    mode: str
    status: str
    input_rgb_sha256: str
    geometry_provider_name: str | None
    geometry_provider_version: str | None
    checkpoint_sha256: str | None
    geometry_latency_ms: float
    neural_latency_ms: float
    total_latency_ms: float
    geometry_supported_count: int
    geometry_unknown_count: int
    yolo_box_count: int
    matched_count: int
    failure_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "mode": self.mode,
            "status": self.status,
            "input_rgb_sha256": self.input_rgb_sha256,
            "geometry_provider_name": self.geometry_provider_name,
            "geometry_provider_version": self.geometry_provider_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "geometry_latency_ms": self.geometry_latency_ms,
            "neural_latency_ms": self.neural_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "geometry_supported_count": self.geometry_supported_count,
            "geometry_unknown_count": self.geometry_unknown_count,
            "yolo_box_count": self.yolo_box_count,
            "matched_count": self.matched_count,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class BenchmarkVisionResult:
    observation: SceneObservation
    report: VisionGateReport


class BenchmarkVisionAbstention(VisionError):
    """YOLO/geometry disagreement with a structured report for evaluation."""

    def __init__(self, message: str, report: VisionGateReport) -> None:
        super().__init__(message)
        self.report = report


class BenchmarkVisionProvider:
    """Instrument deterministic geometry and optional YOLO corroboration.

    The wrapped hybrid provider owns calibration, metric pose recovery, and
    temporal tracking.  In YOLO mode every supported geometry target and every
    neural box in the configured crop must participate in exactly one
    class-consistent match.  Any disagreement abstains instead of deleting an
    object or manufacturing clear space.
    """

    PROVIDER_VERSION = "1"

    def __init__(
        self,
        *,
        geometry_provider: VisionProvider,
        config: BenchmarkVisionConfig,
        yolo_model: Any | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not callable(getattr(geometry_provider, "observe", None)):
            raise ValueError("geometry_provider must expose observe()")
        if not isinstance(config, BenchmarkVisionConfig):
            raise TypeError("config must be a BenchmarkVisionConfig")
        if not callable(clock):
            raise ValueError("clock must be callable")
        if config.mode == YOLO_ASSISTED_HYBRID:
            if yolo_model is None or not callable(getattr(yolo_model, "predict", None)):
                raise ValueError("YOLO-assisted mode requires a model with predict()")
            names = _normalise_model_names(getattr(yolo_model, "names", None))
            if names != YOLO_CLASS_NAMES:
                raise ValueError(
                    f"YOLO class map must be exactly {YOLO_CLASS_NAMES!r}; "
                    f"received {names!r}"
                )
        elif yolo_model is not None:
            raise ValueError("deterministic mode must not receive a YOLO model")
        self._geometry_provider = geometry_provider
        self.config = config
        self._model = yolo_model
        self._clock = clock
        self.last_report: VisionGateReport | None = None

    @property
    def provider_name(self) -> str:
        if self.config.mode == DETERMINISTIC_HYBRID:
            return "benchmark-deterministic-hybrid-geometry"
        return "benchmark-yolo11n-assisted-hybrid-geometry"

    def observe(self, request: PerceptionRequest) -> SceneObservation:
        return self.observe_with_report(request).observation

    def observe_with_report(self, request: PerceptionRequest) -> BenchmarkVisionResult:
        if not isinstance(request, PerceptionRequest):
            raise TypeError("request must be a PerceptionRequest")
        frame = request.frame(self.config.camera_id)
        if frame.rgb is None:
            raise VisionError(f"camera {self.config.camera_id} has no RGB frame")
        image = np.asarray(frame.rgb)
        expected_width, expected_height = self.config.expected_image_size
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape != (expected_height, expected_width, 3)
        ):
            raise VisionError(
                "benchmark vision requires an uncropped native RGB uint8 image "
                f"with shape {(expected_height, expected_width, 3)!r}; "
                f"received shape={image.shape!r} dtype={image.dtype}"
            )
        input_digest = hashlib.sha256(
            np.ascontiguousarray(image).tobytes()
        ).hexdigest()
        total_started = self._clock()
        geometry_started = self._clock()
        geometry_scene = validate_scene_for_request(
            request, self._geometry_provider.observe(request)
        )
        geometry_latency_ms = (self._clock() - geometry_started) * 1000.0
        supported = tuple(
            detected
            for detected in geometry_scene.objects
            if detected.class_id in CLASS_TO_YOLO_ID
        )
        unknown_count = len(geometry_scene.objects) - len(supported)

        if self.config.mode == DETERMINISTIC_HYBRID:
            objects = tuple(
                self._annotate_geometry_only(detected, input_digest)
                for detected in geometry_scene.objects
            )
            observation = replace(
                geometry_scene,
                objects=objects,
                provider_name=self.provider_name,
                provider_version=(
                    f"{self.PROVIDER_VERSION}+geometry:"
                    f"{geometry_scene.provider_name}@{geometry_scene.provider_version}"
                ),
            )
            total_latency_ms = (self._clock() - total_started) * 1000.0
            report = VisionGateReport(
                scene_id=request.scene_id,
                mode=self.config.mode,
                status="accepted",
                input_rgb_sha256=input_digest,
                geometry_provider_name=geometry_scene.provider_name,
                geometry_provider_version=geometry_scene.provider_version,
                checkpoint_sha256=None,
                geometry_latency_ms=geometry_latency_ms,
                neural_latency_ms=0.0,
                total_latency_ms=total_latency_ms,
                geometry_supported_count=len(supported),
                geometry_unknown_count=unknown_count,
                yolo_box_count=0,
                matched_count=0,
            )
            self.last_report = report
            return BenchmarkVisionResult(
                observation=validate_scene_for_request(request, observation),
                report=report,
            )

        neural_started = self._clock()
        boxes: tuple[YoloBox, ...] = ()
        try:
            boxes = self._predict(image)
            neural_latency_ms = (self._clock() - neural_started) * 1000.0
            matched = self._match_supported_objects(supported, boxes)
        except VisionError as exc:
            neural_latency_ms = (self._clock() - neural_started) * 1000.0
            report = VisionGateReport(
                scene_id=request.scene_id,
                mode=self.config.mode,
                status="abstained",
                input_rgb_sha256=input_digest,
                geometry_provider_name=geometry_scene.provider_name,
                geometry_provider_version=geometry_scene.provider_version,
                checkpoint_sha256=self.config.checkpoint_sha256,
                geometry_latency_ms=geometry_latency_ms,
                neural_latency_ms=neural_latency_ms,
                total_latency_ms=(self._clock() - total_started) * 1000.0,
                geometry_supported_count=len(supported),
                geometry_unknown_count=unknown_count,
                yolo_box_count=len(boxes),
                matched_count=0,
                failure_reason=str(exc),
            )
            self.last_report = report
            raise BenchmarkVisionAbstention(str(exc), report) from exc

        objects: list[DetectedObject] = []
        for detected in geometry_scene.objects:
            box = matched.get(detected.track_id)
            if box is None:
                objects.append(self._annotate_unknown(detected, input_digest))
            else:
                objects.append(self._annotate_match(detected, box, input_digest))
        observation = replace(
            geometry_scene,
            objects=tuple(objects),
            provider_name=self.provider_name,
            provider_version=(
                f"{self.PROVIDER_VERSION}+checkpoint:"
                f"{self.config.checkpoint_sha256}+geometry:"
                f"{geometry_scene.provider_name}@{geometry_scene.provider_version}"
            ),
        )
        report = VisionGateReport(
            scene_id=request.scene_id,
            mode=self.config.mode,
            status="accepted",
            input_rgb_sha256=input_digest,
            geometry_provider_name=geometry_scene.provider_name,
            geometry_provider_version=geometry_scene.provider_version,
            checkpoint_sha256=self.config.checkpoint_sha256,
            geometry_latency_ms=geometry_latency_ms,
            neural_latency_ms=neural_latency_ms,
            total_latency_ms=(self._clock() - total_started) * 1000.0,
            geometry_supported_count=len(supported),
            geometry_unknown_count=unknown_count,
            yolo_box_count=len(boxes),
            matched_count=len(matched),
        )
        self.last_report = report
        return BenchmarkVisionResult(
            observation=validate_scene_for_request(request, observation),
            report=report,
        )

    def _predict(self, image: np.ndarray) -> tuple[YoloBox, ...]:
        left, top, right, bottom = self.config.crop_xyxy
        crop_rgb = image[top:bottom, left:right]
        crop_bgr = np.ascontiguousarray(crop_rgb[:, :, ::-1])
        try:
            raw_results = tuple(
                self._model.predict(
                    source=crop_bgr,
                    imgsz=self.config.inference_image_size,
                    conf=self.config.confidence_threshold,
                    iou=self.config.iou_threshold,
                    verbose=False,
                )
            )
        except Exception as exc:
            raise VisionError("YOLO11n benchmark inference failed") from exc
        if len(raw_results) != 1:
            raise VisionError("single-image YOLO inference must return one result")
        raw_boxes = getattr(raw_results[0], "boxes", None)
        if raw_boxes is None:
            return ()
        xyxy = _as_numpy(getattr(raw_boxes, "xyxy", ()))
        confidence = _as_numpy(getattr(raw_boxes, "conf", ()))
        classes = _as_numpy(getattr(raw_boxes, "cls", ()))
        if xyxy.size == confidence.size == classes.size == 0:
            return ()
        if xyxy.ndim != 2 or xyxy.shape[1] != 4:
            raise VisionError("YOLO boxes.xyxy must have shape (N, 4)")
        if confidence.ndim != 1 or classes.ndim != 1:
            raise VisionError("YOLO boxes.conf and boxes.cls must be vectors")
        if not (len(xyxy) == len(confidence) == len(classes)):
            raise VisionError("YOLO box arrays have inconsistent lengths")
        crop_width = right - left
        crop_height = bottom - top
        parsed: list[YoloBox] = []
        for raw_xyxy, raw_confidence, raw_class in zip(
            xyxy, confidence, classes, strict=True
        ):
            coordinates = tuple(float(value) for value in raw_xyxy)
            score = float(raw_confidence)
            class_value = float(raw_class)
            if not all(
                math.isfinite(value)
                for value in (*coordinates, score, class_value)
            ):
                raise VisionError("YOLO returned non-finite box data")
            if score < self.config.confidence_threshold:
                continue
            if not 0.0 <= score <= 1.0:
                raise VisionError("YOLO returned confidence outside [0, 1]")
            if not class_value.is_integer():
                raise VisionError("YOLO returned a non-integer class ID")
            class_id = int(class_value)
            if class_id not in YOLO_CLASS_NAMES:
                raise VisionError(f"YOLO returned unsupported class ID {class_id}")
            box_left, box_top, box_right, box_bottom = coordinates
            if not (
                0.0 <= box_left < box_right <= float(crop_width)
                and 0.0 <= box_top < box_bottom <= float(crop_height)
            ):
                raise VisionError("YOLO returned a box outside the configured crop")
            parsed.append(
                YoloBox(
                    class_id=class_id,
                    class_name=YOLO_CLASS_NAMES[class_id],
                    confidence=score,
                    xyxy_full=(
                        box_left + left,
                        box_top + top,
                        box_right + left,
                        box_bottom + top,
                    ),
                )
            )
        return tuple(parsed)

    def _match_supported_objects(
        self,
        objects: Sequence[DetectedObject],
        boxes: Sequence[YoloBox],
    ) -> dict[str, YoloBox]:
        left, top, right, bottom = self.config.crop_xyxy
        candidates: list[tuple[float, int, int]] = []
        centres: list[tuple[float, float]] = []
        for object_index, detected in enumerate(objects):
            raw_centre = detected.attributes.get("pixel_center_uv")
            if isinstance(raw_centre, (str, bytes)):
                raise VisionError(
                    f"geometry target {detected.track_id!r} lacks a pixel centre"
                )
            try:
                centre = tuple(float(value) for value in raw_centre)
            except (TypeError, ValueError) as exc:
                raise VisionError(
                    f"geometry target {detected.track_id!r} lacks a pixel centre"
                ) from exc
            if len(centre) != 2 or not all(math.isfinite(value) for value in centre):
                raise VisionError(
                    f"geometry target {detected.track_id!r} has an invalid pixel centre"
                )
            u, v = centre
            if not (left <= u < right and top <= v < bottom):
                raise VisionError(
                    f"geometry target {detected.track_id!r} lies outside the YOLO crop"
                )
            centres.append((u, v))
            expected_class = CLASS_TO_YOLO_ID[detected.class_id]
            for box_index, box in enumerate(boxes):
                if box.class_id != expected_class:
                    continue
                box_left, box_top, box_right, box_bottom = box.xyxy_full
                margin = float(self.config.roi_margin_px)
                if not (
                    box_left - margin <= u <= box_right + margin
                    and box_top - margin <= v <= box_bottom + margin
                ):
                    continue
                half_width = max((box_right - box_left) / 2.0, 1e-9)
                half_height = max((box_bottom - box_top) / 2.0, 1e-9)
                du = (u - (box_left + box_right) / 2.0) / half_width
                dv = (v - (box_top + box_bottom) / 2.0) / half_height
                candidates.append((du * du + dv * dv, object_index, box_index))
        assigned_objects: set[int] = set()
        assigned_boxes: set[int] = set()
        matched: dict[str, YoloBox] = {}
        for _, object_index, box_index in sorted(candidates):
            if object_index in assigned_objects or box_index in assigned_boxes:
                continue
            assigned_objects.add(object_index)
            assigned_boxes.add(box_index)
            matched[objects[object_index].track_id] = boxes[box_index]
        unmatched_objects = [
            {
                "track_id": objects[index].track_id,
                "class_id": objects[index].class_id,
                "pixel_center_uv": centres[index],
            }
            for index in range(len(objects))
            if index not in assigned_objects
        ]
        unmatched_boxes = [
            {
                "class_name": boxes[index].class_name,
                "confidence": round(boxes[index].confidence, 6),
                "xyxy_full": boxes[index].xyxy_full,
            }
            for index in range(len(boxes))
            if index not in assigned_boxes
        ]
        if unmatched_objects or unmatched_boxes:
            raise VisionError(
                "YOLO and calibrated hybrid geometry disagree: "
                f"unmatched_geometry={unmatched_objects!r}, "
                f"unmatched_yolo={unmatched_boxes!r}"
            )
        return matched

    def _common_attributes(
        self, detected: DetectedObject, input_digest: str
    ) -> dict[str, Any]:
        attributes = dict(detected.attributes)
        attributes.update(
            {
                "benchmark_vision_mode": self.config.mode,
                "benchmark_input_rgb_sha256": input_digest,
                "benchmark_geometry_provider": {
                    "name": getattr(
                        self._geometry_provider,
                        "PROVIDER_NAME",
                        self._geometry_provider.__class__.__name__,
                    ),
                    "version": getattr(
                        self._geometry_provider, "PROVIDER_VERSION", "unknown"
                    ),
                },
            }
        )
        return attributes

    def _annotate_geometry_only(
        self, detected: DetectedObject, input_digest: str
    ) -> DetectedObject:
        attributes = self._common_attributes(detected, input_digest)
        attributes["neural_gate_status"] = "not_requested"
        return replace(detected, attributes=attributes)

    def _annotate_unknown(
        self, detected: DetectedObject, input_digest: str
    ) -> DetectedObject:
        attributes = self._common_attributes(detected, input_digest)
        attributes.update(
            {
                "neural_gate_status": "geometry_only_non_target_obstacle",
                "checkpoint_sha256": self.config.checkpoint_sha256,
            }
        )
        return replace(detected, attributes=attributes)

    def _annotate_match(
        self, detected: DetectedObject, box: YoloBox, input_digest: str
    ) -> DetectedObject:
        attributes = self._common_attributes(detected, input_digest)
        attributes.update(
            {
                "neural_gate_status": "corroborated",
                "checkpoint_sha256": self.config.checkpoint_sha256,
                "yolo_class_id": box.class_id,
                "yolo_class_name": box.class_name,
                "yolo_confidence": box.confidence,
                "yolo_bbox_xyxy_full": box.xyxy_full,
                "geometry_confidence": detected.confidence,
                "metric_pose_source": "calibrated_hybrid_geometry",
                "neural_output_scope": "class_and_2d_region_only",
            }
        )
        return replace(
            detected,
            confidence=min(detected.confidence, box.confidence),
            attributes=attributes,
        )


def load_pinned_yolo11n(checkpoint: Path | str) -> Any:
    """Hash-check and lazily load the reviewed seed-55 Ultralytics model."""

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = _sha256_file(path)
    if digest != PINNED_YOLO11N_SHA256:
        raise ValueError(
            "YOLO11n checkpoint hash does not match the reviewed seed-55 artifact"
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - runtime-only dependency
        raise RuntimeError(
            "YOLO-assisted mode requires the plugin-owned Ultralytics runtime"
        ) from exc
    model = YOLO(str(path), task="detect")
    names = _normalise_model_names(getattr(model, "names", None))
    if names != YOLO_CLASS_NAMES:
        raise ValueError(
            f"YOLO class map must be exactly {YOLO_CLASS_NAMES!r}; "
            f"received {names!r}"
        )
    return model
