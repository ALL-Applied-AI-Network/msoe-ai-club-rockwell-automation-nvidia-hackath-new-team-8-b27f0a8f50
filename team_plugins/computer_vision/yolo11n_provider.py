"""Pinned YOLO11n cube corroboration plus existing calibrated cube geometry.

The neural detector is deliberately limited to proposing two-dimensional cube
regions. Colour, robot-base position, and square-symmetric yaw continue to come
from the existing deterministic colour/known-size geometry detector operating
on the complete frame. Inside the neural crop, both detectors must agree
one-to-one; disagreement aborts the observation rather than turning a missed
object into clear space. A YOLO screw prediction also aborts the observation
because the pilot did not contain any validation or final-test screws and a
box is not an executable screw pose.

Ultralytics is imported only by :func:`_load_yolo_model`, which is called by
the opt-in profile factory.  Importing this module therefore leaves the
profile-unset baseline dependency-free and unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import logging
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from sorting_core.contracts import CubeDetection, SceneDetections
from sorting_core.perception import (
    DetectedObject,
    PerceptionRequest,
    SceneObservation,
    validate_scene_for_request,
)
from sorting_core.task_perception import cube_scene_to_observation
from sorting_core.vision import (
    VisionError,
    detect_coloured_cubes,
    detection_in_pickup_region,
)

if TYPE_CHECKING:
    from sorting_core.integration_plugins import VisionFactoryContext
    from sorting_core.perception import VisionProvider


PLUGIN_API_VERSION = 1
PROVIDER_NAME = "yolo11n-cube-proposals-calibrated-geometry"
PROVIDER_REVISION = "2"
EXPECTED_CHECKPOINT_SHA256 = (
    "010bb71b1b07fc77490a2bc2bd774102205640bdf9ba07d6fa8b3bd758788fbb"
)
EXPECTED_CLASS_NAMES = {
    0: "cube",
    1: "screw_iso10642_m6x30",
}
EXPECTED_IMAGE_SIZE = (1920, 1080)
EXPECTED_CROP_XYXY = (500, 400, 1500, 1050)
EXPECTED_INFERENCE_IMAGE_SIZE = 960
EXPECTED_CONFIDENCE_THRESHOLD = 0.6596596596596597
EXPECTED_IOU_THRESHOLD = 0.7
EXPECTED_WARMUP_PASSES = 3
GEOMETRY_AGREEMENT_SCOPES = frozenset(
    {"all_inference_crop", "pickup_region"}
)

_LOGGER = logging.getLogger(__name__)

def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _integer_tuple(value: Any, size: int, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain {size} integers")
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain {size} integers") from exc
    if len(raw) != size or any(type(item) is not int for item in raw):
        raise ValueError(f"{name} must contain {size} integers")
    return tuple(int(item) for item in raw)


def _normalise_model_names(value: Any) -> dict[int, str]:
    if isinstance(value, Mapping):
        result: dict[int, str] = {}
        for key, name in value.items():
            if isinstance(key, bool):
                raise ValueError("YOLO class-map keys must be integer class IDs")
            try:
                class_id = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "YOLO class-map keys must be integer class IDs"
                ) from exc
            if str(class_id) != str(key) and class_id != key:
                raise ValueError("YOLO class-map keys must be integer class IDs")
            result[class_id] = str(name)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {index: str(name) for index, name in enumerate(value)}
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


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _load_yolo_model(checkpoint: Path) -> Any:
    """Load Ultralytics lazily so the default provider never imports it."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - exercised in runtime env
        raise RuntimeError(
            "the YOLO11n profile requires the plugin-owned Ultralytics runtime"
        ) from exc
    return YOLO(str(checkpoint), task="detect")


@dataclass(frozen=True)
class _YoloBox:
    class_id: int
    confidence: float
    xyxy_full: tuple[float, float, float, float]


@dataclass(frozen=True)
class _GatedCube:
    cube: CubeDetection
    box: _YoloBox | None
    geometry_confidence: float
    yolo_status: str


class Yolo11nCubeVisionProvider:
    """Use pinned YOLO boxes to gate calibrated deterministic cube poses."""

    def __init__(
        self,
        *,
        config: Any,
        camera_projectors: Mapping[str, Any],
        calibration_ids: Mapping[str, str],
        model: Any,
        checkpoint_sha256: str,
        camera_id: str = "table",
        expected_cube_edge_m: float | None = None,
        detector: Any | None = None,
        expected_image_size: tuple[int, int] = EXPECTED_IMAGE_SIZE,
        crop_xyxy: tuple[int, int, int, int] = EXPECTED_CROP_XYXY,
        inference_image_size: int = EXPECTED_INFERENCE_IMAGE_SIZE,
        confidence_threshold: float = EXPECTED_CONFIDENCE_THRESHOLD,
        iou_threshold: float = EXPECTED_IOU_THRESHOLD,
        roi_margin_px: int = 4,
        geometry_agreement_scope: str = "all_inference_crop",
        maximum_frame_age_s: float = 0.15,
        profile_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config.validate()
        self.camera_id = str(camera_id).strip()
        if self.camera_id != "table":
            raise ValueError("the pinned YOLO11n provider supports only camera 'table'")
        if self.camera_id not in camera_projectors:
            raise ValueError(f"missing projector for camera {self.camera_id}")
        if self.camera_id not in calibration_ids:
            raise ValueError(f"missing calibration id for camera {self.camera_id}")
        self._projector = camera_projectors[self.camera_id]
        self._calibration_id = str(calibration_ids[self.camera_id]).strip()
        if not self._calibration_id:
            raise ValueError("table calibration id must not be empty")

        digest = str(checkpoint_sha256).strip().lower()
        if digest != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError(
                "YOLO11n checkpoint hash is not the pinned seed-55 artifact"
            )
        self._checkpoint_sha256 = digest

        names = _normalise_model_names(getattr(model, "names", None))
        if names != EXPECTED_CLASS_NAMES:
            raise ValueError(
                "YOLO11n class map must be exactly "
                f"{EXPECTED_CLASS_NAMES!r}; received {names!r}"
            )
        if not callable(getattr(model, "predict", None)):
            raise ValueError("YOLO model must expose predict()")
        self._model = model

        self._expected_image_size = tuple(expected_image_size)
        if self._expected_image_size != EXPECTED_IMAGE_SIZE:
            raise ValueError(
                f"expected_image_size must remain pinned to {EXPECTED_IMAGE_SIZE}"
            )
        self._crop_xyxy = tuple(crop_xyxy)
        if self._crop_xyxy != EXPECTED_CROP_XYXY:
            raise ValueError(f"crop_xyxy must remain pinned to {EXPECTED_CROP_XYXY}")
        self._inference_image_size = int(inference_image_size)
        if self._inference_image_size != EXPECTED_INFERENCE_IMAGE_SIZE:
            raise ValueError(
                "inference_image_size must remain pinned to "
                f"{EXPECTED_INFERENCE_IMAGE_SIZE}"
            )
        self._confidence_threshold = _finite_float(
            confidence_threshold, "confidence_threshold"
        )
        if self._confidence_threshold != EXPECTED_CONFIDENCE_THRESHOLD:
            raise ValueError(
                "confidence_threshold must remain pinned to "
                f"{EXPECTED_CONFIDENCE_THRESHOLD}"
            )
        self._iou_threshold = _finite_float(iou_threshold, "iou_threshold")
        if self._iou_threshold != EXPECTED_IOU_THRESHOLD:
            raise ValueError(
                f"iou_threshold must remain pinned to {EXPECTED_IOU_THRESHOLD}"
            )
        if type(roi_margin_px) is not int or not 0 <= roi_margin_px <= 32:
            raise ValueError("roi_margin_px must be an integer in [0, 32]")
        self._roi_margin_px = roi_margin_px
        if not isinstance(geometry_agreement_scope, str):
            raise ValueError(
                "geometry_agreement_scope must be one of "
                f"{sorted(GEOMETRY_AGREEMENT_SCOPES)}"
            )
        agreement_scope = geometry_agreement_scope.strip()
        if agreement_scope not in GEOMETRY_AGREEMENT_SCOPES:
            raise ValueError(
                "geometry_agreement_scope must be one of "
                f"{sorted(GEOMETRY_AGREEMENT_SCOPES)}"
            )
        self._geometry_agreement_scope = agreement_scope
        self._maximum_frame_age_s = _finite_float(
            maximum_frame_age_s, "maximum_frame_age_s"
        )
        if self._maximum_frame_age_s < 0.0:
            raise ValueError("maximum_frame_age_s must be non-negative")

        self._detector = detector or detect_coloured_cubes
        self._cube_edge_m = (
            self.config.cube_edge_m
            if expected_cube_edge_m is None
            else _finite_float(expected_cube_edge_m, "expected_cube_edge_m")
        )
        if self._cube_edge_m <= 0.0:
            raise ValueError("expected_cube_edge_m must be positive")
        self._profile_provenance = dict(
            profile_provenance
            or {
                "profile_id": "direct-constructor",
                "profile_path": None,
                "profile_sha256": None,
            }
        )
        self._warmup_latency_ms: float | None = None
        self._warmup_pass_latencies_ms: tuple[float, ...] = ()

    @property
    def provider_version(self) -> str:
        return f"{PROVIDER_REVISION}+sha256:{self._checkpoint_sha256}"

    def warm_up(self) -> float:
        """Pay one-time model initialization before the first live frame.

        The physical policy's observation-age budget starts after the cell
        sends a frame. Cold Torch/Ultralytics inference can be much slower than
        steady state, so the opt-in factory warms the exact preprocessing shape
        while ``Policy`` is still being constructed.
        """

        if self._warmup_latency_ms is not None:
            return self._warmup_latency_ms
        left, top, right, bottom = self._crop_xyxy
        blank_bgr = np.zeros((bottom - top, right - left, 3), dtype=np.uint8)
        latencies: list[float] = []
        for _ in range(EXPECTED_WARMUP_PASSES):
            started = time.perf_counter()
            self._predict(blank_bgr)
            latencies.append((time.perf_counter() - started) * 1000.0)
        self._warmup_pass_latencies_ms = tuple(latencies)
        self._warmup_latency_ms = sum(latencies)
        _LOGGER.info(
            "YOLO11n backend warm-up completed in %.1f ms across %d passes: %s",
            self._warmup_latency_ms,
            EXPECTED_WARMUP_PASSES,
            tuple(round(value, 1) for value in latencies),
        )
        return self._warmup_latency_ms

    def _predict(self, crop_bgr: np.ndarray) -> tuple[_YoloBox, ...]:
        try:
            raw_results = self._model.predict(
                source=crop_bgr,
                imgsz=self._inference_image_size,
                conf=self._confidence_threshold,
                iou=self._iou_threshold,
                verbose=False,
            )
            results = tuple(raw_results)
        except Exception as exc:
            raise VisionError("YOLO11n inference failed") from exc
        if len(results) != 1:
            raise VisionError(
                "YOLO11n single-image inference must return exactly one result"
            )
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return ()
        xyxy = _as_numpy(getattr(boxes, "xyxy", ()))
        confidence = _as_numpy(getattr(boxes, "conf", ()))
        classes = _as_numpy(getattr(boxes, "cls", ()))
        if xyxy.size == 0 and confidence.size == 0 and classes.size == 0:
            return ()
        if xyxy.ndim != 2 or xyxy.shape[1] != 4:
            raise VisionError("YOLO11n boxes.xyxy must have shape (N, 4)")
        if confidence.ndim != 1 or classes.ndim != 1:
            raise VisionError("YOLO11n boxes.conf and boxes.cls must be vectors")
        if not (len(xyxy) == len(confidence) == len(classes)):
            raise VisionError("YOLO11n box arrays have inconsistent lengths")

        crop_left, crop_top, crop_right, crop_bottom = self._crop_xyxy
        crop_width = crop_right - crop_left
        crop_height = crop_bottom - crop_top
        parsed: list[_YoloBox] = []
        for raw_xyxy, raw_confidence, raw_class in zip(
            xyxy, confidence, classes, strict=True
        ):
            values = tuple(float(value) for value in raw_xyxy)
            score = float(raw_confidence)
            class_value = float(raw_class)
            if not all(math.isfinite(value) for value in (*values, score, class_value)):
                raise VisionError("YOLO11n returned non-finite box data")
            if not 0.0 <= score <= 1.0:
                raise VisionError("YOLO11n returned confidence outside [0, 1]")
            if score < self._confidence_threshold:
                continue
            if not class_value.is_integer():
                raise VisionError("YOLO11n returned a non-integer class ID")
            class_id = int(class_value)
            if class_id not in EXPECTED_CLASS_NAMES:
                raise VisionError(f"YOLO11n returned unsupported class ID {class_id}")
            left, top, right, bottom = values
            if not (
                0.0 <= left < right <= float(crop_width)
                and 0.0 <= top < bottom <= float(crop_height)
            ):
                raise VisionError("YOLO11n returned a box outside the pinned crop")
            parsed.append(
                _YoloBox(
                    class_id=class_id,
                    confidence=score,
                    xyxy_full=(
                        left + crop_left,
                        top + crop_top,
                        right + crop_left,
                        bottom + crop_top,
                    ),
                )
            )
        return tuple(parsed)

    def _gate_geometry(
        self,
        geometry: SceneDetections,
        boxes: Sequence[_YoloBox],
    ) -> tuple[_GatedCube, ...]:
        candidates: list[tuple[int, float, int, int]] = []
        crop_left, crop_top, crop_right, crop_bottom = self._crop_xyxy
        geometry_in_crop: set[int] = set()
        geometry_in_pickup_region = {
            index
            for index, cube in enumerate(geometry.cubes)
            if detection_in_pickup_region(cube, self.config)
        }
        for geometry_index, cube in enumerate(geometry.cubes):
            if cube.pixel_center_uv is None:
                raise VisionError(
                    "calibrated cube geometry must retain its pixel centre"
                )
            u, v = cube.pixel_center_uv
            if not (
                crop_left <= u < crop_right and crop_top <= v < crop_bottom
            ):
                continue
            geometry_in_crop.add(geometry_index)
            for box_index, box in enumerate(boxes):
                left, top, right, bottom = box.xyxy_full
                margin = float(self._roi_margin_px)
                if not (
                    left - margin <= u <= right + margin
                    and top - margin <= v <= bottom + margin
                ):
                    continue
                half_width = max((right - left) / 2.0, 1e-9)
                half_height = max((bottom - top) / 2.0, 1e-9)
                centre_u = (left + right) / 2.0
                centre_v = (top + bottom) / 2.0
                distance = ((u - centre_u) / half_width) ** 2 + (
                    (v - centre_v) / half_height
                ) ** 2
                required_priority = (
                    0
                    if self._geometry_agreement_scope == "all_inference_crop"
                    or geometry_index in geometry_in_pickup_region
                    else 1
                )
                candidates.append(
                    (required_priority, distance, geometry_index, box_index)
                )

        geometry_requiring_yolo = (
            geometry_in_crop
            if self._geometry_agreement_scope == "all_inference_crop"
            else geometry_in_crop & geometry_in_pickup_region
        )

        assigned_geometry: set[int] = set()
        assigned_boxes: set[int] = set()
        matched_boxes: dict[int, _YoloBox] = {}
        for _, _, geometry_index, box_index in sorted(candidates):
            if geometry_index in assigned_geometry or box_index in assigned_boxes:
                continue
            cube = geometry.cubes[geometry_index]
            box = boxes[box_index]
            assigned_geometry.add(geometry_index)
            assigned_boxes.add(box_index)
            matched_boxes[geometry_index] = box

        unmatched_boxes = tuple(
            index for index in range(len(boxes)) if index not in assigned_boxes
        )
        unmatched_required_geometry = tuple(
            index
            for index in sorted(geometry_requiring_yolo)
            if index not in assigned_geometry
        )
        if unmatched_boxes or unmatched_required_geometry:
            unmatched_box_details = [
                {
                    "xyxy_full": tuple(
                        round(value, 2) for value in boxes[index].xyxy_full
                    ),
                    "confidence": round(boxes[index].confidence, 4),
                }
                for index in unmatched_boxes
            ]
            unmatched_geometry_details = [
                {
                    "track_id": geometry.cubes[index].track_id,
                    "colour": geometry.cubes[index].colour,
                    "pixel_center_uv": tuple(
                        round(value, 2)
                        for value in geometry.cubes[index].pixel_center_uv or ()
                    ),
                }
                for index in unmatched_required_geometry
            ]
            raise VisionError(
                "YOLO11n and calibrated geometry disagree inside the inference "
                f"crop: unmatched_yolo_boxes={len(unmatched_boxes)}, "
                "unmatched_geometry_detections="
                f"{len(unmatched_required_geometry)}, "
                f"geometry_agreement_scope={self._geometry_agreement_scope!r}, "
                f"unmatched_yolo={unmatched_box_details!r}, "
                f"unmatched_geometry={unmatched_geometry_details!r}"
            )

        gated: list[_GatedCube] = []
        for geometry_index, cube in enumerate(geometry.cubes):
            if geometry_index in matched_boxes:
                box = matched_boxes[geometry_index]
                gated.append(
                    _GatedCube(
                        cube=replace(
                            cube,
                            confidence=min(cube.confidence, box.confidence),
                        ),
                        box=box,
                        geometry_confidence=cube.confidence,
                        yolo_status="corroborated",
                    )
                )
                continue

            if geometry_index in geometry_in_crop:
                # Only the opt-in pickup-region scope can reach this branch.
                # Preserve off-region geometry for destination verification and
                # obstacle evidence, but expose its missing neural corroboration.
                gated.append(
                    _GatedCube(
                        cube=cube,
                        box=None,
                        geometry_confidence=cube.confidence,
                        yolo_status=(
                            "inside_inference_crop_outside_pickup_geometry_only"
                        ),
                    )
                )
                continue

            pick_eligible = geometry_index in geometry_in_pickup_region
            if pick_eligible:
                raise VisionError(
                    "calibrated geometry found a pick-eligible cube outside the "
                    "pinned YOLO11n crop; neural coverage is incomplete"
                )
            gated.append(
                _GatedCube(
                    cube=cube,
                    box=None,
                    geometry_confidence=cube.confidence,
                    yolo_status="outside_inference_crop_geometry_only",
                )
            )
        return tuple(gated)

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

        image = np.asarray(frame.rgb)
        expected_width, expected_height = self._expected_image_size
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape != (expected_height, expected_width, 3)
        ):
            raise VisionError(
                "YOLO11n requires an uncropped 1920x1080 HxWx3 uint8 RGB "
                f"table-camera frame; received shape={image.shape!r} "
                f"dtype={image.dtype}"
            )

        crop_left, crop_top, crop_right, crop_bottom = self._crop_xyxy
        crop_rgb = image[crop_top:crop_bottom, crop_left:crop_right]
        # The pilot was trained from cv2.imread images, so its ndarray input is
        # BGR even though CameraFrame explicitly stores native RGB.
        crop_bgr = np.ascontiguousarray(crop_rgb[:, :, ::-1])
        predictions = self._predict(crop_bgr)
        screw_count = sum(box.class_id == 1 for box in predictions)
        if screw_count:
            raise VisionError(
                "YOLO11n reported an unvalidated screw detection; this cube-only "
                "profile abstains because screw recall and pose were never tested"
            )
        cube_boxes = tuple(box for box in predictions if box.class_id == 0)
        geometry = self._detector(
            image,
            self._projector,
            self.config,
            scene_id=request.scene_id,
            capture_time_s=request.capture_time_s,
            expected_cube_edge_m=self._cube_edge_m,
        )
        if not isinstance(geometry, SceneDetections):
            raise VisionError("calibrated cube detector must return SceneDetections")
        if geometry.scene_id != request.scene_id:
            raise VisionError("calibrated cube detector returned a different scene_id")
        if abs(geometry.capture_time_s - request.capture_time_s) > 1e-6:
            raise VisionError(
                "calibrated cube detector returned a stale capture timestamp"
            )

        matches = self._gate_geometry(geometry, cube_boxes)
        gated_scene = SceneDetections(
            scene_id=request.scene_id,
            capture_time_s=request.capture_time_s,
            cubes=tuple(match.cube for match in matches),
        )
        observation = cube_scene_to_observation(
            gated_scene,
            class_id="cube",
            dimensions_m=(self._cube_edge_m,) * 3,
            provider_name=PROVIDER_NAME,
            provider_version=self.provider_version,
            calibration_ids=(self._calibration_id,),
            source_camera_ids=(self.camera_id,),
        )
        objects: list[DetectedObject] = []
        for detected, match in zip(
            observation.objects, matches, strict=True
        ):
            attributes = dict(detected.attributes)
            attributes.update({
                "yolo_status": match.yolo_status,
                "geometry_confidence": match.geometry_confidence,
                "geometry_agreement_scope": self._geometry_agreement_scope,
                "in_pickup_region": detection_in_pickup_region(
                    match.cube, self.config
                ),
                "checkpoint_sha256": self._checkpoint_sha256,
                "inference_crop_xyxy": self._crop_xyxy,
                "inference_image_size": self._inference_image_size,
                "model_warmup_latency_ms": self._warmup_latency_ms,
                "model_warmup_pass_latencies_ms": (
                    self._warmup_pass_latencies_ms
                ),
                "detector_family": "yolo11n_box_plus_hsv_known_size_geometry_v1",
                "symmetry_id": "upright_square_yaw_modulo_90_deg",
                "pose_constraint": "upright_on_table",
                "absolute_base_accuracy_verified": False,
                "source_frame_capture_time_s": frame.capture_time_s,
                "source_frame_age_s": max(0.0, frame_age_s),
                "integration_profile": self._profile_provenance,
            })
            if match.box is not None:
                attributes.update(
                    {
                        "yolo_bbox_xyxy_full": match.box.xyxy_full,
                        "yolo_confidence": match.box.confidence,
                    }
                )
            objects.append(replace(detected, attributes=attributes))
        return validate_scene_for_request(
            request,
            replace(observation, objects=tuple(objects)),
        )


def create(context: "VisionFactoryContext") -> "VisionProvider":
    """Create the pinned provider from an explicit integration profile."""

    options = dict(context.options)
    camera_id = str(options.pop("camera_id", "table"))
    maximum_frame_age_s = options.pop("maximum_frame_age_s", 0.15)
    expected_image_size = _integer_tuple(
        options.pop("expected_image_size", EXPECTED_IMAGE_SIZE),
        2,
        "expected_image_size",
    )
    crop_xyxy = _integer_tuple(
        options.pop("crop_xyxy", EXPECTED_CROP_XYXY), 4, "crop_xyxy"
    )
    inference_image_size = options.pop(
        "inference_image_size", EXPECTED_INFERENCE_IMAGE_SIZE
    )
    confidence_threshold = options.pop(
        "confidence_threshold", EXPECTED_CONFIDENCE_THRESHOLD
    )
    iou_threshold = options.pop("iou_threshold", EXPECTED_IOU_THRESHOLD)
    roi_margin_px = options.pop("roi_margin_px", 4)
    geometry_agreement_scope = options.pop(
        "geometry_agreement_scope", "all_inference_crop"
    )
    declared_checkpoint_sha256 = str(
        options.pop("checkpoint_sha256", EXPECTED_CHECKPOINT_SHA256)
    ).strip().lower()
    if options:
        raise ValueError(f"unknown YOLO11n provider options: {sorted(options)}")
    if declared_checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "profile checkpoint_sha256 does not identify the pinned seed-55 model"
        )

    if context.checkpoint is None or not str(context.checkpoint).strip():
        raise ValueError(
            "the YOLO11n profile requires an external checkpoint path via "
            "ROCKWELL_VISION_CHECKPOINT"
        )
    checkpoint = Path(str(context.checkpoint)).expanduser()
    if not checkpoint.is_absolute():
        raise ValueError("YOLO11n checkpoint path must be absolute")
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise ValueError(f"YOLO11n checkpoint does not exist: {checkpoint}")
    repository_root = Path(__file__).resolve().parents[2]
    if _is_within(checkpoint, repository_root):
        raise ValueError("YOLO11n checkpoint must remain outside the Git repository")
    actual_checkpoint_sha256 = _sha256_file(checkpoint)
    if actual_checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "YOLO11n checkpoint SHA-256 mismatch: expected "
            f"{EXPECTED_CHECKPOINT_SHA256}, received {actual_checkpoint_sha256}"
        )

    model = _load_yolo_model(checkpoint)
    provider = Yolo11nCubeVisionProvider(
        config=context.config,
        camera_projectors=context.camera_projectors,
        calibration_ids=context.calibration_ids,
        model=model,
        checkpoint_sha256=actual_checkpoint_sha256,
        camera_id=camera_id,
        expected_cube_edge_m=context.expected_cube_edge_m,
        detector=context.detector,
        expected_image_size=expected_image_size,
        crop_xyxy=crop_xyxy,
        inference_image_size=inference_image_size,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        roi_margin_px=roi_margin_px,
        geometry_agreement_scope=geometry_agreement_scope,
        maximum_frame_age_s=maximum_frame_age_s,
        profile_provenance={
            "profile_id": context.profile_id,
            "profile_path": context.profile_path,
            "profile_sha256": context.profile_sha256,
        },
    )
    provider.warm_up()
    return provider


__all__ = [
    "EXPECTED_CHECKPOINT_SHA256",
    "EXPECTED_CLASS_NAMES",
    "EXPECTED_CONFIDENCE_THRESHOLD",
    "EXPECTED_CROP_XYXY",
    "EXPECTED_IMAGE_SIZE",
    "EXPECTED_INFERENCE_IMAGE_SIZE",
    "EXPECTED_IOU_THRESHOLD",
    "EXPECTED_WARMUP_PASSES",
    "GEOMETRY_AGREEMENT_SCOPES",
    "PLUGIN_API_VERSION",
    "Yolo11nCubeVisionProvider",
    "create",
]
