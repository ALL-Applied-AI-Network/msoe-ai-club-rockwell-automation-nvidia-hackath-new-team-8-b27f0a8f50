"""Backend-neutral contracts for replaceable perception providers.

This module deliberately has no NumPy, OpenCV, simulator, or robot-client
dependency.  A provider may use any of those internally, but it must translate
its result into robot-base SI units before returning a :class:`SceneObservation`.
The small contracts here are therefore safe to share between Isaac, hardware,
offline data tools, and lightweight unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


SCENE_SCHEMA_VERSION = 1
ROBOT_BASE_FRAME = "robot_base"


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _finite_tuple(values: Any, size: int, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain {size} finite numbers")
    try:
        result = tuple(_finite_number(value, name) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain {size} finite numbers") from exc
    if len(result) != size:
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a non-empty string")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must be a non-empty string")
    return result


def _string_tuple(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of non-empty strings")
    try:
        result = tuple(_nonempty_string(value, name) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of non-empty strings") from exc
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _freeze_json_value(value: Any, name: str) -> Any:
    """Validate and freeze one JSON-compatible value.

    Model-specific tensors, masks, and arrays stay outside the stable scene
    contract.  Providers may put compact, serializable evidence such as pixel
    centres, bounding boxes, or category metadata here.
    """

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{name} keys must be non-empty strings")
            frozen[key] = _freeze_json_value(nested, f"{name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json_value(nested, f"{name}[{index}]")
            for index, nested in enumerate(value)
        )
    raise ValueError(f"{name} must contain only JSON-compatible values")


def _freeze_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return _freeze_json_value(value, name)


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(nested) for nested in value]
    return value


@dataclass(frozen=True)
class Pose3D:
    """One rigid pose expressed in the canonical robot-base frame.

    Quaternion component order is ``(x, y, z, w)``.  The quaternion must
    already be normalized; silently normalizing a malformed model result could
    hide a provider error at the robot boundary.  Orientation rotates vectors
    from the class-defined object-local frame into robot base; each class's
    origin, axes, dimensions and symmetries belong in the object catalog.
    """

    position_base_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        position = _finite_tuple(self.position_base_m, 3, "position_base_m")
        orientation = _finite_tuple(self.orientation_xyzw, 4, "orientation_xyzw")
        norm = math.sqrt(sum(value * value for value in orientation))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("orientation_xyzw must be a unit quaternion")
        object.__setattr__(self, "position_base_m", position)
        object.__setattr__(self, "orientation_xyzw", orientation)

    @classmethod
    def from_yaw(
        cls, position_base_m: Sequence[float], yaw_rad: float
    ) -> "Pose3D":
        """Construct a robot-base pose whose only rotation is about base Z."""

        yaw = _finite_number(yaw_rad, "yaw_rad")
        half = yaw / 2.0
        return cls(
            position_base_m=tuple(position_base_m),
            orientation_xyzw=(0.0, 0.0, math.sin(half), math.cos(half)),
        )


@dataclass(frozen=True)
class DepthEncoding:
    """Explicit conversion and validity rules for one raw depth image."""

    meters_per_unit: float
    invalid_values: tuple[float, ...] = ()
    aligned_to_rgb: bool = False

    def __post_init__(self) -> None:
        scale = _finite_number(self.meters_per_unit, "meters_per_unit")
        if scale <= 0.0:
            raise ValueError("meters_per_unit must be positive")
        invalid = tuple(
            _finite_number(value, "invalid_values")
            for value in self.invalid_values
        )
        if len(set(invalid)) != len(invalid):
            raise ValueError("invalid_values must not contain duplicates")
        if not isinstance(self.aligned_to_rgb, bool):
            raise ValueError("aligned_to_rgb must be a boolean")
        object.__setattr__(self, "meters_per_unit", scale)
        object.__setattr__(self, "invalid_values", invalid)


@dataclass(frozen=True)
class CameraFrame:
    """One camera sample with distinct RGB and depth calibration revisions."""

    camera_id: str
    capture_time_s: float
    calibration_id: str | None
    rgb: Any = None
    depth: Any = None
    depth_calibration_id: str | None = None
    depth_encoding: DepthEncoding | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_id", _nonempty_string(self.camera_id, "camera_id"))
        object.__setattr__(
            self,
            "capture_time_s",
            _finite_number(self.capture_time_s, "capture_time_s"),
        )
        if self.rgb is None and self.depth is None:
            raise ValueError("a camera frame must contain rgb, depth, or both")
        if self.calibration_id is None:
            rgb_calibration_id = None
        else:
            rgb_calibration_id = _nonempty_string(
                self.calibration_id, "calibration_id"
            )
        if self.depth_calibration_id is None:
            depth_calibration_id = None
        else:
            depth_calibration_id = _nonempty_string(
                self.depth_calibration_id, "depth_calibration_id"
            )
        if self.rgb is not None and rgb_calibration_id is None:
            raise ValueError("an RGB frame requires calibration_id")
        if self.depth is not None and depth_calibration_id is None:
            raise ValueError("a depth frame requires depth_calibration_id")
        if self.depth is not None and not isinstance(
            self.depth_encoding, DepthEncoding
        ):
            raise ValueError("a depth frame requires DepthEncoding")
        if self.depth is None and self.depth_encoding is not None:
            raise ValueError("depth_encoding requires a depth frame")
        object.__setattr__(self, "calibration_id", rgb_calibration_id)
        object.__setattr__(self, "depth_calibration_id", depth_calibration_id)


@dataclass(frozen=True)
class RobotState:
    """Synchronized robot state in the shared SI/runtime convention.

    Joint values use the project's six-arm-joint convention in radians.
    ``tcp_pose_base_m_wpr_rad`` contains base-frame ``(x, y, z)`` in metres
    followed by the FANUC ``(w, p, r)`` orientation parameters in radians.
    The orientation representation is named explicitly; it is not a generic
    roll/pitch/yaw claim.  Gripper state is normalized to ``[0, 1]``.
    """

    arm_joints_rad: tuple[float, ...] | None = None
    tcp_pose_base_m_wpr_rad: tuple[float, ...] | None = None
    gripper_state: float | None = None

    def __post_init__(self) -> None:
        joints = (
            None
            if self.arm_joints_rad is None
            else _finite_tuple(self.arm_joints_rad, 6, "arm_joints_rad")
        )
        tcp = (
            None
            if self.tcp_pose_base_m_wpr_rad is None
            else _finite_tuple(
                self.tcp_pose_base_m_wpr_rad,
                6,
                "tcp_pose_base_m_wpr_rad",
            )
        )
        if self.gripper_state is None:
            gripper = None
        else:
            gripper = _finite_number(self.gripper_state, "gripper_state")
            if not 0.0 <= gripper <= 1.0:
                raise ValueError("gripper_state must be in [0, 1]")
        if joints is None and tcp is None and gripper is None:
            raise ValueError("RobotState must contain at least one measured value")
        object.__setattr__(self, "arm_joints_rad", joints)
        object.__setattr__(self, "tcp_pose_base_m_wpr_rad", tcp)
        object.__setattr__(self, "gripper_state", gripper)


@dataclass(frozen=True)
class PerceptionRequest:
    """Grouped runtime inputs presented to one perception provider.

    Individual camera timestamps remain explicit because the official camera
    service returns the latest frame from each source; callers must not claim
    perfect multi-camera synchronization.  Moving-camera providers can use the
    joints from this same robot observation when applying wrist extrinsics.
    """

    scene_id: str
    capture_time_s: float
    frames: tuple[CameraFrame, ...]
    robot_state: RobotState | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        scene_id = _nonempty_string(self.scene_id, "scene_id")
        capture_time = _finite_number(self.capture_time_s, "capture_time_s")
        frames = tuple(self.frames)
        if not frames:
            raise ValueError("frames must contain at least one camera frame")
        if not all(isinstance(frame, CameraFrame) for frame in frames):
            raise ValueError("frames must contain only CameraFrame values")
        camera_ids = tuple(frame.camera_id for frame in frames)
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("frames must not contain duplicate camera IDs")
        if self.robot_state is not None and not isinstance(
            self.robot_state, RobotState
        ):
            raise ValueError("robot_state must be a RobotState or None")
        metadata = _freeze_mapping(self.metadata, "metadata")
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "capture_time_s", capture_time)
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "metadata", metadata)

    def frame(self, camera_id: str) -> CameraFrame:
        """Return a named camera frame or raise ``KeyError`` when absent."""

        requested = _nonempty_string(camera_id, "camera_id")
        for frame in self.frames:
            if frame.camera_id == requested:
                return frame
        raise KeyError(requested)


@dataclass(frozen=True)
class DetectedObject:
    """One provider result in robot-base metres and quaternion orientation."""

    track_id: str
    class_id: str
    sort_label: str | None
    capture_time_s: float
    pose: Pose3D
    dimensions_m: tuple[float, float, float] | None = None
    confidence: float = 1.0
    pose_quality: float | None = None
    visibility: float | None = None
    source_camera_ids: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        track_id = _nonempty_string(self.track_id, "track_id")
        class_id = _nonempty_string(self.class_id, "class_id")
        if self.sort_label is None:
            sort_label = None
        else:
            sort_label = _nonempty_string(self.sort_label, "sort_label")
        capture_time = _finite_number(self.capture_time_s, "capture_time_s")
        if not isinstance(self.pose, Pose3D):
            raise ValueError("pose must be a Pose3D")
        dimensions = None
        if self.dimensions_m is not None:
            dimensions = _finite_tuple(self.dimensions_m, 3, "dimensions_m")
            if any(value <= 0.0 for value in dimensions):
                raise ValueError("dimensions_m must contain positive values")
        confidence = _finite_number(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        pose_quality = self._optional_score(self.pose_quality, "pose_quality")
        visibility = self._optional_score(self.visibility, "visibility")
        source_camera_ids = _string_tuple(
            self.source_camera_ids, "source_camera_ids"
        )
        attributes = _freeze_mapping(self.attributes, "attributes")
        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "class_id", class_id)
        object.__setattr__(self, "sort_label", sort_label)
        object.__setattr__(self, "capture_time_s", capture_time)
        object.__setattr__(self, "dimensions_m", dimensions)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "pose_quality", pose_quality)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "source_camera_ids", source_camera_ids)
        object.__setattr__(self, "attributes", attributes)

    @staticmethod
    def _optional_score(value: Any, name: str) -> float | None:
        if value is None:
            return None
        score = _finite_number(value, name)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
        return score


@dataclass(frozen=True, kw_only=True)
class SceneObservation:
    """Versioned, serializable result returned by every vision provider."""

    schema_version: int = SCENE_SCHEMA_VERSION
    scene_id: str
    capture_time_s: float
    coordinate_frame: str = ROBOT_BASE_FRAME
    objects: tuple[DetectedObject, ...] = ()
    provider_name: str
    provider_version: str
    calibration_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCENE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported scene schema version {self.schema_version!r}"
            )
        scene_id = _nonempty_string(self.scene_id, "scene_id")
        capture_time = _finite_number(self.capture_time_s, "capture_time_s")
        coordinate_frame = _nonempty_string(
            self.coordinate_frame, "coordinate_frame"
        )
        if coordinate_frame != ROBOT_BASE_FRAME:
            raise ValueError("coordinate_frame must be 'robot_base'")
        objects = tuple(self.objects)
        if not all(isinstance(value, DetectedObject) for value in objects):
            raise ValueError("objects must contain only DetectedObject values")
        track_ids = tuple(value.track_id for value in objects)
        if len(set(track_ids)) != len(track_ids):
            raise ValueError("objects must not contain duplicate track IDs")
        if any(abs(value.capture_time_s - capture_time) > 1e-6 for value in objects):
            raise ValueError("all objects must use the scene capture timestamp")
        provider_name = _nonempty_string(self.provider_name, "provider_name")
        provider_version = _nonempty_string(self.provider_version, "provider_version")
        calibration_ids = _string_tuple(self.calibration_ids, "calibration_ids")
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "capture_time_s", capture_time)
        object.__setattr__(self, "coordinate_frame", coordinate_frame)
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "provider_name", provider_name)
        object.__setattr__(self, "provider_version", provider_version)
        object.__setattr__(self, "calibration_ids", calibration_ids)


@runtime_checkable
class VisionProvider(Protocol):
    """Structural interface implemented by interchangeable perception systems."""

    def observe(self, request: PerceptionRequest) -> SceneObservation:
        """Interpret one sensor request without reading simulator ground truth."""


def validate_scene_for_request(
    request: PerceptionRequest, scene: SceneObservation
) -> SceneObservation:
    """Reject stale/misattributed provider output at the runtime boundary."""

    if not isinstance(request, PerceptionRequest):
        raise TypeError("request must be a PerceptionRequest")
    if not isinstance(scene, SceneObservation):
        raise TypeError("vision provider must return a SceneObservation")
    if scene.scene_id != request.scene_id:
        raise ValueError("vision provider returned a different scene_id")
    if abs(scene.capture_time_s - request.capture_time_s) > 1e-6:
        raise ValueError("vision provider returned a stale capture timestamp")
    camera_ids = {frame.camera_id for frame in request.frames}
    frames_by_camera = {frame.camera_id: frame for frame in request.frames}
    calibration_ids = {
        calibration_id
        for frame in request.frames
        for calibration_id in (
            frame.calibration_id,
            frame.depth_calibration_id,
        )
        if calibration_id is not None
    }
    if not scene.calibration_ids:
        raise ValueError("vision provider must identify its calibration IDs")
    unknown_calibrations = set(scene.calibration_ids) - calibration_ids
    if unknown_calibrations:
        raise ValueError(
            "vision provider cited unavailable calibration IDs: "
            + ", ".join(sorted(unknown_calibrations))
        )
    for detected in scene.objects:
        if not detected.source_camera_ids:
            raise ValueError(
                f"object {detected.track_id} must identify at least one source camera"
            )
        unknown_cameras = set(detected.source_camera_ids) - camera_ids
        if unknown_cameras:
            raise ValueError(
                f"object {detected.track_id} cites unavailable cameras: "
                + ", ".join(sorted(unknown_cameras))
            )
        for camera_id in detected.source_camera_ids:
            source_frame = frames_by_camera[camera_id]
            source_calibrations = {
                calibration_id
                for calibration_id in (
                    source_frame.calibration_id,
                    source_frame.depth_calibration_id,
                )
                if calibration_id is not None
            }
            if not source_calibrations.intersection(scene.calibration_ids):
                raise ValueError(
                    f"object {detected.track_id} cites source camera "
                    f"{camera_id} without citing one of its calibrations"
                )
    return scene


_SCENE_KEYS = {
    "schema_version",
    "scene_id",
    "capture_time_s",
    "coordinate_frame",
    "objects",
    "provider_name",
    "provider_version",
    "calibration_ids",
}
_OBJECT_KEYS = {
    "track_id",
    "class_id",
    "sort_label",
    "capture_time_s",
    "pose",
    "dimensions_m",
    "confidence",
    "pose_quality",
    "visibility",
    "source_camera_ids",
    "attributes",
}
_POSE_KEYS = {"position_base_m", "orientation_xyzw"}


def _strict_mapping(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    actual = set(value)
    missing = sorted(keys - actual)
    unknown = sorted(actual - keys)
    if missing or unknown:
        parts = []
        if missing:
            parts.append("missing " + ", ".join(missing))
        if unknown:
            parts.append("unknown " + ", ".join(unknown))
        raise ValueError(f"{name} has " + "; ".join(parts))
    return value


def _json_sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a JSON array")
    return value


def scene_observation_to_dict(scene: SceneObservation) -> dict[str, Any]:
    """Return the canonical JSON-compatible representation of one scene."""

    if not isinstance(scene, SceneObservation):
        raise TypeError("scene must be a SceneObservation")
    objects = []
    for detected in scene.objects:
        objects.append(
            {
                "track_id": detected.track_id,
                "class_id": detected.class_id,
                "sort_label": detected.sort_label,
                "capture_time_s": detected.capture_time_s,
                "pose": {
                    "position_base_m": list(detected.pose.position_base_m),
                    "orientation_xyzw": list(detected.pose.orientation_xyzw),
                },
                "dimensions_m": (
                    None
                    if detected.dimensions_m is None
                    else list(detected.dimensions_m)
                ),
                "confidence": detected.confidence,
                "pose_quality": detected.pose_quality,
                "visibility": detected.visibility,
                "source_camera_ids": list(detected.source_camera_ids),
                "attributes": _thaw_json_value(detected.attributes),
            }
        )
    return {
        "schema_version": scene.schema_version,
        "scene_id": scene.scene_id,
        "capture_time_s": scene.capture_time_s,
        "coordinate_frame": scene.coordinate_frame,
        "objects": objects,
        "provider_name": scene.provider_name,
        "provider_version": scene.provider_version,
        "calibration_ids": list(scene.calibration_ids),
    }


def scene_observation_from_dict(value: Mapping[str, Any]) -> SceneObservation:
    """Parse the strict version-1 JSON representation of one scene.

    Unknown fields are rejected rather than ignored so independently developed
    providers cannot accidentally disagree about units, frames, or meanings.
    A future schema change must increment ``schema_version`` and add an explicit
    migration instead of silently changing this parser.
    """

    raw = _strict_mapping(value, _SCENE_KEYS, "scene")
    objects_raw = _json_sequence(raw["objects"], "scene.objects")
    objects: list[DetectedObject] = []
    for index, item in enumerate(objects_raw):
        object_raw = _strict_mapping(item, _OBJECT_KEYS, f"scene.objects[{index}]")
        pose_raw = _strict_mapping(
            object_raw["pose"], _POSE_KEYS, f"scene.objects[{index}].pose"
        )
        dimensions_raw = object_raw["dimensions_m"]
        if dimensions_raw is not None:
            dimensions_raw = _json_sequence(
                dimensions_raw, f"scene.objects[{index}].dimensions_m"
            )
        source_ids = _json_sequence(
            object_raw["source_camera_ids"],
            f"scene.objects[{index}].source_camera_ids",
        )
        attributes = object_raw["attributes"]
        if not isinstance(attributes, Mapping):
            raise ValueError(f"scene.objects[{index}].attributes must be a mapping")
        objects.append(
            DetectedObject(
                track_id=object_raw["track_id"],
                class_id=object_raw["class_id"],
                sort_label=object_raw["sort_label"],
                capture_time_s=object_raw["capture_time_s"],
                pose=Pose3D(
                    position_base_m=_json_sequence(
                        pose_raw["position_base_m"],
                        f"scene.objects[{index}].pose.position_base_m",
                    ),
                    orientation_xyzw=_json_sequence(
                        pose_raw["orientation_xyzw"],
                        f"scene.objects[{index}].pose.orientation_xyzw",
                    ),
                ),
                dimensions_m=dimensions_raw,
                confidence=object_raw["confidence"],
                pose_quality=object_raw["pose_quality"],
                visibility=object_raw["visibility"],
                source_camera_ids=source_ids,
                attributes=attributes,
            )
        )
    calibration_ids = _json_sequence(
        raw["calibration_ids"], "scene.calibration_ids"
    )
    return SceneObservation(
        schema_version=raw["schema_version"],
        scene_id=raw["scene_id"],
        capture_time_s=raw["capture_time_s"],
        coordinate_frame=raw["coordinate_frame"],
        objects=tuple(objects),
        provider_name=raw["provider_name"],
        provider_version=raw["provider_version"],
        calibration_ids=tuple(calibration_ids),
    )


__all__ = [
    "CameraFrame",
    "DepthEncoding",
    "DetectedObject",
    "PerceptionRequest",
    "Pose3D",
    "ROBOT_BASE_FRAME",
    "RobotState",
    "SCENE_SCHEMA_VERSION",
    "SceneObservation",
    "VisionProvider",
    "scene_observation_from_dict",
    "scene_observation_to_dict",
    "validate_scene_for_request",
]
