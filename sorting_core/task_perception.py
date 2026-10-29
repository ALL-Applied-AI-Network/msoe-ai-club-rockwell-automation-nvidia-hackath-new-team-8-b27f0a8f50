"""Compatibility adapters between generic perception and the cube task.

The current planner intentionally remains narrow and proven: it accepts one
``CubeDetection`` per configured colour.  Vision teammates can work against
the generic contracts in :mod:`sorting_core.perception`; this module is the
single explicit boundary that projects their output into the existing task.
Unsupported objects remain visible to the caller instead of being silently
discarded as free space.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

from .contracts import CubeDetection, SceneDetections, normalise_square_yaw_rad
from .perception import DetectedObject, Pose3D, SceneObservation


def _dimensions(values: Any) -> tuple[float, float, float]:
    if isinstance(values, bool):
        raise ValueError("dimensions_m must contain three positive finite values")
    if isinstance(values, (int, float)):
        values = (values, values, values)
    if isinstance(values, (str, bytes)):
        raise ValueError("dimensions_m must contain three positive finite values")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "dimensions_m must contain three positive finite values"
        ) from exc
    if (
        len(result) != 3
        or not all(math.isfinite(value) and value > 0.0 for value in result)
    ):
        raise ValueError("dimensions_m must contain three positive finite values")
    return result


def _upright_yaw_from_quaternion_xyzw(
    values: Sequence[float], *, maximum_tilt_rad: float
) -> float:
    """Return robot-base Z yaw after enforcing the cube-task upright assumption.

    A generic provider may estimate full 6D orientation.  The legacy cube task
    consumes only yaw and applies square symmetry in ``CubeDetection``.  It
    must not silently discard material roll or pitch from a tilted object.
    """

    x, y, z, w = values
    # Base-frame Z component of the object's local Z axis.  Pose3D has already
    # checked that the quaternion is finite and unit length.
    local_z_in_base_z = 1.0 - 2.0 * (x * x + y * y)
    local_z_in_base_z = min(1.0, max(-1.0, local_z_in_base_z))
    tilt_rad = math.acos(local_z_in_base_z)
    if tilt_rad > maximum_tilt_rad:
        raise ValueError(
            "supported cube orientation exceeds the upright-task tilt limit: "
            f"{math.degrees(tilt_rad):.3f} deg > "
            f"{math.degrees(maximum_tilt_rad):.3f} deg"
        )
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw, cos_yaw)


def _optional_attribute_pair(value: Any, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain two finite numbers")
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two finite numbers") from exc
    if len(result) != 2 or not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain two finite numbers")
    return result


def _optional_positive_attribute(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be positive and finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _optional_finite_attribute(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class CubeSceneProjection:
    """Legacy cube scene plus generic objects the cube task cannot yet pick."""

    scene: SceneDetections
    unsupported_objects: tuple[DetectedObject, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scene, SceneDetections):
            raise ValueError("scene must be a SceneDetections")
        unsupported = tuple(self.unsupported_objects)
        if not all(isinstance(value, DetectedObject) for value in unsupported):
            raise ValueError(
                "unsupported_objects must contain only DetectedObject values"
            )
        object.__setattr__(self, "unsupported_objects", unsupported)


def cube_scene_to_observation(
    scene: SceneDetections,
    *,
    dimensions_m: Sequence[float] | float,
    class_id: str = "cube",
    provider_name: str = "legacy-cube-detector",
    provider_version: str = "1",
    calibration_ids: Iterable[str] = (),
    source_camera_ids: Iterable[str] = (),
) -> SceneObservation:
    """Lift the current cube contract into the generic scene schema.

    Pixel centre and mask-area evidence are retained in serializable
    ``attributes`` so converting back is lossless for every legacy field.
    """

    if not isinstance(scene, SceneDetections):
        raise TypeError("scene must be a SceneDetections")
    dimensions = _dimensions(dimensions_m)
    source_ids = tuple(source_camera_ids)
    objects: list[DetectedObject] = []
    for cube in scene.cubes:
        attributes: dict[str, Any] = {}
        if cube.pixel_center_uv is not None:
            attributes["pixel_center_uv"] = tuple(cube.pixel_center_uv)
        if cube.mask_area_px is not None:
            attributes["mask_area_px"] = cube.mask_area_px
        # Preserve the legacy square-normalized value exactly. Reconstructing
        # through sin/cos can cross the floating-point -pi/4 boundary and
        # select the 90-degree-equivalent representation instead.
        attributes["legacy_square_yaw_rad"] = cube.yaw_rad
        objects.append(
            DetectedObject(
                track_id=cube.track_id,
                class_id=class_id,
                sort_label=cube.colour,
                capture_time_s=cube.capture_time_s,
                pose=Pose3D.from_yaw(cube.position_base_m, cube.yaw_rad),
                dimensions_m=dimensions,
                confidence=cube.confidence,
                source_camera_ids=source_ids,
                attributes=attributes,
            )
        )
    return SceneObservation(
        scene_id=scene.scene_id,
        capture_time_s=scene.capture_time_s,
        objects=tuple(objects),
        provider_name=provider_name,
        provider_version=provider_version,
        calibration_ids=tuple(calibration_ids),
    )


def cube_scene_from_observation(
    observation: SceneObservation,
    *,
    class_ids: Iterable[str] = ("cube",),
    maximum_tilt_rad: float = math.radians(10.0),
) -> CubeSceneProjection:
    """Project supported generic objects into the unchanged cube planner input.

    Objects with other class IDs are returned separately.  A caller can use
    them as obstacles or route them to a future object-specific planner; this
    adapter never turns an unsupported observation into implicit empty space.
    """

    if not isinstance(observation, SceneObservation):
        raise TypeError("observation must be a SceneObservation")
    try:
        tilt_limit = float(maximum_tilt_rad)
    except (TypeError, ValueError) as exc:
        raise ValueError("maximum_tilt_rad must be finite and in [0, pi]") from exc
    if not math.isfinite(tilt_limit) or not 0.0 <= tilt_limit <= math.pi:
        raise ValueError("maximum_tilt_rad must be finite and in [0, pi]")
    if isinstance(class_ids, (str, bytes)):
        raise ValueError("class_ids must be a non-empty sequence of class IDs")
    try:
        supported = tuple(str(value).strip() for value in class_ids)
    except TypeError as exc:
        raise ValueError(
            "class_ids must be a non-empty sequence of class IDs"
        ) from exc
    if not supported or any(not value for value in supported):
        raise ValueError("class_ids must be a non-empty sequence of class IDs")
    if len(set(supported)) != len(supported):
        raise ValueError("class_ids must not contain duplicates")
    supported_set = set(supported)

    cubes: list[CubeDetection] = []
    unsupported: list[DetectedObject] = []
    for detected in observation.objects:
        if detected.class_id not in supported_set:
            unsupported.append(detected)
            continue
        if detected.sort_label is None:
            raise ValueError(
                f"supported object {detected.track_id!r} has no sort_label"
            )
        pixel_center = _optional_attribute_pair(
            detected.attributes.get("pixel_center_uv"), "pixel_center_uv"
        )
        mask_area = _optional_positive_attribute(
            detected.attributes.get("mask_area_px"), "mask_area_px"
        )
        legacy_yaw = _optional_finite_attribute(
            detected.attributes.get("legacy_square_yaw_rad"),
            "legacy_square_yaw_rad",
        )
        pose_yaw = _upright_yaw_from_quaternion_xyzw(
            detected.pose.orientation_xyzw,
            maximum_tilt_rad=tilt_limit,
        )
        if legacy_yaw is not None:
            square_yaw_disagreement = abs(
                normalise_square_yaw_rad(legacy_yaw - pose_yaw)
            )
            if square_yaw_disagreement > 1e-9:
                raise ValueError(
                    "legacy_square_yaw_rad conflicts with the object pose"
                )
        yaw = legacy_yaw if legacy_yaw is not None else pose_yaw
        cubes.append(
            CubeDetection(
                track_id=detected.track_id,
                colour=detected.sort_label,
                capture_time_s=detected.capture_time_s,
                position_base_m=detected.pose.position_base_m,
                yaw_rad=yaw,
                confidence=detected.confidence,
                pixel_center_uv=pixel_center,
                mask_area_px=mask_area,
            )
        )
    return CubeSceneProjection(
        scene=SceneDetections(
            scene_id=observation.scene_id,
            capture_time_s=observation.capture_time_s,
            cubes=tuple(cubes),
        ),
        unsupported_objects=tuple(unsupported),
    )


__all__ = [
    "CubeSceneProjection",
    "cube_scene_from_observation",
    "cube_scene_to_observation",
]
