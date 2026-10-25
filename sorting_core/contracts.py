"""Backend-neutral observations, plans, poses, and semantic motion phases."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


KNOWN_COLOURS = ("red", "yellow", "blue")


def _finite_tuple(values, size: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {size} numbers") from exc
    if len(result) != size or not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


def normalise_square_yaw_rad(yaw_rad: float) -> float:
    """Return square yaw in the equivalent interval [-pi/4, pi/4)."""

    yaw = float(yaw_rad)
    if not math.isfinite(yaw):
        raise ValueError("yaw must be finite")
    quarter_turn = math.pi / 2.0
    return (yaw + quarter_turn / 2.0) % quarter_turn - quarter_turn / 2.0


@dataclass(frozen=True)
class CubeDetection:
    """One camera-derived cube observation in robot-base coordinates.

    ``position_base_m`` is the cube centre.  ``yaw_rad`` is equivalent modulo
    90 degrees.  No simulator pose or evaluation ground truth is represented by
    this type.
    """

    track_id: str
    colour: str
    capture_time_s: float
    position_base_m: tuple[float, float, float]
    yaw_rad: float
    confidence: float = 1.0
    pixel_center_uv: Optional[tuple[float, float]] = None
    mask_area_px: Optional[float] = None

    def __post_init__(self) -> None:
        track_id = str(self.track_id).strip()
        colour = str(self.colour).strip().lower()
        if not track_id:
            raise ValueError("track_id must not be empty")
        capture_time = float(self.capture_time_s)
        confidence = float(self.confidence)
        if not math.isfinite(capture_time):
            raise ValueError("capture_time_s must be finite")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "colour", colour)
        object.__setattr__(self, "capture_time_s", capture_time)
        object.__setattr__(
            self, "position_base_m", _finite_tuple(self.position_base_m, 3, "position")
        )
        object.__setattr__(self, "yaw_rad", normalise_square_yaw_rad(self.yaw_rad))
        object.__setattr__(self, "confidence", confidence)
        if self.pixel_center_uv is not None:
            object.__setattr__(
                self,
                "pixel_center_uv",
                _finite_tuple(self.pixel_center_uv, 2, "pixel centre"),
            )
        if self.mask_area_px is not None:
            area = float(self.mask_area_px)
            if not math.isfinite(area) or area <= 0:
                raise ValueError("mask_area_px must be positive and finite")
            object.__setattr__(self, "mask_area_px", area)


@dataclass(frozen=True)
class SceneDetections:
    scene_id: str
    capture_time_s: float
    cubes: tuple[CubeDetection, ...]

    def __post_init__(self) -> None:
        scene_id = str(self.scene_id).strip()
        capture_time = float(self.capture_time_s)
        if not scene_id:
            raise ValueError("scene_id must not be empty")
        if not math.isfinite(capture_time):
            raise ValueError("capture_time_s must be finite")
        cubes = tuple(self.cubes)
        if any(abs(c.capture_time_s - capture_time) > 1e-6 for c in cubes):
            raise ValueError("all detections must use the scene capture timestamp")
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "capture_time_s", capture_time)
        object.__setattr__(self, "cubes", cubes)


@dataclass(frozen=True)
class TipPoseBase:
    """Desired physical fingertip pose in robot-base metres/radians."""

    xyz_m: tuple[float, float, float]
    yaw_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "xyz_m", _finite_tuple(self.xyz_m, 3, "tip pose"))
        yaw = float(self.yaw_rad)
        if not math.isfinite(yaw):
            raise ValueError("tip yaw must be finite")
        object.__setattr__(self, "yaw_rad", yaw)


@dataclass(frozen=True)
class BinTarget:
    colour: str
    centre_base_m: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "colour", str(self.colour).strip().lower())
        object.__setattr__(
            self, "centre_base_m", _finite_tuple(self.centre_base_m, 2, "bin centre")
        )


@dataclass(frozen=True)
class PickPlacePlan:
    """One plan tied to a specific camera scene and target identity."""

    plan_id: str
    scene_id: str
    capture_time_s: float
    target: CubeDetection
    destination: BinTarget
    pick_tip_pose: TipPoseBase
    release_tip_pose: TipPoseBase

    def __post_init__(self) -> None:
        if not str(self.plan_id).strip() or not str(self.scene_id).strip():
            raise ValueError("plan_id and scene_id must not be empty")
        if self.target.colour != self.destination.colour:
            raise ValueError("target and destination colours must match")
        if self.target.capture_time_s != float(self.capture_time_s):
            raise ValueError("plan and target timestamps differ")


@dataclass(frozen=True)
class MotionPhase:
    """A semantic phase shared by Isaac and the physical-robot adapter.

    A phase with no target is a synchronization gate, not a robot action.
    Backends must stop the action stream at ``requires_reobservation_before``
    gates so the target can be invalidated before descent.
    """

    name: str
    target: Optional[TipPoseBase]
    gripper: float
    dwell_s: float = 0.0
    requires_reobservation_before: bool = False

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        gripper = float(self.gripper)
        dwell = float(self.dwell_s)
        if not name:
            raise ValueError("phase name must not be empty")
        if not math.isfinite(gripper) or not 0 <= gripper <= 1:
            raise ValueError("phase gripper command must be in [0, 1]")
        if not math.isfinite(dwell) or dwell < 0:
            raise ValueError("phase dwell must be non-negative")
        if self.requires_reobservation_before and self.target is not None:
            raise ValueError("a reobservation gate must not command motion")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "gripper", gripper)
        object.__setattr__(self, "dwell_s", dwell)


@dataclass(frozen=True)
class TrajectoryChunk:
    """One bounded FANUC action chunk or an explicit reobservation gate."""

    phase_name: str
    actions: tuple[tuple[float, ...], ...]
    dt_s: float
    part_index: int = 0
    requires_reobservation_before: bool = False

    def __post_init__(self) -> None:
        actions = tuple(_finite_tuple(row, 7, "TCP action") for row in self.actions)
        if self.requires_reobservation_before and actions:
            raise ValueError("reobservation gate cannot include actions")
        object.__setattr__(self, "actions", actions)
