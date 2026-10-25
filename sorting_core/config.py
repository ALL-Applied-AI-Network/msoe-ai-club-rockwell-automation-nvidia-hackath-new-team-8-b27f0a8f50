"""Validated configuration for the shared colour-bucket sorting task.

The default JSON is the single source of truth for dimensions and frame
conventions used by both backends.  This module has no simulator or robot-client
dependency, so it is safe to import from Isaac Python, the FANUC policy client,
and host-side tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "colour_bucket_sort.json"
)


class ConfigurationError(ValueError):
    """Raised when a task configuration is incomplete or physically inconsistent."""


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if not math.isfinite(number):
        raise ConfigurationError(f"{name} must be finite")
    return number


def _pair(value: Any, name: str) -> tuple[float, float]:
    try:
        pair = tuple(_finite(v, name) for v in value)
    except TypeError as exc:
        raise ConfigurationError(f"{name} must contain two numbers") from exc
    if len(pair) != 2:
        raise ConfigurationError(f"{name} must contain two numbers")
    return pair


def _triple(value: Any, name: str) -> tuple[float, float, float]:
    try:
        triple = tuple(_finite(v, name) for v in value)
    except TypeError as exc:
        raise ConfigurationError(f"{name} must contain three numbers") from exc
    if len(triple) != 3:
        raise ConfigurationError(f"{name} must contain three numbers")
    return triple


@dataclass(frozen=True)
class TaskSpec:
    colour_order: tuple[str, ...]
    cube_spawns_base_xy_yaw_deg: Mapping[str, tuple[float, float, float]]
    bin_centres_base_m: Mapping[str, tuple[float, float]]
    pick_tip_clearance_m: float
    approach_clearance_m: float
    bin_release_clearance_m: float
    minimum_distinct_centres_m: float


@dataclass(frozen=True)
class BinSpec:
    outer_m: float
    pocket_m: float
    height_m: float
    floor_m: float

    @property
    def pocket_depth_m(self) -> float:
        return self.height_m - self.floor_m


@dataclass(frozen=True)
class FrameSpec:
    surface_base_z_m: float
    uf0_z_offset_mm: float
    tool_down_wpr_deg: tuple[float, float, float]
    tcp_beyond_fingertips_mm: float
    tcp_conflict_note: str


@dataclass(frozen=True)
class GripperSpec:
    model: str
    jaw_travel_per_side_m: float
    straight_open_gap_m: float
    hook_closed_gap_m: float
    hook_open_gap_m: float
    finger_length_m: float
    finger_width_m: float
    finger_thickness_m: float
    j6_to_fingertip_m: float
    open_command: float
    closed_command: float


@dataclass(frozen=True)
class HardwareSpec:
    tcp_bounds_uf0_mm: Mapping[str, tuple[float, float]]
    angle_limit_deg: float
    dt_range_s: tuple[float, float]
    max_chunk_actions: int
    maximum_radial_reach_m: float
    default_dt_s: float
    max_linear_step_mm: float
    max_angular_step_deg: float


@dataclass(frozen=True)
class VisionSpec:
    # Each OpenCV HSV range is (h_min, h_max, s_min, v_min).
    hsv_ranges_opencv: Mapping[str, tuple[tuple[int, int, int, int], ...]]
    minimum_component_area_px: float
    maximum_component_area_fraction: float
    maximum_component_aspect_ratio: float
    cube_edge_minimum_factor: float
    cube_edge_maximum_factor: float
    morphology_kernel_px: int


@dataclass(frozen=True)
class SortConfig:
    schema_version: int
    cube_edge_m: float
    cube_size_note: str
    task: TaskSpec
    bin: BinSpec
    frames: FrameSpec
    gripper: GripperSpec
    hardware: HardwareSpec
    vision: VisionSpec

    @property
    def cube_top_base_z_m(self) -> float:
        return self.frames.surface_base_z_m + self.cube_edge_m

    @property
    def cube_centre_base_z_m(self) -> float:
        return self.frames.surface_base_z_m + self.cube_edge_m / 2.0

    @property
    def pick_tip_base_z_m(self) -> float:
        return self.frames.surface_base_z_m + self.task.pick_tip_clearance_m

    @property
    def bin_release_tip_base_z_m(self) -> float:
        return (
            self.frames.surface_base_z_m
            + self.bin.height_m
            + self.task.bin_release_clearance_m
        )

    def validate(self) -> "SortConfig":
        if self.schema_version != 1:
            raise ConfigurationError(
                f"unsupported colour-bucket config schema {self.schema_version}"
            )
        if self.cube_edge_m <= 0:
            raise ConfigurationError("cube_edge_m must be positive")
        if not self.cube_size_note.strip():
            raise ConfigurationError("cube_size_note must preserve the size assumption")

        colours = self.task.colour_order
        if colours != ("red", "yellow", "blue"):
            raise ConfigurationError("colour_order must be red, yellow, blue")
        if len(set(colours)) != len(colours):
            raise ConfigurationError("colour_order contains duplicates")
        expected = set(colours)
        for name, values in (
            ("cube_spawns_base_xy_yaw_deg", self.task.cube_spawns_base_xy_yaw_deg),
            ("bin_centres_base_m", self.task.bin_centres_base_m),
            ("hsv_ranges_opencv", self.vision.hsv_ranges_opencv),
        ):
            if set(values) != expected:
                raise ConfigurationError(f"{name} must define exactly {sorted(expected)}")

        positive = {
            "pick_tip_clearance_m": self.task.pick_tip_clearance_m,
            "approach_clearance_m": self.task.approach_clearance_m,
            "bin_release_clearance_m": self.task.bin_release_clearance_m,
            "minimum_distinct_centres_m": self.task.minimum_distinct_centres_m,
            "bin.outer_m": self.bin.outer_m,
            "bin.pocket_m": self.bin.pocket_m,
            "bin.height_m": self.bin.height_m,
            "bin.floor_m": self.bin.floor_m,
            "gripper.jaw_travel_per_side_m": self.gripper.jaw_travel_per_side_m,
            "gripper.straight_open_gap_m": self.gripper.straight_open_gap_m,
            "gripper.hook_open_gap_m": self.gripper.hook_open_gap_m,
            "gripper.finger_length_m": self.gripper.finger_length_m,
            "gripper.finger_width_m": self.gripper.finger_width_m,
            "gripper.finger_thickness_m": self.gripper.finger_thickness_m,
            "gripper.j6_to_fingertip_m": self.gripper.j6_to_fingertip_m,
            "hardware.angle_limit_deg": self.hardware.angle_limit_deg,
            "hardware.max_linear_step_mm": self.hardware.max_linear_step_mm,
            "hardware.max_angular_step_deg": self.hardware.max_angular_step_deg,
            "hardware.maximum_radial_reach_m": self.hardware.maximum_radial_reach_m,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"{name} must be positive and finite")

        if not 0 <= self.frames.tcp_beyond_fingertips_mm < 100:
            raise ConfigurationError("tcp_beyond_fingertips_mm is implausible")
        if not self.frames.tcp_conflict_note.strip():
            raise ConfigurationError("the TCP convention conflict must remain documented")
        if "10 mm" not in self.frames.tcp_conflict_note:
            raise ConfigurationError("tcp_conflict_note must mention the live-site 10 mm value")
        if not self.bin.floor_m < self.bin.height_m:
            raise ConfigurationError("bin floor must be below the rim")
        if not self.cube_edge_m < self.bin.pocket_m:
            raise ConfigurationError("cube must fit inside the bin pocket")
        if not self.cube_edge_m < self.gripper.straight_open_gap_m:
            raise ConfigurationError("cube must fit inside the straight-face opening")
        if self.cube_edge_m * math.sqrt(2.0) >= self.gripper.hook_open_gap_m:
            raise ConfigurationError("rotated cube diagonal does not fit the open hook gap")
        if not math.isclose(
            self.gripper.straight_open_gap_m,
            2.0 * self.gripper.jaw_travel_per_side_m,
            abs_tol=1e-12,
        ):
            raise ConfigurationError("straight opening must equal twice per-jaw travel")
        if not 0 <= self.gripper.open_command <= 1:
            raise ConfigurationError("open gripper command must be in [0, 1]")
        if not 0 <= self.gripper.closed_command <= 1:
            raise ConfigurationError("closed gripper command must be in [0, 1]")

        if set(self.hardware.tcp_bounds_uf0_mm) != {"x", "y", "z"}:
            raise ConfigurationError("TCP bounds must define x, y and z")
        for axis, (lower, upper) in self.hardware.tcp_bounds_uf0_mm.items():
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ConfigurationError(f"invalid TCP bounds for {axis}")
        dt_min, dt_max = self.hardware.dt_range_s
        if not 0 < dt_min <= self.hardware.default_dt_s <= dt_max:
            raise ConfigurationError("default dt must be inside the permitted range")
        if self.hardware.max_chunk_actions < 1:
            raise ConfigurationError("max_chunk_actions must be positive")

        if not 0 < self.vision.maximum_component_area_fraction <= 1:
            raise ConfigurationError("maximum component area fraction must be in (0, 1]")
        if self.vision.minimum_component_area_px <= 0:
            raise ConfigurationError("minimum component area must be positive")
        if self.vision.maximum_component_aspect_ratio < 1:
            raise ConfigurationError("maximum component aspect ratio must be >= 1")
        if not 0 < self.vision.cube_edge_minimum_factor < 1:
            raise ConfigurationError("cube edge minimum factor must be in (0, 1)")
        if self.vision.cube_edge_maximum_factor <= 1:
            raise ConfigurationError("cube edge maximum factor must be greater than 1")
        if self.vision.cube_edge_minimum_factor >= self.vision.cube_edge_maximum_factor:
            raise ConfigurationError("cube edge factors are reversed")
        if self.vision.morphology_kernel_px < 1 or self.vision.morphology_kernel_px % 2 == 0:
            raise ConfigurationError("morphology kernel must be a positive odd integer")
        for colour, ranges in self.vision.hsv_ranges_opencv.items():
            if not ranges:
                raise ConfigurationError(f"no HSV range configured for {colour}")
            for h0, h1, s0, v0 in ranges:
                if not (0 <= h0 <= h1 <= 179 and 0 <= s0 <= 255 and 0 <= v0 <= 255):
                    raise ConfigurationError(f"invalid HSV range for {colour}")
        return self


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{key} must be an object")
    return value


def config_from_dict(raw: Mapping[str, Any]) -> SortConfig:
    """Parse and validate a JSON-compatible mapping."""

    assumptions = _mapping(raw, "assumptions")
    task_raw = _mapping(raw, "task")
    bin_raw = _mapping(raw, "bin")
    frames_raw = _mapping(raw, "frames")
    gripper_raw = _mapping(raw, "gripper")
    hardware_raw = _mapping(raw, "hardware")
    vision_raw = _mapping(raw, "vision")

    colours = tuple(str(c).strip().lower() for c in task_raw["colour_order"])
    spawns_raw = _mapping(task_raw, "cube_spawns_base_xy_yaw_deg")
    bins_raw = _mapping(task_raw, "bin_centres_base_m")
    hsv_raw = _mapping(vision_raw, "hsv_ranges_opencv")

    config = SortConfig(
        schema_version=int(raw["schema_version"]),
        cube_edge_m=_finite(assumptions["cube_edge_m"], "cube_edge_m"),
        cube_size_note=str(assumptions["cube_size_note"]),
        task=TaskSpec(
            colour_order=colours,
            cube_spawns_base_xy_yaw_deg={
                str(k).lower(): _triple(v, f"cube spawn {k}")
                for k, v in spawns_raw.items()
            },
            bin_centres_base_m={
                str(k).lower(): _pair(v, f"bin centre {k}")
                for k, v in bins_raw.items()
            },
            pick_tip_clearance_m=_finite(
                task_raw["pick_tip_clearance_m"], "pick_tip_clearance_m"
            ),
            approach_clearance_m=_finite(
                task_raw["approach_clearance_m"], "approach_clearance_m"
            ),
            bin_release_clearance_m=_finite(
                task_raw["bin_release_clearance_m"], "bin_release_clearance_m"
            ),
            minimum_distinct_centres_m=_finite(
                task_raw["minimum_distinct_centres_m"],
                "minimum_distinct_centres_m",
            ),
        ),
        bin=BinSpec(
            outer_m=_finite(bin_raw["outer_m"], "bin.outer_m"),
            pocket_m=_finite(bin_raw["pocket_m"], "bin.pocket_m"),
            height_m=_finite(bin_raw["height_m"], "bin.height_m"),
            floor_m=_finite(bin_raw["floor_m"], "bin.floor_m"),
        ),
        frames=FrameSpec(
            surface_base_z_m=_finite(
                frames_raw["surface_base_z_m"], "surface_base_z_m"
            ),
            uf0_z_offset_mm=_finite(
                frames_raw["uf0_z_offset_mm"], "uf0_z_offset_mm"
            ),
            tool_down_wpr_deg=_triple(
                frames_raw["tool_down_wpr_deg"], "tool_down_wpr_deg"
            ),
            tcp_beyond_fingertips_mm=_finite(
                assumptions["tcp_beyond_fingertips_mm"],
                "tcp_beyond_fingertips_mm",
            ),
            tcp_conflict_note=str(assumptions["tcp_conflict_note"]),
        ),
        gripper=GripperSpec(
            model=str(gripper_raw["model"]),
            jaw_travel_per_side_m=_finite(
                gripper_raw["jaw_travel_per_side_m"],
                "gripper.jaw_travel_per_side_m",
            ),
            straight_open_gap_m=_finite(
                gripper_raw["straight_open_gap_m"], "gripper.straight_open_gap_m"
            ),
            hook_closed_gap_m=_finite(
                gripper_raw["hook_closed_gap_m"], "gripper.hook_closed_gap_m"
            ),
            hook_open_gap_m=_finite(
                gripper_raw["hook_open_gap_m"], "gripper.hook_open_gap_m"
            ),
            finger_length_m=_finite(
                gripper_raw["finger_length_m"], "gripper.finger_length_m"
            ),
            finger_width_m=_finite(
                gripper_raw["finger_width_m"], "gripper.finger_width_m"
            ),
            finger_thickness_m=_finite(
                gripper_raw["finger_thickness_m"], "gripper.finger_thickness_m"
            ),
            j6_to_fingertip_m=_finite(
                gripper_raw["j6_to_fingertip_m"], "gripper.j6_to_fingertip_m"
            ),
            open_command=_finite(gripper_raw["open_command"], "open_command"),
            closed_command=_finite(
                gripper_raw["closed_command"], "closed_command"
            ),
        ),
        hardware=HardwareSpec(
            tcp_bounds_uf0_mm={
                str(k): _pair(v, f"TCP {k} bounds")
                for k, v in _mapping(hardware_raw, "tcp_bounds_uf0_mm").items()
            },
            angle_limit_deg=_finite(
                hardware_raw["angle_limit_deg"], "angle_limit_deg"
            ),
            dt_range_s=_pair(hardware_raw["dt_range_s"], "dt_range_s"),
            max_chunk_actions=int(hardware_raw["max_chunk_actions"]),
            maximum_radial_reach_m=_finite(
                hardware_raw["maximum_radial_reach_m"], "maximum_radial_reach_m"
            ),
            default_dt_s=_finite(hardware_raw["default_dt_s"], "default_dt_s"),
            max_linear_step_mm=_finite(
                hardware_raw["max_linear_step_mm"], "max_linear_step_mm"
            ),
            max_angular_step_deg=_finite(
                hardware_raw["max_angular_step_deg"], "max_angular_step_deg"
            ),
        ),
        vision=VisionSpec(
            hsv_ranges_opencv={
                str(colour).lower(): tuple(
                    tuple(int(v) for v in hsv_range) for hsv_range in ranges
                )
                for colour, ranges in hsv_raw.items()
            },
            minimum_component_area_px=_finite(
                vision_raw["minimum_component_area_px"],
                "minimum_component_area_px",
            ),
            maximum_component_area_fraction=_finite(
                vision_raw["maximum_component_area_fraction"],
                "maximum_component_area_fraction",
            ),
            maximum_component_aspect_ratio=_finite(
                vision_raw["maximum_component_aspect_ratio"],
                "maximum_component_aspect_ratio",
            ),
            cube_edge_minimum_factor=_finite(
                vision_raw["cube_edge_minimum_factor"],
                "cube_edge_minimum_factor",
            ),
            cube_edge_maximum_factor=_finite(
                vision_raw["cube_edge_maximum_factor"],
                "cube_edge_maximum_factor",
            ),
            morphology_kernel_px=int(vision_raw["morphology_kernel_px"]),
        ),
    )
    return config.validate()


def load_config(path: str | Path | None = None) -> SortConfig:
    """Load the default checked-in configuration or an explicit replacement."""

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot load sorting config {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigurationError("sorting config root must be an object")
    return config_from_dict(raw)
