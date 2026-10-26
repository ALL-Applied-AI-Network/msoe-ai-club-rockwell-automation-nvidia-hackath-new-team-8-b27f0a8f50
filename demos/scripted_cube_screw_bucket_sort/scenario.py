"""Pure-Python contract for the sibling cube-and-screw Isaac demonstration.

This module deliberately has no Isaac Sim or NumPy dependency.  It validates
the exact supplied ISO 10642 M6x30 STL, defines the five-item scripted scene,
and provides geometry/report helpers shared by the Isaac entry point and host
tests.  Simulator truth is suitable for scoring this scripted demonstration;
it must not be described as camera or AI perception.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping, Sequence

from sorting_core.config import SortConfig, load_config


DEFAULT_SCENARIO_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "cube_screw_bucket_sort.json"
)

SUPPLIED_SCREW_FILENAME = "ISO10642-M6x30-countersunk-screw.stl"
SUPPLIED_SCREW_SHA256 = (
    "6a842f1de0923083b0eb2a3a209eacae87cc83d4b62e15b768a0a8cd277cec2a"
)
SUPPLIED_SCREW_TRIANGLES = 1910
SUPPLIED_SCREW_BOUNDS_MM = (
    (-6.0, -5.998135089874268, -30.0),
    (6.0, 5.998135089874268, 0.0),
)

Point3 = tuple[float, float, float]
Triangle = tuple[Point3, Point3, Point3]


class ScenarioError(ValueError):
    """Raised when the mixed-scene contract or supplied CAD is invalid."""


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ScenarioError(f"{name} must be a number") from exc
    if not math.isfinite(number):
        raise ScenarioError(f"{name} must be finite")
    return number


def _triple(value: Any, name: str) -> Point3:
    if isinstance(value, (str, bytes)):
        raise ScenarioError(f"{name} must contain three numbers")
    try:
        result = tuple(_finite(component, name) for component in value)
    except TypeError as exc:
        raise ScenarioError(f"{name} must contain three numbers") from exc
    if len(result) != 3:
        raise ScenarioError(f"{name} must contain three numbers")
    return result  # type: ignore[return-value]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScenarioError(f"{name} must be an object")
    return value


@dataclass(frozen=True)
class ScrewSpec:
    model_filename: str
    expected_sha256: str
    expected_triangles: int
    expected_bounds_mm: tuple[Point3, Point3]
    bounds_tolerance_mm: float
    mass_kg: float
    length_m: float
    head_diameter_m: float
    shank_diameter_m: float
    shank_start_from_head_m: float
    grasp_offset_from_head_m: float
    jaw_seating_margin_m: float
    spawn_head_height_above_surface_m: float
    placement_note: str

    @property
    def grasp_point_local_m(self) -> Point3:
        """CAD-local grasp point; the screw points from the head along local -Z."""

        return (0.0, 0.0, -self.grasp_offset_from_head_m)


@dataclass(frozen=True)
class ScrewInstance:
    item_id: str
    spawn_head_base_m: Point3
    spawn_yaw_deg: float
    destination_colour: str
    placement_grasp_point_offset_from_bin_centre_m: Point3
    target_axis_yaw_deg: float


@dataclass(frozen=True)
class ScenarioConfig:
    schema_version: int
    path: Path
    base_colour_sort_config_path: Path
    cube_config: SortConfig
    screw: ScrewSpec
    screws: tuple[ScrewInstance, ...]
    sort_order: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedScrewModel:
    path: Path
    sha256: str
    triangles_mm: tuple[Triangle, ...]
    bounds_mm: tuple[Point3, Point3]

    @property
    def triangle_count(self) -> int:
        return len(self.triangles_mm)

    @property
    def vertices_mm(self) -> tuple[Point3, ...]:
        return tuple(vertex for triangle in self.triangles_mm for vertex in triangle)

    @property
    def vertices_m(self) -> tuple[Point3, ...]:
        return tuple(
            (x_mm * 0.001, y_mm * 0.001, z_mm * 0.001)
            for x_mm, y_mm, z_mm in self.vertices_mm
        )


@dataclass(frozen=True)
class SortItem:
    item_id: str
    kind: str
    destination_colour: str
    spawn_reference_base_m: Point3
    spawn_grasp_point_base_m: Point3
    spawn_yaw_deg: float
    placement_reference_base_m: Point3
    placement_grasp_point_base_m: Point3
    target_axis_yaw_deg: float


@dataclass(frozen=True)
class BinContainment:
    vertex_count: int
    inside_pocket_xy: bool
    above_floor: bool
    below_rim: bool
    minimum_wall_clearance_m: float
    floor_clearance_m: float
    rim_clearance_m: float
    bounds_world_m: tuple[Point3, Point3]

    @property
    def passed(self) -> bool:
        return self.inside_pocket_xy and self.above_floor and self.below_rim

    def as_dict(self) -> dict[str, Any]:
        return {
            "vertex_count": self.vertex_count,
            "inside_pocket_xy": self.inside_pocket_xy,
            "above_floor": self.above_floor,
            "below_rim": self.below_rim,
            "minimum_wall_clearance_m": self.minimum_wall_clearance_m,
            "floor_clearance_m": self.floor_clearance_m,
            "rim_clearance_m": self.rim_clearance_m,
            "bounds_world_m": [list(bound) for bound in self.bounds_world_m],
            "passed": self.passed,
        }


def _parse_screw_spec(raw: Mapping[str, Any]) -> ScrewSpec:
    bounds_raw = raw.get("expected_bounds_mm")
    if not isinstance(bounds_raw, Sequence) or isinstance(bounds_raw, (str, bytes)):
        raise ScenarioError("screw.expected_bounds_mm must contain min and max")
    bounds = tuple(
        _triple(value, "screw.expected_bounds_mm") for value in bounds_raw
    )
    if len(bounds) != 2:
        raise ScenarioError("screw.expected_bounds_mm must contain min and max")
    spec = ScrewSpec(
        model_filename=str(raw.get("model_filename", "")),
        expected_sha256=str(raw.get("expected_sha256", "")).lower(),
        expected_triangles=int(raw.get("expected_triangles", 0)),
        expected_bounds_mm=(bounds[0], bounds[1]),
        bounds_tolerance_mm=_finite(
            raw.get("bounds_tolerance_mm"), "screw.bounds_tolerance_mm"
        ),
        mass_kg=_finite(raw.get("mass_kg"), "screw.mass_kg"),
        length_m=_finite(raw.get("length_m"), "screw.length_m"),
        head_diameter_m=_finite(
            raw.get("head_diameter_m"), "screw.head_diameter_m"
        ),
        shank_diameter_m=_finite(
            raw.get("shank_diameter_m"), "screw.shank_diameter_m"
        ),
        shank_start_from_head_m=_finite(
            raw.get("shank_start_from_head_m"),
            "screw.shank_start_from_head_m",
        ),
        grasp_offset_from_head_m=_finite(
            raw.get("grasp_offset_from_head_m"),
            "screw.grasp_offset_from_head_m",
        ),
        jaw_seating_margin_m=_finite(
            raw.get("jaw_seating_margin_m"), "screw.jaw_seating_margin_m"
        ),
        spawn_head_height_above_surface_m=_finite(
            raw.get("spawn_head_height_above_surface_m"),
            "screw.spawn_head_height_above_surface_m",
        ),
        placement_note=str(raw.get("placement_note", "")),
    )
    return spec


def _validate_screw_spec(spec: ScrewSpec) -> None:
    # These constants describe the exact separately supplied model.  Keeping
    # them independent of JSON prevents a config edit from blessing a different
    # mesh while still claiming to use the supplied screw.
    if spec.model_filename != SUPPLIED_SCREW_FILENAME:
        raise ScenarioError(f"screw model must be {SUPPLIED_SCREW_FILENAME}")
    if spec.expected_sha256 != SUPPLIED_SCREW_SHA256:
        raise ScenarioError("screw expected_sha256 does not identify the supplied STL")
    if spec.expected_triangles != SUPPLIED_SCREW_TRIANGLES:
        raise ScenarioError("supplied screw STL must contain 1910 triangles")
    if spec.bounds_tolerance_mm <= 0.0:
        raise ScenarioError("screw.bounds_tolerance_mm must be positive")
    for configured, supplied in zip(
        spec.expected_bounds_mm[0] + spec.expected_bounds_mm[1],
        SUPPLIED_SCREW_BOUNDS_MM[0] + SUPPLIED_SCREW_BOUNDS_MM[1],
    ):
        if not math.isclose(
            configured, supplied, abs_tol=spec.bounds_tolerance_mm, rel_tol=0.0
        ):
            raise ScenarioError("configured screw bounds do not match the supplied STL")
    positive = {
        "mass_kg": spec.mass_kg,
        "length_m": spec.length_m,
        "head_diameter_m": spec.head_diameter_m,
        "shank_diameter_m": spec.shank_diameter_m,
        "shank_start_from_head_m": spec.shank_start_from_head_m,
        "grasp_offset_from_head_m": spec.grasp_offset_from_head_m,
        "spawn_head_height_above_surface_m": spec.spawn_head_height_above_surface_m,
    }
    for name, value in positive.items():
        if value <= 0.0:
            raise ScenarioError(f"screw.{name} must be positive")
    if not math.isclose(spec.mass_kg, 0.0075, abs_tol=1e-12, rel_tol=0.0):
        raise ScenarioError("screw.mass_kg must preserve the 0.0075 kg CAD estimate")
    if not math.isclose(spec.length_m, 0.03, abs_tol=1e-12, rel_tol=0.0):
        raise ScenarioError("screw.length_m must match the supplied M6x30 model")
    if not math.isclose(
        spec.shank_diameter_m, 0.006, abs_tol=1e-12, rel_tol=0.0
    ):
        raise ScenarioError("screw.shank_diameter_m must match the M6 shank")
    if not (
        spec.shank_start_from_head_m
        < spec.grasp_offset_from_head_m
        < spec.length_m
    ):
        raise ScenarioError("screw grasp offset must lie on the 6 mm shank")
    if not 0.0 <= spec.jaw_seating_margin_m < spec.shank_diameter_m / 2.0:
        raise ScenarioError("screw jaw seating margin is outside the shank radius")
    if not spec.placement_note.strip():
        raise ScenarioError("screw placement frame convention must remain documented")


def load_scenario_config(path: str | Path | None = None) -> ScenarioConfig:
    """Load and validate the mixed scene and its inherited cube contract."""

    config_path = (
        DEFAULT_SCENARIO_CONFIG_PATH if path is None else Path(path).expanduser()
    ).resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScenarioError(f"cannot load mixed-scene config {config_path}: {exc}") from exc
    root = _mapping(raw, "mixed-scene config")
    schema_version = int(root.get("schema_version", 0))
    if schema_version != 1:
        raise ScenarioError(f"unsupported mixed-scene schema {schema_version}")

    base_name = str(root.get("base_colour_sort_config", ""))
    if not base_name:
        raise ScenarioError("base_colour_sort_config is required")
    base_path = (config_path.parent / base_name).resolve()
    cube_config = load_config(base_path)

    screw_spec = _parse_screw_spec(_mapping(root.get("screw"), "screw"))
    _validate_screw_spec(screw_spec)
    task = _mapping(root.get("task"), "task")
    screw_rows = task.get("screws")
    if not isinstance(screw_rows, Sequence) or isinstance(screw_rows, (str, bytes)):
        raise ScenarioError("task.screws must be a list")
    screws = tuple(
        ScrewInstance(
            item_id=str(_mapping(row, f"task.screws[{index}]").get("id", "")),
            spawn_head_base_m=_triple(
                _mapping(row, f"task.screws[{index}]").get("spawn_head_base_m"),
                f"task.screws[{index}].spawn_head_base_m",
            ),
            spawn_yaw_deg=_finite(
                _mapping(row, f"task.screws[{index}]").get("spawn_yaw_deg"),
                f"task.screws[{index}].spawn_yaw_deg",
            ),
            destination_colour=str(
                _mapping(row, f"task.screws[{index}]").get(
                    "destination_colour", ""
                )
            ).lower(),
            placement_grasp_point_offset_from_bin_centre_m=_triple(
                _mapping(row, f"task.screws[{index}]").get(
                    "placement_grasp_point_offset_from_bin_centre_m"
                ),
                (
                    f"task.screws[{index}]."
                    "placement_grasp_point_offset_from_bin_centre_m"
                ),
            ),
            target_axis_yaw_deg=_finite(
                _mapping(row, f"task.screws[{index}]").get(
                    "target_axis_yaw_deg"
                ),
                f"task.screws[{index}].target_axis_yaw_deg",
            ),
        )
        for index, row in enumerate(screw_rows)
    )
    if len(screws) != 2:
        raise ScenarioError("the mixed demonstration must configure exactly two screws")
    screw_ids = tuple(screw.item_id for screw in screws)
    if any(not item_id for item_id in screw_ids) or len(set(screw_ids)) != len(screw_ids):
        raise ScenarioError("screw IDs must be non-empty and unique")
    if any(screw.destination_colour != "red" for screw in screws):
        raise ScenarioError("both mixed-demo screws must map to the red bin")

    cube_ids = tuple(f"cube-{colour}" for colour in cube_config.task.colour_order)
    all_ids = cube_ids + screw_ids
    if len(set(all_ids)) != len(all_ids):
        raise ScenarioError("cube and screw IDs must be globally unique")
    order_raw = task.get("sort_order")
    if not isinstance(order_raw, Sequence) or isinstance(order_raw, (str, bytes)):
        raise ScenarioError("task.sort_order must be a list")
    sort_order = tuple(str(item_id) for item_id in order_raw)
    if len(sort_order) != len(all_ids) or set(sort_order) != set(all_ids):
        raise ScenarioError("task.sort_order must contain every item ID exactly once")

    x_bounds = cube_config.task.pickup_region_base_m["x"]
    y_bounds = cube_config.task.pickup_region_base_m["y"]
    surface_z = cube_config.frames.surface_base_z_m
    expected_spawn_z = surface_z + screw_spec.spawn_head_height_above_surface_m
    head_radius = screw_spec.head_diameter_m / 2.0
    for screw in screws:
        if not math.isclose(
            screw.spawn_head_base_m[2], expected_spawn_z, abs_tol=1e-9, rel_tol=0.0
        ):
            raise ScenarioError(
                f"{screw.item_id} head Z must equal surface plus configured head height"
            )
        yaw = math.radians(screw.spawn_yaw_deg)
        axis = (math.cos(yaw), math.sin(yaw))
        tip_xy = (
            screw.spawn_head_base_m[0] + axis[0] * screw_spec.length_m,
            screw.spawn_head_base_m[1] + axis[1] * screw_spec.length_m,
        )
        # Conservatively include the 12 mm head radius around both axial ends.
        if not all(
            x_bounds[0] + head_radius <= x <= x_bounds[1] - head_radius
            for x in (screw.spawn_head_base_m[0], tip_xy[0])
        ):
            raise ScenarioError(f"{screw.item_id} lies outside pickup-region x bounds")
        if not all(
            y_bounds[0] + head_radius <= y <= y_bounds[1] - head_radius
            for y in (screw.spawn_head_base_m[1], tip_xy[1])
        ):
            raise ScenarioError(f"{screw.item_id} lies outside pickup-region y bounds")
        placement_z = screw.placement_grasp_point_offset_from_bin_centre_m[2]
        if placement_z <= head_radius:
            raise ScenarioError(
                f"{screw.item_id} placement does not clear the bin floor"
            )
        if placement_z + head_radius >= cube_config.bin.pocket_depth_m:
            raise ScenarioError(f"{screw.item_id} placement reaches the bin rim")

    return ScenarioConfig(
        schema_version=schema_version,
        path=config_path,
        base_colour_sort_config_path=base_path,
        cube_config=cube_config,
        screw=screw_spec,
        screws=screws,
        sort_order=sort_order,
    )


def resolve_screw_model(
    value: str | Path | None = None,
    *,
    reference_assets: str | Path | None = None,
) -> Path:
    """Resolve the exact screw STL from an explicit path or reference bundle."""

    if value is not None:
        path = Path(value).expanduser().resolve()
    else:
        assets_value = (
            reference_assets
            if reference_assets is not None
            else os.environ.get("ROCKWELL_REFERENCE_ASSETS")
        )
        if not assets_value:
            raise ScenarioError(
                "ROCKWELL_REFERENCE_ASSETS is unset; pass --screw-model explicitly"
            )
        path = (
            Path(assets_value).expanduser().resolve()
            / "models"
            / SUPPLIED_SCREW_FILENAME
        )
    if not path.is_file():
        raise FileNotFoundError(f"Supplied screw model not found: {path}")
    return path


def load_stl_triangles(path: str | Path) -> tuple[Triangle, ...]:
    """Load binary or ASCII STL triangles without changing native millimetres."""

    stl_path = Path(path)
    data = stl_path.read_bytes()
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        if 84 + triangle_count * 50 == len(data):
            triangles: list[Triangle] = []
            for index in range(triangle_count):
                values = struct.unpack_from("<9f", data, 84 + index * 50 + 12)
                triangles.append(
                    (
                        (float(values[0]), float(values[1]), float(values[2])),
                        (float(values[3]), float(values[4]), float(values[5])),
                        (float(values[6]), float(values[7]), float(values[8])),
                    )
                )
            return tuple(triangles)

    vertices: list[Point3] = []
    try:
        text = data.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ScenarioError(f"unsupported or malformed STL: {stl_path}") from exc
    for raw_line in text.splitlines():
        fields = raw_line.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            vertices.append(
                (
                    _finite(fields[1], "STL vertex"),
                    _finite(fields[2], "STL vertex"),
                    _finite(fields[3], "STL vertex"),
                )
            )
    if not vertices or len(vertices) % 3:
        raise ScenarioError(f"unsupported or malformed STL: {stl_path}")
    return tuple(
        (vertices[index], vertices[index + 1], vertices[index + 2])
        for index in range(0, len(vertices), 3)
    )


def _bounds(vertices: Sequence[Point3]) -> tuple[Point3, Point3]:
    if not vertices:
        raise ScenarioError("cannot calculate bounds for zero vertices")
    return (
        tuple(min(vertex[axis] for vertex in vertices) for axis in range(3)),
        tuple(max(vertex[axis] for vertex in vertices) for axis in range(3)),
    )  # type: ignore[return-value]


def validate_screw_model(
    path: str | Path,
    spec: ScrewSpec | None = None,
) -> ValidatedScrewModel:
    """Fail unless ``path`` is the exact supplied M6x30 screw STL."""

    active_spec = load_scenario_config().screw if spec is None else spec
    _validate_screw_spec(active_spec)
    stl_path = Path(path).expanduser().resolve()
    data = stl_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != active_spec.expected_sha256:
        raise ScenarioError(
            f"screw STL SHA-256 mismatch: expected {active_spec.expected_sha256}, got {digest}"
        )
    triangles = load_stl_triangles(stl_path)
    if len(triangles) != active_spec.expected_triangles:
        raise ScenarioError(
            "screw STL triangle count mismatch: "
            f"expected {active_spec.expected_triangles}, got {len(triangles)}"
        )
    vertices = tuple(vertex for triangle in triangles for vertex in triangle)
    bounds = _bounds(vertices)
    for actual, expected in zip(
        bounds[0] + bounds[1],
        active_spec.expected_bounds_mm[0] + active_spec.expected_bounds_mm[1],
    ):
        if not math.isclose(
            actual,
            expected,
            abs_tol=active_spec.bounds_tolerance_mm,
            rel_tol=0.0,
        ):
            raise ScenarioError(
                f"unexpected screw STL bounds in millimetres: {bounds}"
            )
    return ValidatedScrewModel(
        path=stl_path,
        sha256=digest,
        triangles_mm=triangles,
        bounds_mm=bounds,
    )


def local_to_world_vertices(
    vertices_local_m: Iterable[Sequence[float]],
    *,
    head_world_m: Sequence[float],
    yaw_rad: float,
) -> tuple[Point3, ...]:
    """Transform lying-screw CAD points using ``Rz(yaw) * Ry(-90 deg)``.

    The STL origin is the head-face centre and its tip points along local -Z.
    After this transform local -Z points along ``(cos(yaw), sin(yaw), 0)`` and
    local +X points upward.  Inputs and outputs use metres.
    """

    head = _triple(head_world_m, "head_world_m")
    yaw = _finite(yaw_rad, "yaw_rad")
    sin_yaw = math.sin(yaw)
    cos_yaw = math.cos(yaw)
    transformed: list[Point3] = []
    for index, value in enumerate(vertices_local_m):
        local_x, local_y, local_z = _triple(value, f"vertices_local_m[{index}]")
        transformed.append(
            (
                head[0] - sin_yaw * local_y - cos_yaw * local_z,
                head[1] + cos_yaw * local_y - sin_yaw * local_z,
                head[2] + local_x,
            )
        )
    if not transformed:
        raise ScenarioError("vertices_local_m must not be empty")
    return tuple(transformed)


def screw_jaw_half_target(
    *,
    shank_diameter_m: float,
    finger_inner_tip_m: float,
    seating_margin_m: float,
    jaw_closed_m: float = 0.0,
    jaw_open_m: float | None = None,
) -> float:
    """Return one jaw's target from the shank radius and finger-tip geometry.

    ``jaw + finger_inner_tip`` is the modelled half-gap.  The seating margin
    commands a small additional closure, following the existing cube formula.
    The mechanical closed stop clamps the result; an optional open stop guards
    callers against inconsistent gripper geometry.
    """

    diameter = _finite(shank_diameter_m, "shank_diameter_m")
    inner_tip = _finite(finger_inner_tip_m, "finger_inner_tip_m")
    seating = _finite(seating_margin_m, "seating_margin_m")
    closed = _finite(jaw_closed_m, "jaw_closed_m")
    if diameter <= 0.0:
        raise ScenarioError("shank_diameter_m must be positive")
    if inner_tip < 0.0 or seating < 0.0 or closed < 0.0:
        raise ScenarioError("finger, seating and jaw values must be non-negative")
    if seating >= diameter / 2.0:
        raise ScenarioError("seating margin must be smaller than the shank radius")
    target = max(closed, diameter / 2.0 - inner_tip - seating)
    if jaw_open_m is not None:
        opened = _finite(jaw_open_m, "jaw_open_m")
        if opened <= closed:
            raise ScenarioError("jaw_open_m must be greater than jaw_closed_m")
        if target > opened:
            raise ScenarioError("screw jaw target exceeds the configured open stop")
    return target


def verify_vertices_inside_bin(
    vertices_world_m: Iterable[Sequence[float]],
    *,
    bin_centre_xy_m: Sequence[float],
    pocket_m: float,
    floor_z_m: float,
    rim_z_m: float,
    corner_radius_m: float = 0.0,
    tolerance_m: float = 1e-6,
) -> BinContainment:
    """Evaluate every transformed screw vertex against a rounded-square pocket."""

    vertices = tuple(
        _triple(value, f"vertices_world_m[{index}]")
        for index, value in enumerate(vertices_world_m)
    )
    if not vertices:
        raise ScenarioError("vertices_world_m must not be empty")
    try:
        bin_x, bin_y = tuple(
            _finite(value, "bin_centre_xy_m") for value in bin_centre_xy_m
        )
    except (TypeError, ValueError) as exc:
        raise ScenarioError("bin_centre_xy_m must contain two numbers") from exc
    pocket = _finite(pocket_m, "pocket_m")
    floor = _finite(floor_z_m, "floor_z_m")
    rim = _finite(rim_z_m, "rim_z_m")
    corner_radius = _finite(corner_radius_m, "corner_radius_m")
    tolerance = _finite(tolerance_m, "tolerance_m")
    if (
        pocket <= 0.0
        or rim <= floor
        or not 0.0 <= corner_radius <= pocket / 2.0
        or tolerance < 0.0
    ):
        raise ScenarioError("bin dimensions or tolerance are invalid")

    bounds = _bounds(vertices)
    half = pocket / 2.0
    wall_clearances = (
        bounds[0][0] - (bin_x - half),
        (bin_x + half) - bounds[1][0],
        bounds[0][1] - (bin_y - half),
        (bin_y + half) - bounds[1][1],
    )
    rounded_corner_centre = half - corner_radius

    def inside_rounded_pocket(vertex: Point3) -> bool:
        dx = abs(vertex[0] - bin_x)
        dy = abs(vertex[1] - bin_y)
        if dx > half + tolerance or dy > half + tolerance:
            return False
        if (
            corner_radius <= 0.0
            or dx <= rounded_corner_centre
            or dy <= rounded_corner_centre
        ):
            return True
        return math.hypot(
            dx - rounded_corner_centre, dy - rounded_corner_centre
        ) <= corner_radius + tolerance

    floor_clearance = bounds[0][2] - floor
    rim_clearance = rim - bounds[1][2]
    return BinContainment(
        vertex_count=len(vertices),
        inside_pocket_xy=all(inside_rounded_pocket(vertex) for vertex in vertices),
        above_floor=floor_clearance >= -tolerance,
        below_rim=rim_clearance >= -tolerance,
        minimum_wall_clearance_m=min(wall_clearances),
        floor_clearance_m=floor_clearance,
        rim_clearance_m=rim_clearance,
        bounds_world_m=bounds,
    )


def build_sort_items(config: ScenarioConfig) -> tuple[SortItem, ...]:
    """Build the configured five-item sequence with absolute grasp references."""

    by_id: dict[str, SortItem] = {}
    cube = config.cube_config
    floor_z = cube.frames.surface_base_z_m + cube.bin.floor_m
    for colour in cube.task.colour_order:
        spawn_x, spawn_y, yaw_deg = cube.task.cube_spawns_base_xy_yaw_deg[colour]
        spawn_centre = (spawn_x, spawn_y, cube.cube_centre_base_z_m)
        bin_x, bin_y = cube.task.bin_centres_base_m[colour]
        placement_centre = (bin_x, bin_y, floor_z + cube.cube_edge_m / 2.0)
        item_id = f"cube-{colour}"
        by_id[item_id] = SortItem(
            item_id=item_id,
            kind="cube",
            destination_colour=colour,
            spawn_reference_base_m=spawn_centre,
            spawn_grasp_point_base_m=spawn_centre,
            spawn_yaw_deg=yaw_deg,
            placement_reference_base_m=placement_centre,
            placement_grasp_point_base_m=placement_centre,
            target_axis_yaw_deg=0.0,
        )

    grasp_local = (config.screw.grasp_point_local_m,)
    for screw in config.screws:
        spawn_grasp = local_to_world_vertices(
            grasp_local,
            head_world_m=screw.spawn_head_base_m,
            yaw_rad=math.radians(screw.spawn_yaw_deg),
        )[0]
        bin_x, bin_y = cube.task.bin_centres_base_m[screw.destination_colour]
        offset_x, offset_y, offset_z = (
            screw.placement_grasp_point_offset_from_bin_centre_m
        )
        placement_grasp = (
            bin_x + offset_x,
            bin_y + offset_y,
            floor_z + offset_z,
        )
        target_yaw = math.radians(screw.target_axis_yaw_deg)
        target_axis = (math.cos(target_yaw), math.sin(target_yaw), 0.0)
        placement_head = (
            placement_grasp[0]
            - target_axis[0] * config.screw.grasp_offset_from_head_m,
            placement_grasp[1]
            - target_axis[1] * config.screw.grasp_offset_from_head_m,
            placement_grasp[2],
        )
        by_id[screw.item_id] = SortItem(
            item_id=screw.item_id,
            kind="screw",
            destination_colour=screw.destination_colour,
            spawn_reference_base_m=screw.spawn_head_base_m,
            spawn_grasp_point_base_m=spawn_grasp,
            spawn_yaw_deg=screw.spawn_yaw_deg,
            placement_reference_base_m=placement_head,
            placement_grasp_point_base_m=placement_grasp,
            target_axis_yaw_deg=screw.target_axis_yaw_deg,
        )
    return tuple(by_id[item_id] for item_id in config.sort_order)


def build_completion_report(
    config: ScenarioConfig,
    events: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build dynamic denominators and completion counts for a mixed run."""

    items = build_sort_items(config)
    by_id = {item.item_id: item for item in items}
    event_rows = tuple(events)
    observed_ids: list[str] = []
    verified_pickups = 0
    verified_placements = 0
    placed_by_kind = {kind: 0 for kind in sorted({item.kind for item in items})}
    placed_by_destination = {
        colour: 0 for colour in config.cube_config.task.colour_order
    }
    for index, event in enumerate(event_rows):
        if not isinstance(event, Mapping):
            raise ScenarioError(f"events[{index}] must be an object")
        item_id = str(event.get("item_id", ""))
        if item_id not in by_id:
            raise ScenarioError(f"events[{index}] has unknown item_id {item_id!r}")
        if item_id in observed_ids:
            raise ScenarioError(f"events contain duplicate item_id {item_id!r}")
        observed_ids.append(item_id)
        pickup = bool(event.get("pickup_verified", False))
        placement = bool(event.get("placement_verified", False))
        if placement and not pickup:
            raise ScenarioError(f"{item_id} placement cannot pass when pickup failed")
        verified_pickups += int(pickup)
        verified_placements += int(placement)
        if placement:
            item = by_id[item_id]
            placed_by_kind[item.kind] += 1
            placed_by_destination[item.destination_colour] += 1

    expected_by_kind = {
        kind: sum(item.kind == kind for item in items)
        for kind in sorted({item.kind for item in items})
    }
    expected_by_destination = {
        colour: sum(item.destination_colour == colour for item in items)
        for colour in config.cube_config.task.colour_order
    }
    expected_ids = [item.item_id for item in items]
    sequence_matches_plan = observed_ids == expected_ids[: len(observed_ids)]
    runtime_sequence_complete = (
        sequence_matches_plan and len(observed_ids) == len(expected_ids)
    )
    task_complete = (
        runtime_sequence_complete and verified_placements == len(expected_ids)
    )
    return {
        "expected_items": len(expected_ids),
        "expected_item_ids": expected_ids,
        "expected_by_kind": expected_by_kind,
        "expected_by_destination": expected_by_destination,
        "processed_items": len(observed_ids),
        "processed_item_ids": observed_ids,
        "sequence_matches_plan": sequence_matches_plan,
        "verified_pickups": verified_pickups,
        "verified_placements": verified_placements,
        "verified_placements_by_kind": placed_by_kind,
        "verified_placements_by_destination": placed_by_destination,
        "runtime_sequence_complete": runtime_sequence_complete,
        "task_complete": task_complete,
        "completion_summary": (
            f"pickups={verified_pickups}/{len(expected_ids)} "
            f"placements={verified_placements}/{len(expected_ids)}"
        ),
    }
