"""Immutable, replayable scene manifests for the four-combination benchmark.

The manifest is deliberately independent of Isaac Sim.  One manifest is the
experimental condition and must be replayed unchanged for deterministic/YOLO
vision crossed with deterministic/AI grasp selection.  Simulator truth in this
file describes only the initial condition and later evaluation labels; runtime
providers must still make decisions from camera observations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
GENERATOR_ID = "experiments.randomized_sort_benchmark.manifest/v1"
SURFACE_BASE_Z_M = -0.0051
CUBE_EDGE_M = 0.010
SCREW_LENGTH_M = 0.030
SCREW_HEAD_DIAMETER_M = 0.012
SCREW_HEAD_HEIGHT_ABOVE_SURFACE_M = 0.0065
DEFAULT_MINIMUM_SPAWN_CLEARANCE_M = 0.0005
DEFAULT_SCREW_PROBABILITY = 0.35
SUPPORTED_CUBE_COLOURS = ("red", "yellow", "blue")
SUPPORTED_CLASSES = ("cube_10mm", "screw_iso10642_m6x30")


class BenchmarkManifestError(ValueError):
    """Raised when a benchmark condition is invalid or cannot be replayed."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkManifestError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise BenchmarkManifestError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise BenchmarkManifestError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkManifestError(f"{name} must be an integer") from exc
    if result != value:
        raise BenchmarkManifestError(f"{name} must be an integer")
    return result


def _pair(value: Any, name: str) -> tuple[float, float]:
    try:
        values = tuple(_finite(entry, name) for entry in value)
    except TypeError as exc:
        raise BenchmarkManifestError(f"{name} must contain two values") from exc
    if len(values) != 2:
        raise BenchmarkManifestError(f"{name} must contain two values")
    return values


def _triple(value: Any, name: str) -> tuple[float, float, float]:
    try:
        values = tuple(_finite(entry, name) for entry in value)
    except TypeError as exc:
        raise BenchmarkManifestError(f"{name} must contain three values") from exc
    if len(values) != 3:
        raise BenchmarkManifestError(f"{name} must contain three values")
    return values


def _derive_seed(base_seed: int, namespace: str) -> int:
    digest = hashlib.sha256(
        f"rockwell-randomized-sort-v1:{base_seed}:{namespace}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


@dataclass(frozen=True)
class PickupRegion:
    """Rectangular robot-base area in which complete object footprints must fit."""

    x_m: tuple[float, float] = (0.30, 0.55)
    y_m: tuple[float, float] = (-0.30, 0.30)

    def validate(self) -> "PickupRegion":
        if not all(math.isfinite(value) for value in (*self.x_m, *self.y_m)):
            raise BenchmarkManifestError("pickup-region bounds must be finite")
        if self.x_m[0] >= self.x_m[1] or self.y_m[0] >= self.y_m[1]:
            raise BenchmarkManifestError("pickup-region bounds must be increasing")
        return self

    def as_dict(self) -> dict[str, list[float]]:
        return {"x_m": list(self.x_m), "y_m": list(self.y_m)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PickupRegion":
        return cls(
            x_m=_pair(value.get("x_m"), "pickup_region.x_m"),
            y_m=_pair(value.get("y_m"), "pickup_region.y_m"),
        ).validate()


@dataclass(frozen=True)
class SeedBundle:
    """Independent deterministic seeds shared by all four policy cells."""

    condition_seed: int
    scene_seed: int
    appearance_seed: int
    physics_seed: int
    sensor_seed: int

    def validate(self) -> "SeedBundle":
        for name, value in self.as_dict().items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise BenchmarkManifestError(f"seed_bundle.{name} must be non-negative")
        return self

    def as_dict(self) -> dict[str, int]:
        return {
            "condition_seed": self.condition_seed,
            "scene_seed": self.scene_seed,
            "appearance_seed": self.appearance_seed,
            "physics_seed": self.physics_seed,
            "sensor_seed": self.sensor_seed,
        }

    @classmethod
    def for_condition(cls, condition_seed: int) -> "SeedBundle":
        if isinstance(condition_seed, bool) or not isinstance(condition_seed, int):
            raise BenchmarkManifestError("condition_seed must be an integer")
        if condition_seed < 0:
            raise BenchmarkManifestError("condition_seed must be non-negative")
        return cls(
            condition_seed=condition_seed,
            scene_seed=_derive_seed(condition_seed, "scene"),
            appearance_seed=_derive_seed(condition_seed, "appearance"),
            physics_seed=_derive_seed(condition_seed, "physics"),
            sensor_seed=_derive_seed(condition_seed, "sensor"),
        ).validate()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SeedBundle":
        result = cls(
            condition_seed=_integer(value.get("condition_seed"), "condition_seed"),
            scene_seed=_integer(value.get("scene_seed"), "scene_seed"),
            appearance_seed=_integer(value.get("appearance_seed"), "appearance_seed"),
            physics_seed=_integer(value.get("physics_seed"), "physics_seed"),
            sensor_seed=_integer(value.get("sensor_seed"), "sensor_seed"),
        ).validate()
        expected = cls.for_condition(result.condition_seed)
        if result != expected:
            raise BenchmarkManifestError(
                "seed bundle does not match the versioned derivation from condition_seed"
            )
        return result


@dataclass(frozen=True)
class BenchmarkObject:
    """One initial object pose in the robot-base frame."""

    instance_id: str
    class_name: str
    colour: str
    destination_colour: str
    pose_reference: str
    position_base_m: tuple[float, float, float]
    yaw_deg: float
    dimensions_m: tuple[float, float, float]
    yaw_symmetry_period_deg: float
    local_axis_convention: str

    @property
    def footprint(self) -> tuple[tuple[float, float], float, float, float]:
        """Return OBB centre, half-length, half-width and yaw in radians."""

        yaw = math.radians(self.yaw_deg)
        if self.class_name == "cube_10mm":
            return (
                (self.position_base_m[0], self.position_base_m[1]),
                CUBE_EDGE_M / 2.0,
                CUBE_EDGE_M / 2.0,
                yaw,
            )
        axis = (math.cos(yaw), math.sin(yaw))
        centre = (
            self.position_base_m[0] + axis[0] * SCREW_LENGTH_M / 2.0,
            self.position_base_m[1] + axis[1] * SCREW_LENGTH_M / 2.0,
        )
        return (centre, SCREW_LENGTH_M / 2.0, SCREW_HEAD_DIAMETER_M / 2.0, yaw)

    def validate(self) -> "BenchmarkObject":
        if not self.instance_id.strip():
            raise BenchmarkManifestError("object instance_id must not be empty")
        if self.class_name not in SUPPORTED_CLASSES:
            raise BenchmarkManifestError(f"unsupported class {self.class_name!r}")
        if self.destination_colour not in SUPPORTED_CUBE_COLOURS:
            raise BenchmarkManifestError("destination must be red, yellow, or blue")
        if not all(math.isfinite(value) for value in self.position_base_m):
            raise BenchmarkManifestError(f"{self.instance_id} position must be finite")
        if not math.isfinite(self.yaw_deg) or not 0.0 <= self.yaw_deg < 360.0:
            raise BenchmarkManifestError(f"{self.instance_id} yaw must be in [0, 360)")
        if any(value <= 0.0 or not math.isfinite(value) for value in self.dimensions_m):
            raise BenchmarkManifestError(f"{self.instance_id} dimensions must be positive")
        if self.class_name == "cube_10mm":
            if self.colour not in SUPPORTED_CUBE_COLOURS:
                raise BenchmarkManifestError("cube colour must be red, yellow, or blue")
            if self.destination_colour != self.colour:
                raise BenchmarkManifestError("a cube must map to its matching destination")
            if self.pose_reference != "centre":
                raise BenchmarkManifestError("cube pose reference must be centre")
            if self.dimensions_m != (CUBE_EDGE_M,) * 3:
                raise BenchmarkManifestError("cube dimensions must be exactly 10 mm")
            if not math.isclose(self.yaw_symmetry_period_deg, 90.0):
                raise BenchmarkManifestError("cube yaw symmetry must be 90 degrees")
            expected_z = SURFACE_BASE_Z_M + CUBE_EDGE_M / 2.0
        else:
            if self.colour != "metal" or self.destination_colour != "red":
                raise BenchmarkManifestError("every supplied screw must map to red")
            if self.pose_reference != "head_centre":
                raise BenchmarkManifestError("screw pose reference must be head_centre")
            if self.dimensions_m != (
                SCREW_LENGTH_M,
                SCREW_HEAD_DIAMETER_M,
                SCREW_HEAD_DIAMETER_M,
            ):
                raise BenchmarkManifestError("screw dimensions do not match M6x30 CAD")
            if not math.isclose(self.yaw_symmetry_period_deg, 360.0):
                raise BenchmarkManifestError("directed screw yaw period must be 360 degrees")
            expected_z = SURFACE_BASE_Z_M + SCREW_HEAD_HEIGHT_ABOVE_SURFACE_M
        if not math.isclose(self.position_base_m[2], expected_z, abs_tol=1e-9):
            raise BenchmarkManifestError(f"{self.instance_id} has an invalid spawn height")
        if not self.local_axis_convention.strip():
            raise BenchmarkManifestError("local axis convention must be documented")
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "class_name": self.class_name,
            "colour": self.colour,
            "destination_colour": self.destination_colour,
            "pose_reference": self.pose_reference,
            "position_base_m": list(self.position_base_m),
            "yaw_deg": self.yaw_deg,
            "dimensions_m": list(self.dimensions_m),
            "yaw_symmetry_period_deg": self.yaw_symmetry_period_deg,
            "local_axis_convention": self.local_axis_convention,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkObject":
        return cls(
            instance_id=str(value.get("instance_id", "")),
            class_name=str(value.get("class_name", "")),
            colour=str(value.get("colour", "")),
            destination_colour=str(value.get("destination_colour", "")),
            pose_reference=str(value.get("pose_reference", "")),
            position_base_m=_triple(value.get("position_base_m"), "position_base_m"),
            yaw_deg=_finite(value.get("yaw_deg"), "yaw_deg"),
            dimensions_m=_triple(value.get("dimensions_m"), "dimensions_m"),
            yaw_symmetry_period_deg=_finite(
                value.get("yaw_symmetry_period_deg"), "yaw_symmetry_period_deg"
            ),
            local_axis_convention=str(value.get("local_axis_convention", "")),
        ).validate()


def _obb_corners(obj: BenchmarkObject) -> tuple[tuple[float, float], ...]:
    centre, half_length, half_width, yaw = obj.footprint
    forward = (math.cos(yaw), math.sin(yaw))
    side = (-forward[1], forward[0])
    return tuple(
        (
            centre[0] + sign_l * half_length * forward[0] + sign_w * half_width * side[0],
            centre[1] + sign_l * half_length * forward[1] + sign_w * half_width * side[1],
        )
        for sign_l in (-1.0, 1.0)
        for sign_w in (-1.0, 1.0)
    )


def _inside_region(obj: BenchmarkObject, region: PickupRegion) -> bool:
    return all(
        region.x_m[0] <= x <= region.x_m[1]
        and region.y_m[0] <= y <= region.y_m[1]
        for x, y in _obb_corners(obj)
    )


def _obb_overlaps(
    left: BenchmarkObject,
    right: BenchmarkObject,
    *,
    minimum_clearance_m: float,
) -> bool:
    """Separating-axis test for conservative rectangular object footprints."""

    left_centre, left_l, left_w, left_yaw = left.footprint
    right_centre, right_l, right_w, right_yaw = right.footprint
    left_axes = (
        (math.cos(left_yaw), math.sin(left_yaw)),
        (-math.sin(left_yaw), math.cos(left_yaw)),
    )
    right_axes = (
        (math.cos(right_yaw), math.sin(right_yaw)),
        (-math.sin(right_yaw), math.cos(right_yaw)),
    )
    delta = (right_centre[0] - left_centre[0], right_centre[1] - left_centre[1])
    padding = minimum_clearance_m / 2.0
    for axis in (*left_axes, *right_axes):
        left_radius = (
            (left_l + padding) * abs(left_axes[0][0] * axis[0] + left_axes[0][1] * axis[1])
            + (left_w + padding)
            * abs(left_axes[1][0] * axis[0] + left_axes[1][1] * axis[1])
        )
        right_radius = (
            (right_l + padding)
            * abs(right_axes[0][0] * axis[0] + right_axes[0][1] * axis[1])
            + (right_w + padding)
            * abs(right_axes[1][0] * axis[0] + right_axes[1][1] * axis[1])
        )
        separation = abs(delta[0] * axis[0] + delta[1] * axis[1])
        if separation > left_radius + right_radius:
            return False
    return True


@dataclass(frozen=True)
class BenchmarkManifest:
    """One matched experimental condition, independent of a policy cell."""

    condition_id: str
    seeds: SeedBundle
    pickup_region: PickupRegion
    minimum_spawn_clearance_m: float
    objects: tuple[BenchmarkObject, ...]
    schema_version: int = SCHEMA_VERSION
    generator: str = GENERATOR_ID

    def validate(self) -> "BenchmarkManifest":
        if self.schema_version != SCHEMA_VERSION or self.generator != GENERATOR_ID:
            raise BenchmarkManifestError("unsupported benchmark manifest version")
        if not self.condition_id.strip():
            raise BenchmarkManifestError("condition_id must not be empty")
        self.seeds.validate()
        self.pickup_region.validate()
        if (
            not math.isfinite(self.minimum_spawn_clearance_m)
            or self.minimum_spawn_clearance_m < 0.0
        ):
            raise BenchmarkManifestError("minimum spawn clearance must be non-negative")
        if not 1 <= len(self.objects) <= 10:
            raise BenchmarkManifestError("a condition must contain 1-10 objects")
        ids: set[str] = set()
        for obj in self.objects:
            obj.validate()
            if obj.instance_id in ids:
                raise BenchmarkManifestError(f"duplicate object ID {obj.instance_id}")
            ids.add(obj.instance_id)
            if not _inside_region(obj, self.pickup_region):
                raise BenchmarkManifestError(
                    f"{obj.instance_id} footprint leaves the pickup region"
                )
        for index, left in enumerate(self.objects):
            for right in self.objects[index + 1 :]:
                if _obb_overlaps(
                    left,
                    right,
                    minimum_clearance_m=self.minimum_spawn_clearance_m,
                ):
                    raise BenchmarkManifestError(
                        f"initial footprints overlap: {left.instance_id}, {right.instance_id}"
                    )
        return self

    @property
    def object_count(self) -> int:
        return len(self.objects)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generator": self.generator,
            "condition_id": self.condition_id,
            "seed_bundle": self.seeds.as_dict(),
            "pickup_region_base_m": self.pickup_region.as_dict(),
            "minimum_spawn_clearance_m": self.minimum_spawn_clearance_m,
            "object_count": self.object_count,
            "objects": [obj.as_dict() for obj in self.objects],
            "runtime_input_warning": (
                "Initial simulator truth is condition/evaluator data only; runtime "
                "vision and grasp decisions must use camera-derived SceneObservation."
            ),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.as_dict())

    def envelope(self) -> dict[str, Any]:
        return {"manifest": self.as_dict(), "manifest_sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkManifest":
        objects_raw = value.get("objects")
        if not isinstance(objects_raw, Sequence) or isinstance(objects_raw, (str, bytes)):
            raise BenchmarkManifestError("objects must be a list")
        seed_raw = value.get("seed_bundle")
        region_raw = value.get("pickup_region_base_m")
        if not isinstance(seed_raw, Mapping) or not isinstance(region_raw, Mapping):
            raise BenchmarkManifestError("seed bundle and pickup region are required")
        result = cls(
            schema_version=_integer(value.get("schema_version"), "schema_version"),
            generator=str(value.get("generator", "")),
            condition_id=str(value.get("condition_id", "")),
            seeds=SeedBundle.from_dict(seed_raw),
            pickup_region=PickupRegion.from_dict(region_raw),
            minimum_spawn_clearance_m=_finite(
                value.get("minimum_spawn_clearance_m"),
                "minimum_spawn_clearance_m",
            ),
            objects=tuple(BenchmarkObject.from_dict(row) for row in objects_raw),
        ).validate()
        stated_count = _integer(value.get("object_count"), "object_count")
        if stated_count != result.object_count:
            raise BenchmarkManifestError("object_count does not match objects")
        return result


@dataclass(frozen=True)
class BenchmarkSuite:
    """A pre-registered set of matched conditions with count-stratified coverage."""

    base_seed: int
    trials_per_object_count: int
    conditions: tuple[BenchmarkManifest, ...]

    def validate(self) -> "BenchmarkSuite":
        if self.trials_per_object_count < 1:
            raise BenchmarkManifestError("trials_per_object_count must be positive")
        expected = self.trials_per_object_count * 10
        if len(self.conditions) != expected:
            raise BenchmarkManifestError(f"suite must contain exactly {expected} conditions")
        ids = {condition.condition_id for condition in self.conditions}
        hashes = {condition.sha256 for condition in self.conditions}
        if len(ids) != expected or len(hashes) != expected:
            raise BenchmarkManifestError("suite condition IDs and manifests must be unique")
        counts = {count: 0 for count in range(1, 11)}
        for condition in self.conditions:
            condition.validate()
            counts[condition.object_count] += 1
        if any(value != self.trials_per_object_count for value in counts.values()):
            raise BenchmarkManifestError("suite object-count strata are incomplete")
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "generator": GENERATOR_ID,
            "base_seed": self.base_seed,
            "trials_per_object_count": self.trials_per_object_count,
            "condition_count": len(self.conditions),
            "condition_manifest_sha256": [condition.sha256 for condition in self.conditions],
            "conditions": [condition.as_dict() for condition in self.conditions],
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.as_dict())


def _make_candidate(
    *, rng: random.Random, index: int, force_class: str | None = None
) -> BenchmarkObject:
    class_name = force_class or (
        "screw_iso10642_m6x30"
        if rng.random() < DEFAULT_SCREW_PROBABILITY
        else "cube_10mm"
    )
    if class_name == "cube_10mm":
        colour = rng.choice(SUPPORTED_CUBE_COLOURS)
        return BenchmarkObject(
            instance_id=f"object-{index:02d}-cube-{colour}",
            class_name=class_name,
            colour=colour,
            destination_colour=colour,
            pose_reference="centre",
            position_base_m=(
                rng.uniform(0.30, 0.55),
                rng.uniform(-0.30, 0.30),
                SURFACE_BASE_Z_M + CUBE_EDGE_M / 2.0,
            ),
            # Cube orientation is sampled over its unique physical period.
            yaw_deg=rng.uniform(0.0, 90.0),
            dimensions_m=(CUBE_EDGE_M,) * 3,
            yaw_symmetry_period_deg=90.0,
            local_axis_convention=(
                "+X/+Y follow the square sides; +Z points away from the table"
            ),
        )
    if class_name != "screw_iso10642_m6x30":
        raise BenchmarkManifestError(f"cannot generate unsupported class {class_name}")
    return BenchmarkObject(
        instance_id=f"object-{index:02d}-screw",
        class_name=class_name,
        colour="metal",
        destination_colour="red",
        pose_reference="head_centre",
        position_base_m=(
            rng.uniform(0.30, 0.55),
            rng.uniform(-0.30, 0.30),
            SURFACE_BASE_Z_M + SCREW_HEAD_HEIGHT_ABOVE_SURFACE_M,
        ),
        yaw_deg=rng.uniform(0.0, 360.0),
        dimensions_m=(SCREW_LENGTH_M, SCREW_HEAD_DIAMETER_M, SCREW_HEAD_DIAMETER_M),
        yaw_symmetry_period_deg=360.0,
        local_axis_convention=(
            "position is countersunk-head centre; yaw 0 directs CAD local -Z/head-to-tip along base +X"
        ),
    )


def generate_condition(
    condition_seed: int,
    *,
    object_count: int | None = None,
    pickup_region: PickupRegion = PickupRegion(),
    minimum_spawn_clearance_m: float = DEFAULT_MINIMUM_SPAWN_CLEARANCE_M,
    maximum_attempts_per_object: int = 10_000,
) -> BenchmarkManifest:
    """Generate one deterministic nonpenetrating condition from a single seed."""

    seeds = SeedBundle.for_condition(condition_seed)
    pickup_region.validate()
    if object_count is not None:
        object_count = _integer(object_count, "object_count")
        if not 1 <= object_count <= 10:
            raise BenchmarkManifestError("object_count must be in [1, 10]")
    if maximum_attempts_per_object < 1:
        raise BenchmarkManifestError("maximum_attempts_per_object must be positive")
    if minimum_spawn_clearance_m < 0.0 or not math.isfinite(minimum_spawn_clearance_m):
        raise BenchmarkManifestError("minimum spawn clearance must be non-negative")

    rng = random.Random(seeds.scene_seed)
    selected_count = rng.randint(1, 10) if object_count is None else object_count
    objects: list[BenchmarkObject] = []
    for index in range(selected_count):
        for _attempt in range(maximum_attempts_per_object):
            candidate = _make_candidate(rng=rng, index=index)
            if not _inside_region(candidate, pickup_region):
                continue
            if any(
                _obb_overlaps(
                    candidate,
                    existing,
                    minimum_clearance_m=minimum_spawn_clearance_m,
                )
                for existing in objects
            ):
                continue
            objects.append(candidate)
            break
        else:
            raise BenchmarkManifestError(
                f"could not place object {index + 1}/{selected_count} after "
                f"{maximum_attempts_per_object} attempts"
            )

    digest = hashlib.sha256(
        f"{GENERATOR_ID}:{condition_seed}:{selected_count}".encode("utf-8")
    ).hexdigest()[:12]
    return BenchmarkManifest(
        condition_id=f"condition-n{selected_count:02d}-{digest}",
        seeds=seeds,
        pickup_region=pickup_region,
        minimum_spawn_clearance_m=minimum_spawn_clearance_m,
        objects=tuple(objects),
    ).validate()


def generate_matched_suite(
    base_seed: int,
    *,
    trials_per_object_count: int = 4,
    pickup_region: PickupRegion = PickupRegion(),
) -> BenchmarkSuite:
    """Generate equal coverage of object counts 1 through 10.

    Four trials per count yields the planned 40 conditions.  Each condition is
    later evaluated in all four policy cells for 160 end-to-end runs.
    """

    base_seed = _integer(base_seed, "base_seed")
    trials_per_object_count = _integer(
        trials_per_object_count, "trials_per_object_count"
    )
    if base_seed < 0 or trials_per_object_count < 1:
        raise BenchmarkManifestError("suite seeds/counts must be positive")
    conditions: list[BenchmarkManifest] = []
    for count in range(1, 11):
        for trial_index in range(trials_per_object_count):
            condition_seed = _derive_seed(base_seed, f"count-{count}:trial-{trial_index}")
            conditions.append(
                generate_condition(
                    condition_seed,
                    object_count=count,
                    pickup_region=pickup_region,
                )
            )
    return BenchmarkSuite(
        base_seed=base_seed,
        trials_per_object_count=trials_per_object_count,
        conditions=tuple(conditions),
    ).validate()


def write_manifest(path: str | Path, manifest: BenchmarkManifest) -> None:
    """Atomically write a hash-wrapped condition manifest."""

    manifest.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(_canonical_json_bytes(manifest.envelope()))
    temporary.replace(destination)


def read_manifest(path: str | Path) -> BenchmarkManifest:
    """Load and hash-check a condition so edited inputs cannot be compared."""

    try:
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkManifestError(f"cannot read benchmark manifest: {exc}") from exc
    if not isinstance(envelope, Mapping) or not isinstance(
        envelope.get("manifest"), Mapping
    ):
        raise BenchmarkManifestError("manifest envelope is invalid")
    manifest = BenchmarkManifest.from_dict(envelope["manifest"])
    expected = str(envelope.get("manifest_sha256", ""))
    if expected != manifest.sha256:
        raise BenchmarkManifestError(
            f"manifest SHA-256 mismatch: expected {expected}, calculated {manifest.sha256}"
        )
    return manifest


__all__ = [
    "BenchmarkManifest",
    "BenchmarkManifestError",
    "BenchmarkObject",
    "BenchmarkSuite",
    "DEFAULT_MINIMUM_SPAWN_CLEARANCE_M",
    "GENERATOR_ID",
    "PickupRegion",
    "SCHEMA_VERSION",
    "SeedBundle",
    "generate_condition",
    "generate_matched_suite",
    "read_manifest",
    "write_manifest",
]
