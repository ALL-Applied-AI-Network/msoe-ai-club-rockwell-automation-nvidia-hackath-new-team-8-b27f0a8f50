"""Host-safe validation helpers for simulation camera and task geometry.

The functions in this module deliberately accept plain mappings and import no
Isaac or USD packages.  An Isaac entry point can therefore reduce live stage
readback to JSON-like values, validate those values here, and store the result
in a run report.  Host tests can exercise the same validation without a GPU.

Two kinds of provenance are kept explicit:

* camera projection and lens distortion must name the calibration record that
  supplied them; and
* geometry dimensions and silhouettes must name the source asset/profile that
  supplied them.

Matching extents alone does not prove that a mesh has the correct silhouette.
Callers should use an immutable asset hash or a versioned mesh fingerprint for
``silhouette_provenance_id`` and separately retain rendered-mask tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


class SimulationValidationError(ValueError):
    """Raised when simulation readback disagrees with its declared source."""


def isaac_camera_config_from_calibration(
    calibration: Mapping[str, Any],
    *,
    parent: str,
    horizontal_aperture_mm: float,
    parent_origin_offset_m: Sequence[Any] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Convert the physical +Z-forward calibration into an Isaac look-at camera.

    The checked-in calibration matrices use conventional column vectors with
    camera ``+X`` right, ``+Y`` down and ``+Z`` forward. USD cameras use ``+X``
    right, ``+Y`` up and look down local ``-Z``. The returned eye/look-at/up
    values make that axis conversion explicit instead of copying rounded scene
    constants. ``parent_origin_offset_m`` is used for base-frame cameras whose
    USD world origin differs from the robot-base origin; wrist/flange cameras
    leave it at zero.
    """

    if not isinstance(calibration, Mapping):
        raise SimulationValidationError("camera calibration must be a mapping")
    parent_name = _nonempty_text(parent, "parent")
    aperture = _positive_float(horizontal_aperture_mm, "horizontal_aperture_mm")
    expectation = CameraCalibrationExpectation.from_calibration_mapping(
        "camera", calibration, provenance_id="conversion-input", include_extrinsics=True
    )
    if expectation.camera_to_parent_row_major is None:
        raise SimulationValidationError("camera calibration has no extrinsic")
    offset = tuple(
        _finite_float(value, f"parent_origin_offset_m[{index}]")
        for index, value in enumerate(parent_origin_offset_m)
    )
    if len(offset) != 3:
        raise SimulationValidationError("parent_origin_offset_m must contain 3 values")
    transform = expectation.camera_to_parent_row_major
    eye = tuple(transform[index][3] + offset[index] for index in range(3))
    forward = tuple(transform[index][2] for index in range(3))
    up = tuple(-transform[index][1] for index in range(3))
    for label, vector in (("forward", forward), ("up", up)):
        norm = math.sqrt(sum(value * value for value in vector))
        if abs(norm - 1.0) > 1e-4:
            raise SimulationValidationError(
                f"calibration {label} axis is not unit length"
            )
    if abs(sum(a * b for a, b in zip(forward, up))) > 1e-4:
        raise SimulationValidationError("calibration forward/up axes are not orthogonal")
    width, height = expectation.resolution
    return {
        "parent": parent_name,
        "eye": eye,
        "look_at": tuple(eye[index] + forward[index] for index in range(3)),
        "up": up,
        "resolution": (width, height),
        "focal_length": expectation.focal_px * aperture / width,
        "horiz_aperture": aperture,
        "fisheye": False,
    }


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SimulationValidationError(f"{label} must be a number") from exc
    if not math.isfinite(result):
        raise SimulationValidationError(f"{label} must be finite")
    return result


def _positive_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise SimulationValidationError(f"{label} must be positive")
    return result


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SimulationValidationError(f"{label} must be a non-empty string")
    return value.strip()


def _resolution(value: Any, label: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SimulationValidationError(f"{label} must contain width and height")
    if len(value) != 2:
        raise SimulationValidationError(f"{label} must contain width and height")
    result: list[int] = []
    for axis, item in zip(("width", "height"), value):
        if isinstance(item, bool):
            raise SimulationValidationError(f"{label} {axis} must be an integer")
        try:
            integer = int(item)
        except (TypeError, ValueError) as exc:
            raise SimulationValidationError(
                f"{label} {axis} must be an integer"
            ) from exc
        if integer != item or integer <= 0:
            raise SimulationValidationError(
                f"{label} {axis} must be a positive integer"
            )
        result.append(integer)
    return (result[0], result[1])


def _matrix4(value: Any, label: str) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SimulationValidationError(f"{label} must be a 4x4 matrix")
    if len(value) != 4:
        raise SimulationValidationError(f"{label} must be a 4x4 matrix")
    rows: list[tuple[float, ...]] = []
    for row_index, row in enumerate(value):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise SimulationValidationError(f"{label} must be a 4x4 matrix")
        if len(row) != 4:
            raise SimulationValidationError(f"{label} must be a 4x4 matrix")
        rows.append(
            tuple(
                _finite_float(item, f"{label}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row)
            )
        )
    expected_last_row = (0.0, 0.0, 0.0, 1.0)
    if any(abs(rows[3][index] - expected_last_row[index]) > 1e-9 for index in range(4)):
        raise SimulationValidationError(
            f"{label} must be a rigid-transform homogeneous matrix"
        )
    return tuple(rows)


def _transform_errors(
    expected: tuple[tuple[float, ...], ...],
    measured: tuple[tuple[float, ...], ...],
) -> tuple[float, float]:
    translation_error_m = math.sqrt(
        sum((measured[index][3] - expected[index][3]) ** 2 for index in range(3))
    )

    # Relative rotation R_expected^T R_measured.  The trace identity gives the
    # shortest angular separation for proper 3D rotation matrices.
    relative_trace = 0.0
    for diagonal in range(3):
        relative_trace += sum(
            expected[row][diagonal] * measured[row][diagonal] for row in range(3)
        )
    cosine = max(-1.0, min(1.0, (relative_trace - 1.0) / 2.0))
    rotation_error_deg = math.degrees(math.acos(cosine))
    return translation_error_m, rotation_error_deg


@dataclass(frozen=True)
class CameraCalibrationExpectation:
    """Expected camera record derived from one named calibration entry."""

    camera_id: str
    resolution: tuple[int, int]
    focal_px: float
    radial_k1: float
    provenance_id: str
    extrinsic_frame: str | None = None
    camera_to_parent_row_major: tuple[tuple[float, ...], ...] | None = None

    @classmethod
    def from_calibration_mapping(
        cls,
        camera_id: str,
        calibration: Mapping[str, Any],
        *,
        provenance_id: str,
        include_extrinsics: bool = True,
    ) -> "CameraCalibrationExpectation":
        """Build an expectation from the checked-in calibration JSON shape.

        ``calibration`` must contain ``size``, ``fx`` and ``k1``.  When
        ``include_extrinsics`` is true, at most one transform named
        ``T_<parent>_cam`` is accepted and retained in that parent frame.
        """

        if not isinstance(calibration, Mapping):
            raise SimulationValidationError("camera calibration must be a mapping")
        camera_name = _nonempty_text(camera_id, "camera_id")
        source = _nonempty_text(provenance_id, "provenance_id")
        resolution = _resolution(calibration.get("size"), "calibration size")
        focal_px = _positive_float(calibration.get("fx"), "calibration fx")
        radial_k1 = _finite_float(calibration.get("k1"), "calibration k1")

        extrinsic_frame = None
        transform = None
        if include_extrinsics:
            transform_keys = [
                key
                for key in calibration
                if isinstance(key, str)
                and key.startswith("T_")
                and key.endswith("_cam")
            ]
            if len(transform_keys) > 1:
                raise SimulationValidationError(
                    "camera calibration contains more than one camera extrinsic"
                )
            if transform_keys:
                transform_key = transform_keys[0]
                extrinsic_frame = transform_key[2:-4]
                transform = _matrix4(calibration[transform_key], transform_key)

        return cls(
            camera_id=camera_name,
            resolution=resolution,
            focal_px=focal_px,
            radial_k1=radial_k1,
            provenance_id=source,
            extrinsic_frame=extrinsic_frame,
            camera_to_parent_row_major=transform,
        )


@dataclass(frozen=True)
class CameraValidationResult:
    camera_id: str
    resolution: tuple[int, int]
    expected_focal_px: float
    measured_focal_px: float
    focal_error_px: float
    expected_radial_k1: float
    measured_radial_k1: float
    distortion_application: str
    provenance_id: str
    extrinsic_frame: str | None
    translation_error_m: float | None
    rotation_error_deg: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "resolution": list(self.resolution),
            "expected_focal_px": self.expected_focal_px,
            "measured_focal_px": self.measured_focal_px,
            "focal_error_px": self.focal_error_px,
            "expected_radial_k1": self.expected_radial_k1,
            "measured_radial_k1": self.measured_radial_k1,
            "distortion_application": self.distortion_application,
            "provenance_id": self.provenance_id,
            "extrinsic_frame": self.extrinsic_frame,
            "translation_error_m": self.translation_error_m,
            "rotation_error_deg": self.rotation_error_deg,
        }


def validate_camera_readback(
    expected: CameraCalibrationExpectation,
    readback: Mapping[str, Any],
    *,
    focal_tolerance_px: float = 0.05,
    radial_tolerance: float = 1e-9,
    require_extrinsics: bool = False,
    translation_tolerance_m: float = 0.0001,
    rotation_tolerance_deg: float = 0.05,
) -> CameraValidationResult:
    """Validate authored/read-back camera metadata against one calibration.

    ``readback`` uses this JSON-friendly schema::

        {
          "resolution": [1920, 1080],
          "focal_length_mm": 11.1854,
          "horizontal_aperture_mm": 20.955,
          "projection_provenance_id": "cameras.json#table",
          "distortion": {
            "k1": 0.00886,
            "application": "postprocess",
            "provenance_id": "cameras.json#table"
          },
          "extrinsic_frame": "base",                 # optional
          "camera_to_parent_row_major": [[...], ...]  # optional
        }

    ``focal_px`` may replace the two millimetre-valued focal fields.  If both
    forms are supplied, they must agree.  Non-zero calibrated distortion must
    be explicitly applied by either the renderer or post-processing.
    """

    if not isinstance(readback, Mapping):
        raise SimulationValidationError("camera readback must be a mapping")
    focal_tolerance = _positive_float(focal_tolerance_px, "focal_tolerance_px")
    radial_limit = _positive_float(radial_tolerance, "radial_tolerance")
    translation_limit = _positive_float(
        translation_tolerance_m, "translation_tolerance_m"
    )
    rotation_limit = _positive_float(
        rotation_tolerance_deg, "rotation_tolerance_deg"
    )

    measured_resolution = _resolution(readback.get("resolution"), "readback resolution")
    if measured_resolution != expected.resolution:
        raise SimulationValidationError(
            f"{expected.camera_id} resolution {measured_resolution} does not match "
            f"calibration {expected.resolution}"
        )

    focal_candidates: list[tuple[str, float]] = []
    if "focal_px" in readback:
        focal_candidates.append(
            ("focal_px", _positive_float(readback["focal_px"], "readback focal_px"))
        )
    has_focal_mm = "focal_length_mm" in readback
    has_aperture_mm = "horizontal_aperture_mm" in readback
    if has_focal_mm != has_aperture_mm:
        raise SimulationValidationError(
            "focal_length_mm and horizontal_aperture_mm must be supplied together"
        )
    if has_focal_mm:
        focal_mm = _positive_float(readback["focal_length_mm"], "focal_length_mm")
        aperture_mm = _positive_float(
            readback["horizontal_aperture_mm"], "horizontal_aperture_mm"
        )
        focal_candidates.append(
            ("USD focal/aperture", focal_mm * measured_resolution[0] / aperture_mm)
        )
    if not focal_candidates:
        raise SimulationValidationError(
            "camera readback needs focal_px or focal/aperture millimetres"
        )
    measured_focal_px = focal_candidates[0][1]
    for label, candidate in focal_candidates[1:]:
        if abs(candidate - measured_focal_px) > focal_tolerance:
            raise SimulationValidationError(
                f"camera readback focal forms disagree: {measured_focal_px:.6f} px "
                f"versus {candidate:.6f} px from {label}"
            )
    focal_error_px = abs(measured_focal_px - expected.focal_px)
    if focal_error_px > focal_tolerance:
        raise SimulationValidationError(
            f"{expected.camera_id} focal error {focal_error_px:.6f} px exceeds "
            f"{focal_tolerance:.6f} px"
        )

    projection_source = _nonempty_text(
        readback.get("projection_provenance_id"), "projection_provenance_id"
    )
    if projection_source != expected.provenance_id:
        raise SimulationValidationError(
            f"{expected.camera_id} projection provenance {projection_source!r} does "
            f"not match {expected.provenance_id!r}"
        )

    distortion = readback.get("distortion")
    if not isinstance(distortion, Mapping):
        raise SimulationValidationError("camera readback distortion must be a mapping")
    measured_k1 = _finite_float(distortion.get("k1"), "distortion k1")
    if abs(measured_k1 - expected.radial_k1) > radial_limit:
        raise SimulationValidationError(
            f"{expected.camera_id} radial k1 {measured_k1} does not match "
            f"{expected.radial_k1}"
        )
    distortion_source = _nonempty_text(
        distortion.get("provenance_id"), "distortion provenance_id"
    )
    if distortion_source != expected.provenance_id:
        raise SimulationValidationError(
            f"{expected.camera_id} distortion provenance {distortion_source!r} does "
            f"not match {expected.provenance_id!r}"
        )
    distortion_application = _nonempty_text(
        distortion.get("application"), "distortion application"
    )
    valid_applications = {"renderer", "postprocess", "ideal_pinhole", "none"}
    if distortion_application not in valid_applications:
        raise SimulationValidationError(
            f"unsupported distortion application {distortion_application!r}"
        )
    if abs(expected.radial_k1) > radial_limit and distortion_application not in {
        "renderer",
        "postprocess",
    }:
        raise SimulationValidationError(
            f"{expected.camera_id} has non-zero calibrated distortion but the "
            f"readback declares {distortion_application!r}"
        )

    translation_error = None
    rotation_error = None
    has_readback_transform = "camera_to_parent_row_major" in readback
    if require_extrinsics and expected.camera_to_parent_row_major is None:
        raise SimulationValidationError(
            f"{expected.camera_id} calibration has no camera extrinsic"
        )
    if require_extrinsics and not has_readback_transform:
        raise SimulationValidationError(
            f"{expected.camera_id} readback has no camera extrinsic"
        )
    if has_readback_transform:
        if expected.camera_to_parent_row_major is None:
            raise SimulationValidationError(
                f"{expected.camera_id} readback has an extrinsic but calibration does not"
            )
        measured_frame = _nonempty_text(
            readback.get("extrinsic_frame"), "extrinsic_frame"
        )
        if measured_frame != expected.extrinsic_frame:
            raise SimulationValidationError(
                f"{expected.camera_id} extrinsic frame {measured_frame!r} does not "
                f"match {expected.extrinsic_frame!r}"
            )
        measured_transform = _matrix4(
            readback["camera_to_parent_row_major"],
            "camera_to_parent_row_major",
        )
        translation_error, rotation_error = _transform_errors(
            expected.camera_to_parent_row_major, measured_transform
        )
        if translation_error > translation_limit:
            raise SimulationValidationError(
                f"{expected.camera_id} extrinsic translation error "
                f"{translation_error:.9f} m exceeds {translation_limit:.9f} m"
            )
        if rotation_error > rotation_limit:
            raise SimulationValidationError(
                f"{expected.camera_id} extrinsic rotation error "
                f"{rotation_error:.6f} deg exceeds {rotation_limit:.6f} deg"
            )

    return CameraValidationResult(
        camera_id=expected.camera_id,
        resolution=measured_resolution,
        expected_focal_px=expected.focal_px,
        measured_focal_px=measured_focal_px,
        focal_error_px=focal_error_px,
        expected_radial_k1=expected.radial_k1,
        measured_radial_k1=measured_k1,
        distortion_application=distortion_application,
        provenance_id=expected.provenance_id,
        extrinsic_frame=expected.extrinsic_frame if has_readback_transform else None,
        translation_error_m=translation_error,
        rotation_error_deg=rotation_error,
    )


@dataclass(frozen=True)
class GeometryExpectation:
    """Expected named measurements and immutable geometry provenance."""

    geometry_id: str
    measurements_m: tuple[tuple[str, float], ...]
    geometry_provenance_id: str
    silhouette_provenance_id: str | None = None

    @classmethod
    def from_measurements(
        cls,
        geometry_id: str,
        measurements_m: Mapping[str, Any],
        *,
        geometry_provenance_id: str,
        silhouette_provenance_id: str | None = None,
    ) -> "GeometryExpectation":
        if not isinstance(measurements_m, Mapping) or not measurements_m:
            raise SimulationValidationError(
                "geometry measurements_m must be a non-empty mapping"
            )
        normalized: list[tuple[str, float]] = []
        for name, value in measurements_m.items():
            measurement_name = _nonempty_text(name, "geometry measurement name")
            normalized.append(
                (
                    measurement_name,
                    _positive_float(value, f"geometry measurement {measurement_name}"),
                )
            )
        silhouette = None
        if silhouette_provenance_id is not None:
            silhouette = _nonempty_text(
                silhouette_provenance_id, "silhouette_provenance_id"
            )
        return cls(
            geometry_id=_nonempty_text(geometry_id, "geometry_id"),
            measurements_m=tuple(sorted(normalized)),
            geometry_provenance_id=_nonempty_text(
                geometry_provenance_id, "geometry_provenance_id"
            ),
            silhouette_provenance_id=silhouette,
        )


@dataclass(frozen=True)
class GeometryValidationResult:
    geometry_id: str
    measurements_m: tuple[tuple[str, float], ...]
    absolute_errors_m: tuple[tuple[str, float], ...]
    geometry_provenance_id: str
    silhouette_provenance_id: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "geometry_id": self.geometry_id,
            "measurements_m": dict(self.measurements_m),
            "absolute_errors_m": dict(self.absolute_errors_m),
            "geometry_provenance_id": self.geometry_provenance_id,
            "silhouette_provenance_id": self.silhouette_provenance_id,
        }


def measurements_from_bounds(
    bounds_min_m: Sequence[Any],
    bounds_max_m: Sequence[Any],
    *,
    labels: Sequence[str] = ("size_x", "size_y", "size_z"),
) -> dict[str, float]:
    """Convert two 3D bounds into named positive extents."""

    if len(bounds_min_m) != 3 or len(bounds_max_m) != 3 or len(labels) != 3:
        raise SimulationValidationError("bounds and labels must each contain 3 values")
    result: dict[str, float] = {}
    for index, raw_label in enumerate(labels):
        label = _nonempty_text(raw_label, f"bounds label {index}")
        minimum = _finite_float(bounds_min_m[index], f"bounds minimum {label}")
        maximum = _finite_float(bounds_max_m[index], f"bounds maximum {label}")
        if maximum <= minimum:
            raise SimulationValidationError(
                f"bounds maximum for {label} must exceed its minimum"
            )
        result[label] = maximum - minimum
    return result


def validate_geometry_readback(
    expected: GeometryExpectation,
    readback: Mapping[str, Any],
    *,
    tolerance_m: float | Mapping[str, Any] = 0.00005,
) -> GeometryValidationResult:
    """Validate measured stage geometry and its visual-mesh provenance.

    ``readback`` must contain ``measurements_m`` and
    ``geometry_provenance_id``.  If the expectation names a silhouette source,
    ``silhouette_provenance_id`` is also required.  Extra measured fields are
    allowed, but every expected field must exist and be within tolerance.
    ``tolerance_m`` may be one scalar or a per-measurement mapping.
    """

    if not isinstance(readback, Mapping):
        raise SimulationValidationError("geometry readback must be a mapping")
    measured = readback.get("measurements_m")
    if not isinstance(measured, Mapping):
        raise SimulationValidationError(
            "geometry readback measurements_m must be a mapping"
        )
    geometry_source = _nonempty_text(
        readback.get("geometry_provenance_id"), "geometry_provenance_id"
    )
    if geometry_source != expected.geometry_provenance_id:
        raise SimulationValidationError(
            f"{expected.geometry_id} geometry provenance {geometry_source!r} does "
            f"not match {expected.geometry_provenance_id!r}"
        )

    silhouette_source = readback.get("silhouette_provenance_id")
    if expected.silhouette_provenance_id is not None:
        silhouette_source = _nonempty_text(
            silhouette_source, "silhouette_provenance_id"
        )
        if silhouette_source != expected.silhouette_provenance_id:
            raise SimulationValidationError(
                f"{expected.geometry_id} silhouette provenance "
                f"{silhouette_source!r} does not match "
                f"{expected.silhouette_provenance_id!r}"
            )
    elif silhouette_source is not None:
        silhouette_source = _nonempty_text(
            silhouette_source, "silhouette_provenance_id"
        )

    scalar_tolerance = None
    tolerance_mapping = None
    if isinstance(tolerance_m, Mapping):
        tolerance_mapping = tolerance_m
    else:
        scalar_tolerance = _positive_float(tolerance_m, "tolerance_m")

    measured_values: list[tuple[str, float]] = []
    errors: list[tuple[str, float]] = []
    for measurement_name, expected_value in expected.measurements_m:
        if measurement_name not in measured:
            raise SimulationValidationError(
                f"{expected.geometry_id} readback is missing {measurement_name!r}"
            )
        measured_value = _positive_float(
            measured[measurement_name],
            f"{expected.geometry_id} measurement {measurement_name}",
        )
        if tolerance_mapping is not None:
            if measurement_name not in tolerance_mapping:
                raise SimulationValidationError(
                    f"tolerance_m is missing {measurement_name!r}"
                )
            allowed_error = _positive_float(
                tolerance_mapping[measurement_name],
                f"tolerance for {measurement_name}",
            )
        else:
            assert scalar_tolerance is not None
            allowed_error = scalar_tolerance
        error = abs(measured_value - expected_value)
        if error > allowed_error:
            raise SimulationValidationError(
                f"{expected.geometry_id} {measurement_name} error {error:.9f} m "
                f"exceeds {allowed_error:.9f} m"
            )
        measured_values.append((measurement_name, measured_value))
        errors.append((measurement_name, error))

    return GeometryValidationResult(
        geometry_id=expected.geometry_id,
        measurements_m=tuple(measured_values),
        absolute_errors_m=tuple(errors),
        geometry_provenance_id=expected.geometry_provenance_id,
        silhouette_provenance_id=silhouette_source,
    )

