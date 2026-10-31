"""Evaluate camera pose estimates against isolated simulator ground truth.

The estimator and evaluator intentionally use different input files.  A
perception provider may read RGB-D, calibration, masks, and CAD assets, but it
must never receive the truth document consumed here.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


class PoseEvaluationError(ValueError):
    """Raised when estimate or truth data cannot be evaluated safely."""


def _vector(values: Sequence[Any], size: int, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != size:
        raise PoseEvaluationError(f"{name} must contain {size} values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise PoseEvaluationError(f"{name} must contain finite values")
    return result


def _unit(values: Sequence[Any], size: int, name: str) -> tuple[float, ...]:
    result = _vector(values, size, name)
    magnitude = math.sqrt(sum(value * value for value in result))
    if magnitude <= 1e-12:
        raise PoseEvaluationError(f"{name} must be nonzero")
    return tuple(value / magnitude for value in result)


def _quaternion_xyzw(values: Sequence[Any], name: str) -> tuple[float, ...]:
    quaternion = _unit(values, 4, name)
    # Canonicalize the double cover so reports remain stable across runtimes.
    if quaternion[3] < 0.0:
        quaternion = tuple(-value for value in quaternion)
    return quaternion


def rotation_error_deg(
    estimate_xyzw: Sequence[Any], truth_xyzw: Sequence[Any]
) -> float:
    """Return the shortest SO(3) geodesic angle between two quaternions."""

    estimate = _quaternion_xyzw(estimate_xyzw, "estimate orientation")
    truth = _quaternion_xyzw(truth_xyzw, "truth orientation")
    dot = min(1.0, abs(sum(a * b for a, b in zip(estimate, truth))))
    return math.degrees(2.0 * math.acos(dot))


def _rotate_vector_xyzw(
    quaternion_xyzw: Sequence[Any], vector: Sequence[Any]
) -> tuple[float, float, float]:
    x, y, z, w = _quaternion_xyzw(quaternion_xyzw, "orientation")
    vx, vy, vz = _unit(vector, 3, "local axis")
    # R(q) @ v, expanded to avoid a mandatory numerical dependency.
    return (
        (1 - 2 * (y * y + z * z)) * vx
        + 2 * (x * y - z * w) * vy
        + 2 * (x * z + y * w) * vz,
        2 * (x * y + z * w) * vx
        + (1 - 2 * (x * x + z * z)) * vy
        + 2 * (y * z - x * w) * vz,
        2 * (x * z - y * w) * vx
        + 2 * (y * z + x * w) * vy
        + (1 - 2 * (x * x + y * y)) * vz,
    )


def axis_error_deg(
    estimate_xyzw: Sequence[Any],
    truth_xyzw: Sequence[Any],
    local_axis: Sequence[Any],
    *,
    undirected: bool = False,
) -> float:
    """Return the angle between a model axis in two estimated orientations."""

    estimate_axis = _rotate_vector_xyzw(estimate_xyzw, local_axis)
    truth_axis = _rotate_vector_xyzw(truth_xyzw, local_axis)
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(estimate_axis, truth_axis))))
    if undirected:
        dot = abs(dot)
    return math.degrees(math.acos(dot))


def _pose(raw: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise PoseEvaluationError(f"{label} must be an object")
    return {
        "position_base_m": _vector(raw["position_base_m"], 3, f"{label} position"),
        "orientation_xyzw": _quaternion_xyzw(
            raw["orientation_xyzw"], f"{label} orientation"
        ),
    }


def evaluate_pose_documents(
    estimates: Mapping[str, Any], truth: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two versioned pose documents without exposing truth to inference."""

    if estimates.get("schema_version") != 1 or truth.get("schema_version") != 1:
        raise PoseEvaluationError("estimate and truth schema_version must be 1")
    estimate_frame = estimates.get("coordinate_frame")
    truth_frame = truth.get("coordinate_frame")
    if estimate_frame != truth_frame or not isinstance(estimate_frame, str):
        raise PoseEvaluationError("estimate and truth coordinate frames must match")
    estimate_objects = estimates.get("objects")
    truth_objects = truth.get("objects")
    if not isinstance(estimate_objects, Mapping) or not isinstance(
        truth_objects, Mapping
    ):
        raise PoseEvaluationError("documents must contain object mappings")

    rows: list[dict[str, Any]] = []
    missing = sorted(set(truth_objects) - set(estimate_objects))
    unexpected = sorted(set(estimate_objects) - set(truth_objects))
    for object_id in sorted(set(truth_objects) & set(estimate_objects)):
        estimated_raw = estimate_objects[object_id]
        truth_raw = truth_objects[object_id]
        estimated = _pose(estimated_raw, f"estimate {object_id}")
        expected = _pose(truth_raw, f"truth {object_id}")
        confidence = float(estimated_raw["confidence"])
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise PoseEvaluationError(
                f"estimate {object_id} confidence must be in [0, 1]"
            )
        delta = tuple(
            a - b
            for a, b in zip(
                estimated["position_base_m"], expected["position_base_m"]
            )
        )
        position_error_m = math.sqrt(sum(value * value for value in delta))
        row: dict[str, Any] = {
            "object_id": object_id,
            "class_id": truth_raw.get("class_id"),
            "estimated_position_base_m": list(estimated["position_base_m"]),
            "truth_position_base_m": list(expected["position_base_m"]),
            "position_error_m": position_error_m,
            "position_error_mm": position_error_m * 1000.0,
            "estimated_orientation_xyzw": list(estimated["orientation_xyzw"]),
            "truth_orientation_xyzw": list(expected["orientation_xyzw"]),
            "rotation_error_deg": rotation_error_deg(
                estimated["orientation_xyzw"], expected["orientation_xyzw"]
            ),
            "confidence": confidence,
            "confidence_source": estimated_raw.get(
                "confidence_source", "foundationpose_score"
            ),
            "foundationpose_raw_score": estimated_raw.get(
                "foundationpose_raw_score"
            ),
            "confidence_is_calibrated_probability": bool(
                estimated_raw.get("confidence_is_calibrated_probability", False)
            ),
        }
        local_axis = truth_raw.get("evaluation_axis_local")
        if local_axis is not None:
            row["axis_error_deg"] = axis_error_deg(
                estimated["orientation_xyzw"],
                expected["orientation_xyzw"],
                local_axis,
                undirected=bool(truth_raw.get("axis_undirected", False)),
            )
        rows.append(row)

    position_errors = [row["position_error_m"] for row in rows]
    rotation_errors = [row["rotation_error_deg"] for row in rows]
    confidences = [row["confidence"] for row in rows]

    def mean(values: list[float]) -> float | None:
        return None if not values else sum(values) / len(values)

    def rmse(values: list[float]) -> float | None:
        return None if not values else math.sqrt(sum(v * v for v in values) / len(values))

    return {
        "schema_version": 1,
        "coordinate_frame": estimate_frame,
        "estimator": estimates.get("provider", "foundationpose"),
        "truth_source": truth.get("truth_source", "isaac_sim_rigid_body_state"),
        "objects": rows,
        "missing_estimates": missing,
        "unexpected_estimates": unexpected,
        "summary": {
            "truth_object_count": len(truth_objects),
            "estimated_object_count": len(estimate_objects),
            "matched_object_count": len(rows),
            "mean_position_error_mm": (
                None if mean(position_errors) is None else mean(position_errors) * 1000.0
            ),
            "rmse_position_error_mm": (
                None if rmse(position_errors) is None else rmse(position_errors) * 1000.0
            ),
            "mean_rotation_error_deg": mean(rotation_errors),
            "mean_confidence": mean(confidences),
            "complete": not missing and not unexpected,
        },
    }


__all__ = [
    "PoseEvaluationError",
    "axis_error_deg",
    "evaluate_pose_documents",
    "rotation_error_deg",
]
