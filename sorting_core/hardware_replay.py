"""Validated, read-only shadow replay of recorded FANUC observations.

The downloaded LeRobot exports contain measured joints, a measured TCP pose,
and a field called ``action``.  The first action rows in every inspected
episode are *joint-state vectors* even though the export metadata calls the
action kind ``tcp``.  This module therefore never turns recorded actions into
commands.  It validates measured observation state for an Isaac-only pose
replay used to compare camera framing and appearance.

Nothing in this module talks to the physical robot.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class HardwareReplayError(ValueError):
    """Raised when recorded observations are unsafe or internally inconsistent."""


# Exact limits copied from the supplied ER-4iA twin's scene_config.py.  Keeping
# the values here makes the host-side validator independent of Isaac imports.
ER4IA_JOINT_LIMITS_DEG: tuple[tuple[float, float], ...] = (
    (-170.0, 170.0),
    (-110.0, 120.0),
    (-69.0, 205.0),
    (-190.0, 190.0),
    (-120.0, 120.0),
    (-360.0, 360.0),
)

_CHAIN = (
    ((0.0, 0.0, 0.33), "z", +1.0),
    ((0.0, 0.0, 0.00), "y", +1.0),
    ((0.0, 0.0, 0.26), "y", -1.0),
    ((0.0, 0.0, 0.02), "x", -1.0),
    ((0.29, 0.0, 0.0), "y", -1.0),
    ((0.0, 0.0, 0.00), "x", -1.0),
)
_TOOL_OFFSET_M = (0.293, 0.0, 0.0)
_UF0_Z_OFFSET_MM = 330.0
# Fixed rotation from the twin chain's final X-forward link frame to the FANUC
# fingertip TCP frame (Z points along the tool).  The recorded W/P/R convention
# is Rz(R) @ Ry(P) @ Rx(W).
_TOOL_FRAME_ROTATION = (
    (0.0, 0.0, 1.0),
    (0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0),
)


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HardwareReplayError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise HardwareReplayError(f"{name} must be finite")
    return result


def _vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(_finite(item, name) for item in value)
    except TypeError as exc:
        raise HardwareReplayError(f"{name} must contain {size} values") from exc
    if len(result) != size:
        raise HardwareReplayError(f"{name} must contain {size} values")
    return result


def _matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]):
    return tuple(
        tuple(sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]):
    return tuple(sum(matrix[row][k] * vector[k] for k in range(3)) for row in range(3))


def _rotation(axis: str, angle: float):
    cosine = math.cos(angle)
    sine = math.sin(angle)
    if axis == "x":
        return ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine))
    if axis == "y":
        return ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine))
    return ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))


def _predicted_chain_pose(joints_deg: Sequence[float]):
    """Return final-link position and rotation in the twin base frame."""

    joints = _vector(joints_deg, 6, "joints_deg")
    rotation = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    position = (0.0, 0.0, 0.0)
    for joint_deg, (offset, axis, sign) in zip(joints, _CHAIN):
        moved = _matvec(rotation, offset)
        position = tuple(position[index] + moved[index] for index in range(3))
        rotation = _matmul(
            rotation,
            _rotation(axis, sign * math.radians(joint_deg)),
        )
    moved = _matvec(rotation, _TOOL_OFFSET_M)
    position = tuple(position[index] + moved[index] for index in range(3))
    return position, rotation


def predicted_tcp_xyz_uf0_mm(joints_deg: Sequence[float]) -> tuple[float, float, float]:
    """Return fingertip XYZ in controller UF0 for the twin's joint convention.

    Recorded ``observation.state`` already uses the twin convention.  In
    particular, callers must **not** add J2 to J3 again.
    """

    position, _ = _predicted_chain_pose(joints_deg)
    return (
        position[0] * 1000.0,
        position[1] * 1000.0,
        position[2] * 1000.0 - _UF0_Z_OFFSET_MM,
    )


def _predicted_tcp_rotation(joints_deg: Sequence[float]):
    _, final_link_rotation = _predicted_chain_pose(joints_deg)
    return _matmul(final_link_rotation, _TOOL_FRAME_ROTATION)


def _wpr_rotation(wpr_deg: Sequence[float]):
    w, p, r = _vector(wpr_deg, 3, "W/P/R")
    return _matmul(
        _rotation("z", math.radians(r)),
        _matmul(
            _rotation("y", math.radians(p)),
            _rotation("x", math.radians(w)),
        ),
    )


def predicted_tcp_wpr_uf0_deg(joints_deg: Sequence[float]) -> tuple[float, float, float]:
    """Return one FANUC W/P/R representation of the predicted TCP orientation."""

    rotation = _predicted_tcp_rotation(joints_deg)
    pitch = math.asin(max(-1.0, min(1.0, -rotation[2][0])))
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(rotation[2][1], rotation[2][2])
        yaw = math.atan2(rotation[1][0], rotation[0][0])
    else:
        roll = 0.0
        yaw = math.atan2(-rotation[0][1], rotation[1][1])
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def _orientation_error_deg(first, second) -> float:
    relative = _matmul(tuple(zip(*first)), second)
    cosine = max(-1.0, min(1.0, (sum(relative[i][i] for i in range(3)) - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


@dataclass(frozen=True)
class ReplaySample:
    frame_index: int
    timestamp_s: float
    joints_deg: tuple[float, ...]
    gripper_state: float
    tcp_pose_uf0_mm_deg: tuple[float, ...]
    fk_position_error_mm: float
    fk_orientation_error_deg: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "joints_deg": list(self.joints_deg),
            "gripper_state": self.gripper_state,
            "tcp_pose_uf0_mm_deg": list(self.tcp_pose_uf0_mm_deg),
            "fk_position_error_mm": self.fk_position_error_mm,
            "fk_orientation_error_deg": self.fk_orientation_error_deg,
        }


def build_replay_sample(
    *,
    frame_index: Any,
    timestamp_s: Any,
    observation_state: Sequence[float],
    tcp_pose: Sequence[float],
) -> ReplaySample:
    """Validate one measured observation row and calculate its FK residual."""

    if isinstance(frame_index, bool):
        raise HardwareReplayError("frame_index must be an integer")
    try:
        frame = int(frame_index)
    except (TypeError, ValueError) as exc:
        raise HardwareReplayError("frame_index must be an integer") from exc
    if frame < 0 or frame != frame_index:
        raise HardwareReplayError("frame_index must be a non-negative integer")
    timestamp = _finite(timestamp_s, "timestamp_s")
    if timestamp < 0.0:
        raise HardwareReplayError("timestamp_s must be non-negative")
    state = _vector(observation_state, 7, "observation_state")
    joints = state[:6]
    for index, (joint, limits) in enumerate(zip(joints, ER4IA_JOINT_LIMITS_DEG), 1):
        if not limits[0] <= joint <= limits[1]:
            raise HardwareReplayError(
                f"J{index}={joint} deg is outside [{limits[0]}, {limits[1]}]"
            )
    gripper = state[6]
    if not 0.0 <= gripper <= 1.0:
        raise HardwareReplayError("gripper state must be within [0, 1]")
    measured_tcp = _vector(tcp_pose, 6, "tcp_pose")
    predicted_xyz = predicted_tcp_xyz_uf0_mm(joints)
    residual = math.sqrt(
        sum((predicted_xyz[index] - measured_tcp[index]) ** 2 for index in range(3))
    )
    orientation_residual = _orientation_error_deg(
        _predicted_tcp_rotation(joints),
        _wpr_rotation(measured_tcp[3:]),
    )
    return ReplaySample(
        frame_index=frame,
        timestamp_s=timestamp,
        joints_deg=joints,
        gripper_state=gripper,
        tcp_pose_uf0_mm_deg=measured_tcp,
        fk_position_error_mm=residual,
        fk_orientation_error_deg=orientation_residual,
    )


def initial_joint_action_sentinel_length(
    observation_states: Sequence[Sequence[float]],
    actions: Sequence[Sequence[float]],
    *,
    absolute_tolerance: float = 1e-5,
) -> int:
    """Count the initial rows whose nominal TCP action equals joint state.

    A nonzero result identifies the reset/export sentinel seen in every audited
    hardware episode.  Such rows must not be interpreted as TCP commands.
    """

    if len(observation_states) != len(actions):
        raise HardwareReplayError("state and action row counts differ")
    tolerance = _finite(absolute_tolerance, "absolute_tolerance")
    if tolerance < 0.0:
        raise HardwareReplayError("absolute_tolerance must be non-negative")
    count = 0
    for index, (state_raw, action_raw) in enumerate(zip(observation_states, actions)):
        state = _vector(state_raw, 7, f"state[{index}]")
        action = _vector(action_raw, 7, f"action[{index}]")
        if all(abs(left - right) <= tolerance for left, right in zip(state, action)):
            count += 1
        else:
            break
    return count


def validate_replay_samples(
    samples: Iterable[ReplaySample],
    *,
    maximum_median_fk_error_mm: float = 0.25,
    maximum_row_fk_error_mm: float = 10.0,
    maximum_median_orientation_error_deg: float = 0.25,
    maximum_row_orientation_error_deg: float = 2.0,
) -> dict[str, Any]:
    """Validate ordering and direct-joint FK agreement for selected samples."""

    items = tuple(samples)
    if not items:
        raise HardwareReplayError("replay must contain at least one sample")
    frames = [item.frame_index for item in items]
    times = [item.timestamp_s for item in items]
    if any(right <= left for left, right in zip(frames, frames[1:])):
        raise HardwareReplayError("replay frame indices must be strictly increasing")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise HardwareReplayError("replay timestamps must be strictly increasing")
    errors = sorted(item.fk_position_error_mm for item in items)
    middle = len(errors) // 2
    median = (
        errors[middle]
        if len(errors) % 2
        else 0.5 * (errors[middle - 1] + errors[middle])
    )
    maximum = max(errors)
    if median > _finite(maximum_median_fk_error_mm, "maximum median FK error"):
        raise HardwareReplayError(
            f"median FK/TCP position error {median:.3f} mm exceeds limit"
        )
    if maximum > _finite(maximum_row_fk_error_mm, "maximum row FK error"):
        raise HardwareReplayError(
            f"maximum FK/TCP position error {maximum:.3f} mm exceeds limit"
        )
    orientation_errors = sorted(item.fk_orientation_error_deg for item in items)
    orientation_middle = len(orientation_errors) // 2
    orientation_median = (
        orientation_errors[orientation_middle]
        if len(orientation_errors) % 2
        else 0.5
        * (
            orientation_errors[orientation_middle - 1]
            + orientation_errors[orientation_middle]
        )
    )
    orientation_maximum = max(orientation_errors)
    if orientation_median > _finite(
        maximum_median_orientation_error_deg,
        "maximum median orientation error",
    ):
        raise HardwareReplayError(
            f"median FK/TCP orientation error {orientation_median:.3f} deg exceeds limit"
        )
    if orientation_maximum > _finite(
        maximum_row_orientation_error_deg,
        "maximum row orientation error",
    ):
        raise HardwareReplayError(
            f"maximum FK/TCP orientation error {orientation_maximum:.3f} deg exceeds limit"
        )
    return {
        "sample_count": len(items),
        "first_frame": frames[0],
        "last_frame": frames[-1],
        "median_fk_position_error_mm": median,
        "maximum_fk_position_error_mm": maximum,
        "median_fk_orientation_error_deg": orientation_median,
        "maximum_fk_orientation_error_deg": orientation_maximum,
        "joint_mapping": (
            "direct v0.5 twin convention with full TCP orientation validation; "
            "do not add J2 to J3"
        ),
    }


def allowed_frame_ranges(
    curation: Mapping[str, Any],
    *,
    dataset_id: str,
    episode_index: int,
    split_name: str,
) -> tuple[tuple[int, int], ...]:
    """Return reviewed half-open ranges from the privacy curation manifest."""

    if curation.get("default_policy") != "deny":
        raise HardwareReplayError("privacy curation must be default-deny")
    split_root = curation.get("splits")
    try:
        records = split_root[split_name]
        excluded_records = split_root["excluded_human_or_setup"]
    except (KeyError, TypeError) as exc:
        raise HardwareReplayError(f"unknown curation split {split_name!r}") from exc
    matches = [
        item
        for item in records
        if str(item.get("dataset")) == str(dataset_id)
        and int(item.get("episode", -1)) == int(episode_index)
    ]
    if len(matches) != 1:
        raise HardwareReplayError(
            f"curation split {split_name!r} has no unique record for "
            f"{dataset_id} episode {episode_index}"
        )
    ranges = []
    for index, raw in enumerate(matches[0].get("ranges", ())):
        if len(raw) != 2:
            raise HardwareReplayError(f"curation range {index} must have two values")
        start, end = int(raw[0]), int(raw[1])
        if start < 0 or end <= start:
            raise HardwareReplayError(f"curation range {index} is invalid")
        ranges.append((start, end))
    if not ranges:
        raise HardwareReplayError("curation record contains no allowed ranges")
    ordered = sorted(ranges)
    if any(right[0] < left[1] for left, right in zip(ordered, ordered[1:])):
        raise HardwareReplayError("curation ranges overlap")
    excluded_ranges = []
    for record in excluded_records:
        if (
            str(record.get("dataset")) != str(dataset_id)
            or int(record.get("episode", -1)) != int(episode_index)
        ):
            continue
        for index, raw in enumerate(record.get("ranges", ())):
            if len(raw) != 2:
                raise HardwareReplayError(
                    f"privacy exclusion range {index} must have two values"
                )
            start, end = int(raw[0]), int(raw[1])
            if start < 0 or end <= start:
                raise HardwareReplayError(
                    f"privacy exclusion range {index} is invalid"
                )
            excluded_ranges.append((start, end))
    for allowed_start, allowed_end in ordered:
        for excluded_start, excluded_end in excluded_ranges:
            if allowed_start < excluded_end and excluded_start < allowed_end:
                raise HardwareReplayError(
                    "allowed replay range overlaps excluded human/setup imagery"
                )
    return tuple(ordered)


def frame_is_allowed(frame_index: int, ranges: Sequence[tuple[int, int]]) -> bool:
    return any(start <= int(frame_index) < end for start, end in ranges)


def validate_replay_document(raw: Mapping[str, Any]) -> tuple[ReplaySample, ...]:
    """Load and validate the compact JSON consumed by the Isaac replay demo."""

    if int(raw.get("schema_version", -1)) != 1:
        raise HardwareReplayError("unsupported hardware replay schema")
    if str(raw.get("replay_kind", "")) != "measured_state_shadow_replay":
        raise HardwareReplayError("replay_kind must be measured_state_shadow_replay")
    if raw.get("physical_robot_commands_permitted") is not False:
        raise HardwareReplayError("replay must explicitly forbid physical commands")
    samples = tuple(
        build_replay_sample(
            frame_index=item.get("frame_index"),
            timestamp_s=item.get("timestamp_s"),
            observation_state=[*item.get("joints_deg", ()), item.get("gripper_state")],
            tcp_pose=item.get("tcp_pose_uf0_mm_deg"),
        )
        for item in raw.get("samples", ())
    )
    validate_replay_samples(samples)
    return samples


def load_replay_document(path: str | Path) -> tuple[Mapping[str, Any], tuple[ReplaySample, ...]]:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HardwareReplayError(f"cannot load replay {source}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise HardwareReplayError("replay document must be a JSON object")
    return raw, validate_replay_document(raw)


__all__ = [
    "ER4IA_JOINT_LIMITS_DEG",
    "HardwareReplayError",
    "ReplaySample",
    "allowed_frame_ranges",
    "build_replay_sample",
    "frame_is_allowed",
    "initial_joint_action_sentinel_length",
    "load_replay_document",
    "predicted_tcp_xyz_uf0_mm",
    "predicted_tcp_wpr_uf0_deg",
    "validate_replay_document",
    "validate_replay_samples",
]
