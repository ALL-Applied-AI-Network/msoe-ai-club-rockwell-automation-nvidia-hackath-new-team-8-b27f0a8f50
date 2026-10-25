"""Conservative interpolation and bounded FANUC action-chunk construction."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .config import SortConfig
from .contracts import MotionPhase, TrajectoryChunk
from .frames import base_tip_pose_to_uf0_tcp, validate_tcp_chunk


def _shortest_degrees(delta: np.ndarray) -> np.ndarray:
    return (delta + 180.0) % 360.0 - 180.0


def interpolate_tcp_rows(
    start: Sequence[float],
    target: Sequence[float],
    config: SortConfig,
    *,
    max_linear_step_mm: float | None = None,
    max_angular_step_deg: float | None = None,
) -> np.ndarray:
    """Interpolate absolute TCP targets without interpolating gripper state.

    The returned rows exclude ``start`` and include ``target``.  Gripper changes
    occur only on the last row, avoiding fractional pneumatic commands.
    """

    first = np.asarray(start, dtype=float)
    last = np.asarray(target, dtype=float)
    if first.shape != (7,) or last.shape != (7,):
        raise ValueError("start and target must each be one seven-value TCP row")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(last)):
        raise ValueError("start and target must be finite")
    linear_step = float(
        max_linear_step_mm or config.hardware.max_linear_step_mm
    )
    angular_step = float(
        max_angular_step_deg or config.hardware.max_angular_step_deg
    )
    if linear_step <= 0 or angular_step <= 0:
        raise ValueError("interpolation steps must be positive")

    distance = float(np.linalg.norm(last[:3] - first[:3]))
    angular_delta = _shortest_degrees(last[3:6] - first[3:6])
    angle = float(np.max(np.abs(angular_delta)))
    count = max(1, math.ceil(distance / linear_step), math.ceil(angle / angular_step))
    fractions = np.arange(1, count + 1, dtype=float) / count
    rows = np.empty((count, 7), dtype=float)
    rows[:, :3] = first[:3] + fractions[:, None] * (last[:3] - first[:3])
    rows[:, 3:6] = first[3:6] + fractions[:, None] * angular_delta
    rows[:, 6] = first[6]
    rows[-1, 6] = last[6]
    rows[-1, :6] = last[:6]
    return rows


def build_trajectory_chunks(
    initial_tcp: Sequence[float],
    phases: Iterable[MotionPhase],
    config: SortConfig,
    *,
    dt_s: float | None = None,
) -> tuple[TrajectoryChunk, ...]:
    """Translate semantic phases into official-size TCP action chunks.

    Reobservation gates become empty chunks, forcing the caller to obtain and
    validate a fresh observation before building/sending later phases.  This
    function performs protocol preflight only; it does not prove IK or collision
    safety on the real cell.
    """

    dt = config.hardware.default_dt_s if dt_s is None else float(dt_s)
    # Validate dt and the starting row with the same checks used for actions.
    current = validate_tcp_chunk([initial_tcp], dt, config)[0]
    output: list[TrajectoryChunk] = []
    maximum = config.hardware.max_chunk_actions

    for phase in phases:
        if phase.requires_reobservation_before:
            output.append(
                TrajectoryChunk(
                    phase_name=phase.name,
                    actions=(),
                    dt_s=dt,
                    requires_reobservation_before=True,
                )
            )
            continue
        if phase.target is None:
            raise ValueError(f"phase {phase.name} has neither a target nor a gate")

        target = np.asarray(
            base_tip_pose_to_uf0_tcp(phase.target, phase.gripper, config),
            dtype=float,
        )
        rows = interpolate_tcp_rows(current, target, config)
        moved = not np.allclose(current[:6], target[:6], atol=1e-9)
        dwell_rows = max(0, math.ceil(phase.dwell_s / dt) - (0 if moved else 1))
        if dwell_rows:
            rows = np.vstack([rows, np.repeat(target[None, :], dwell_rows, axis=0)])

        part = 0
        for offset in range(0, len(rows), maximum):
            checked = validate_tcp_chunk(rows[offset : offset + maximum], dt, config)
            output.append(
                TrajectoryChunk(
                    phase_name=phase.name,
                    actions=tuple(tuple(float(v) for v in row) for row in checked),
                    dt_s=dt,
                    part_index=part,
                )
            )
            part += 1
        current = target
    return tuple(output)
