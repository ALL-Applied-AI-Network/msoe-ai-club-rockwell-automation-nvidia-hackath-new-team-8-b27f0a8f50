"""Frame conversion between canonical robot-base poses and FANUC UF0 TCP rows."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .config import SortConfig
from .contracts import TipPoseBase


class ActionValidationError(ValueError):
    """Raised before an invalid action can reach the cell client."""


def base_tip_pose_to_uf0_tcp(
    pose: TipPoseBase, gripper: float, config: SortConfig
) -> tuple[float, float, float, float, float, float, float]:
    """Convert a tool-down fingertip target into the official TCP action format.

    Canonical poses describe the physical fingertip in robot-base metres.  UF0
    shares base x/y and shifts z by ``uf0_z_offset_mm``.  A positive
    ``tcp_beyond_fingertips_mm`` means the configured TCP lies farther downward
    than the fingertips and is therefore subtracted from tool-down z.  The
    checked-in default is deliberately zero; see the conflict note in the JSON.
    """

    x, y, z = pose.xyz_m
    w, p, r0 = config.frames.tool_down_wpr_deg
    row = (
        x * 1000.0,
        y * 1000.0,
        z * 1000.0
        + config.frames.uf0_z_offset_mm
        - config.frames.tcp_beyond_fingertips_mm,
        w,
        p,
        r0 + math.degrees(pose.yaw_rad),
        float(gripper),
    )
    if not all(math.isfinite(v) for v in row):
        raise ActionValidationError("converted TCP action contains NaN or infinity")
    return row


def uf0_tcp_to_base_tip(row: Sequence[float], config: SortConfig) -> TipPoseBase:
    """Inverse of :func:`base_tip_pose_to_uf0_tcp` for tool-down poses."""

    values = np.asarray(row, dtype=float)
    if values.shape not in ((6,), (7,)) or not np.all(np.isfinite(values)):
        raise ActionValidationError("TCP row must contain six or seven finite values")
    w0, p0, r0 = config.frames.tool_down_wpr_deg
    if not np.allclose(values[3:5], [w0, p0], atol=1e-6):
        raise ActionValidationError("only the configured tool-down W/P is supported")
    return TipPoseBase(
        xyz_m=(
            values[0] / 1000.0,
            values[1] / 1000.0,
            (
                values[2]
                - config.frames.uf0_z_offset_mm
                + config.frames.tcp_beyond_fingertips_mm
            )
            / 1000.0,
        ),
        yaw_rad=math.radians(values[5] - r0),
    )


def validate_tcp_chunk(
    rows: Sequence[Sequence[float]] | np.ndarray,
    dt_s: float,
    config: SortConfig,
) -> np.ndarray:
    """Apply the official mock-cell TCP limits and return a safe copy.

    This is a protocol preflight, not collision, IK, velocity, or grasp
    validation.  It mirrors the official client mock by clipping the gripper
    column to [0, 1].
    """

    actions = np.asarray(rows, dtype=float)
    maximum = config.hardware.max_chunk_actions
    if actions.ndim != 2 or actions.shape[1:] != (7,) or not 1 <= len(actions) <= maximum:
        raise ActionValidationError(
            f"tcp must have shape (N, 7) with 1 <= N <= {maximum}, got {actions.shape}"
        )
    if not np.all(np.isfinite(actions)):
        raise ActionValidationError("tcp contains NaN or infinity")
    for index, axis in enumerate(("x", "y", "z")):
        lower, upper = config.hardware.tcp_bounds_uf0_mm[axis]
        if actions[:, index].min() < lower or actions[:, index].max() > upper:
            raise ActionValidationError(
                f"tcp {axis} outside [{lower}, {upper}] mm (UF0)"
            )
    if np.abs(actions[:, 3:6]).max() > config.hardware.angle_limit_deg:
        raise ActionValidationError(
            f"tcp w, p, r must be within +-{config.hardware.angle_limit_deg:g} deg"
        )
    dt = float(dt_s)
    dt_min, dt_max = config.hardware.dt_range_s
    if not math.isfinite(dt) or not dt_min <= dt <= dt_max:
        raise ActionValidationError(f"dt must be in [{dt_min}, {dt_max}] s")
    result = actions.copy()
    result[:, 6] = np.clip(result[:, 6], 0.0, 1.0)
    return result
