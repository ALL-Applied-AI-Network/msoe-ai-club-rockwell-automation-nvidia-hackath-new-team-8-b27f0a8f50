"""Shared task contract for Isaac simulation and the FANUC policy client.

All coordinates exposed by this package are in the robot-base frame and SI
units unless an API explicitly says ``uf0`` or ``mm``.  The checked-in default
uses the active **10 mm cube assumption**; callers should surface
``SortConfig.cube_size_note`` until the physical parts are measured.

The package root intentionally avoids importing NumPy or OpenCV.  Isaac scripts
may safely load configuration before ``SimulationApp``; numerical and vision
exports below are resolved lazily on first use.
"""

from importlib import import_module

from .config import (
    ConfigurationError,
    DEFAULT_CONFIG_PATH,
    SortConfig,
    config_from_dict,
    load_config,
)
from .contracts import (
    BinTarget,
    CubeDetection,
    MotionPhase,
    PickPlacePlan,
    SceneDetections,
    TipPoseBase,
    TrajectoryChunk,
    normalise_square_yaw_rad,
)
from .planner import PlanningError, build_colour_sort_plans, semantic_phase_plan


_LAZY_EXPORTS = {
    "ActionValidationError": (".frames", "ActionValidationError"),
    "base_tip_pose_to_uf0_tcp": (".frames", "base_tip_pose_to_uf0_tcp"),
    "uf0_tcp_to_base_tip": (".frames", "uf0_tcp_to_base_tip"),
    "validate_tcp_chunk": (".frames", "validate_tcp_chunk"),
    "build_trajectory_chunks": (".trajectory", "build_trajectory_chunks"),
    "interpolate_tcp_rows": (".trajectory", "interpolate_tcp_rows"),
    "VisionError": (".vision", "VisionError"),
    "detect_coloured_cubes": (".vision", "detect_coloured_cubes"),
}


def __getattr__(name):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

__all__ = [
    "ActionValidationError",
    "BinTarget",
    "ConfigurationError",
    "CubeDetection",
    "DEFAULT_CONFIG_PATH",
    "MotionPhase",
    "PickPlacePlan",
    "PlanningError",
    "SceneDetections",
    "SortConfig",
    "TipPoseBase",
    "TrajectoryChunk",
    "VisionError",
    "base_tip_pose_to_uf0_tcp",
    "build_colour_sort_plans",
    "build_trajectory_chunks",
    "config_from_dict",
    "detect_coloured_cubes",
    "interpolate_tcp_rows",
    "load_config",
    "normalise_square_yaw_rad",
    "semantic_phase_plan",
    "uf0_tcp_to_base_tip",
    "validate_tcp_chunk",
]
