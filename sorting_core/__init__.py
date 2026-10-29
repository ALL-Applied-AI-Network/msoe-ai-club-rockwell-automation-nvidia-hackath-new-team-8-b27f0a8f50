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
    GraspSelectionProvenance,
    MotionPhase,
    PickPlacePlan,
    SceneDetections,
    TipPoseBase,
    TrajectoryChunk,
    normalise_square_yaw_rad,
)
from .grasping import (
    ConstantSuccessScorer,
    EvaluatedGrasp,
    FeasibilityResult,
    GraspCandidate,
    GraspFeatures,
    GraspPlanningContext,
    GraspPlanningDecision,
    GraspPlanningError,
    GraspPlanningPipeline,
    GraspScore,
    HighestSuccessSelector,
    LegacyFeatureExtractor,
    LegacyGeometricChecker,
    LegacyTopDownCubeGenerator,
    RejectedGrasp,
    SuccessTimeSelector,
)
from .perception import (
    CameraFrame,
    DepthEncoding,
    DetectedObject,
    PerceptionRequest,
    Pose3D,
    RobotState,
    SceneObservation,
    VisionProvider,
    scene_observation_from_dict,
    scene_observation_to_dict,
    validate_scene_for_request,
)
from .planner import (
    PlanningError,
    build_colour_sort_plans,
    semantic_phase_plan,
    square_yaw_delta_degrees,
    validate_colour_sort_scene,
    validate_reobserved_scene,
)
from .task_perception import (
    CubeSceneProjection,
    cube_scene_from_observation,
    cube_scene_to_observation,
)


_LAZY_EXPORTS = {
    "ActionValidationError": (".frames", "ActionValidationError"),
    "base_tip_pose_to_uf0_tcp": (".frames", "base_tip_pose_to_uf0_tcp"),
    "uf0_tcp_to_base_tip": (".frames", "uf0_tcp_to_base_tip"),
    "validate_tcp_chunk": (".frames", "validate_tcp_chunk"),
    "build_trajectory_chunks": (".trajectory", "build_trajectory_chunks"),
    "interpolate_tcp_rows": (".trajectory", "interpolate_tcp_rows"),
    "VisionError": (".vision", "VisionError"),
    "ColourCubeVisionProvider": (".vision", "ColourCubeVisionProvider"),
    "detection_in_pickup_region": (".vision", "detection_in_pickup_region"),
    "detect_coloured_cubes": (".vision", "detect_coloured_cubes"),
    "filter_scene_to_pickup_region": (
        ".vision",
        "filter_scene_to_pickup_region",
    ),
    "SyntheticDataError": (".synthetic_data", "SyntheticDataError"),
    "augment_sensor_rgb": (".synthetic_data", "augment_sensor_rgb"),
    "build_synchronized_frame_record": (
        ".synthetic_data",
        "build_synchronized_frame_record",
    ),
    "finalize_application_status": (
        ".synthetic_data",
        "finalize_application_status",
    ),
    "load_capture_calibration": (
        ".synthetic_data",
        "load_capture_calibration",
    ),
    "load_synthetic_data_spec": (
        ".synthetic_data",
        "load_synthetic_data_spec",
    ),
    "sample_randomization": (".synthetic_data", "sample_randomization"),
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
    "CameraFrame",
    "ColourCubeVisionProvider",
    "ConfigurationError",
    "ConstantSuccessScorer",
    "CubeDetection",
    "CubeSceneProjection",
    "DEFAULT_CONFIG_PATH",
    "DetectedObject",
    "DepthEncoding",
    "EvaluatedGrasp",
    "FeasibilityResult",
    "GraspCandidate",
    "GraspFeatures",
    "GraspPlanningContext",
    "GraspPlanningDecision",
    "GraspPlanningError",
    "GraspPlanningPipeline",
    "GraspScore",
    "GraspSelectionProvenance",
    "HighestSuccessSelector",
    "LegacyFeatureExtractor",
    "LegacyGeometricChecker",
    "LegacyTopDownCubeGenerator",
    "RejectedGrasp",
    "RobotState",
    "MotionPhase",
    "PickPlacePlan",
    "PlanningError",
    "PerceptionRequest",
    "Pose3D",
    "SceneObservation",
    "SceneDetections",
    "SortConfig",
    "SuccessTimeSelector",
    "SyntheticDataError",
    "TipPoseBase",
    "TrajectoryChunk",
    "VisionError",
    "VisionProvider",
    "base_tip_pose_to_uf0_tcp",
    "build_colour_sort_plans",
    "build_trajectory_chunks",
    "build_synchronized_frame_record",
    "config_from_dict",
    "cube_scene_from_observation",
    "cube_scene_to_observation",
    "detection_in_pickup_region",
    "detect_coloured_cubes",
    "filter_scene_to_pickup_region",
    "augment_sensor_rgb",
    "finalize_application_status",
    "interpolate_tcp_rows",
    "load_config",
    "load_capture_calibration",
    "load_synthetic_data_spec",
    "normalise_square_yaw_rad",
    "scene_observation_from_dict",
    "scene_observation_to_dict",
    "semantic_phase_plan",
    "square_yaw_delta_degrees",
    "sample_randomization",
    "uf0_tcp_to_base_tip",
    "validate_tcp_chunk",
    "validate_colour_sort_scene",
    "validate_reobserved_scene",
    "validate_scene_for_request",
]
