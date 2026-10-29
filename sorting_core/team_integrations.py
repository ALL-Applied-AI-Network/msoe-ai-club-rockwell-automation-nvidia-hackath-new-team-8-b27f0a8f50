"""Single composition point for teammate-provided vision and grasp systems.

The working cube demo imports these factories instead of importing a trained
model directly.  Computer-vision and grasp teammates should implement their
components in separate modules and change only the small factory/constants in
this file to activate them.  Keeping model imports inside the factories means
the baseline and contract tests do not acquire a PyTorch/CUDA dependency.

The checked-in defaults deliberately reproduce the existing HSV cube detector
and single top-down grasp.  See ``docs/TEAM_INTEGRATION.md`` before replacing
either default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .config import SortConfig
    from .grasping import GraspPlanningPipeline
    from .perception import VisionProvider


# These class-level values are consumed by the official FANUC client before it
# constructs Policy.  Add a camera here only when the selected provider really
# consumes it and its calibration/timing behavior has been tested.
PERCEPTION_CAMERA_IDS = ("table",)
PERCEPTION_DEPTH_CAMERA_IDS: tuple[str, ...] = ()
PERCEPTION_CALIBRATION_IDS: Mapping[str, str] = {
    "table": "sidclab-cameras.json:table",
}
PERCEPTION_DEPTH_CALIBRATION_IDS: Mapping[str, str] = {
    "table": "sidclab-cameras.json:table_depth",
    "wrist": "sidclab-cameras.json:wrist_depth",
}
# Generic class IDs that the *current cube task adapter* may convert into the
# legacy colour-cube planner. New object classes remain visible in the generic
# scene but require a task/destination adapter before they become pick targets.
CUBE_TASK_CLASS_IDS = ("cube",)

# The official client exposes one legacy ``checkpoint`` argument. These
# component-specific overrides let independently versioned vision and grasp
# artifacts coexist without both teams interpreting the same file as theirs.
VISION_CHECKPOINT_ENV = "ROCKWELL_VISION_CHECKPOINT"
GRASP_CHECKPOINT_ENV = "ROCKWELL_GRASP_CHECKPOINT"


def create_vision_provider(
    *,
    config: "SortConfig",
    camera_projectors: Mapping[str, Any],
    calibration_ids: Mapping[str, str] | None = None,
    expected_cube_edge_m: float | None = None,
    checkpoint: str | None = None,
    detector: Any | None = None,
) -> "VisionProvider":
    """Build the active perception provider.

    ``checkpoint`` is passed through from the official client for a future
    trained provider.  The baseline does not use it.  A replacement must obey
    :class:`sorting_core.perception.VisionProvider` and return robot-base,
    metre/radian observations; it must not return simulator truth.
    """

    del checkpoint
    from .vision import ColourCubeVisionProvider

    return ColourCubeVisionProvider(
        config=config,
        camera_projectors=camera_projectors,
        calibration_ids=(
            PERCEPTION_CALIBRATION_IDS
            if calibration_ids is None
            else calibration_ids
        ),
        camera_id="table",
        expected_cube_edge_m=expected_cube_edge_m,
        detector=detector,
    )


def create_grasp_pipeline(
    *,
    config: "SortConfig",
    checkpoint: str | None = None,
) -> "GraspPlanningPipeline":
    """Build the active grasp candidate/feasibility/scoring pipeline.

    A learned scorer may load its artifact from ``checkpoint`` in a teammate
    implementation.  The default remains dependency-free and selects the same
    single top-down grasp as the pre-integration code.
    """

    del checkpoint
    from .grasping import GraspPlanningPipeline

    return GraspPlanningPipeline.legacy_default(config)


__all__ = [
    "CUBE_TASK_CLASS_IDS",
    "PERCEPTION_CALIBRATION_IDS",
    "PERCEPTION_CAMERA_IDS",
    "PERCEPTION_DEPTH_CALIBRATION_IDS",
    "PERCEPTION_DEPTH_CAMERA_IDS",
    "GRASP_CHECKPOINT_ENV",
    "VISION_CHECKPOINT_ENV",
    "create_grasp_pipeline",
    "create_vision_provider",
]
