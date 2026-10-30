"""Single composition point for teammate-provided vision and grasp systems.

The working cube demo imports these factories instead of importing a trained
model directly. Computer-vision and grasp teammates may add separate modules
and select them through ``ROCKWELL_INTEGRATION_PROFILE`` without editing this
file. Model modules remain lazy so the baseline and contract tests do not
acquire a PyTorch/CUDA dependency.

The checked-in defaults deliberately reproduce the existing HSV cube detector
and single top-down grasp.  See ``docs/TEAM_INTEGRATION.md`` before replacing
either default.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping

from .integration_plugins import (
    GraspFactoryContext,
    INTEGRATION_PROFILE_ENV,
    IntegrationLoadError,
    VisionFactoryContext,
    create_profile_grasp_pipeline,
    create_profile_vision_provider,
    load_active_integration_profile,
)

if TYPE_CHECKING:
    from .config import SortConfig
    from .grasping import GraspPlanningPipeline
    from .perception import VisionProvider


_LOGGER = logging.getLogger(__name__)
_ACTIVE_INTEGRATION_PROFILE = load_active_integration_profile()
ACTIVE_INTEGRATION_PROFILE = _ACTIVE_INTEGRATION_PROFILE
INTEGRATION_PROFILE_PROVENANCE = (
    None
    if _ACTIVE_INTEGRATION_PROFILE is None
    else _ACTIVE_INTEGRATION_PROFILE.provenance
)
if _ACTIVE_INTEGRATION_PROFILE is not None:
    _LOGGER.info(
        "loaded integration profile id=%s sha256=%s source=%s",
        _ACTIVE_INTEGRATION_PROFILE.profile_id,
        _ACTIVE_INTEGRATION_PROFILE.source_sha256,
        _ACTIVE_INTEGRATION_PROFILE.source_path,
    )


# These class-level values are consumed by the official FANUC client before it
# constructs Policy. Profile metadata is therefore validated at import while
# the referenced model module remains lazy.
_BASELINE_CAMERA_IDS = ("table",)
_BASELINE_DEPTH_CAMERA_IDS: tuple[str, ...] = ()
_BASELINE_CALIBRATION_IDS: Mapping[str, str] = {
    "table": "sidclab-cameras.json:table",
}
_BASELINE_DEPTH_CALIBRATION_IDS: Mapping[str, str] = {
    "table": "sidclab-cameras.json:table_depth",
    "wrist": "sidclab-cameras.json:wrist_depth",
}
_ACTIVE_VISION = (
    None
    if _ACTIVE_INTEGRATION_PROFILE is None
    else _ACTIVE_INTEGRATION_PROFILE.vision
)
PERCEPTION_CAMERA_IDS = (
    _BASELINE_CAMERA_IDS if _ACTIVE_VISION is None else _ACTIVE_VISION.camera_ids
)
PERCEPTION_DEPTH_CAMERA_IDS = (
    _BASELINE_DEPTH_CAMERA_IDS
    if _ACTIVE_VISION is None
    else _ACTIVE_VISION.depth_camera_ids
)
PERCEPTION_CALIBRATION_IDS: Mapping[str, str] = (
    _BASELINE_CALIBRATION_IDS
    if _ACTIVE_VISION is None
    else _ACTIVE_VISION.calibration_ids
)
PERCEPTION_DEPTH_CALIBRATION_IDS: Mapping[str, str] = (
    _BASELINE_DEPTH_CALIBRATION_IDS
    if _ACTIVE_VISION is None
    else _ACTIVE_VISION.depth_calibration_ids
)
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

    if _ACTIVE_INTEGRATION_PROFILE is not None and _ACTIVE_VISION is not None:
        selected_calibration_ids = (
            _ACTIVE_VISION.calibration_ids
            if calibration_ids is None
            else calibration_ids
        )
        _LOGGER.info(
            "creating vision plugin factory=%s profile_id=%s sha256=%s",
            _ACTIVE_VISION.factory,
            _ACTIVE_INTEGRATION_PROFILE.profile_id,
            _ACTIVE_INTEGRATION_PROFILE.source_sha256,
        )
        return create_profile_vision_provider(
            _ACTIVE_INTEGRATION_PROFILE,
            VisionFactoryContext(
                config=config,
                camera_projectors=camera_projectors,
                calibration_ids=selected_calibration_ids,
                expected_cube_edge_m=expected_cube_edge_m,
                checkpoint=checkpoint,
                detector=detector,
                options=_ACTIVE_VISION.options,
                camera_ids=_ACTIVE_VISION.camera_ids,
                depth_camera_ids=_ACTIVE_VISION.depth_camera_ids,
                declared_calibration_ids=_ACTIVE_VISION.calibration_ids,
                declared_depth_calibration_ids=(
                    _ACTIVE_VISION.depth_calibration_ids
                ),
                profile_id=_ACTIVE_INTEGRATION_PROFILE.profile_id,
                profile_path=_ACTIVE_INTEGRATION_PROFILE.source_path,
                profile_sha256=_ACTIVE_INTEGRATION_PROFILE.source_sha256,
            ),
        )

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

    active_grasp = (
        None
        if _ACTIVE_INTEGRATION_PROFILE is None
        else _ACTIVE_INTEGRATION_PROFILE.grasp
    )
    if _ACTIVE_INTEGRATION_PROFILE is not None and active_grasp is not None:
        _LOGGER.info(
            "creating grasp plugin factory=%s profile_id=%s sha256=%s",
            active_grasp.factory,
            _ACTIVE_INTEGRATION_PROFILE.profile_id,
            _ACTIVE_INTEGRATION_PROFILE.source_sha256,
        )
        return create_profile_grasp_pipeline(
            _ACTIVE_INTEGRATION_PROFILE,
            GraspFactoryContext(
                config=config,
                checkpoint=checkpoint,
                options=active_grasp.options,
                profile_id=_ACTIVE_INTEGRATION_PROFILE.profile_id,
                profile_path=_ACTIVE_INTEGRATION_PROFILE.source_path,
                profile_sha256=_ACTIVE_INTEGRATION_PROFILE.source_sha256,
            ),
        )

    del checkpoint
    from .grasping import GraspPlanningPipeline

    return GraspPlanningPipeline.legacy_default(config)


__all__ = [
    "ACTIVE_INTEGRATION_PROFILE",
    "CUBE_TASK_CLASS_IDS",
    "GRASP_CHECKPOINT_ENV",
    "INTEGRATION_PROFILE_ENV",
    "INTEGRATION_PROFILE_PROVENANCE",
    "IntegrationLoadError",
    "PERCEPTION_CALIBRATION_IDS",
    "PERCEPTION_CAMERA_IDS",
    "PERCEPTION_DEPTH_CALIBRATION_IDS",
    "PERCEPTION_DEPTH_CAMERA_IDS",
    "VISION_CHECKPOINT_ENV",
    "create_grasp_pipeline",
    "create_vision_provider",
]
