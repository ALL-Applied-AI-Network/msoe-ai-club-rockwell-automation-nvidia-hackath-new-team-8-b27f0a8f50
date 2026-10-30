"""Dependency-light example vision factory preserving baseline behavior.

Copy this module under a new name and replace only the provider implementation.
Keep heavyweight model imports out of module scope so profile loading and the
baseline do not initialize an unrelated model runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sorting_core.integration_plugins import VisionFactoryContext
    from sorting_core.perception import VisionProvider


PLUGIN_API_VERSION = 1


def create(context: "VisionFactoryContext") -> "VisionProvider":
    """Build the existing camera-derived cube provider from plugin context."""

    from sorting_core.vision import ColourCubeVisionProvider

    return ColourCubeVisionProvider(
        config=context.config,
        camera_projectors=context.camera_projectors,
        calibration_ids=context.calibration_ids,
        camera_id="table",
        expected_cube_edge_m=context.expected_cube_edge_m,
        detector=context.detector,
    )


__all__ = ["PLUGIN_API_VERSION", "create"]
