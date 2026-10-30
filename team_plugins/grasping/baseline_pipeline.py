"""Dependency-light example grasp factory preserving baseline behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sorting_core.grasping import GraspPlanningPipeline
    from sorting_core.integration_plugins import GraspFactoryContext


PLUGIN_API_VERSION = 1


def create(context: "GraspFactoryContext") -> "GraspPlanningPipeline":
    """Build the current single-candidate top-down cube pipeline."""

    from sorting_core.grasping import GraspPlanningPipeline

    return GraspPlanningPipeline.legacy_default(context.config)


__all__ = ["PLUGIN_API_VERSION", "create"]
