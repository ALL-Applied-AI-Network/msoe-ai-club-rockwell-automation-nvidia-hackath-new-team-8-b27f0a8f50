"""Camera-observation bridge between Isaac Sim and :mod:`sorting_core`.

The package deliberately has no Isaac imports.  Isaac entry points can hand it
native RGB pixels and the camera calibration read from USD, while host-side
tests can exercise the same perception and planning path without a GPU.
"""

from .camera_sort import (
    CalibratedPinholeCamera,
    CameraSortDecision,
    IsaacCameraSortAdapter,
    IsaacObservationError,
)

__all__ = [
    "CalibratedPinholeCamera",
    "CameraSortDecision",
    "IsaacCameraSortAdapter",
    "IsaacObservationError",
]
