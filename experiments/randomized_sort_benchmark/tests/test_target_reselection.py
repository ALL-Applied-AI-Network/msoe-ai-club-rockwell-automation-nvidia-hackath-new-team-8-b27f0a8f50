from __future__ import annotations

from types import SimpleNamespace

import pytest

from experiments.randomized_sort_benchmark.isaac_episode import (
    _AttemptedTargetSignature,
    _final_destination_containment,
    _matches_attempted_target,
)
from experiments.randomized_sort_benchmark.manifest import BenchmarkObject


def _detection(
    *,
    track_id: str,
    x_m: float,
    y_m: float = -0.1,
    class_id: str = "cube",
    sort_label: str = "blue",
) -> SimpleNamespace:
    return SimpleNamespace(
        track_id=track_id,
        class_id=class_id,
        sort_label=sort_label,
        pose=SimpleNamespace(position_base_m=(x_m, y_m, 0.005)),
    )


def test_capture_specific_track_id_does_not_allow_stationary_reselection() -> None:
    first = _detection(track_id="scene-1:blue:0", x_m=0.400)
    attempted = [_AttemptedTargetSignature.from_detection(first)]
    next_capture = _detection(track_id="scene-2:blue:0", x_m=0.402)

    assert _matches_attempted_target(next_capture, attempted)


def test_nearby_same_colour_object_remains_eligible() -> None:
    first = _detection(track_id="scene-1:blue:0", x_m=0.400)
    attempted = [_AttemptedTargetSignature.from_detection(first)]
    adjacent = _detection(track_id="scene-2:blue:1", x_m=0.4105)

    assert not _matches_attempted_target(adjacent, attempted)


def test_materially_moved_target_is_eligible_for_a_fresh_attempt() -> None:
    first = _detection(track_id="stable-track", x_m=0.400)
    attempted = [_AttemptedTargetSignature.from_detection(first)]
    moved = _detection(track_id="stable-track", x_m=0.406)

    assert not _matches_attempted_target(moved, attempted)


def test_class_and_sort_label_are_part_of_camera_signature() -> None:
    attempted = [
        _AttemptedTargetSignature.from_detection(
            _detection(track_id="cube", x_m=0.400)
        )
    ]

    assert not _matches_attempted_target(
        _detection(
            track_id="screw",
            x_m=0.400,
            class_id="screw_iso10642_m6x30",
            sort_label="red",
        ),
        attempted,
    )
    assert not _matches_attempted_target(
        _detection(track_id="red-cube", x_m=0.400, sort_label="red"),
        attempted,
    )


def test_invalid_association_radius_is_rejected() -> None:
    item = _detection(track_id="cube", x_m=0.400)

    with pytest.raises(ValueError, match="finite and positive"):
        _matches_attempted_target(item, (), association_radius_m=0.0)


def _cube_reference() -> BenchmarkObject:
    return BenchmarkObject(
        instance_id="cube-red-0",
        class_name="cube_10mm",
        colour="red",
        destination_colour="red",
        pose_reference="centre",
        position_base_m=(0.4, -0.1, -0.0001),
        yaw_deg=0.0,
        dimensions_m=(0.01, 0.01, 0.01),
        yaw_symmetry_period_deg=90.0,
        local_axis_convention="cube local axes",
    ).validate()


def test_final_cube_containment_uses_final_pose_not_action_event() -> None:
    reference = _cube_reference()
    inside = _final_destination_containment(
        reference,
        {"reference_base_m": [0.60, -0.20, 0.01]},
        bin_centre_xy_m=(0.60, -0.20),
        pocket_m=0.035,
        floor_z_m=0.0,
        rim_z_m=0.055,
        cube_edge_m=0.01,
    )
    ejected = _final_destination_containment(
        reference,
        {"reference_base_m": [0.66, -0.20, 0.01]},
        bin_centre_xy_m=(0.60, -0.20),
        pocket_m=0.035,
        floor_z_m=0.0,
        rim_z_m=0.055,
        cube_edge_m=0.01,
    )

    assert inside["passed"] is True
    assert ejected["passed"] is False
