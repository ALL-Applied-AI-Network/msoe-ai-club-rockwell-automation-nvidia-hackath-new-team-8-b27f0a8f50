"""Separate Isaac Sim smoke demo: sort three cubes and two screws into bins.

The known-good cube-only demo and the real-hardware policy are intentionally not
imported as execution backends or modified by this entry point.  This script
builds its own mixed-object task on top of the supplied, disposable Isaac twin.
It uses configured simulator poses and simulator truth, not Cosmos or camera
perception.  Objects remain dynamic rigid bodies throughout each grasp; nothing
is attached to the gripper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


def _find_repo_root() -> Path:
    candidates: list[Path] = []
    configured = os.environ.get("ROCKWELL_REPO_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser().resolve())
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (
            (candidate / "sorting_core" / "__init__.py").is_file()
            and (candidate / "config" / "cube_screw_bucket_sort.json").is_file()
        ):
            return candidate
    raise RuntimeError(
        "Cannot locate the repository root; set ROCKWELL_REPO_ROOT to the clean clone"
    )


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demos.scripted_cube_screw_bucket_sort.scenario import (  # noqa: E402
    build_completion_report,
    build_sort_items,
    load_scenario_config,
    resolve_screw_model,
    screw_jaw_half_target,
    tool_tip_z_for_desired_grasp_height,
    validate_screw_model,
    verify_vertices_inside_bin,
)


FPS = 30
VIDEO_RESOLUTION = (1280, 720)
BIN_MATERIALS = {
    "red": "plastic_red",
    "yellow": "plastic_yellow",
    "blue": "plastic_blue",
}
_DEFERRED_SIMULATION_APP = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="fresh output directory")
    parser.add_argument(
        "--bin-model",
        default=None,
        help="explicit manipbench-bin-50x50x55.stl path",
    )
    parser.add_argument(
        "--screw-model",
        default=None,
        help="explicit ISO10642-M6x30-countersunk-screw.stl path",
    )
    parser.add_argument(
        "--physics-only",
        action="store_true",
        help="run contact physics and verification without writing PNG frames",
    )
    parser.add_argument(
        "--defer-shutdown",
        action="store_true",
        help=(
            "leave Isaac open so an embedding runner can finish its report "
            "before requesting fast shutdown"
        ),
    )
    return parser.parse_args()


def _take_deferred_simulation_app():
    global _DEFERRED_SIMULATION_APP

    app = _DEFERRED_SIMULATION_APP
    _DEFERRED_SIMULATION_APP = None
    return app


def _reference_model(value: str | None, filename: str) -> Path:
    if value:
        path = Path(value).expanduser().resolve()
    else:
        root = os.environ.get("ROCKWELL_REFERENCE_ASSETS")
        if not root:
            raise RuntimeError(
                f"ROCKWELL_REFERENCE_ASSETS is unset; pass the {filename} path explicitly"
            )
        path = Path(root).expanduser().resolve() / "models" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Supplied model not found: {path}")
    return path


def _ease(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def _turn(start, end, fraction: float) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    angle = math.atan2(
        start[0] * end[1] - start[1] * end[0], np.dot(start[:2], end[:2])
    ) * fraction
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        [
            start[0] * cosine - start[1] * sine,
            start[0] * sine + start[1] * cosine,
            0.0,
        ]
    )


def _rotation_matrix_from_quat(quaternion) -> np.ndarray:
    w, x, y, z = [float(value) for value in quaternion]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _axis_delta_degrees(first, second) -> float:
    """Unsigned axial difference; a screw axis is equivalent after 180 degrees."""

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return math.degrees(math.acos(cosine))


def _motion_segments(pick: dict, home: np.ndarray, scene_config):
    source = np.array([*pick["xy"], pick["pick_z"]], dtype=float)
    destination = np.array([*pick["place_xy"], pick["place_z"]], dtype=float)
    clearance = float(pick["approach_clearance_m"])
    source_above = source + [0.0, 0.0, clearance]
    destination_above = destination + [0.0, 0.0, clearance]
    spread = pick["spread_dir"]
    destination_spread = pick["place_spread_dir"]
    shut = float(pick["jaw_half_target_m"])
    opened = scene_config.JAW_OPEN
    in_bin_open = float(pick.get("in_bin_open_jaw_half_m", opened))
    return [
        ("approach_pick", home, source_above, 1.2, opened, opened, spread),
        (
            "revalidate_before_descent",
            source_above,
            source_above,
            0.0,
            opened,
            opened,
            spread,
        ),
        ("descend_pick", source_above, source, 0.8, opened, opened, spread),
        ("close", source, source, 0.6, opened, shut, spread),
        ("dwell", source, source, 0.5, shut, shut, spread),
        ("lift", source, source_above, 0.8, shut, shut, spread),
        (
            "carry",
            source_above,
            destination_above,
            1.4,
            shut,
            shut,
            destination_spread,
        ),
        (
            "lower",
            destination_above,
            destination,
            0.8,
            shut,
            shut,
            destination_spread,
        ),
        (
            "release",
            destination,
            destination,
            0.6,
            shut,
            in_bin_open,
            destination_spread,
        ),
        (
            "settle",
            destination,
            destination,
            0.6,
            in_bin_open,
            in_bin_open,
            destination_spread,
        ),
        (
            "retreat",
            destination,
            destination_above,
            0.8,
            in_bin_open,
            in_bin_open,
            destination_spread,
        ),
        ("home", destination_above, home, 1.1, in_bin_open, opened, None),
    ]


def main() -> None:
    global _DEFERRED_SIMULATION_APP

    args = parse_args()
    scenario = load_scenario_config()
    task_config = scenario.cube_config
    screw_path = resolve_screw_model(args.screw_model)
    screw_model = validate_screw_model(screw_path, scenario.screw)
    bin_path = _reference_model(
        args.bin_model, "manipbench-bin-50x50x55.stl"
    )
    # The scenario STL reader preserves the supplied model's native millimetres.
    from demos.scripted_cube_screw_bucket_sort.scenario import load_stl_triangles

    bin_triangles_mm = load_stl_triangles(bin_path)
    bin_vertices_mm = np.asarray(
        [vertex for triangle in bin_triangles_mm for vertex in triangle], dtype=float
    )
    expected_bin_bounds = np.array(
        [
            [-task_config.bin.outer_m * 500.0] * 2 + [0.0],
            [task_config.bin.outer_m * 500.0] * 2
            + [task_config.bin.height_m * 1000.0],
        ]
    )
    actual_bin_bounds = np.stack(
        [bin_vertices_mm.min(axis=0), bin_vertices_mm.max(axis=0)]
    )
    if not np.allclose(actual_bin_bounds, expected_bin_bounds, atol=0.05):
        raise ValueError(f"Unexpected bin model bounds in millimetres: {actual_bin_bounds}")

    run = Path(args.run).expanduser().resolve()
    if run.exists() and any(run.iterdir()):
        raise SystemExit(f"Refusing to mix outputs in non-empty run directory {run}")
    run.mkdir(parents=True, exist_ok=True)
    frames = run / "frames"
    if not args.physics_only:
        frames.mkdir(parents=True, exist_ok=True)

    # Import the supplied runtime only after all host-side contracts pass.
    import scene_config as C
    import fk
    import ik

    C.SURFACE_DZ = task_config.frames.surface_base_z_m
    C.CUBE_EDGE = task_config.cube_edge_m
    C.CUBE_TIP_CLEARANCE = task_config.task.pick_tip_clearance_m
    C.CUBE_JAW_HALF_TARGET = max(
        C.JAW_CLOSED,
        C.CUBE_EDGE / 2.0 - C.FINGER_INNER_TIP - 0.0003,
    )
    C.SORT_VIALS = []
    C.SORT_CAPS = []
    C.CAP_ORIENTATIONS = []
    C.SHOW_BINS = False
    C.SHOW_FIXTURE = False
    C.TABLE_CAM = dict(C.TABLE_CAM)
    C.TABLE_CAM["resolution"] = VIDEO_RESOLUTION

    from isaacsim import SimulationApp

    if args.defer_shutdown and _DEFERRED_SIMULATION_APP is not None:
        raise RuntimeError("a deferred Isaac app is already registered")
    app = SimulationApp({"headless": True})
    if args.defer_shutdown:
        _DEFERRED_SIMULATION_APP = app
    try:
        import cv2
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera
        import cell

        colours = tuple(task_config.task.colour_order)
        cube_spawns = dict(task_config.task.cube_spawns_base_xy_yaw_deg)
        bin_centres = dict(task_config.task.bin_centres_base_m)
        C.SORT_CUBES = [(*cube_spawns[colour][:2], colour) for colour in colours]
        C.SORT_CUBE_YAWS_DEG = [cube_spawns[colour][2] for colour in colours]

        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1 / 60,
            rendering_dt=1 / FPS,
        )
        stage = omni.usd.get_context().get_stage()
        cell.build(stage)

        def add_bin(colour: str) -> None:
            bin_x, bin_y = bin_centres[colour]
            visual_path = f"/World/ColourBin_{colour}"
            mesh = UsdGeom.Mesh.Define(stage, visual_path)
            points_m = bin_vertices_mm * 0.001
            mesh.CreatePointsAttr(
                [Gf.Vec3f(*map(float, point)) for point in points_m]
            )
            mesh.CreateFaceVertexCountsAttr([3] * len(bin_triangles_mm))
            mesh.CreateFaceVertexIndicesAttr(list(range(len(points_m))))
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateSubdivisionSchemeAttr("none")
            UsdGeom.Xformable(mesh.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3d(bin_x, bin_y, C.TABLE_TOP_Z + C.SURFACE_DZ)
            )
            cell.bind_mat(mesh.GetPrim(), BIN_MATERIALS[colour])
            proxy_path = f"{visual_path}_Proxy"
            cell._make_bin(
                stage,
                proxy_path,
                xy=(bin_x, bin_y),
                outer=(task_config.bin.outer_m, task_config.bin.outer_m),
                pocket_size=task_config.bin.pocket_m,
                pocket_r=0.004,
                wall_h=task_config.bin.height_m,
                floor_t=task_config.bin.floor_m,
                mat_name=BIN_MATERIALS[colour],
            )
            UsdGeom.Imageable(
                stage.GetPrimAtPath(proxy_path)
            ).CreateVisibilityAttr().Set("invisible")

        for colour in colours:
            add_bin(colour)

        screw_roots: dict[str, str] = {}
        screw_points_m = np.asarray(screw_model.vertices_m, dtype=float)
        for screw in scenario.screws:
            root_path = f"/World/{screw.item_id.replace('-', '_')}"
            root = UsdGeom.Xform.Define(stage, root_path)
            root_xform = UsdGeom.Xformable(root.GetPrim())
            root_xform.AddTranslateOp().Set(
                Gf.Vec3d(
                    screw.spawn_head_base_m[0],
                    screw.spawn_head_base_m[1],
                    screw.spawn_head_base_m[2] + C.TABLE_TOP_Z,
                )
            )
            root_xform.AddRotateZOp().Set(screw.spawn_yaw_deg)
            root_xform.AddRotateYOp().Set(-90.0)
            body = UsdGeom.Mesh.Define(stage, f"{root_path}/CadBody")
            body.CreatePointsAttr(
                [Gf.Vec3f(*map(float, point)) for point in screw_points_m]
            )
            body.CreateFaceVertexCountsAttr([3] * screw_model.triangle_count)
            body.CreateFaceVertexIndicesAttr(list(range(len(screw_points_m))))
            body.CreateDoubleSidedAttr(True)
            body.CreateSubdivisionSchemeAttr("none")
            body.CreateDisplayColorAttr([Gf.Vec3f(0.10, 0.10, 0.12)])
            cell.bind_mat(body.GetPrim(), "black_oxide")
            UsdPhysics.CollisionAPI.Apply(body.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(
                body.GetPrim()
            ).CreateApproximationAttr().Set("convexDecomposition")
            decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(
                body.GetPrim()
            )
            decomposition.CreateHullVertexLimitAttr().Set(64)
            decomposition.CreateMaxConvexHullsAttr().Set(8)
            decomposition.CreateErrorPercentageAttr().Set(0.5)
            cell.bind_physics_material(body.GetPrim())
            cell.small_contact_offsets(body.GetPrim())
            cell.make_dynamic(
                root.GetPrim(), scenario.screw.mass_kg, collide_self=False
            )
            UsdPhysics.RigidBodyAPI(root.GetPrim()).CreateKinematicEnabledAttr().Set(True)
            screw_roots[screw.item_id] = root_path

        wrappers: dict[str, SingleXFormPrim] = {}
        for index, colour in enumerate(colours):
            path = f"/World/SortCube{index}"
            if not stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(f"Missing generated cube: {path}")
            UsdPhysics.RigidBodyAPI(
                stage.GetPrimAtPath(path)
            ).CreateKinematicEnabledAttr().Set(True)
            wrappers[f"cube-{colour}"] = SingleXFormPrim(path)
        for item_id, path in screw_roots.items():
            wrappers[item_id] = SingleXFormPrim(path)

        camera = None
        if not args.physics_only:
            camera = Camera(
                prim_path=cell.TABLE_CAM_PATH,
                resolution=VIDEO_RESOLUTION,
            )

        cell.enable_ccd(world)
        world.reset()
        if camera is not None:
            camera.initialize()

        arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
        arm.initialize()
        dof_names = list(arm.dof_names)
        arm_indices = [dof_names.index(f"J{index}") for index in range(1, 7)]
        joint_index = {name: index for index, name in enumerate(dof_names)}
        limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
        tool = tuple(np.asarray(C.TOOL_AXIS) * cell.tip_offset())
        home = np.array([0.30, 0.0, 0.24])
        down = [0.0, 0.0, -1.0]
        q = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
        q, solved = ik.solve(q, home, down, limits, tool_offset=tool)
        if not solved:
            raise RuntimeError("Home IK failed")

        def command(joints, jaw_half_opening: float) -> None:
            full = arm.get_joint_positions().copy()
            full[arm_indices] = joints
            for joint, value in cell.jaw_targets(jaw_half_opening).items():
                full[joint_index[joint]] = value
            arm.apply_action(ArticulationAction(joint_positions=full))

        full = arm.get_joint_positions().copy()
        full[arm_indices] = q
        for joint, value in cell.jaw_targets(C.JAW_OPEN).items():
            full[joint_index[joint]] = value
        arm.set_joint_positions(full)
        for _ in range(120):
            command(q, C.JAW_OPEN)
            world.step(render=camera is not None)

        for index, _colour in enumerate(colours):
            UsdPhysics.RigidBodyAPI(
                stage.GetPrimAtPath(f"/World/SortCube{index}")
            ).CreateKinematicEnabledAttr().Set(False)
        for path in screw_roots.values():
            UsdPhysics.RigidBodyAPI(
                stage.GetPrimAtPath(path)
            ).CreateKinematicEnabledAttr().Set(False)
        for _ in range(90):
            command(q, C.JAW_OPEN)
            world.step(render=camera is not None)

        def object_pose(item_id: str, kind: str) -> dict:
            position, quaternion = wrappers[item_id].get_world_pose()
            position = np.asarray(position, dtype=float)
            quaternion = np.asarray(quaternion, dtype=float)
            rotation = _rotation_matrix_from_quat(quaternion)
            if kind == "cube":
                centre = position + rotation @ np.array(
                    [0.0, 0.0, C.CUBE_EDGE / 2.0]
                )
                yaw_rad = math.atan2(rotation[1, 0], rotation[0, 0])
                return {
                    "origin_world_m": position.tolist(),
                    "reference_world_m": centre.tolist(),
                    "grasp_point_world_m": centre.tolist(),
                    "axis_world": [math.cos(yaw_rad), math.sin(yaw_rad), 0.0],
                    "orientation_wxyz": quaternion.tolist(),
                }
            axis = rotation @ np.array([0.0, 0.0, -1.0])
            grasp_point = position + axis * scenario.screw.grasp_offset_from_head_m
            vertices_world = screw_points_m @ rotation.T + position
            return {
                "origin_world_m": position.tolist(),
                "reference_world_m": position.tolist(),
                "grasp_point_world_m": grasp_point.tolist(),
                "axis_world": axis.tolist(),
                "orientation_wxyz": quaternion.tolist(),
                "cad_bounds_world_m": [
                    vertices_world.min(axis=0).tolist(),
                    vertices_world.max(axis=0).tolist(),
                ],
            }

        frame_number = 0

        def save_frame(item_id: str, kind: str, destination: str, phase: str) -> None:
            nonlocal frame_number
            if camera is None:
                return
            rgba = camera.get_rgba()
            if rgba is None:
                raise RuntimeError("Table camera returned no image")
            image = cv2.cvtColor(rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, (18, 16), (680, 102), (18, 18, 18), -1)
            cv2.putText(
                image,
                "SCRIPTED CUBE + SCREW PHYSICS SIMULATION",
                (34, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"{item_id.upper()} ({kind}) -> {destination.upper()} bin | {phase}",
                (34, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (235, 235, 235),
                2,
                cv2.LINE_AA,
            )
            path = frames / f"f{frame_number:05d}.png"
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"OpenCV could not write {path}")
            frame_number += 1

        items = build_sort_items(scenario)
        initial = {
            item.item_id: object_pose(item.item_id, item.kind) for item in items
        }
        screw_jaw_target = screw_jaw_half_target(
            shank_diameter_m=scenario.screw.shank_diameter_m,
            finger_inner_tip_m=C.FINGER_INNER_TIP,
            seating_margin_m=scenario.screw.jaw_seating_margin_m,
            jaw_closed_m=C.JAW_CLOSED,
            jaw_open_m=C.JAW_OPEN,
        )
        events: list[dict] = []
        semantic_phases = [
            "approach_pick",
            "revalidate_before_descent",
            "descend_pick",
            "close",
            "dwell",
            "lift",
            "carry",
            "lower",
            "release",
            "settle",
            "retreat",
        ]

        for item in items:
            start = object_pose(item.item_id, item.kind)
            grasp = np.asarray(start["grasp_point_world_m"], dtype=float)
            axis_xy = np.asarray(start["axis_world"][:2], dtype=float)
            if np.linalg.norm(axis_xy) <= 1e-9:
                raise RuntimeError(f"{item.item_id} has no horizontal grasp axis")
            axis_xy /= np.linalg.norm(axis_xy)
            source_spread = np.array([-axis_xy[1], axis_xy[0], 0.0])
            if item.kind == "cube":
                destination_spread = np.array([0.0, 1.0, 0.0])
                place_xy = np.asarray(bin_centres[item.destination_colour], dtype=float)
                jaw_target = C.CUBE_JAW_HALF_TARGET
                place_tip_z = task_config.bin_release_tip_base_z_m
                in_bin_open_jaw = C.JAW_OPEN
                placement_height_plan = {
                    "mode": "cube release above bin rim",
                    "commanded_tip_base_z_m": float(place_tip_z),
                }
            else:
                target_yaw = math.radians(item.target_axis_yaw_deg)
                destination_axis = np.array(
                    [math.cos(target_yaw), math.sin(target_yaw), 0.0]
                )
                destination_spread = np.array(
                    [-destination_axis[1], destination_axis[0], 0.0]
                )
                place_xy = np.asarray(
                    item.placement_grasp_point_base_m[:2], dtype=float
                )
                jaw_target = screw_jaw_target
                place_tip_z, grasp_above_tip_m = (
                    tool_tip_z_for_desired_grasp_height(
                        desired_grasp_base_z_m=(
                            item.placement_grasp_point_base_m[2]
                        ),
                        observed_grasp_world_z_m=grasp[2],
                        pickup_tip_base_z_m=task_config.pick_tip_base_z_m,
                        table_top_world_z_m=C.TABLE_TOP_Z,
                    )
                )
                # An approximately 8 mm hook gap clears the 6 mm shank while
                # keeping the in-bin finger envelope smaller than it would be
                # at the 6 mm full stroke. Full opening happens only after the
                # vertical retreat.
                in_bin_open_jaw = min(C.JAW_OPEN, jaw_target + 0.0013)
                placement_height_plan = {
                    "mode": "configured screw grasp height",
                    "desired_grasp_point_base_z_m": float(
                        item.placement_grasp_point_base_m[2]
                    ),
                    "observed_grasp_above_pickup_tip_m": float(
                        grasp_above_tip_m
                    ),
                    "commanded_tip_base_z_m": float(place_tip_z),
                }

            current_rotation, _ = fk.fk(q)
            current_spread = current_rotation @ np.asarray(C.FINGER_SPREAD_AXIS)
            if np.dot(current_spread, source_spread) < 0.0:
                source_spread = -source_spread
            if np.dot(source_spread, destination_spread) < 0.0:
                destination_spread = -destination_spread
            pick = {
                "xy": grasp[:2].tolist(),
                "pick_z": task_config.pick_tip_base_z_m,
                "spread_dir": source_spread.tolist(),
                "place_xy": place_xy.tolist(),
                "place_z": place_tip_z,
                "place_spread_dir": destination_spread.tolist(),
                "jaw_half_target_m": jaw_target,
                "in_bin_open_jaw_half_m": in_bin_open_jaw,
                "approach_clearance_m": task_config.task.approach_clearance_m,
            }
            event = {
                "item_id": item.item_id,
                "object_type": item.kind,
                "destination_colour": item.destination_colour,
                "source_truth": start,
                "scripted_pick_grasp_point_base_xy_m": grasp[:2].tolist(),
                "jaw_half_target_m": float(jaw_target),
                "in_bin_open_jaw_half_m": float(in_bin_open_jaw),
                "placement_height_plan": placement_height_plan,
                "shared_semantic_phases": semantic_phases,
                "pickup_verified": False,
                "placement_verified": False,
                "trajectory_ik_failures": 0,
            }
            segments = _motion_segments(pick, home, C)
            for phase, start_xyz, end_xyz, duration, jaw0, jaw1, spread in segments:
                if phase == "revalidate_before_descent":
                    refreshed = object_pose(item.item_id, item.kind)
                    refreshed_grasp = np.asarray(
                        refreshed["grasp_point_world_m"], dtype=float
                    )
                    movement_xy_m = float(
                        np.linalg.norm(refreshed_grasp[:2] - grasp[:2])
                    )
                    movement_z_m = float(abs(refreshed_grasp[2] - grasp[2]))
                    axis_delta_deg = (
                        _axis_delta_degrees(
                            start["axis_world"], refreshed["axis_world"]
                        )
                        if item.kind == "screw"
                        else None
                    )
                    revalidation_passed = bool(
                        movement_xy_m <= 0.002
                        and movement_z_m <= 0.001
                        and (axis_delta_deg is None or axis_delta_deg <= 3.0)
                    )
                    event["pre_descent_revalidation"] = {
                        "source": "simulator ground truth, scripted demo only",
                        "target_movement_xy_m": movement_xy_m,
                        "maximum_allowed_xy_m": 0.002,
                        "target_movement_z_m": movement_z_m,
                        "maximum_allowed_z_m": 0.001,
                        "screw_axis_delta_deg": axis_delta_deg,
                        "maximum_allowed_screw_axis_delta_deg": (
                            3.0 if item.kind == "screw" else None
                        ),
                        "passed": revalidation_passed,
                    }
                    if not revalidation_passed:
                        raise RuntimeError(
                            f"{item.item_id} changed before descent "
                            f"(xy={movement_xy_m * 1000.0:.2f} mm, "
                            f"z={movement_z_m * 1000.0:.2f} mm, "
                            f"axis={axis_delta_deg}); refusing a stale pick"
                        )
                    continue

                spread_start = None
                if spread is not None and phase in {
                    "approach_pick",
                    "descend_pick",
                    "carry",
                }:
                    rotation, _ = fk.fk(q)
                    current = rotation @ np.asarray(C.FINGER_SPREAD_AXIS)
                    current[2] = 0.0
                    current /= np.linalg.norm(current)
                    spread = np.asarray(spread, dtype=float)
                    if np.dot(current, spread) < 0.0:
                        spread = -spread
                    spread_start = current

                step_count = round(duration * FPS)
                for step in range(step_count):
                    fraction = _ease((step + 1) / step_count)
                    target = np.asarray(start_xyz) + (
                        np.asarray(end_xyz) - np.asarray(start_xyz)
                    ) * fraction
                    direction = (
                        _turn(spread_start, spread, fraction)
                        if spread_start is not None
                        else spread
                    )
                    q, solved = ik.solve(
                        q,
                        target,
                        down,
                        limits,
                        tool_offset=tool,
                        spread_dir=direction,
                    )
                    if not solved:
                        q, solved = ik.solve(
                            q,
                            target,
                            down,
                            limits,
                            tool_offset=tool,
                            spread_dir=direction,
                            iters=500,
                            damping=0.02,
                        )
                    rotation, actual = fk.tool_pose(q, tool)
                    aligned = direction is None or abs(
                        np.dot(
                            rotation @ np.asarray(C.FINGER_SPREAD_AXIS), direction
                        )
                    ) > 0.999
                    if not solved or np.linalg.norm(actual - target) > 0.001 or not aligned:
                        event["trajectory_ik_failures"] += 1
                        raise RuntimeError(
                            f"IK failed for {item.item_id} during {phase} frame {step}"
                        )
                    jaw = jaw0 + (jaw1 - jaw0) * fraction
                    command(q, jaw)
                    for _ in range(2):
                        world.step(render=camera is not None)
                    save_frame(
                        item.item_id,
                        item.kind,
                        item.destination_colour,
                        phase,
                    )

                if phase == "lift":
                    lifted = object_pose(item.item_id, item.kind)
                    lift_m = float(
                        np.asarray(lifted["grasp_point_world_m"])[2]
                        - np.asarray(start["grasp_point_world_m"])[2]
                    )
                    event["lift_height_m"] = lift_m
                    event["pickup_verified"] = lift_m > 0.025

                if item.kind == "screw" and phase in {
                    "lift",
                    "carry",
                    "lower",
                    "release",
                    "settle",
                    "retreat",
                }:
                    event.setdefault("phase_truth", {})[phase] = object_pose(
                        item.item_id, item.kind
                    )

            stability_passed = True
            if item.kind == "screw":
                # Let the released 7.5 g part dissipate its impact, then require
                # a short pose-stability window. A single in-bin snapshot could
                # otherwise accept a screw while it is bouncing through the bin.
                for _ in range(45):
                    command(q, C.JAW_OPEN)
                    world.step(render=camera is not None)
                stability_positions = []
                stability_axes = []
                for _ in range(15):
                    command(q, C.JAW_OPEN)
                    world.step(render=camera is not None)
                    sample = object_pose(item.item_id, item.kind)
                    stability_positions.append(
                        np.asarray(sample["origin_world_m"], dtype=float)
                    )
                    stability_axes.append(
                        np.asarray(sample["axis_world"], dtype=float)
                    )
                position_span_m = max(
                    float(np.linalg.norm(first - second))
                    for first in stability_positions
                    for second in stability_positions
                )
                axis_span_deg = max(
                    _axis_delta_degrees(first, second)
                    for first in stability_axes
                    for second in stability_axes
                )
                stability_passed = bool(
                    position_span_m <= 0.001 and axis_span_deg <= 3.0
                )
                event["post_place_stability"] = {
                    "warmup_physics_steps": 45,
                    "sampled_physics_steps": 15,
                    "position_span_m": position_span_m,
                    "maximum_allowed_position_span_m": 0.001,
                    "axis_span_deg": axis_span_deg,
                    "maximum_allowed_axis_span_deg": 3.0,
                    "passed": stability_passed,
                }

            finish = object_pose(item.item_id, item.kind)
            bin_x, bin_y = bin_centres[item.destination_colour]
            if item.kind == "cube":
                centre = np.asarray(finish["reference_world_m"], dtype=float)
                half_clearance = (
                    task_config.bin.pocket_m - C.CUBE_EDGE
                ) / 2.0
                expected_z = (
                    C.TABLE_TOP_Z
                    + C.SURFACE_DZ
                    + task_config.bin.floor_m
                    + C.CUBE_EDGE / 2.0
                )
                inside_xy = (
                    abs(centre[0] - bin_x) <= half_clearance
                    and abs(centre[1] - bin_y) <= half_clearance
                )
                on_floor = abs(centre[2] - expected_z) <= 0.004
                placement_detail = {
                    "inside_matching_bin_xy": bool(inside_xy),
                    "resting_on_bin_floor": bool(on_floor),
                    "xy_error_mm": float(
                        math.hypot(centre[0] - bin_x, centre[1] - bin_y) * 1000.0
                    ),
                    "z_error_mm": float(abs(centre[2] - expected_z) * 1000.0),
                }
                placement_passed = inside_xy and on_floor
            else:
                position = np.asarray(finish["origin_world_m"], dtype=float)
                rotation = _rotation_matrix_from_quat(
                    finish["orientation_wxyz"]
                )
                vertices_world = screw_points_m @ rotation.T + position
                containment = verify_vertices_inside_bin(
                    vertices_world,
                    bin_centre_xy_m=(bin_x, bin_y),
                    pocket_m=task_config.bin.pocket_m,
                    floor_z_m=(
                        C.TABLE_TOP_Z + C.SURFACE_DZ + task_config.bin.floor_m
                    ),
                    rim_z_m=(
                        C.TABLE_TOP_Z + C.SURFACE_DZ + task_config.bin.height_m
                    ),
                    corner_radius_m=0.004,
                    tolerance_m=0.001,
                )
                placement_detail = {
                    "full_cad_bin_containment": containment.as_dict(),
                    "verification_uses_every_cad_vertex": True,
                }
                placement_passed = containment.passed and stability_passed
            event["final_truth"] = finish
            event["placement_verification"] = placement_detail
            event["placement_verified"] = bool(
                event["pickup_verified"] and placement_passed
            )
            events.append(event)
            (run / "execution_events.json").write_text(
                json.dumps(events, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"[cube-screw-demo] {item.item_id}: "
                f"pickup={event['pickup_verified']} "
                f"placement={event['placement_verified']}",
                flush=True,
            )
            if not event["placement_verified"]:
                break

        # Re-check the whole scene after the final action. Per-item success at
        # an earlier time is not enough: a later insertion can disturb a cube
        # or screw that was already accepted.
        for _ in range(45):
            command(q, C.JAW_OPEN)
            world.step(render=camera is not None)
        final_samples: dict[str, list[dict]] = {
            item.item_id: [] for item in items
        }
        for _ in range(15):
            command(q, C.JAW_OPEN)
            world.step(render=camera is not None)
            for item in items:
                final_samples[item.item_id].append(
                    object_pose(item.item_id, item.kind)
                )
            save_frame(
                "final scene",
                "5 objects",
                "all matching",
                "final verification",
            )

        final = {
            item.item_id: object_pose(item.item_id, item.kind) for item in items
        }
        final_scene_items: dict[str, dict] = {}
        for item in items:
            samples = final_samples[item.item_id]
            positions = [
                np.asarray(sample["origin_world_m"], dtype=float)
                for sample in samples
            ]
            axes = [
                np.asarray(sample["axis_world"], dtype=float)
                for sample in samples
            ]
            position_span_m = max(
                float(np.linalg.norm(first - second))
                for first in positions
                for second in positions
            )
            axis_span_deg = max(
                _axis_delta_degrees(first, second)
                for first in axes
                for second in axes
            )
            stability = {
                "position_span_m": position_span_m,
                "maximum_allowed_position_span_m": 0.001,
                "axis_span_deg": axis_span_deg,
                "maximum_allowed_axis_span_deg": 3.0,
                "passed": bool(
                    position_span_m <= 0.001 and axis_span_deg <= 3.0
                ),
            }

            pose = final[item.item_id]
            bin_x, bin_y = bin_centres[item.destination_colour]
            if item.kind == "cube":
                centre = np.asarray(pose["reference_world_m"], dtype=float)
                half_clearance = (
                    task_config.bin.pocket_m - C.CUBE_EDGE
                ) / 2.0
                expected_z = (
                    C.TABLE_TOP_Z
                    + C.SURFACE_DZ
                    + task_config.bin.floor_m
                    + C.CUBE_EDGE / 2.0
                )
                geometry = {
                    "inside_matching_bin_xy": bool(
                        abs(centre[0] - bin_x) <= half_clearance
                        and abs(centre[1] - bin_y) <= half_clearance
                    ),
                    "resting_on_bin_floor": bool(
                        abs(centre[2] - expected_z) <= 0.004
                    ),
                }
                geometry_passed = all(geometry.values())
            else:
                position = np.asarray(pose["origin_world_m"], dtype=float)
                rotation = _rotation_matrix_from_quat(
                    pose["orientation_wxyz"]
                )
                vertices_world = screw_points_m @ rotation.T + position
                containment = verify_vertices_inside_bin(
                    vertices_world,
                    bin_centre_xy_m=(bin_x, bin_y),
                    pocket_m=task_config.bin.pocket_m,
                    floor_z_m=(
                        C.TABLE_TOP_Z + C.SURFACE_DZ + task_config.bin.floor_m
                    ),
                    rim_z_m=(
                        C.TABLE_TOP_Z + C.SURFACE_DZ + task_config.bin.height_m
                    ),
                    corner_radius_m=0.004,
                    tolerance_m=0.001,
                )
                geometry = {
                    "full_cad_bin_containment": containment.as_dict(),
                    "verification_uses_every_cad_vertex": True,
                }
                geometry_passed = containment.passed
            item_passed = bool(stability["passed"] and geometry_passed)
            final_scene_items[item.item_id] = {
                "object_type": item.kind,
                "destination_colour": item.destination_colour,
                "stability": stability,
                "geometry": geometry,
                "passed": item_passed,
            }

        final_scene_verified = sum(
            int(result["passed"]) for result in final_scene_items.values()
        )
        final_scene_verification = {
            "warmup_physics_steps": 45,
            "sampled_physics_steps": 15,
            "items": final_scene_items,
            "verified_items": final_scene_verified,
            "expected_items": len(items),
            "passed": final_scene_verified == len(items),
        }
        completion = build_completion_report(scenario, events)
        completion["task_complete"] = bool(
            completion["task_complete"]
            and final_scene_verification["passed"]
        )
        completion["completion_summary"] = (
            f"{completion['completion_summary']} "
            f"final_scene={final_scene_verified}/{len(items)}"
        )
        report = {
            "task": "scripted_cube_screw_bucket_sort",
            "scope": (
                "separate configured-pose Isaac smoke demo; existing cube demo and "
                "real-hardware policy are unchanged"
            ),
            "decision_source": "configured simulator poses and simulator truth",
            "decision_uses_simulator_truth": True,
            "camera_perception": False,
            "cosmos_or_ai_perception": False,
            "physics": (
                "Isaac Sim rigid-body contact; cubes and screws are never attached "
                "to the gripper"
            ),
            "scenario_config": {
                "path": scenario.path.relative_to(REPO_ROOT).as_posix(),
                "sha256": hashlib.sha256(scenario.path.read_bytes()).hexdigest(),
                "schema_version": scenario.schema_version,
                "sort_order": list(scenario.sort_order),
            },
            "models": {
                "bin": {
                    "path": str(bin_path),
                    "sha256": hashlib.sha256(bin_path.read_bytes()).hexdigest(),
                    "triangles": len(bin_triangles_mm),
                    "bounds_mm": actual_bin_bounds.tolist(),
                },
                "screw": {
                    "path": str(screw_model.path),
                    "sha256": screw_model.sha256,
                    "triangles": screw_model.triangle_count,
                    "bounds_mm": [list(bound) for bound in screw_model.bounds_mm],
                    "mass_kg": scenario.screw.mass_kg,
                    "collision_approximation": "convexDecomposition",
                },
            },
            "video": {
                "rendered": not args.physics_only,
                "fps": FPS,
                "resolution": list(VIDEO_RESOLUTION),
                "frames": frame_number,
            },
            "initial_truth": initial,
            "final_truth": final,
            "final_scene_verification": final_scene_verification,
            "events": events,
            **completion,
        }
        (run / "execution_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[cube-screw-demo] complete={report['task_complete']} "
            f"{report['completion_summary']} frames={frame_number}",
            flush=True,
        )
        if not report["task_complete"]:
            raise RuntimeError(
                "Cube-and-screw verification failed; inspect execution_report.json"
            )
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        if not args.defer_shutdown:
            app.close(
                exit_code=1 if sys.exc_info()[0] is not None else 0
            )


if __name__ == "__main__":
    main()
