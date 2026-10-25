"""Scripted Isaac Sim demo: sort red, yellow, and blue cubes into matching bins.

This is deliberately a *scripted simulation demonstration*.  It uses configured
simulator poses to exercise the existing FANUC/SMC inverse kinematics, physical
grasping, contact, and rendering stack.  It is not a Cosmos or camera-perception
result and must not be presented as one.

The visual bin mesh is loaded directly from the supplied
``manipbench-bin-50x50x55.stl`` reference asset.  Simple static floor/wall box
proxies provide robust collision physics while preserving the supplied visual
geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct

import numpy as np


COLOURS = ("red", "yellow", "blue")
CUBE_SPAWNS = {
    "red": (0.44, -0.19, 0.0),
    "yellow": (0.44, 0.0, 22.0),
    "blue": (0.44, 0.19, -28.0),
}
BIN_CENTRES = {
    "red": (0.24, -0.19),
    "yellow": (0.24, 0.0),
    "blue": (0.24, 0.19),
}
BIN_OUTER_M = 0.050
BIN_POCKET_M = 0.035
BIN_HEIGHT_M = 0.055
BIN_FLOOR_M = 0.030
BIN_POCKET_DEPTH_M = BIN_HEIGHT_M - BIN_FLOOR_M
BIN_MATERIALS = {
    "red": "plastic_red",
    "yellow": "plastic_yellow",
    "blue": "plastic_blue",
}
VIDEO_RESOLUTION = (1280, 720)
FPS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="fresh output directory")
    parser.add_argument(
        "--bin-model",
        default=None,
        help="path to manipbench-bin-50x50x55.stl; defaults to the runtime reference assets",
    )
    parser.add_argument(
        "--physics-only",
        action="store_true",
        help="run and verify physics without rendering PNG frames",
    )
    return parser.parse_args()


def resolve_bin_model(value: str | None) -> Path:
    if value:
        path = Path(value).resolve()
    else:
        assets = os.environ.get("ROCKWELL_REFERENCE_ASSETS")
        if not assets:
            raise RuntimeError(
                "ROCKWELL_REFERENCE_ASSETS is unset; pass --bin-model explicitly"
            )
        path = Path(assets) / "models" / "manipbench-bin-50x50x55.stl"
    if not path.is_file():
        raise FileNotFoundError(f"Supplied bin model not found: {path}")
    return path


def load_stl_triangles(path: Path) -> np.ndarray:
    """Load binary or ASCII STL vertices as an ``(N, 3, 3)`` float array."""
    data = path.read_bytes()
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if 84 + count * 50 == len(data):
            triangles = np.empty((count, 3, 3), dtype=np.float32)
            for i in range(count):
                triangles[i] = np.asarray(
                    struct.unpack_from("<9f", data, 84 + i * 50 + 12),
                    dtype=np.float32,
                ).reshape(3, 3)
            return triangles

    vertices = []
    for raw in data.decode("ascii", errors="strict").splitlines():
        fields = raw.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            vertices.append([float(v) for v in fields[1:]])
    if not vertices or len(vertices) % 3:
        raise ValueError(f"Unsupported or malformed STL: {path}")
    return np.asarray(vertices, dtype=np.float32).reshape(-1, 3, 3)


def ease(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def turn(a, b, t):
    a, b = np.asarray(a, float), np.asarray(b, float)
    angle = math.atan2(a[0] * b[1] - a[1] * b[0], np.dot(a[:2], b[:2])) * t
    c, s = math.cos(angle), math.sin(angle)
    return np.array([a[0] * c - a[1] * s, a[0] * s + a[1] * c, 0.0])


def motion_segments(pick, home, config):
    source = np.array([*pick["xy"], pick["pick_z"]], dtype=float)
    destination = np.array([*pick["place_xy"], pick["place_z"]], dtype=float)
    source_above = source + [0.0, 0.0, 0.13]
    destination_above = destination + [0.0, 0.0, 0.13]
    spread = pick["spread_dir"]
    destination_spread = pick["place_spread_dir"]
    shut = config.CUBE_JAW_HALF_TARGET
    return [
        ("approach", home, source_above, 1.2, config.JAW_OPEN, config.JAW_OPEN, spread),
        ("descend", source_above, source, 0.8, config.JAW_OPEN, config.JAW_OPEN, spread),
        ("close", source, source, 0.6, config.JAW_OPEN, shut, spread),
        ("dwell", source, source, 0.5, shut, shut, spread),
        ("lift", source, source_above, 0.8, shut, shut, spread),
        ("carry", source_above, destination_above, 1.4, shut, shut, destination_spread),
        ("lower", destination_above, destination, 0.8, shut, shut, destination_spread),
        ("release", destination, destination, 0.6, shut, config.JAW_OPEN, destination_spread),
        ("settle", destination, destination, 0.6, config.JAW_OPEN, config.JAW_OPEN, destination_spread),
        ("retreat", destination, destination_above, 0.8, config.JAW_OPEN, config.JAW_OPEN, destination_spread),
        ("home", destination_above, home, 1.1, config.JAW_OPEN, config.JAW_OPEN, None),
    ]


def rotation_matrix_from_quat(quat) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def main() -> None:
    args = parse_args()
    run = Path(args.run).resolve()
    frames = run / "frames"
    if frames.exists() and any(frames.glob("*.png")):
        raise SystemExit(f"Refusing to overwrite rendered frames in {frames}")
    run.mkdir(parents=True, exist_ok=True)
    bin_model = resolve_bin_model(args.bin_model)
    triangles_mm = load_stl_triangles(bin_model)
    bounds_mm = [
        triangles_mm.reshape(-1, 3).min(axis=0).tolist(),
        triangles_mm.reshape(-1, 3).max(axis=0).tolist(),
    ]
    expected_bounds = np.array([[-25.0, -25.0, 0.0], [25.0, 25.0, 55.0]])
    if not np.allclose(np.asarray(bounds_mm), expected_bounds, atol=0.05):
        raise ValueError(f"Unexpected bin model bounds in millimetres: {bounds_mm}")

    # Pure-Python project modules are safe to load before SimulationApp.
    import scene_config as C
    import fk
    import ik

    # Isolate this demo from the starter's six-cube stacking scene.  These are
    # runtime-only overrides; the supplied source and reference assets stay clean.
    C.SORT_VIALS = []
    C.SORT_CAPS = []
    C.CAP_ORIENTATIONS = []
    C.SORT_CUBES = [(*CUBE_SPAWNS[colour][:2], colour) for colour in COLOURS]
    C.SORT_CUBE_YAWS_DEG = [CUBE_SPAWNS[colour][2] for colour in COLOURS]
    C.SHOW_BINS = False
    C.SHOW_FIXTURE = False
    C.TABLE_CAM = dict(C.TABLE_CAM)
    C.TABLE_CAM["resolution"] = VIDEO_RESOLUTION

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    report = None
    try:
        import cv2
        import omni.usd
        from pxr import Gf, UsdGeom, UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera
        import cell

        world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / FPS)
        stage = omni.usd.get_context().get_stage()
        cell.build(stage)

        def add_bin(colour: str) -> None:
            bx, by = BIN_CENTRES[colour]
            path = f"/World/ColourBin_{colour}"
            mesh = UsdGeom.Mesh.Define(stage, path)
            points = triangles_mm.reshape(-1, 3) * 0.001
            mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
            mesh.CreateFaceVertexCountsAttr([3] * len(triangles_mm))
            mesh.CreateFaceVertexIndicesAttr(list(range(len(points))))
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateSubdivisionSchemeAttr("none")
            UsdGeom.Xformable(mesh.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3d(bx, by, C.TABLE_TOP_Z + C.SURFACE_DZ)
            )
            cell.bind_mat(mesh.GetPrim(), BIN_MATERIALS[colour])

            # Reuse the starter's proven static proxy construction, with exact
            # dimensions from the supplied model.  Hide only its procedural
            # visual mesh; its sibling collider prims remain active.
            proxy_path = f"/World/ColourBin_{colour}_Proxy"
            cell._make_bin(
                stage,
                proxy_path,
                xy=(bx, by),
                outer=(BIN_OUTER_M, BIN_OUTER_M),
                pocket_size=BIN_POCKET_M,
                pocket_r=0.004,
                wall_h=BIN_HEIGHT_M,
                floor_t=BIN_FLOOR_M,
                mat_name=BIN_MATERIALS[colour],
            )
            UsdGeom.Imageable(stage.GetPrimAtPath(proxy_path)).CreateVisibilityAttr().Set(
                "invisible"
            )

        for colour in COLOURS:
            add_bin(colour)

        wrappers = {}
        for index, colour in enumerate(COLOURS):
            path = f"/World/SortCube{index}"
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"Missing generated cube: {path}")
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)
            wrappers[colour] = SingleXFormPrim(path)

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
        arm_indices = [dof_names.index(f"J{i}") for i in range(1, 7)]
        joint_index = {name: i for i, name in enumerate(dof_names)}
        limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
        tool = tuple(np.asarray(C.TOOL_AXIS) * cell.tip_offset())
        home = np.array([0.30, 0.0, 0.24])
        down = [0.0, 0.0, -1.0]
        q = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
        q, ok = ik.solve(q, home, down, limits, tool_offset=tool)
        if not ok:
            raise RuntimeError("Home IK failed")

        def command(q6, jaw):
            full = arm.get_joint_positions().copy()
            full[arm_indices] = q6
            for joint, value in cell.jaw_targets(jaw).items():
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

        for index, colour in enumerate(COLOURS):
            UsdPhysics.RigidBodyAPI(
                stage.GetPrimAtPath(f"/World/SortCube{index}")
            ).CreateKinematicEnabledAttr().Set(False)
        for _ in range(30):
            command(q, C.JAW_OPEN)
            world.step(render=camera is not None)

        def cube_pose(colour: str):
            pos, quat = wrappers[colour].get_world_pose()
            rotation = rotation_matrix_from_quat(quat)
            centre = np.asarray(pos, float) + rotation @ np.array([0.0, 0.0, C.CUBE_EDGE / 2])
            yaw = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
            tilt = math.degrees(math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0))))
            return {
                "origin": np.asarray(pos, float).tolist(),
                "centre": centre.tolist(),
                "orientation_wxyz": np.asarray(quat, float).tolist(),
                "yaw_deg": yaw,
                "tilt_deg": tilt,
            }

        initial = {colour: cube_pose(colour) for colour in COLOURS}
        events = []
        frame_number = 0
        frames.mkdir(parents=True, exist_ok=True)

        def save_frame(colour: str, phase: str) -> None:
            nonlocal frame_number
            if camera is None:
                return
            rgba = camera.get_rgba()
            if rgba is None:
                raise RuntimeError("Table camera returned no image")
            image = cv2.cvtColor(
                rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR
            )
            cv2.rectangle(image, (18, 16), (610, 102), (18, 18, 18), -1)
            cv2.putText(
                image,
                "SCRIPTED PHYSICS SIMULATION",
                (34, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"{colour.upper()} cube  ->  {colour.upper()} bin   |   {phase}",
                (34, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (235, 235, 235),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(frames / f"f{frame_number:05d}.png"), image)
            frame_number += 1

        for colour in COLOURS:
            start_all = {name: cube_pose(name) for name in COLOURS}
            start = start_all[colour]
            yaw = start["yaw_deg"]
            theta = math.radians(yaw)
            pick = {
                "colour": colour,
                "xy": start["centre"][:2],
                "pick_z": C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE,
                "spread_dir": [-math.sin(theta), math.cos(theta), 0.0],
                "place_xy": list(BIN_CENTRES[colour]),
                # Release above the 55 mm rim.  The cube then falls under
                # gravity into the 25 mm-deep pocket.
                "place_z": C.SURFACE_DZ + BIN_HEIGHT_M + 0.015,
                "place_spread_dir": [0.0, 1.0, 0.0],
            }
            # Pick one of each direction's equivalent signs once for the entire
            # action.  This prevents a needless 180-degree wrist turn between
            # carry and lower phases.
            current_rotation, _ = fk.fk(q)
            current_spread = current_rotation @ np.asarray(C.FINGER_SPREAD_AXIS)
            source_spread = np.asarray(pick["spread_dir"], float)
            if np.dot(current_spread, source_spread) < 0.0:
                source_spread = -source_spread
            destination_spread = np.asarray(pick["place_spread_dir"], float)
            if np.dot(source_spread, destination_spread) < 0.0:
                destination_spread = -destination_spread
            pick["spread_dir"] = source_spread.tolist()
            pick["place_spread_dir"] = destination_spread.tolist()
            event = {
                "colour": colour,
                "source": start,
                "destination_bin": {
                    "colour": colour,
                    "centre_xy_m": list(BIN_CENTRES[colour]),
                },
                "pickup_verified": False,
                "placement_verified": False,
                "trajectory_ik_failures": 0,
            }

            for phase, p0, p1, duration, jaw0, jaw1, spread in motion_segments(
                pick, home, C
            ):
                spread_start = None
                if spread is not None and phase in ("approach", "carry"):
                    rotation, _ = fk.fk(q)
                    current = rotation @ np.asarray(C.FINGER_SPREAD_AXIS)
                    current[2] = 0.0
                    current /= np.linalg.norm(current)
                    spread = np.asarray(spread, float)
                    if np.dot(current, spread) < 0.0:
                        spread = -spread
                    spread_start = current

                step_count = round(duration * FPS)
                for step in range(step_count):
                    t = ease((step + 1) / step_count)
                    target = np.asarray(p0) + (np.asarray(p1) - np.asarray(p0)) * t
                    direction = (
                        turn(spread_start, spread, t)
                        if spread_start is not None
                        else spread
                    )
                    q, ok = ik.solve(
                        q,
                        target,
                        down,
                        limits,
                        tool_offset=tool,
                        spread_dir=direction,
                    )
                    if not ok:
                        q, ok = ik.solve(
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
                        np.dot(rotation @ np.asarray(C.FINGER_SPREAD_AXIS), direction)
                    ) > 0.999
                    if not ok or np.linalg.norm(actual - target) > 0.001 or not aligned:
                        event["trajectory_ik_failures"] += 1
                        raise RuntimeError(
                            f"IK failed for {colour} during {phase} frame {step}; "
                            "refusing off-target motion"
                        )
                    command(q, jaw0 + (jaw1 - jaw0) * t)
                    for _ in range(2):
                        world.step(render=camera is not None)
                    save_frame(colour, phase)

                if phase == "lift":
                    lifted = cube_pose(colour)
                    lift_m = lifted["centre"][2] - start["centre"][2]
                    event["lift_height_m"] = lift_m
                    event["pickup_verified"] = bool(lift_m > 0.025)

            finish = cube_pose(colour)
            bx, by = BIN_CENTRES[colour]
            cx, cy, cz = finish["centre"]
            half_clearance = (BIN_POCKET_M - C.CUBE_EDGE) / 2.0
            expected_z = C.TABLE_TOP_Z + C.SURFACE_DZ + BIN_FLOOR_M + C.CUBE_EDGE / 2.0
            inside_xy = abs(cx - bx) <= half_clearance and abs(cy - by) <= half_clearance
            on_floor = abs(cz - expected_z) <= 0.004
            event.update(
                final=finish,
                xy_error_mm=float(math.hypot(cx - bx, cy - by) * 1000.0),
                expected_centre_z_m=float(expected_z),
                z_error_mm=float(abs(cz - expected_z) * 1000.0),
                inside_matching_bin_xy=bool(inside_xy),
                resting_on_bin_floor=bool(on_floor),
            )
            event["placement_verified"] = bool(
                event["pickup_verified"] and inside_xy and on_floor
            )
            events.append(event)
            (run / "execution_events.json").write_text(
                json.dumps(events, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"[bucket-demo] {colour}: pickup={event['pickup_verified']} "
                f"matching-bin={event['placement_verified']} "
                f"xy-error={event['xy_error_mm']:.2f} mm "
                f"z-error={event['z_error_mm']:.2f} mm",
                flush=True,
            )
            if not event["placement_verified"]:
                break

        final = {colour: cube_pose(colour) for colour in COLOURS}
        report = {
            "task": "scripted_colour_bucket_sort",
            "decision_source": "configured simulator poses; no camera perception or AI",
            "physics": "Isaac Sim rigid-body contact; cubes are not attached to the gripper",
            "bin_model": {
                "path": str(bin_model),
                "sha256": hashlib.sha256(bin_model.read_bytes()).hexdigest(),
                "triangles": int(len(triangles_mm)),
                "bounds_mm": bounds_mm,
                "outer_m": BIN_OUTER_M,
                "pocket_m": BIN_POCKET_M,
                "height_m": BIN_HEIGHT_M,
                "floor_m": BIN_FLOOR_M,
                "pocket_depth_m": BIN_POCKET_DEPTH_M,
            },
            "video": {
                "rendered": not args.physics_only,
                "fps": FPS,
                "resolution": list(VIDEO_RESOLUTION),
                "frames": frame_number,
            },
            "initial": initial,
            "final": final,
            "events": events,
            "verified_pickups": sum(e["pickup_verified"] for e in events),
            "verified_placements": sum(e["placement_verified"] for e in events),
            "task_complete": len(events) == len(COLOURS)
            and all(e["placement_verified"] for e in events),
        }
        (run / "execution_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[bucket-demo] complete={report['task_complete']} "
            f"pickups={report['verified_pickups']}/3 "
            f"placements={report['verified_placements']}/3 "
            f"frames={frame_number}",
            flush=True,
        )
        if not report["task_complete"]:
            raise RuntimeError("Bucket-sort verification failed; inspect execution_report.json")
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
