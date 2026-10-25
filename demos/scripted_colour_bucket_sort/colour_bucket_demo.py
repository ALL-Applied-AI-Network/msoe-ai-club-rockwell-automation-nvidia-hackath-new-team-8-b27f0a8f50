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
import sys

import numpy as np


def _find_repo_root() -> Path:
    """Find the checkout even when the runtime twin is the current directory."""
    candidates = []
    configured = os.environ.get("ROCKWELL_REPO_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser().resolve())
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (
            (candidate / "sorting_core" / "__init__.py").is_file()
            and (candidate / "config" / "colour_bucket_sort.json").is_file()
        ):
            return candidate
    raise RuntimeError(
        "Cannot locate the repository root containing sorting_core and "
        "config/colour_bucket_sort.json; set ROCKWELL_REPO_ROOT"
    )


# The demo is normally invoked by absolute path after ``cd`` into the disposable
# twin runtime.  Import the backend-neutral task package from the checkout, not
# from the current working directory or from the pristine twin source.
REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sorting_core import (  # noqa: E402
    CubeDetection,
    DEFAULT_CONFIG_PATH,
    SceneDetections,
    build_colour_sort_plans,
    load_config,
    semantic_phase_plan,
)


TASK_CONFIG = load_config()
COLOURS = TASK_CONFIG.task.colour_order
CUBE_SPAWNS = dict(TASK_CONFIG.task.cube_spawns_base_xy_yaw_deg)
BIN_CENTRES = dict(TASK_CONFIG.task.bin_centres_base_m)
BIN_OUTER_M = TASK_CONFIG.bin.outer_m
BIN_POCKET_M = TASK_CONFIG.bin.pocket_m
BIN_HEIGHT_M = TASK_CONFIG.bin.height_m
BIN_FLOOR_M = TASK_CONFIG.bin.floor_m
BIN_POCKET_DEPTH_M = TASK_CONFIG.bin.pocket_depth_m


def _shared_semantic_phase_names() -> tuple[str, ...]:
    """Resolve names through the same planner contract used by hardware."""
    capture_time_s = 0.0
    scene = SceneDetections(
        scene_id="configured-isaac-smoke-scene",
        capture_time_s=capture_time_s,
        cubes=tuple(
            CubeDetection(
                track_id=f"configured-{colour}",
                colour=colour,
                capture_time_s=capture_time_s,
                position_base_m=(
                    CUBE_SPAWNS[colour][0],
                    CUBE_SPAWNS[colour][1],
                    TASK_CONFIG.cube_centre_base_z_m,
                ),
                yaw_rad=math.radians(CUBE_SPAWNS[colour][2]),
            )
            for colour in COLOURS
        ),
    )
    plans = build_colour_sort_plans(scene, TASK_CONFIG)
    phase_sequences = {
        tuple(phase.name for phase in semantic_phase_plan(plan, TASK_CONFIG))
        for plan in plans
    }
    if len(phase_sequences) != 1:
        raise RuntimeError("Shared planner produced inconsistent semantic phases")
    return phase_sequences.pop()


SHARED_SEMANTIC_PHASE_NAMES = _shared_semantic_phase_names()

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


def motion_segments(pick, home, scene_config):
    source = np.array([*pick["xy"], pick["pick_z"]], dtype=float)
    destination = np.array([*pick["place_xy"], pick["place_z"]], dtype=float)
    clearance = TASK_CONFIG.task.approach_clearance_m
    source_above = source + [0.0, 0.0, clearance]
    destination_above = destination + [0.0, 0.0, clearance]
    spread = pick["spread_dir"]
    destination_spread = pick["place_spread_dir"]
    shut = scene_config.CUBE_JAW_HALF_TARGET
    return [
        (
            "approach_pick",
            home,
            source_above,
            1.2,
            scene_config.JAW_OPEN,
            scene_config.JAW_OPEN,
            spread,
        ),
        (
            "revalidate_before_descent",
            source_above,
            source_above,
            0.0,
            scene_config.JAW_OPEN,
            scene_config.JAW_OPEN,
            spread,
        ),
        (
            "descend_pick",
            source_above,
            source,
            0.8,
            scene_config.JAW_OPEN,
            scene_config.JAW_OPEN,
            spread,
        ),
        ("close", source, source, 0.6, scene_config.JAW_OPEN, shut, spread),
        ("dwell", source, source, 0.5, shut, shut, spread),
        ("lift", source, source_above, 0.8, shut, shut, spread),
        ("carry", source_above, destination_above, 1.4, shut, shut, destination_spread),
        ("lower", destination_above, destination, 0.8, shut, shut, destination_spread),
        (
            "release",
            destination,
            destination,
            0.6,
            shut,
            scene_config.JAW_OPEN,
            destination_spread,
        ),
        (
            "settle",
            destination,
            destination,
            0.6,
            scene_config.JAW_OPEN,
            scene_config.JAW_OPEN,
            destination_spread,
        ),
        (
            "retreat",
            destination,
            destination_above,
            0.8,
            scene_config.JAW_OPEN,
            scene_config.JAW_OPEN,
            destination_spread,
        ),
        (
            "home",
            destination_above,
            home,
            1.1,
            scene_config.JAW_OPEN,
            scene_config.JAW_OPEN,
            None,
        ),
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
    print(
        f"[bucket-demo] shared config schema={TASK_CONFIG.schema_version}; "
        f"cube edge={TASK_CONFIG.cube_edge_m * 1000.0:.1f} mm",
        flush=True,
    )
    print(f"[bucket-demo] {TASK_CONFIG.cube_size_note}", flush=True)
    print(
        "[bucket-demo] ACTIVE TCP CONVENTION: UTOOL1/TCP is at the fingertips "
        f"({TASK_CONFIG.frames.tcp_beyond_fingertips_mm:.1f} mm beyond).",
        flush=True,
    )
    print(
        f"[bucket-demo] UNRESOLVED TCP CONFLICT: {TASK_CONFIG.frames.tcp_conflict_note}",
        flush=True,
    )
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
    half_outer_mm = BIN_OUTER_M * 500.0
    expected_bounds = np.array(
        [
            [-half_outer_mm, -half_outer_mm, 0.0],
            [half_outer_mm, half_outer_mm, BIN_HEIGHT_M * 1000.0],
        ]
    )
    if not np.allclose(np.asarray(bounds_mm), expected_bounds, atol=0.05):
        raise ValueError(f"Unexpected bin model bounds in millimetres: {bounds_mm}")

    # Pure-Python project modules are safe to load before SimulationApp.
    import scene_config as C
    import fk
    import ik

    # Isolate this demo from the starter's six-cube stacking scene.  These are
    # runtime-only overrides; the supplied source and reference assets stay clean.
    C.SURFACE_DZ = TASK_CONFIG.frames.surface_base_z_m
    C.CUBE_EDGE = TASK_CONFIG.cube_edge_m
    C.CUBE_TIP_CLEARANCE = TASK_CONFIG.task.pick_tip_clearance_m
    C.CUBE_JAW_HALF_TARGET = max(
        C.JAW_CLOSED,
        C.CUBE_EDGE / 2.0 - C.FINGER_INNER_TIP - 0.0003,
    )
    if 2.0 * (C.JAW_OPEN + C.FINGER_INNER_TIP) <= C.CUBE_EDGE * math.sqrt(2.0):
        raise RuntimeError(
            "Configured cube diagonal does not fit the simulated open jaws"
        )
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

        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1 / 60,
            rendering_dt=1 / FPS,
        )
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
            centre = np.asarray(pos, float) + rotation @ np.array(
                [0.0, 0.0, C.CUBE_EDGE / 2]
            )
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
                "pick_z": TASK_CONFIG.pick_tip_base_z_m,
                "spread_dir": [-math.sin(theta), math.cos(theta), 0.0],
                "place_xy": list(BIN_CENTRES[colour]),
                # Release above the 55 mm rim.  The cube then falls under
                # gravity into the 25 mm-deep pocket.
                "place_z": TASK_CONFIG.bin_release_tip_base_z_m,
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
                "shared_semantic_phases": list(SHARED_SEMANTIC_PHASE_NAMES),
            }

            segments = motion_segments(pick, home, C)
            actual_semantic_names = tuple(segment[0] for segment in segments[:-1])
            if actual_semantic_names != SHARED_SEMANTIC_PHASE_NAMES:
                raise RuntimeError(
                    "Isaac adapter semantic phases differ from the shared planner: "
                    f"{actual_semantic_names!r} != {SHARED_SEMANTIC_PHASE_NAMES!r}"
                )
            for phase, p0, p1, duration, jaw0, jaw1, spread in segments:
                if phase == "revalidate_before_descent":
                    current = cube_pose(colour)
                    moved_xy_m = float(
                        np.linalg.norm(
                            np.asarray(current["centre"][:2])
                            - np.asarray(start["centre"][:2])
                        )
                    )
                    # This smoke demo has no perception backend.  Preserve the
                    # shared stop/revalidate gate using simulator truth and label
                    # that limitation explicitly; the hardware policy must use a
                    # fresh camera observation instead.
                    tolerance_m = min(0.002, TASK_CONFIG.cube_edge_m / 4.0)
                    event["pre_descent_revalidation"] = {
                        "source": "simulator_ground_truth_not_camera",
                        "target_movement_xy_m": moved_xy_m,
                        "maximum_allowed_m": tolerance_m,
                        "passed": bool(moved_xy_m <= tolerance_m),
                    }
                    if moved_xy_m > tolerance_m:
                        raise RuntimeError(
                            f"{colour} target moved {moved_xy_m * 1000.0:.2f} mm "
                            "before descent; refusing stale scripted pick"
                        )
                    continue

                spread_start = None
                if spread is not None and phase in ("approach_pick", "carry"):
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
            "shared_task_contract": {
                "config_path": DEFAULT_CONFIG_PATH.relative_to(REPO_ROOT).as_posix(),
                "config_sha256": hashlib.sha256(
                    DEFAULT_CONFIG_PATH.read_bytes()
                ).hexdigest(),
                "schema_version": TASK_CONFIG.schema_version,
                "colour_order": list(COLOURS),
                "cube_edge_m": TASK_CONFIG.cube_edge_m,
                "cube_size_note": TASK_CONFIG.cube_size_note,
                "cube_spawns_base_xy_yaw_deg": {
                    colour: list(CUBE_SPAWNS[colour]) for colour in COLOURS
                },
                "bin_centres_base_m": {
                    colour: list(BIN_CENTRES[colour]) for colour in COLOURS
                },
                "surface_base_z_m": TASK_CONFIG.frames.surface_base_z_m,
                "pick_tip_clearance_m": (
                    TASK_CONFIG.task.pick_tip_clearance_m
                ),
                "approach_clearance_m": (
                    TASK_CONFIG.task.approach_clearance_m
                ),
                "bin_release_clearance_m": (
                    TASK_CONFIG.task.bin_release_clearance_m
                ),
                "tcp_convention": "UTOOL1/TCP at fingertips",
                "tcp_beyond_fingertips_mm": (
                    TASK_CONFIG.frames.tcp_beyond_fingertips_mm
                ),
                "tcp_conflict_unresolved": True,
                "tcp_conflict_note": TASK_CONFIG.frames.tcp_conflict_note,
                "semantic_phase_names": list(SHARED_SEMANTIC_PHASE_NAMES),
                "isaac_adapter_extension_phases": ["home"],
                "isaac_revalidation_source": "simulator ground truth, not camera",
            },
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
