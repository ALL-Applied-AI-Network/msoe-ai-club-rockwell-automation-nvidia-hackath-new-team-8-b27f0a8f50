"""Scripted Isaac Sim demo: sort red, yellow, and blue cubes into matching bins.

The default mode is deliberately a *scripted simulation demonstration*.  It
uses configured simulator poses to exercise the existing FANUC/SMC inverse
kinematics, physical grasping, contact, and rendering stack.  That default is
not a Cosmos or camera-perception result and must not be presented as one.  The
explicit ``--camera-perception`` and ``--training-data`` modes instead make
runtime target decisions from raw rendered pixels; neither mode uses Cosmos.

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
    TipPoseBase,
    augment_sensor_rgb,
    base_tip_pose_to_uf0_tcp,
    build_colour_sort_plans,
    build_synchronized_frame_record,
    finalize_application_status,
    load_capture_calibration,
    load_config,
    load_synthetic_data_spec,
    sample_randomization,
    semantic_phase_plan,
    square_yaw_delta_degrees,
    validate_reobserved_scene,
)
from sorting_core.synthetic_data import distort_points_uv, write_json_atomic  # noqa: E402
from isaac_bridge import CalibratedPinholeCamera, IsaacCameraSortAdapter  # noqa: E402
from isaac_bridge.training_episode import (  # noqa: E402
    apply_postbuild_randomization,
    prepare_prebuild_randomization,
    readback_effective_camera,
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
NATIVE_TABLE_RESOLUTION = (1920, 1080)
FPS = 30
CAMERA_REVALIDATION_MAXIMUM_MOTION_M = 0.005
CAMERA_REVALIDATION_MAXIMUM_YAW_DEG = 15.0


def _should_stop_after_truth_evaluation(
    *, camera_perception: bool, placement_verified: bool
) -> bool:
    """Keep evaluator-only truth from controlling camera/training actions."""

    return (not camera_perception) and (not placement_verified)


def _phase_interpolates_spread(phase: str) -> bool:
    """Identify motion phases that must rotate from measured current yaw."""

    return phase in {"approach_pick", "descend_pick", "carry"}


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
    parser.add_argument(
        "--camera-perception",
        action="store_true",
        help=(
            "select and revalidate picks from raw table-camera RGB using the "
            "same deterministic detector/planner as hardware"
        ),
    )
    parser.add_argument(
        "--training-data",
        action="store_true",
        help=(
            "enable native-resolution seeded domain randomization, lossless raw "
            "frames, derived sensor frames, and synchronized labels"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261025,
        help="episode seed used only with --training-data",
    )
    parser.add_argument(
        "--dataset-stride",
        type=int,
        default=15,
        help="save one training record every N rendered presentation frames",
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


def _clipped_amodal_bbox(
    projected_vertices_uv: np.ndarray,
    resolution: tuple[int, int],
) -> tuple[list[float] | None, str]:
    """Bound a projected full 3D cuboid without implying visible pixels.

    The result is clipped to image boundaries.  A cuboid whose projection has
    no positive-area intersection with the image is explicitly invalid rather
    than emitted as a degenerate box.
    """

    vertices = np.asarray(projected_vertices_uv, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1:] != (2,) or not np.all(
        np.isfinite(vertices)
    ):
        raise ValueError("projected cuboid vertices must be finite Nx2 values")
    width, height = (float(value) for value in resolution)
    unbounded = np.array(
        [
            vertices[:, 0].min(),
            vertices[:, 1].min(),
            vertices[:, 0].max(),
            vertices[:, 1].max(),
        ],
        dtype=float,
    )
    if (
        unbounded[2] <= 0.0
        or unbounded[3] <= 0.0
        or unbounded[0] >= width
        or unbounded[1] >= height
    ):
        return None, "invalid_outside_image"
    clipped = np.array(
        [
            np.clip(unbounded[0], 0.0, width),
            np.clip(unbounded[1], 0.0, height),
            np.clip(unbounded[2], 0.0, width),
            np.clip(unbounded[3], 0.0, height),
        ],
        dtype=float,
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None, "invalid_outside_image"
    status = "valid" if np.array_equal(clipped, unbounded) else "valid_clipped"
    return clipped.tolist(), status


def main() -> None:
    args = parse_args()
    if args.dataset_stride < 1:
        raise SystemExit("--dataset-stride must be at least 1")
    camera_perception = bool(args.camera_perception or args.training_data)
    if args.physics_only and camera_perception:
        raise SystemExit(
            "--physics-only cannot be combined with camera perception or training-data capture"
        )
    render_resolution = (
        NATIVE_TABLE_RESOLUTION if camera_perception else VIDEO_RESOLUTION
    )
    synthetic_spec = None
    randomization = None
    capture_calibration = None
    applied_randomization: set[str] = set()
    stage_randomization_readback = None
    effective_camera = None
    if args.training_data:
        synthetic_spec = load_synthetic_data_spec()
        randomization = sample_randomization(
            synthetic_spec,
            episode_seed=args.seed,
            episode_id=Path(args.run).resolve().name,
        )
        capture_calibration = load_capture_calibration(synthetic_spec)
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
        "[bucket-demo] TCP FAQ DISCREPANCY RESOLVED; live installed frame "
        f"still unverified: {TASK_CONFIG.frames.tcp_conflict_note}",
        flush=True,
    )
    run = Path(args.run).resolve()
    frames = run / "frames"
    if run.exists() and any(run.iterdir()):
        raise SystemExit(f"Refusing to mix outputs in non-empty run directory {run}")
    run.mkdir(parents=True, exist_ok=True)
    decision_frames = run / "decision_frames"
    if camera_perception:
        decision_frames.mkdir(parents=True, exist_ok=True)
    if randomization is not None:
        write_json_atomic(run / "randomization_requested.json", randomization.as_dict())
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
    C.SHOW_BINS = False
    C.SHOW_FIXTURE = False
    C.TABLE_CAM = dict(C.TABLE_CAM)
    C.TABLE_CAM["resolution"] = render_resolution

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

        active_spawns = dict(CUBE_SPAWNS)
        if randomization is not None:
            active_spawns, prebuild_applied = prepare_prebuild_randomization(
                C, cell, randomization, CUBE_SPAWNS
            )
            applied_randomization.update(prebuild_applied)
        C.TABLE_CAM["resolution"] = render_resolution
        C.SORT_CUBES = [(*active_spawns[colour][:2], colour) for colour in COLOURS]
        C.SORT_CUBE_YAWS_DEG = [active_spawns[colour][2] for colour in COLOURS]

        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1 / 60,
            rendering_dt=1 / FPS,
        )
        stage = omni.usd.get_context().get_stage()
        cell.build(stage)
        if randomization is not None:
            (
                postbuild_applied,
                stage_randomization_readback,
            ) = apply_postbuild_randomization(stage, cell, randomization)
            applied_randomization.update(postbuild_applied)
            write_json_atomic(
                run / "isaac_randomization_readback.json",
                stage_randomization_readback,
            )

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
                resolution=render_resolution,
            )

        cell.enable_ccd(world)
        world.reset()
        if camera is not None:
            camera.initialize()

        camera_model = None
        camera_adapter = None
        if camera is not None:
            if randomization is not None:
                effective_camera, camera_applied = readback_effective_camera(
                    stage,
                    camera_prim_path=cell.TABLE_CAM_PATH,
                    resolution_px=render_resolution,
                    robot_base_origin_world_m=(0.0, 0.0, C.TABLE_TOP_Z),
                    expected_camera=C.TABLE_CAM,
                    base_camera=C._TRAINING_RANDOMIZATION_BASE["camera"],
                    manifest=randomization,
                )
                applied_randomization.update(camera_applied)
                write_json_atomic(run / "effective_camera.json", effective_camera)
                intrinsics = effective_camera["intrinsics"]
                camera_to_base = effective_camera["extrinsics"]["robot_base"][
                    "camera_to_robot_base_row_major"
                ]
                camera_model = CalibratedPinholeCamera(
                    resolution_px=tuple(intrinsics["resolution_px"]),
                    focal_length_mm=intrinsics["focal_length_mm"],
                    horizontal_aperture_mm=intrinsics[
                        "horizontal_aperture_mm"
                    ],
                    vertical_aperture_mm=intrinsics["vertical_aperture_mm"],
                    camera_to_base_row_major=tuple(
                        tuple(row) for row in camera_to_base
                    ),
                    default_plane_base_z_m=C.SURFACE_DZ,
                )
            else:
                eye_base = np.asarray(C.TABLE_CAM["eye"], dtype=float)
                look_at_base = np.asarray(C.TABLE_CAM["look_at"], dtype=float)
                # The supplied USD stage places the robot-base origin at world
                # TABLE_TOP_Z.  The shared policy contract uses robot-base Z=0.
                eye_base[2] -= C.TABLE_TOP_Z
                look_at_base[2] -= C.TABLE_TOP_Z
                camera_model = CalibratedPinholeCamera.from_look_at(
                    eye_base_m=eye_base,
                    look_at_base_m=look_at_base,
                    up_base=C.TABLE_CAM["up"],
                    resolution_px=render_resolution,
                    focal_length_mm=C.TABLE_CAM["focal_length"],
                    horizontal_aperture_mm=C.TABLE_CAM["horiz_aperture"],
                    default_plane_base_z_m=C.SURFACE_DZ,
                )
            camera_adapter = IsaacCameraSortAdapter(camera_model, TASK_CONFIG)

        arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
        arm.initialize()
        dof_names = list(arm.dof_names)
        arm_indices = [dof_names.index(f"J{i}") for i in range(1, 7)]
        joint_index = {name: i for i, name in enumerate(dof_names)}
        jaw_open_joint_targets = cell.jaw_targets(C.JAW_OPEN)
        jaw_closed_joint_targets = cell.jaw_targets(C.JAW_CLOSED)
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

        def fk_tip_pose(q6):
            rotation, tip_xyz = fk.tool_pose(q6, tool)
            spread_xy = (
                rotation @ np.asarray(C.FINGER_SPREAD_AXIS, dtype=float)
            )[:2]
            if np.linalg.norm(spread_xy) <= 1e-9:
                raise RuntimeError("FK has no horizontal finger-spread axis")
            return TipPoseBase(
                tuple(float(value) for value in tip_xyz),
                math.atan2(-float(spread_xy[0]), float(spread_xy[1])),
            )

        def articulation_readback():
            """Read measured joints/gripper and derive the measured FK tip pose."""

            full = np.asarray(arm.get_joint_positions(), dtype=float)
            if full.shape != (len(dof_names),) or not np.all(np.isfinite(full)):
                raise RuntimeError("Isaac articulation returned invalid joint positions")
            measured_q = full[arm_indices]
            closure_samples = []
            for joint, open_target in jaw_open_joint_targets.items():
                closed_target = jaw_closed_joint_targets[joint]
                span = closed_target - open_target
                if abs(span) <= 1e-12:
                    raise RuntimeError(f"Jaw joint {joint} has zero configured stroke")
                closure_samples.append(
                    (full[joint_index[joint]] - open_target) / span
                )
            raw_closure_fraction = float(np.mean(closure_samples))
            # PhysX drive readback can exceed an authored endpoint by a tiny
            # floating-point epsilon. Preserve that raw value in the label,
            # but normalize only within a deliberately small tolerance.
            closure_tolerance = 1e-4
            if not -closure_tolerance <= raw_closure_fraction <= 1.0 + closure_tolerance:
                raise RuntimeError(
                    "Measured jaw closure is outside the configured open/closed "
                    f"stroke: {raw_closure_fraction:.6f}"
                )
            closure_fraction = float(np.clip(raw_closure_fraction, 0.0, 1.0))
            jaw_half_opening_m = float(
                C.JAW_OPEN - closure_fraction * (C.JAW_OPEN - C.JAW_CLOSED)
            )

            measured_tip = fk_tip_pose(measured_q)
            protocol_gripper = (
                TASK_CONFIG.gripper.open_command
                + closure_fraction
                * (
                    TASK_CONFIG.gripper.closed_command
                    - TASK_CONFIG.gripper.open_command
                )
            )
            tcp = base_tip_pose_to_uf0_tcp(
                measured_tip, protocol_gripper, TASK_CONFIG
            )
            return (
                measured_q,
                jaw_half_opening_m,
                closure_fraction,
                raw_closure_fraction,
                tcp[:6],
            )

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

        def capture_rgb() -> np.ndarray:
            if camera is None:
                raise RuntimeError("Camera capture requested in physics-only mode")
            rgba = camera.get_rgba()
            if rgba is None:
                raise RuntimeError("Table camera returned no image")
            return rgba[:, :, :3].astype(np.uint8)

        def save_decision_frame(filename: str, rgb: np.ndarray) -> str:
            """Persist the exact unannotated RGB used for a runtime decision."""

            path = decision_frames / filename
            if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"OpenCV could not write {path}")
            return path.relative_to(run).as_posix()

        initial = {colour: cube_pose(colour) for colour in COLOURS}
        initial_camera_decision = None
        initial_camera_observation_relpath = None
        observed_targets = {}
        if camera_perception:
            initial_rgb = capture_rgb()
            initial_camera_observation_relpath = save_decision_frame(
                "initial.png", initial_rgb
            )
            initial_camera_decision = camera_adapter.observe_and_plan(
                initial_rgb,
                scene_id="isaac-camera-initial",
                capture_time_s=float(world.current_time),
            )
            observed_targets = {
                plan.target.colour: plan.target
                for plan in initial_camera_decision.plans
            }
            print(
                "[bucket-demo] camera perception found "
                + ", ".join(
                    f"{item.colour}@({item.position_base_m[0]:.4f},"
                    f"{item.position_base_m[1]:.4f})"
                    for item in initial_camera_decision.scene.cubes
                ),
                flush=True,
            )
        events = []
        frame_number = 0
        frames.mkdir(parents=True, exist_ok=True)
        raw_frames = run / "raw_frames"
        sensor_frames = run / "sensor_frames"
        frame_labels = run / "frame_labels"
        if randomization is not None:
            raw_frames.mkdir(parents=True, exist_ok=True)
            sensor_frames.mkdir(parents=True, exist_ok=True)
            frame_labels.mkdir(parents=True, exist_ok=True)
        frame_tip_pose = TipPoseBase(tuple(float(v) for v in home), 0.0)
        frame_jaw_target = C.JAW_OPEN
        frame_gripper_command = TASK_CONFIG.gripper.open_command

        def truth_object_labels():
            labels = []
            half = C.CUBE_EDGE / 2.0
            for name in COLOURS:
                pose = cube_pose(name)
                centre_world = np.asarray(pose["centre"], dtype=float)
                centre_base = centre_world.copy()
                centre_base[2] -= C.TABLE_TOP_Z
                rotation = rotation_matrix_from_quat(pose["orientation_wxyz"])
                corners_base = []
                for dx in (-half, half):
                    for dy in (-half, half):
                        for dz in (-half, half):
                            corner_world = centre_world + rotation @ np.array(
                                [dx, dy, dz], dtype=float
                            )
                            corner_base = corner_world.copy()
                            corner_base[2] -= C.TABLE_TOP_Z
                            corners_base.append(corner_base)
                pixels = [camera_model.project(point) for point in corners_base]
                raw_bbox = None
                raw_bbox_status = "invalid_behind_camera"
                derived_bbox = None
                derived_bbox_status = "invalid_behind_camera"
                if all(pixel is not None for pixel in pixels):
                    pixel_array = np.asarray(pixels, dtype=float)
                    raw_bbox, raw_bbox_status = _clipped_amodal_bbox(
                        pixel_array, render_resolution
                    )
                    if capture_calibration is not None:
                        distorted = distort_points_uv(
                            pixel_array,
                            width=render_resolution[0],
                            height=render_resolution[1],
                            fx=float(capture_calibration["fx"]),
                            k1=float(capture_calibration["k1"]),
                        )
                        distorted = np.asarray(distorted, dtype=float)
                        derived_bbox, derived_bbox_status = _clipped_amodal_bbox(
                            distorted, render_resolution
                        )
                    else:
                        derived_bbox_status = "not_generated"
                labels.append(
                    {
                        "track_id": f"isaac-{name}-0",
                        "class_name": "cube",
                        "colour": name,
                        "position_base_m": centre_base.tolist(),
                        "quaternion_base_wxyz": pose["orientation_wxyz"],
                        "dimensions_m": [C.CUBE_EDGE] * 3,
                        # Occlusion requires an instance render pass.  Do not
                        # pretend a projected 3D box measured visibility.
                        "visible_fraction": None,
                        "visibility_status": "unknown_not_annotated",
                        "amodal_bbox_type": (
                            "projected_3d_cuboid_clipped_to_image"
                        ),
                        "raw_amodal_bbox_xyxy_px": raw_bbox,
                        "raw_amodal_bbox_status": raw_bbox_status,
                        "derived_amodal_bbox_xyxy_px": derived_bbox,
                        "derived_amodal_bbox_status": derived_bbox_status,
                        "instance_mask_relpath": None,
                    }
                )
            return labels

        def save_frame(colour: str, phase: str) -> None:
            nonlocal frame_number
            if camera is None:
                return
            raw_rgb = capture_rgb()
            image = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, (18, 16), (610, 102), (18, 18, 18), -1)
            cv2.putText(
                image,
                (
                    "CAMERA-PERCEPTION PHYSICS SIMULATION"
                    if camera_perception
                    else "SCRIPTED PHYSICS SIMULATION"
                ),
                (34, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62 if camera_perception else 0.72,
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
            presentation_path = frames / f"f{frame_number:05d}.png"
            if not cv2.imwrite(str(presentation_path), image):
                raise RuntimeError(f"OpenCV could not write {presentation_path}")

            if randomization is not None and frame_number % args.dataset_stride == 0:
                raw_relative = f"raw_frames/f{frame_number:05d}.png"
                sensor_relative = f"sensor_frames/f{frame_number:05d}.png"
                raw_path = run / raw_relative
                if not cv2.imwrite(
                    str(raw_path), cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
                ):
                    raise RuntimeError(f"OpenCV could not write {raw_path}")
                sensor_result = augment_sensor_rgb(
                    raw_rgb,
                    randomization,
                    frame_index=frame_number,
                    camera_calibration=capture_calibration,
                )
                applied_randomization.update(sensor_result.applied_parameters)
                sensor_path = run / sensor_relative
                if not cv2.imwrite(
                    str(sensor_path),
                    cv2.cvtColor(sensor_result.image_rgb, cv2.COLOR_RGB2BGR),
                ):
                    raise RuntimeError(f"OpenCV could not write {sensor_path}")
                capture_time = float(world.current_time)
                (
                    measured_q,
                    measured_jaw_half_opening_m,
                    measured_closure_fraction,
                    measured_closure_fraction_raw,
                    measured_tcp,
                ) = articulation_readback()
                robot_state_time = float(world.current_time)
                object_labels = truth_object_labels()
                scene_state_time = float(world.current_time)
                commanded_tip = fk_tip_pose(q)
                commanded_mechanical_closure_fraction = float(
                    (C.JAW_OPEN - frame_jaw_target)
                    / max(C.JAW_OPEN - C.JAW_CLOSED, 1e-12)
                )
                commanded_protocol_gripper = (
                    TASK_CONFIG.gripper.open_command
                    + frame_gripper_command
                    * (
                        TASK_CONFIG.gripper.closed_command
                        - TASK_CONFIG.gripper.open_command
                    )
                )
                commanded_tcp = base_tip_pose_to_uf0_tcp(
                    commanded_tip, commanded_protocol_gripper, TASK_CONFIG
                )
                record = build_synchronized_frame_record(
                    synthetic_spec,
                    randomization,
                    frame_index=frame_number,
                    raw_image_relpath=raw_relative,
                    derived_image_relpath=sensor_relative,
                    capture_time_s=capture_time,
                    robot_state_time_s=robot_state_time,
                    scene_state_time_s=scene_state_time,
                    measured_arm_joints_deg=np.rad2deg(measured_q),
                    measured_gripper_half_opening_m=(
                        measured_jaw_half_opening_m
                    ),
                    measured_gripper_closure_fraction=(
                        measured_closure_fraction
                    ),
                    measured_tcp_pose_uf0_mm_deg=measured_tcp,
                    commanded_arm_joints_deg=np.rad2deg(q),
                    commanded_gripper_half_opening_m=frame_jaw_target,
                    commanded_gripper_closure_fraction=(
                        commanded_mechanical_closure_fraction
                    ),
                    commanded_tcp_pose_uf0_mm_deg=commanded_tcp[:6],
                    scene_version=f"{run.name}:frame-{frame_number}",
                    phase=phase,
                    objects=object_labels,
                )
                record["camera"]["sensor_processing"] = dict(
                    sensor_result.processing
                )
                record["policy_input"]["measured_robot_state"]["gripper"].update(
                    {
                        "raw_closure_fraction": measured_closure_fraction_raw,
                        "endpoint_tolerance": 1e-4,
                        "normalized_within_endpoint_tolerance": bool(
                            measured_closure_fraction_raw
                            != measured_closure_fraction
                        ),
                    }
                )
                record["camera"]["effective_isaac_readback"] = effective_camera
                record["randomization"].update(
                    {
                        "isaac_stage_readback_relpath": (
                            "isaac_randomization_readback.json"
                        ),
                        "isaac_stage_readback_sha256": (
                            stage_randomization_readback["sha256"]
                        ),
                        "effective_camera_relpath": "effective_camera.json",
                        "effective_camera_sha256": effective_camera["sha256"],
                    }
                )
                record["synchronization"]["clock_source"] = (
                    "Isaac world.current_time; image, articulation readback, and "
                    "scene truth sampled without advancing the simulation tick"
                )
                write_json_atomic(
                    frame_labels / f"f{frame_number:05d}.json", record
                )
            frame_number += 1

        for colour_index, colour in enumerate(COLOURS):
            start_all = {name: cube_pose(name) for name in COLOURS}
            start = start_all[colour]
            runtime_detection = observed_targets.get(colour)
            if camera_perception:
                if runtime_detection is None:
                    raise RuntimeError(f"Camera perception did not find {colour} cube")
                pick_xy = list(runtime_detection.position_base_m[:2])
                yaw = math.degrees(runtime_detection.yaw_rad)
            else:
                pick_xy = start["centre"][:2]
                yaw = start["yaw_deg"]
            theta = math.radians(yaw)
            pick = {
                "colour": colour,
                "xy": pick_xy,
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
                # Keep the original key for report consumers while making the
                # evaluator-only role explicit in the new alias.
                "source": start,
                "source_truth_evaluator_only": start,
                "runtime_observation": (
                    None
                    if runtime_detection is None
                    else {
                        "source": "raw_isaac_table_camera_rgb",
                        "track_id": runtime_detection.track_id,
                        "capture_time_s": runtime_detection.capture_time_s,
                        "position_base_m": list(runtime_detection.position_base_m),
                        "yaw_rad": runtime_detection.yaw_rad,
                        "confidence": runtime_detection.confidence,
                        "pixel_center_uv": list(runtime_detection.pixel_center_uv),
                        "observation_relpath": initial_camera_observation_relpath,
                    }
                ),
                "destination_bin": {
                    "colour": colour,
                    "centre_xy_m": list(BIN_CENTRES[colour]),
                },
                "pickup_verified": False,
                "placement_verified": False,
                "trajectory_ik_failures": 0,
                "shared_semantic_phases": list(SHARED_SEMANTIC_PHASE_NAMES),
            }
            if runtime_detection is not None:
                truth_xy = np.asarray(start["centre"][:2], dtype=float)
                detected_xy = np.asarray(
                    runtime_detection.position_base_m[:2], dtype=float
                )
                event["perception_evaluation_truth_only"] = {
                    "xy_error_mm": float(
                        np.linalg.norm(detected_xy - truth_xy) * 1000.0
                    ),
                    "truth_xy_m": truth_xy.tolist(),
                    "detected_xy_m": detected_xy.tolist(),
                }

            segments = motion_segments(pick, home, C)
            actual_semantic_names = tuple(segment[0] for segment in segments[:-1])
            if actual_semantic_names != SHARED_SEMANTIC_PHASE_NAMES:
                raise RuntimeError(
                    "Isaac adapter semantic phases differ from the shared planner: "
                    f"{actual_semantic_names!r} != {SHARED_SEMANTIC_PHASE_NAMES!r}"
                )
            for segment_index, (
                phase,
                p0,
                p1,
                duration,
                jaw0,
                jaw1,
                spread,
            ) in enumerate(segments):
                if phase == "revalidate_before_descent":
                    if camera_perception:
                        refreshed_rgb = capture_rgb()
                        refreshed_observation_relpath = save_decision_frame(
                            f"{colour}-pre-descent.png", refreshed_rgb
                        )
                        refreshed_scene = camera_adapter.observe(
                            refreshed_rgb,
                            scene_id=f"isaac-camera-{colour}-pre-descent",
                            capture_time_s=float(world.current_time),
                        )
                        refreshed_by_colour = validate_reobserved_scene(
                            refreshed_scene,
                            observed_targets,
                            COLOURS[:colour_index],
                            TASK_CONFIG,
                            maximum_motion_m=(
                                CAMERA_REVALIDATION_MAXIMUM_MOTION_M
                            ),
                            maximum_yaw_motion_deg=(
                                CAMERA_REVALIDATION_MAXIMUM_YAW_DEG
                            ),
                        )
                        refreshed = refreshed_by_colour[colour]
                        per_colour_changes = {}
                        for remaining_colour, current_detection in (
                            refreshed_by_colour.items()
                        ):
                            reference_detection = observed_targets[remaining_colour]
                            per_colour_changes[remaining_colour] = {
                                "movement_xy_m": float(
                                    np.linalg.norm(
                                        np.asarray(
                                            current_detection.position_base_m[:2]
                                        )
                                        - np.asarray(
                                            reference_detection.position_base_m[:2]
                                        )
                                    )
                                ),
                                "square_yaw_change_deg": (
                                    square_yaw_delta_degrees(
                                        current_detection.yaw_rad,
                                        reference_detection.yaw_rad,
                                    )
                                ),
                            }
                        moved_xy_m = per_colour_changes[colour]["movement_xy_m"]
                        yaw_change_deg = per_colour_changes[colour][
                            "square_yaw_change_deg"
                        ]
                        tolerance_m = CAMERA_REVALIDATION_MAXIMUM_MOTION_M
                        yaw_tolerance_deg = CAMERA_REVALIDATION_MAXIMUM_YAW_DEG
                        revalidation_source = "fresh_raw_isaac_table_camera_rgb"
                        # Rebind every post-gate motion target to the fresh
                        # target XY/yaw. The first descent begins at the actual
                        # old hover pose and converges on the refreshed pick;
                        # simulator truth is never used to repair the plan.
                        refreshed_theta = refreshed.yaw_rad
                        refreshed_spread = np.array(
                            [
                                -math.sin(refreshed_theta),
                                math.cos(refreshed_theta),
                                0.0,
                            ],
                            dtype=float,
                        )
                        (
                            measured_rebound_q,
                            _,
                            _,
                            _,
                            _,
                        ) = articulation_readback()
                        # Seed rebound IK and its start pose from articulation
                        # readback, not the last commanded solver state. This
                        # bounds the first step even if a drive lagged at hover.
                        q = np.asarray(measured_rebound_q, dtype=float).copy()
                        current_rotation, _ = fk.fk(q)
                        current_spread = (
                            current_rotation @ np.asarray(C.FINGER_SPREAD_AXIS)
                        )
                        if np.dot(current_spread, refreshed_spread) < 0.0:
                            refreshed_spread = -refreshed_spread
                        refreshed_destination_spread = np.array(
                            [0.0, 1.0, 0.0], dtype=float
                        )
                        if (
                            np.dot(
                                refreshed_spread, refreshed_destination_spread
                            )
                            < 0.0
                        ):
                            refreshed_destination_spread = (
                                -refreshed_destination_spread
                            )
                        pick["xy"] = list(refreshed.position_base_m[:2])
                        pick["spread_dir"] = refreshed_spread.tolist()
                        pick["place_spread_dir"] = (
                            refreshed_destination_spread.tolist()
                        )
                        refreshed_segments = motion_segments(pick, home, C)
                        _, actual_hover = fk.tool_pose(q, tool)
                        rebound_descent = list(refreshed_segments[2])
                        rebound_descent[1] = np.asarray(actual_hover, dtype=float)
                        segments[segment_index + 1 :] = [
                            tuple(rebound_descent),
                            *refreshed_segments[3:],
                        ]
                        runtime_detection = refreshed
                        event["motion_rebound_to_refreshed_detection"] = True
                        event["rebound_measured_arm_joints_rad"] = q.tolist()
                        event["pre_descent_revalidation_detection"] = {
                            "track_id": refreshed.track_id,
                            "capture_time_s": refreshed.capture_time_s,
                            "position_base_m": list(refreshed.position_base_m),
                            "yaw_rad": refreshed.yaw_rad,
                            "confidence": refreshed.confidence,
                            "observation_relpath": refreshed_observation_relpath,
                        }
                    else:
                        tolerance_m = min(0.002, TASK_CONFIG.cube_edge_m / 4.0)
                        yaw_tolerance_deg = None
                        current = cube_pose(colour)
                        moved_xy_m = float(
                            np.linalg.norm(
                                np.asarray(current["centre"][:2])
                                - np.asarray(start["centre"][:2])
                            )
                        )
                        yaw_change_deg = None
                        per_colour_changes = None
                        event["motion_rebound_to_refreshed_detection"] = False
                        revalidation_source = "simulator_ground_truth_not_camera"
                    event["pre_descent_revalidation"] = {
                        "source": revalidation_source,
                        "target_movement_xy_m": moved_xy_m,
                        "maximum_allowed_m": tolerance_m,
                        "target_square_yaw_change_deg": yaw_change_deg,
                        "maximum_allowed_yaw_deg": yaw_tolerance_deg,
                        "remaining_target_changes": per_colour_changes,
                        "passed": bool(
                            moved_xy_m <= tolerance_m
                            and (
                                yaw_tolerance_deg is None
                                or yaw_change_deg <= yaw_tolerance_deg
                            )
                        ),
                    }
                    if moved_xy_m > tolerance_m:
                        raise RuntimeError(
                            f"{colour} target moved {moved_xy_m * 1000.0:.2f} mm "
                            "before descent; refusing stale pick"
                        )
                    continue

                spread_start = None
                if spread is not None and _phase_interpolates_spread(phase):
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
                    frame_jaw_target = jaw0 + (jaw1 - jaw0) * t
                    close_span = max(C.JAW_OPEN - C.CUBE_JAW_HALF_TARGET, 1e-9)
                    frame_gripper_command = float(
                        np.clip((C.JAW_OPEN - frame_jaw_target) / close_span, 0.0, 1.0)
                    )
                    if direction is not None:
                        frame_yaw = math.atan2(
                            -float(direction[0]), float(direction[1])
                        )
                    else:
                        frame_yaw = frame_tip_pose.yaw_rad
                    frame_tip_pose = TipPoseBase(
                        tuple(float(value) for value in target), frame_yaw
                    )
                    command(q, frame_jaw_target)
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
            event["simulator_truth_affects_runtime_control"] = not camera_perception
            event["runtime_post_place_verification"] = (
                {
                    "source": "simulator_truth",
                    "passed": event["placement_verified"],
                    "limitation": "default scripted simulation mode only",
                }
                if not camera_perception
                else {
                    "source": None,
                    "passed": None,
                    "limitation": (
                        "camera-based post-place verification is not implemented; "
                        "simulator truth scores the result but cannot stop or alter "
                        "later actions"
                    ),
                }
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
            if _should_stop_after_truth_evaluation(
                camera_perception=camera_perception,
                placement_verified=event["placement_verified"],
            ):
                break

        final = {colour: cube_pose(colour) for colour in COLOURS}
        finalized_randomization = None
        if randomization is not None:
            finalized_randomization = finalize_application_status(
                randomization,
                applied=applied_randomization,
                skipped={},
            )
            write_json_atomic(
                run / "randomization_manifest.json", finalized_randomization
            )
        report = {
            "task": "scripted_colour_bucket_sort",
            "decision_source": (
                "raw Isaac table-camera RGB through shared deterministic HSV geometry"
                if camera_perception
                else "configured simulator poses; no camera perception or AI"
            ),
            "decision_uses_simulator_truth": not camera_perception,
            "simulator_truth_affects_runtime_control": not camera_perception,
            "camera_post_place_runtime_verification": (
                "not implemented; evaluator truth never branches camera/training actions"
                if camera_perception
                else "not applicable to configured-pose scripted mode"
            ),
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
                    colour: list(active_spawns[colour]) for colour in COLOURS
                },
                "bin_centres_base_m": {
                    colour: list(BIN_CENTRES[colour]) for colour in COLOURS
                },
                "pickup_region_base_m": {
                    axis: list(bounds)
                    for axis, bounds in TASK_CONFIG.task.pickup_region_base_m.items()
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
                # Retain the old field for report-reader compatibility while
                # recording the two distinct facts explicitly.
                "tcp_conflict_unresolved": False,
                "tcp_faq_discrepancy_resolved": True,
                "live_hardware_frame_verified": False,
                "tcp_conflict_note": TASK_CONFIG.frames.tcp_conflict_note,
                "semantic_phase_names": list(SHARED_SEMANTIC_PHASE_NAMES),
                "isaac_adapter_extension_phases": ["home"],
                "isaac_revalidation_source": (
                    "fresh raw Isaac table-camera RGB"
                    if camera_perception
                    else "simulator ground truth, not camera"
                ),
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
                "resolution": list(render_resolution),
                "frames": frame_number,
            },
            "camera_observations": {
                "enabled": camera_perception,
                "initial_observation_relpath": initial_camera_observation_relpath,
                "raw_unannotated": True,
                "count": (
                    len(list(decision_frames.glob("*.png")))
                    if camera_perception
                    else 0
                ),
            },
            "training_data": {
                "enabled": randomization is not None,
                "episode_seed": args.seed if randomization is not None else None,
                "raw_frame_count": (
                    len(list(raw_frames.glob("*.png")))
                    if randomization is not None
                    else 0
                ),
                "sensor_frame_count": (
                    len(list(sensor_frames.glob("*.png")))
                    if randomization is not None
                    else 0
                ),
                "label_record_count": (
                    len(list(frame_labels.glob("*.json")))
                    if randomization is not None
                    else 0
                ),
                "raw_frames_have_overlays": False,
                "sensor_frames_have_overlays": False,
                "simulator_truth_role": "evaluator-only labels",
                "visibility_and_instance_masks": "not yet annotated",
                "ranges": (
                    None if randomization is None else synthetic_spec.range_status
                ),
                "range_warning": (
                    None if randomization is None else synthetic_spec.range_warning
                ),
                "manifest_finalized": finalized_randomization is not None,
                "isaac_stage_readback": (
                    None
                    if stage_randomization_readback is None
                    else {
                        "path": "isaac_randomization_readback.json",
                        "sha256": stage_randomization_readback["sha256"],
                    }
                ),
                "effective_camera_readback": (
                    None
                    if effective_camera is None
                    else {
                        "path": "effective_camera.json",
                        "sha256": effective_camera["sha256"],
                        "coordinate_frames": ["usd_world", "robot_base"],
                    }
                ),
                "camera_decision_input": (
                    "raw pinhole render; derived sensor images are dataset artifacts"
                    if camera_perception
                    else None
                ),
            },
            "initial": initial,
            "final": final,
            "initial_truth_evaluator_only": initial,
            "final_truth_evaluator_only": final,
            "events": events,
            "verified_pickups": sum(e["pickup_verified"] for e in events),
            "verified_placements": sum(e["placement_verified"] for e in events),
            "runtime_sequence_complete": len(events) == len(COLOURS),
            "task_complete": len(events) == len(COLOURS)
            and all(e["placement_verified"] for e in events),
        }
        (run / "execution_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[bucket-demo] runtime-sequence-complete="
            f"{report['runtime_sequence_complete']} "
            f"evaluator-task-complete={report['task_complete']} "
            f"pickups={report['verified_pickups']}/3 "
            f"placements={report['verified_placements']}/3 "
            f"frames={frame_number}",
            flush=True,
        )
        if _should_stop_after_truth_evaluation(
            camera_perception=camera_perception,
            placement_verified=report["task_complete"],
        ):
            raise RuntimeError("Bucket-sort verification failed; inspect execution_report.json")
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
