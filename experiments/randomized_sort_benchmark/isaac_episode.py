"""Run one camera-driven randomized cube/screw sorting episode in Isaac Sim.

This entry point is deliberately separate from the working cube demo and the
physical FANUC policy.  It reuses the supplied digital twin, CAD, calibrated
camera geometry, catalog grasp candidates, and realistic-domain ranges while
allowing an arbitrary 1--10 object condition.  Runtime motion is based on the
sensor-processed camera observation.  Simulator transforms are read only for
outcome labels and metrics after a decision has been made.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


def _repo_root() -> Path:
    configured = os.environ.get("ROCKWELL_REPO_ROOT")
    candidates = ([Path(configured)] if configured else []) + list(Path(__file__).parents)
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "sorting_core" / "__init__.py").is_file():
            return candidate
    raise RuntimeError("set ROCKWELL_REPO_ROOT to the repository checkout")


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demos.scripted_cube_screw_bucket_sort.cube_screw_bucket_demo import (  # noqa: E402
    BIN_MATERIALS,
    _ease,
    _motion_segments,
    _rotation_matrix_from_quat,
    _turn,
)
from demos.scripted_cube_screw_bucket_sort.scenario import (  # noqa: E402
    load_scenario_config,
    load_stl_triangles,
    resolve_screw_model,
    screw_jaw_half_target,
    tool_tip_z_for_desired_grasp_height,
    validate_screw_model,
    verify_vertices_inside_bin,
)
from experiments.randomized_sort_benchmark.grasp_selection import (  # noqa: E402
    BenchmarkGraspPlanner,
)
from experiments.randomized_sort_benchmark.manifest import (  # noqa: E402
    BenchmarkManifest,
    BenchmarkObject,
    read_manifest,
)
from experiments.randomized_sort_benchmark.randomization import (  # noqa: E402
    build_matched_randomization,
    classify_application,
    generalized_postbuild,
    generalized_prebuild,
    write_json_atomic,
)
from experiments.randomized_sort_benchmark.vision import (  # noqa: E402
    BenchmarkVisionAbstention,
    BenchmarkVisionConfig,
    BenchmarkVisionProvider,
    DETERMINISTIC_HYBRID,
    PINNED_YOLO11N_SHA256,
    YOLO_ASSISTED_HYBRID,
)
from isaac_bridge import CalibratedPinholeCamera, IsaacCameraSortAdapter  # noqa: E402
from isaac_bridge.sim_validation import (  # noqa: E402
    isaac_camera_config_from_calibration,
)
from isaac_bridge.training_episode import (  # noqa: E402
    apply_destination_material_randomization,
    readback_effective_camera,
)
from sorting_core import (  # noqa: E402
    CameraFrame,
    PerceptionRequest,
    SceneObservation,
    TipPoseBase,
    augment_sensor_rgb,
    load_capture_calibration,
    load_config,
    load_synthetic_data_spec,
)
from sorting_core.integration_plugins import GraspFactoryContext  # noqa: E402
from team_plugins.computer_vision.hybrid_geometry_provider import (  # noqa: E402
    CubeScrewHybridVisionProvider,
    ScrewDetectorConfig,
)
from team_plugins.grasping.ai_providers.cosmos3 import (  # noqa: E402
    Cosmos3OpenAIProvider,
)


FPS = 30
RENDER_RESOLUTION = (1920, 1080)
SUPPORTED_CLASSES = {"cube", "screw_iso10642_m6x30"}
STABILIZATION_FRAMES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--vision",
        choices=(DETERMINISTIC_HYBRID, YOLO_ASSISTED_HYBRID),
        required=True,
    )
    parser.add_argument("--grasp", choices=("deterministic", "ai"), required=True)
    parser.add_argument("--yolo-checkpoint")
    parser.add_argument("--ai-endpoint")
    parser.add_argument("--ai-model", default="nvidia/Cosmos3-Nano")
    parser.add_argument(
        "--ai-model-revision",
        default="e59a53c25979a090fa8706c9acc0c254a6e89b92",
    )
    parser.add_argument("--bin-model")
    parser.add_argument("--screw-model")
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="save rendered motion frames; disabled for high-throughput trials",
    )
    parser.add_argument(
        "--maximum-actions",
        type=int,
        default=10,
        help="bounded attempts; one per reference object by default",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset_path(value: str | None, filename: str) -> Path:
    if value:
        result = Path(value).expanduser().resolve()
    else:
        assets = os.environ.get("ROCKWELL_REFERENCE_ASSETS")
        if not assets:
            raise RuntimeError(
                f"ROCKWELL_REFERENCE_ASSETS is unset; pass the {filename} path"
            )
        result = Path(assets).expanduser().resolve() / "models" / filename
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _atomic_report(path: Path, value: Mapping[str, Any]) -> None:
    write_json_atomic(path, value)


def _object_class(reference: BenchmarkObject) -> str:
    return "cube" if reference.class_name == "cube_10mm" else reference.class_name


def _within_pickup(item: Any, config: Any) -> bool:
    x_m, y_m = item.pose.position_base_m[:2]
    x_bounds = config.task.pickup_region_base_m["x"]
    y_bounds = config.task.pickup_region_base_m["y"]
    return x_bounds[0] <= x_m <= x_bounds[1] and y_bounds[0] <= y_m <= y_bounds[1]


def _filtered_pickup_observation(observation: SceneObservation, config: Any) -> SceneObservation:
    return replace(
        observation,
        objects=tuple(item for item in observation.objects if _within_pickup(item, config)),
    )


def _target_sort_key(item: Any) -> tuple[float, str, str]:
    x_m, y_m = item.pose.position_base_m[:2]
    return (math.hypot(x_m - 0.30, y_m), item.class_id, item.track_id)


def _match_reference(
    detected: Any,
    references: Sequence[BenchmarkObject],
    truth: Mapping[str, Mapping[str, Any]],
    excluded: set[str],
) -> tuple[str | None, float | None]:
    candidates = []
    for reference in references:
        if reference.instance_id in excluded or _object_class(reference) != detected.class_id:
            continue
        if detected.class_id == "cube" and reference.colour != detected.sort_label:
            continue
        row = truth[reference.instance_id]
        distance = math.dist(
            detected.pose.position_base_m[:2], row["reference_base_m"][:2]
        )
        candidates.append((distance, reference.instance_id))
    if not candidates:
        return None, None
    distance, instance_id = min(candidates)
    return (instance_id, distance) if distance <= 0.04 else (None, distance)


def _placement_slot(destination_index: int, *, screw: bool) -> tuple[float, float]:
    if screw:
        # Alternating y offsets retain the proven long-axis-through-bin-centre
        # layout. Higher loads intentionally become a measurable capacity test.
        return (0.0, (-1.0 if destination_index % 2 == 0 else 1.0) * 0.004)
    grid = (-0.008, 0.0, 0.008)
    return (grid[destination_index % 3], grid[(destination_index // 3) % 3])


def _yaw_from_orientation_xyzw(quaternion: Sequence[float]) -> float:
    x, y, z, w = (float(value) for value in quaternion)
    # Head-to-tip is local -Z for the validated screw CAD.
    axis_x = -2.0 * (x * z + w * y)
    axis_y = -2.0 * (y * z - w * x)
    return math.atan2(axis_y, axis_x)


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_manifest(args.manifest)
    run_dir = Path(args.run).expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"refusing to mix outputs in non-empty {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    decision_frames = run_dir / "decision_frames"
    decision_frames.mkdir()
    video_frames = run_dir / "video_frames"
    if args.record_video:
        video_frames.mkdir()

    config = load_config()
    mixed_config = load_scenario_config()
    synthetic_spec = load_synthetic_data_spec()
    capture_calibration = load_capture_calibration(synthetic_spec)
    matched = build_matched_randomization(
        synthetic_spec,
        condition_id=manifest.condition_id,
        appearance_seed=manifest.seeds.appearance_seed,
        physics_seed=manifest.seeds.physics_seed,
        sensor_seed=manifest.seeds.sensor_seed,
    )
    randomization = matched.combined
    write_json_atomic(run_dir / "condition_manifest.json", manifest.envelope())
    write_json_atomic(run_dir / "matched_randomization.json", matched.as_dict())

    screw_path = resolve_screw_model(args.screw_model)
    screw_model = validate_screw_model(screw_path, mixed_config.screw)
    screw_points_m = np.asarray(screw_model.vertices_m, dtype=float)
    bin_path = _asset_path(args.bin_model, "manipbench-bin-50x50x55.stl")
    bin_triangles_mm = load_stl_triangles(bin_path)
    bin_points_m = np.asarray(
        [vertex for triangle in bin_triangles_mm for vertex in triangle], dtype=float
    ) * 0.001

    cube_references = tuple(
        reference for reference in manifest.objects if reference.class_name == "cube_10mm"
    )
    screw_references = tuple(
        reference
        for reference in manifest.objects
        if reference.class_name == "screw_iso10642_m6x30"
    )

    # Only now import/start Isaac. Host-side manifest/CAD validation fails
    # before consuming a GPU session.
    import scene_config as C
    import fk
    import ik
    from isaacsim import SimulationApp

    C.SURFACE_DZ = config.frames.surface_base_z_m
    C.CUBE_EDGE = config.cube_edge_m
    C.CUBE_TIP_CLEARANCE = config.task.pick_tip_clearance_m
    C.CUBE_JAW_HALF_TARGET = max(
        C.JAW_CLOSED,
        C.CUBE_EDGE / 2.0 - C.FINGER_INNER_TIP - 0.0003,
    )
    C.SORT_VIALS = []
    C.SORT_CAPS = []
    C.CAP_ORIENTATIONS = []
    C.SHOW_BINS = False
    C.SHOW_FIXTURE = False
    table_previous = dict(C.TABLE_CAM)
    C.TABLE_CAM = isaac_camera_config_from_calibration(
        capture_calibration,
        parent=table_previous["parent"],
        horizontal_aperture_mm=float(table_previous["horiz_aperture"]),
        parent_origin_offset_m=(0.0, 0.0, C.TABLE_TOP_Z),
    )
    C.TABLE_CAM["resolution"] = RENDER_RESOLUTION

    app = SimulationApp({"headless": True})
    wall_started = time.perf_counter()
    applied_randomization: set[str] = set()
    report: dict[str, Any] | None = None
    try:
        import cv2
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera
        import cell

        appearance_profile = generalized_prebuild(C, cell, randomization)
        C.TABLE_CAM["resolution"] = RENDER_RESOLUTION
        C.SORT_CUBES = [
            (
                reference.position_base_m[0],
                reference.position_base_m[1],
                reference.colour,
            )
            for reference in cube_references
        ]
        C.SORT_CUBE_YAWS_DEG = [reference.yaw_deg for reference in cube_references]

        world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / FPS)
        stage = omni.usd.get_context().get_stage()
        cell.build(stage)
        cube_rows = [reference.as_dict() for reference in cube_references]
        post_applied, stage_readback = generalized_postbuild(
            stage, cell, randomization, cube_rows
        )
        applied_randomization.update(post_applied)

        bin_centres = dict(config.task.bin_centres_base_m)

        def add_bin(colour: str) -> None:
            bin_x, bin_y = bin_centres[colour]
            path = f"/World/ColourBin_{colour}"
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreatePointsAttr(
                [Gf.Vec3f(*map(float, point)) for point in bin_points_m]
            )
            mesh.CreateFaceVertexCountsAttr([3] * len(bin_triangles_mm))
            mesh.CreateFaceVertexIndicesAttr(list(range(len(bin_points_m))))
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateSubdivisionSchemeAttr("none")
            UsdGeom.Xformable(mesh.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3d(bin_x, bin_y, C.TABLE_TOP_Z + C.SURFACE_DZ)
            )
            cell.bind_mat(mesh.GetPrim(), BIN_MATERIALS[colour])
            proxy_path = f"{path}_Proxy"
            cell._make_bin(
                stage,
                proxy_path,
                xy=(bin_x, bin_y),
                outer=(config.bin.outer_m, config.bin.outer_m),
                pocket_size=config.bin.pocket_m,
                pocket_r=0.004,
                wall_h=config.bin.height_m,
                floor_t=config.bin.floor_m,
                mat_name=BIN_MATERIALS[colour],
            )
            UsdGeom.Imageable(stage.GetPrimAtPath(proxy_path)).CreateVisibilityAttr().Set(
                "invisible"
            )

        for colour in config.task.colour_order:
            add_bin(colour)
        destination_applied, stage_readback = apply_destination_material_randomization(
            stage, randomization, stage_readback=stage_readback
        )
        applied_randomization.update(destination_applied)
        write_json_atomic(run_dir / "isaac_randomization_readback.json", stage_readback)

        screw_roots: dict[str, str] = {}
        for reference in screw_references:
            root_path = f"/World/{reference.instance_id.replace('-', '_')}"
            root = UsdGeom.Xform.Define(stage, root_path)
            transform = UsdGeom.Xformable(root.GetPrim())
            transform.AddTranslateOp().Set(
                Gf.Vec3d(
                    reference.position_base_m[0],
                    reference.position_base_m[1],
                    reference.position_base_m[2] + C.TABLE_TOP_Z,
                )
            )
            transform.AddRotateZOp().Set(reference.yaw_deg)
            transform.AddRotateYOp().Set(-90.0)
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
            UsdPhysics.MeshCollisionAPI.Apply(body.GetPrim()).CreateApproximationAttr().Set(
                "convexDecomposition"
            )
            decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(
                body.GetPrim()
            )
            decomposition.CreateHullVertexLimitAttr().Set(64)
            decomposition.CreateMaxConvexHullsAttr().Set(8)
            decomposition.CreateErrorPercentageAttr().Set(0.5)
            cell.bind_physics_material(body.GetPrim())
            cell.small_contact_offsets(body.GetPrim())
            cell.make_dynamic(root.GetPrim(), mixed_config.screw.mass_kg, collide_self=False)
            UsdPhysics.RigidBodyAPI(root.GetPrim()).CreateKinematicEnabledAttr().Set(True)
            screw_roots[reference.instance_id] = root_path

        wrappers: dict[str, Any] = {}
        for index, reference in enumerate(cube_references):
            path = f"/World/SortCube{index}"
            UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path)).CreateKinematicEnabledAttr().Set(
                True
            )
            wrappers[reference.instance_id] = SingleXFormPrim(path)
        for instance_id, path in screw_roots.items():
            wrappers[instance_id] = SingleXFormPrim(path)

        camera = Camera(prim_path=cell.TABLE_CAM_PATH, resolution=RENDER_RESOLUTION)
        cell.enable_ccd(world)
        world.reset()
        camera.initialize()

        effective_camera, camera_applied = readback_effective_camera(
            stage,
            camera_prim_path=cell.TABLE_CAM_PATH,
            resolution_px=RENDER_RESOLUTION,
            robot_base_origin_world_m=(0.0, 0.0, C.TABLE_TOP_Z),
            expected_camera=C.TABLE_CAM,
            base_camera=C._TRAINING_RANDOMIZATION_BASE["camera"],
            manifest=randomization,
        )
        applied_randomization.update(camera_applied)
        write_json_atomic(run_dir / "effective_camera.json", effective_camera)
        intrinsics = effective_camera["intrinsics"]
        camera_model = CalibratedPinholeCamera(
            resolution_px=tuple(intrinsics["resolution_px"]),
            focal_length_mm=intrinsics["focal_length_mm"],
            horizontal_aperture_mm=intrinsics["horizontal_aperture_mm"],
            vertical_aperture_mm=intrinsics["vertical_aperture_mm"],
            camera_to_base_row_major=tuple(
                tuple(row)
                for row in effective_camera["extrinsics"]["robot_base"][
                    "camera_to_robot_base_row_major"
                ]
            ),
            default_plane_base_z_m=C.SURFACE_DZ,
        )

        geometry_provider = CubeScrewHybridVisionProvider(
            config=config,
            camera_projectors={"table": camera_model},
            calibration_ids={"table": "benchmark-effective-camera-v1"},
            expected_cube_edge_m=config.cube_edge_m,
            screw_config=ScrewDetectorConfig(
                expected_image_size=RENDER_RESOLUTION,
                require_expected_image_size=True,
                minimum_track_observations=STABILIZATION_FRAMES,
            ),
            maximum_frame_age_s=0.2,
            profile_provenance={
                "profile_id": "randomized-benchmark-hybrid-geometry-v1",
                "profile_path": "experiments/randomized_sort_benchmark/isaac_episode.py",
                "profile_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
        )
        yolo_model = None
        if args.vision == YOLO_ASSISTED_HYBRID:
            if not args.yolo_checkpoint:
                raise RuntimeError("YOLO-assisted mode requires --yolo-checkpoint")
            checkpoint = Path(args.yolo_checkpoint).expanduser().resolve()
            digest = _sha256_file(checkpoint)
            if digest != PINNED_YOLO11N_SHA256:
                raise RuntimeError(
                    f"YOLO checkpoint mismatch: {digest} != {PINNED_YOLO11N_SHA256}"
                )
            from ultralytics import YOLO

            yolo_model = YOLO(str(checkpoint), task="detect")
        vision_config = BenchmarkVisionConfig(
            mode=args.vision,
            checkpoint_sha256=(
                PINNED_YOLO11N_SHA256
                if args.vision == YOLO_ASSISTED_HYBRID
                else None
            ),
        )
        vision_provider = BenchmarkVisionProvider(
            geometry_provider=geometry_provider,
            config=vision_config,
            yolo_model=yolo_model,
        )
        camera_adapter = IsaacCameraSortAdapter(
            camera_model,
            config,
            perception_provider=vision_provider,
            grasp_pipeline=object(),
            ignore_bucket_pixels=True,
        )

        ai_provider = None
        if args.grasp == "ai":
            if not args.ai_endpoint:
                raise RuntimeError("AI grasp mode requires --ai-endpoint")
            ai_provider = Cosmos3OpenAIProvider(
                endpoint=args.ai_endpoint,
                model=args.ai_model,
                model_revision=args.ai_model_revision,
            )
        grasp_profile_payload = {
            "profile_id": "randomized-benchmark-grasp-v1",
            "simulation_only": True,
            "scorer": "geometric_prior",
            "minimum_pose_quality": 0.0,
        }
        grasp_profile_sha = hashlib.sha256(
            json.dumps(grasp_profile_payload, sort_keys=True).encode()
        ).hexdigest()
        grasp_context = GraspFactoryContext(
            config=config,
            checkpoint=None,
            options={
                "simulation_only": True,
                "scorer": "geometric_prior",
                "minimum_pose_quality": 0.0,
            },
            profile_id="randomized-benchmark-grasp-v1",
            profile_path="experiments/randomized_sort_benchmark/isaac_episode.py",
            profile_sha256=grasp_profile_sha,
        )
        grasp_planner = BenchmarkGraspPlanner(
            grasp_context,
            mode=args.grasp,
            ai_provider=ai_provider,
            receipt_directory=(run_dir / "ai_receipts") if ai_provider else None,
            minimum_confidence=0.60,
            minimum_score_margin=0.03,
            timeout_s=20.0,
        )

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
            raise RuntimeError("home IK failed")

        def command(joints: np.ndarray, jaw_half_opening: float) -> None:
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
            world.step(render=False)
        for index in range(len(cube_references)):
            UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(f"/World/SortCube{index}")).CreateKinematicEnabledAttr().Set(False)
        for path in screw_roots.values():
            UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path)).CreateKinematicEnabledAttr().Set(False)
        for _ in range(90):
            command(q, C.JAW_OPEN)
            world.step(render=False)
        world.step(render=True)

        def truth_pose(reference: BenchmarkObject) -> dict[str, Any]:
            position, quaternion = wrappers[reference.instance_id].get_world_pose()
            position = np.asarray(position, dtype=float)
            quaternion = np.asarray(quaternion, dtype=float)
            rotation = _rotation_matrix_from_quat(quaternion)
            if reference.class_name == "cube_10mm":
                centre = position + rotation @ np.array([0.0, 0.0, C.CUBE_EDGE / 2.0])
                centre[2] -= C.TABLE_TOP_Z
                yaw = math.atan2(rotation[1, 0], rotation[0, 0])
                return {
                    "reference_base_m": centre.tolist(),
                    "yaw_rad": yaw,
                    "orientation_wxyz": quaternion.tolist(),
                }
            head_base = position.copy()
            head_base[2] -= C.TABLE_TOP_Z
            axis = rotation @ np.array([0.0, 0.0, -1.0])
            vertices_world = screw_points_m @ rotation.T + position
            vertices_base = vertices_world.copy()
            vertices_base[:, 2] -= C.TABLE_TOP_Z
            return {
                "reference_base_m": head_base.tolist(),
                "yaw_rad": math.atan2(axis[1], axis[0]),
                "axis_world": axis.tolist(),
                "orientation_wxyz": quaternion.tolist(),
                "vertices_base_m": vertices_base.tolist(),
            }

        def all_truth() -> dict[str, dict[str, Any]]:
            return {reference.instance_id: truth_pose(reference) for reference in manifest.objects}

        frame_index = 0
        video_index = 0
        vision_reports: list[dict[str, Any]] = []
        sensor_parameters = {
            name
            for name, entry in randomization.parameters.items()
            if entry["consumer"] == "sensor_postprocess"
        }

        def capture_for_decision(tag: str) -> tuple[SceneObservation | None, str | None]:
            nonlocal frame_index
            world.step(render=True)
            rgba = camera.get_rgba()
            if rgba is None:
                raise RuntimeError("table camera returned no image")
            raw_rgb = rgba[:, :, :3].astype(np.uint8)
            sensor_result = augment_sensor_rgb(
                raw_rgb,
                randomization,
                frame_index=frame_index,
                camera_calibration=capture_calibration,
                camera_id="table",
                expected_resolution=RENDER_RESOLUTION,
                # The metric projector is pinhole. Radial distortion is saved
                # in dataset generation, but decision pixels remain undistorted
                # until the provider has an explicit undistortion stage.
                apply_calibrated_radial_distortion=False,
            )
            applied_randomization.update(sensor_result.applied_parameters)
            rgb = sensor_result.image_rgb
            filename = f"f{frame_index:05d}-{tag}.png"
            cv2.imwrite(
                str(decision_frames / filename),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )
            capture_time = float(world.current_time)
            request = PerceptionRequest(
                scene_id=f"{manifest.condition_id}:{tag}:{frame_index}",
                capture_time_s=capture_time,
                frames=(
                    CameraFrame(
                        camera_id="table",
                        capture_time_s=capture_time,
                        calibration_id="benchmark-effective-camera-v1",
                        rgb=rgb,
                    ),
                ),
            )
            try:
                observation = camera_adapter.observe_request(request)
                report_row = vision_provider.last_report.as_dict()
                report_row["decision_frame"] = f"decision_frames/{filename}"
                report_row["sensor_processing"] = dict(sensor_result.processing)
                vision_reports.append(report_row)
                result = _filtered_pickup_observation(observation, config)
                reason = None
            except BenchmarkVisionAbstention as exc:
                report_row = exc.report.as_dict()
                report_row["decision_frame"] = f"decision_frames/{filename}"
                report_row["sensor_processing"] = dict(sensor_result.processing)
                vision_reports.append(report_row)
                result = None
                reason = str(exc)
            frame_index += 1
            return result, reason

        def stable_observation(tag: str) -> tuple[SceneObservation | None, str | None]:
            last = None
            reasons = []
            for stabilization_index in range(STABILIZATION_FRAMES):
                last, reason = capture_for_decision(
                    f"{tag}-stabilize-{stabilization_index}"
                )
                if reason:
                    reasons.append(reason)
            if last is None:
                return None, "; ".join(reasons) or "no accepted observation"
            return last, None

        def save_video_frame(label: str) -> None:
            nonlocal video_index
            if not args.record_video:
                return
            rgba = camera.get_rgba()
            if rgba is None:
                return
            image = cv2.cvtColor(rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
            cv2.putText(
                image,
                label,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(video_frames / f"f{video_index:06d}.png"), image)
            video_index += 1

        initial_truth = all_truth()
        attempted_reference_ids: set[str] = set()
        attempted_track_ids: set[str] = set()
        destination_counts = {colour: 0 for colour in config.task.colour_order}
        events: list[dict[str, Any]] = []
        screw_jaw_target = screw_jaw_half_target(
            shank_diameter_m=mixed_config.screw.shank_diameter_m,
            finger_inner_tip_m=C.FINGER_INNER_TIP,
            seating_margin_m=mixed_config.screw.jaw_seating_margin_m,
            jaw_closed_m=C.JAW_CLOSED,
            jaw_open_m=C.JAW_OPEN,
        )

        for action_index in range(min(args.maximum_actions, manifest.object_count)):
            observation, vision_failure = stable_observation(f"action-{action_index}-observe")
            if observation is None:
                events.append(
                    {
                        "action_index": action_index,
                        "status": "vision_abstained",
                        "reason": vision_failure,
                    }
                )
                break
            supported = tuple(
                item
                for item in observation.objects
                if item.class_id in SUPPORTED_CLASSES
                and item.track_id not in attempted_track_ids
            )
            if not supported:
                events.append(
                    {
                        "action_index": action_index,
                        "status": "no_supported_target",
                        "observed_object_count": len(observation.objects),
                    }
                )
                break
            target = min(supported, key=_target_sort_key)
            truth_before = all_truth()
            reference_id, association_distance = _match_reference(
                target,
                manifest.objects,
                truth_before,
                attempted_reference_ids,
            )
            attempted_track_ids.add(target.track_id)
            if reference_id is not None:
                attempted_reference_ids.add(reference_id)

            destination_colour = target.sort_label
            destination_index = destination_counts[destination_colour]
            offset_x, offset_y = _placement_slot(
                destination_index, screw=target.class_id != "cube"
            )
            bin_x, bin_y = bin_centres[destination_colour]
            if target.class_id == "cube":
                release_z = config.bin_release_tip_base_z_m
                place_spread = np.array([0.0, 1.0, 0.0])
            else:
                desired_grasp_z = (
                    config.frames.surface_base_z_m
                    + config.bin.floor_m
                    + 0.0165
                )
                observed_grasp_world_z = (
                    target.pose.position_base_m[2] + C.TABLE_TOP_Z
                )
                release_z, _ = tool_tip_z_for_desired_grasp_height(
                    desired_grasp_base_z_m=desired_grasp_z,
                    observed_grasp_world_z_m=observed_grasp_world_z,
                    pickup_tip_base_z_m=config.pick_tip_base_z_m,
                    table_top_world_z_m=C.TABLE_TOP_Z,
                )
                place_spread = np.array([0.0, 1.0, 0.0])
            release_pose = TipPoseBase(
                (bin_x + offset_x, bin_y + offset_y, release_z), 0.0
            )
            event: dict[str, Any] = {
                "action_index": action_index,
                "scene_id": observation.scene_id,
                "target_track_id": target.track_id,
                "target_class_id": target.class_id,
                "target_sort_label": target.sort_label,
                "associated_reference_id_evaluator_only": reference_id,
                "association_distance_m_evaluator_only": association_distance,
                "runtime_target_pose": {
                    "position_base_m": list(target.pose.position_base_m),
                    "orientation_xyzw": list(target.pose.orientation_xyzw),
                    "confidence": target.confidence,
                    "pose_quality": target.pose_quality,
                },
            }
            try:
                decision = grasp_planner.decide(
                    observation,
                    target.track_id,
                    release_tip_pose=release_pose,
                )
            except Exception as exc:
                event.update(
                    {
                        "status": "grasp_planning_failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                events.append(event)
                continue
            event["grasp_decision"] = decision.to_record()
            candidate = decision.selected.candidate
            source_spread = np.array(
                [
                    -math.sin(candidate.pick_tip_pose.yaw_rad),
                    math.cos(candidate.pick_tip_pose.yaw_rad),
                    0.0,
                ]
            )
            current_rotation, _ = fk.fk(q)
            current_spread = current_rotation @ np.asarray(C.FINGER_SPREAD_AXIS)
            if np.dot(current_spread, source_spread) < 0.0:
                source_spread = -source_spread
            if np.dot(source_spread, place_spread) < 0.0:
                place_spread = -place_spread
            jaw_target = (
                C.CUBE_JAW_HALF_TARGET
                if target.class_id == "cube"
                else screw_jaw_target
            )
            pick = {
                "xy": list(candidate.pick_tip_pose.xyz_m[:2]),
                "pick_z": float(candidate.pick_tip_pose.xyz_m[2]),
                "spread_dir": source_spread.tolist(),
                "place_xy": [bin_x + offset_x, bin_y + offset_y],
                "place_z": float(release_z),
                "place_spread_dir": place_spread.tolist(),
                "jaw_half_target_m": float(jaw_target),
                "in_bin_open_jaw_half_m": (
                    C.JAW_OPEN
                    if target.class_id == "cube"
                    else min(C.JAW_OPEN, screw_jaw_target + 0.0013)
                ),
                "approach_clearance_m": config.task.approach_clearance_m,
            }
            event["commanded_pick"] = pick
            event["phase_truth"] = {}
            event["trajectory_ik_failures"] = 0
            execution_failed = None
            segments = _motion_segments(pick, home, C)
            for phase, start_xyz, end_xyz, duration, jaw0, jaw1, spread in segments:
                if phase == "revalidate_before_descent":
                    refreshed, refresh_failure = stable_observation(
                        f"action-{action_index}-revalidate"
                    )
                    if refreshed is None:
                        execution_failed = "revalidation_vision_abstained: " + str(refresh_failure)
                        break
                    refreshed_target = next(
                        (
                            item
                            for item in refreshed.objects
                            if item.track_id == target.track_id
                        ),
                        None,
                    )
                    if refreshed_target is None:
                        # Track IDs may be regenerated after a provider
                        # abstention; nearest same-class association remains a
                        # camera-only revalidation, not truth repair.
                        candidates = [
                            item
                            for item in refreshed.objects
                            if item.class_id == target.class_id
                            and item.sort_label == target.sort_label
                        ]
                        refreshed_target = (
                            min(
                                candidates,
                                key=lambda item: math.dist(
                                    item.pose.position_base_m[:2],
                                    target.pose.position_base_m[:2],
                                ),
                            )
                            if candidates
                            else None
                        )
                    if refreshed_target is None:
                        execution_failed = "target_missing_at_pre_descent_revalidation"
                        break
                    movement = math.dist(
                        refreshed_target.pose.position_base_m[:2],
                        target.pose.position_base_m[:2],
                    )
                    event["pre_descent_revalidation"] = {
                        "movement_xy_m": movement,
                        "maximum_allowed_m": 0.005,
                    }
                    if movement > 0.005:
                        execution_failed = "target_moved_before_descent"
                        break
                    continue

                spread_start = None
                if spread is not None and phase in {"approach_pick", "descend_pick", "carry"}:
                    rotation, _ = fk.fk(q)
                    current = rotation @ np.asarray(C.FINGER_SPREAD_AXIS)
                    current[2] = 0.0
                    current /= np.linalg.norm(current)
                    spread = np.asarray(spread, dtype=float)
                    if np.dot(current, spread) < 0.0:
                        spread = -spread
                    spread_start = current
                step_count = max(1, round(duration * FPS))
                for step in range(step_count):
                    fraction = _ease((step + 1) / step_count)
                    target_xyz = np.asarray(start_xyz) + (
                        np.asarray(end_xyz) - np.asarray(start_xyz)
                    ) * fraction
                    direction = (
                        _turn(spread_start, spread, fraction)
                        if spread_start is not None
                        else spread
                    )
                    q, solved = ik.solve(
                        q,
                        target_xyz,
                        down,
                        limits,
                        tool_offset=tool,
                        spread_dir=direction,
                    )
                    if not solved:
                        q, solved = ik.solve(
                            q,
                            target_xyz,
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
                    if not solved or np.linalg.norm(actual - target_xyz) > 0.001 or not aligned:
                        event["trajectory_ik_failures"] += 1
                        execution_failed = f"IK failed during {phase} step {step}"
                        break
                    jaw = jaw0 + (jaw1 - jaw0) * fraction
                    command(q, jaw)
                    world.step(render=args.record_video)
                    world.step(render=args.record_video)
                    if args.record_video and step % 3 == 0:
                        save_video_frame(f"{target.class_id} -> {destination_colour} | {phase}")
                if execution_failed:
                    break
                event["phase_truth"][phase] = all_truth().get(reference_id) if reference_id else None

            if execution_failed:
                event.update({"status": "execution_failed", "reason": execution_failed})
                events.append(event)
                continue
            for _ in range(90):
                command(q, C.JAW_OPEN)
                world.step(render=args.record_video)
            world.step(render=True)
            destination_counts[destination_colour] += 1
            final_truth = all_truth()
            reference = next(
                (item for item in manifest.objects if item.instance_id == reference_id),
                None,
            )
            placed = False
            containment_record = None
            if reference is not None:
                row = final_truth[reference.instance_id]
                if reference.class_name == "cube_10mm":
                    x_m, y_m, z_m = row["reference_base_m"]
                    half = config.bin.pocket_m / 2.0 - config.cube_edge_m / 2.0
                    placed = bool(
                        abs(x_m - bin_x) <= half + 0.002
                        and abs(y_m - bin_y) <= half + 0.002
                        and config.frames.surface_base_z_m + config.bin.floor_m - 0.002
                        <= z_m
                        <= config.frames.surface_base_z_m + config.bin.height_m + 0.01
                    )
                else:
                    containment = verify_vertices_inside_bin(
                        row["vertices_base_m"],
                        bin_centre_xy_m=(bin_x, bin_y),
                        pocket_m=config.bin.pocket_m,
                        floor_z_m=config.frames.surface_base_z_m + config.bin.floor_m,
                        rim_z_m=config.frames.surface_base_z_m + config.bin.height_m,
                        corner_radius_m=0.004,
                        tolerance_m=0.002,
                    )
                    containment_record = containment.as_dict()
                    placed = bool(
                        containment.inside_pocket_xy
                        and containment.above_floor
                        and containment.below_rim
                    )
            event.update(
                {
                    "status": "correctly_placed" if placed else "placement_failed",
                    "complete_grasp_success": placed,
                    "destination_containment": containment_record,
                    "final_truth_evaluator_only": (
                        None if reference_id is None else final_truth[reference_id]
                    ),
                }
            )
            events.append(event)

        final_truth = all_truth()
        correct_ids = {
            event.get("associated_reference_id_evaluator_only")
            for event in events
            if event.get("complete_grasp_success") is True
        }
        correct_ids.discard(None)
        unresolved_ids = sorted(
            {reference.instance_id for reference in manifest.objects} - correct_ids
        )
        correct = len(correct_ids)
        simulated_episode_time_s = float(world.current_time)
        applied_randomization.update(sensor_parameters)
        ledger = classify_application(randomization, applied=sorted(applied_randomization))
        write_json_atomic(run_dir / "randomization_application.json", ledger)
        report = {
            "schema_version": "randomized-sort-episode-v1",
            "condition_id": manifest.condition_id,
            "condition_manifest_sha256": manifest.sha256,
            "policy_cell": {
                "vision": args.vision,
                "grasp": args.grasp,
                "id": (
                    ("Y" if args.vision == YOLO_ASSISTED_HYBRID else "D")
                    + ("A" if args.grasp == "ai" else "D")
                ),
            },
            "realistic_mode": True,
            "runtime_decision_source": "sensor-processed Isaac table-camera RGB",
            "simulator_truth_use": "outcome evaluation and metrics only",
            "object_count": manifest.object_count,
            "reference_object_ids": [item.instance_id for item in manifest.objects],
            "correctly_sorted_object_ids": sorted(correct_ids),
            "correctly_sorted_objects": correct,
            "unresolved_object_ids": unresolved_ids,
            "object_completion_fraction": correct / manifest.object_count,
            "complete_scene_success": correct == manifest.object_count,
            "simulated_episode_time_s": simulated_episode_time_s,
            "correctly_sorted_per_simulated_minute": (
                0.0
                if simulated_episode_time_s <= 0.0
                else correct * 60.0 / simulated_episode_time_s
            ),
            "wall_time_s": time.perf_counter() - wall_started,
            "vision_reports": vision_reports,
            "events": events,
            "initial_truth_evaluator_only": initial_truth,
            "final_truth_evaluator_only": final_truth,
            "appearance_profile": appearance_profile,
            "randomization_manifest_sha256": randomization.sha256,
            "randomization_application_relpath": "randomization_application.json",
            "limitations": [
                "YOLO is a box-level corroborator; calibrated geometry supplies pose.",
                "Seed-55 YOLO has no held-out physical screw evaluation.",
                "Cosmos ranking confidence is not a calibrated grasp-success probability.",
                "Only the table camera drives this benchmark revision.",
                "Isaac outcomes do not prove physical-robot screw performance.",
            ],
        }
        _atomic_report(run_dir / "episode_report.json", report)
        return report
    finally:
        if report is None:
            failure = {
                "schema_version": "randomized-sort-episode-failure-v1",
                "condition_id": manifest.condition_id,
                "condition_manifest_sha256": manifest.sha256,
                "vision": args.vision,
                "grasp": args.grasp,
                "wall_time_s": time.perf_counter() - wall_started,
            }
            _atomic_report(run_dir / "episode_failure.json", failure)
        app.close()


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except Exception as exc:
        print(f"[randomized-benchmark] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "condition_id": report["condition_id"],
        "policy_cell": report["policy_cell"]["id"],
        "correctly_sorted_objects": report["correctly_sorted_objects"],
        "object_count": report["object_count"],
        "complete_scene_success": report["complete_scene_success"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
