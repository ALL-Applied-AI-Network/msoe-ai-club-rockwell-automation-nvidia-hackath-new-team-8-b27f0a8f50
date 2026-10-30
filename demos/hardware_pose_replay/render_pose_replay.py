"""Render all three Isaac cameras at curated measured hardware joint poses.

This is a read-only *shadow replay*: it sets the Isaac articulation to measured
``observation.state`` values.  It never consumes recorded actions and has no
physical-robot backend.  Run it from the disposable twin runtime created by
``scripts/setup_brev.sh``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


def _repo_root() -> Path:
    configured = os.environ.get("ROCKWELL_REPO_ROOT")
    candidates = [Path(configured)] if configured else []
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "sorting_core" / "hardware_replay.py").is_file():
            return resolved
    raise RuntimeError("cannot locate repository; set ROCKWELL_REPO_ROOT")


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sorting_core.camera_appearance import (  # noqa: E402
    create_deployment_preview,
    load_camera_appearance_profiles,
)
from sorting_core.hardware_replay import load_replay_document  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--camera-matched-appearance",
        action="store_true",
        help="enable the existing training-only real-table appearance override",
    )
    parser.add_argument(
        "--wrist-hose-stress",
        action="store_true",
        help=(
            "add the measured blue-hose stress augmentation to derived wrist "
            "previews; native raw renders remain unchanged"
        ),
    )
    parser.add_argument(
        "--disable-neutral-match",
        action="store_true",
        help="disable bounded per-camera neutral-median matching in previews",
    )
    parser.add_argument(
        "--show-default-task-objects",
        action="store_true",
        help=(
            "show the twin's default cubes/vials/caps; hidden by default because "
            "their poses do not match the recorded hardware scene"
        ),
    )
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument(
        "--maximum-set-pose-error-deg",
        type=float,
        default=0.05,
        help="fail if any measured arm joint differs from its rendered pose",
    )
    return parser.parse_args()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = _args()
    replay_path = args.replay.expanduser().resolve()
    run = args.run.expanduser().resolve()
    if args.warmup_frames < 1:
        raise SystemExit("--warmup-frames must be at least one")
    if not np.isfinite(args.maximum_set_pose_error_deg) or args.maximum_set_pose_error_deg <= 0:
        raise SystemExit("--maximum-set-pose-error-deg must be finite and positive")
    if run.exists() and any(run.iterdir()):
        raise SystemExit(f"refusing to mix outputs in non-empty directory {run}")
    run.mkdir(parents=True, exist_ok=True)
    source_document, samples = load_replay_document(replay_path)
    camera_profiles = load_camera_appearance_profiles()
    replay_source = source_document.get("source") or {}
    preview_episode_seed = int(replay_source.get("episode_index", 0))

    # Isaac changes the process working directory during startup.  Preserve the
    # disposable twin runtime so its scene_config/cell modules remain importable.
    twin_runtime = Path.cwd().resolve()
    if not (twin_runtime / "scene_config.py").is_file():
        raise SystemExit(
            "run this script from ROCKWELL_TWIN_RUNTIME (scene_config.py is missing)"
        )
    if str(twin_runtime) not in sys.path:
        sys.path.insert(0, str(twin_runtime))

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    report = None
    try:
        import cv2
        import omni.usd
        from pxr import UsdGeom, UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.sensors.camera import Camera

        import scene_config as C
        import cell

        appearance_evidence = None
        if args.camera_matched_appearance:
            from isaac_bridge.training_episode import apply_training_appearance_profile

            appearance_evidence = apply_training_appearance_profile(C)

        world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 30)
        stage = omni.usd.get_context().get_stage()
        cell.build(stage)

        # Scene objects are context only.  Freeze them so pose replay cannot
        # accidentally become a physics/data-generation claim.
        for prim in stage.Traverse():
            if prim.GetName().startswith(("SortCube", "SortVial", "SortCap")) and prim.HasAPI(
                UsdPhysics.RigidBodyAPI
            ):
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)
                if not args.show_default_task_objects:
                    UsdGeom.Imageable(prim).MakeInvisible()

        camera_specs = {
            "table": (cell.TABLE_CAM_PATH, dict(C.TABLE_CAM)),
            "side": (cell.SIDE_CAM_PATH, dict(C.SIDE_CAM)),
            "wrist": (cell.WRIST_CAM_PATH, dict(C.WRIST_CAM)),
        }
        cameras = {
            name: Camera(prim_path=path, resolution=tuple(config["resolution"]))
            for name, (path, config) in camera_specs.items()
        }

        cell.enable_ccd(world)
        world.reset()
        for camera in cameras.values():
            camera.initialize()

        arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia-shadow-replay")
        arm.initialize()
        names = list(arm.dof_names)
        arm_indices = [names.index(f"J{index}") for index in range(1, 7)]
        joint_index = {name: index for index, name in enumerate(names)}

        def set_measured_pose(sample) -> np.ndarray:
            full = np.asarray(arm.get_joint_positions(), dtype=float).copy()
            full[arm_indices] = np.deg2rad(sample.joints_deg)
            jaw_half_opening = C.JAW_OPEN + sample.gripper_state * (
                C.JAW_CLOSED - C.JAW_OPEN
            )
            for name, value in cell.jaw_targets(jaw_half_opening).items():
                full[joint_index[name]] = value
            arm.set_joint_positions(full)
            return full

        set_measured_pose(samples[0])
        for _ in range(args.warmup_frames):
            set_measured_pose(samples[0])
            world.step(render=True)

        frame_reports = []
        for sample in samples:
            # Reset twice around rendering so articulation drives cannot turn
            # this into controller-dynamics replay.
            set_measured_pose(sample)
            world.step(render=True)
            set_measured_pose(sample)
            world.step(render=True)
            measured = np.asarray(arm.get_joint_positions(), dtype=float)[arm_indices]
            joint_error_deg = np.rad2deg(measured) - np.asarray(sample.joints_deg)
            maximum_joint_error_deg = float(np.max(np.abs(joint_error_deg)))
            if maximum_joint_error_deg > args.maximum_set_pose_error_deg:
                raise RuntimeError(
                    f"frame {sample.frame_index} pose error "
                    f"{maximum_joint_error_deg:.6f} deg exceeds "
                    f"{args.maximum_set_pose_error_deg:.6f} deg"
                )

            camera_reports = {}
            for camera_name, camera in cameras.items():
                rgba = camera.get_rgba()
                if rgba is None:
                    raise RuntimeError(f"{camera_name} camera returned no image")
                rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8)
                raw_relative = Path("raw") / camera_name / f"f{sample.frame_index:05d}.png"
                preview_relative = (
                    Path("deployment") / camera_name / f"f{sample.frame_index:05d}.png"
                )
                mask_relative = (
                    Path("occluder_masks")
                    / camera_name
                    / f"f{sample.frame_index:05d}.png"
                )
                raw_path = run / raw_relative
                preview_path = run / preview_relative
                mask_path = run / mask_relative
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(raw_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f"failed to write {raw_path}")
                preview_result = create_deployment_preview(
                    rgb,
                    camera_profiles,
                    camera_id=camera_name,
                    episode_seed=preview_episode_seed,
                    frame_index=sample.frame_index,
                    add_wrist_occlusion=(
                        args.wrist_hose_stress and camera_name == "wrist"
                    ),
                    match_curated_neutral_median=(
                        not args.disable_neutral_match
                    ),
                )
                if not cv2.imwrite(
                    str(preview_path),
                    cv2.cvtColor(preview_result.image_rgb, cv2.COLOR_RGB2BGR),
                ):
                    raise RuntimeError(f"failed to write {preview_path}")
                mask_relpath = None
                if np.count_nonzero(preview_result.occluder_mask):
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(mask_path), preview_result.occluder_mask):
                        raise RuntimeError(f"failed to write {mask_path}")
                    mask_relpath = mask_relative.as_posix()

                prim_path, config = camera_specs[camera_name]
                prim = stage.GetPrimAtPath(prim_path)
                usd_camera = UsdGeom.Camera(prim)
                transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
                camera_reports[camera_name] = {
                    "raw_image_relpath": raw_relative.as_posix(),
                    "raw_resolution": [int(rgb.shape[1]), int(rgb.shape[0])],
                    "deployment_preview_relpath": preview_relative.as_posix(),
                    "deployment_resolution": [640, 360],
                    "deployment_processing": dict(preview_result.processing),
                    "synthetic_occluder_mask_relpath": mask_relpath,
                    "deployment_preview_label_role": (
                        "unlabeled robustness view"
                        if mask_relpath is not None
                        else "appearance comparison; no synthetic occluder"
                    ),
                    "camera_prim_path": prim_path,
                    "parent": config["parent"],
                    "focal_length_mm": float(usd_camera.GetFocalLengthAttr().Get()),
                    "horizontal_aperture_mm": float(
                        usd_camera.GetHorizontalApertureAttr().Get()
                    ),
                    "camera_to_world_row_major": [
                        [float(transform[row][column]) for column in range(4)]
                        for row in range(4)
                    ],
                }

            frame_reports.append(
                {
                    "source_frame_index": sample.frame_index,
                    "source_timestamp_s": sample.timestamp_s,
                    "source_joints_deg": list(sample.joints_deg),
                    "source_gripper_state": sample.gripper_state,
                    "source_tcp_pose_uf0_mm_deg": list(sample.tcp_pose_uf0_mm_deg),
                    "source_fk_position_error_mm": sample.fk_position_error_mm,
                    "source_fk_orientation_error_deg": (
                        sample.fk_orientation_error_deg
                    ),
                    "maximum_set_pose_joint_error_deg": maximum_joint_error_deg,
                    "cameras": camera_reports,
                }
            )

        report = {
            "schema_version": 1,
            "replay_kind": "measured_state_shadow_replay",
            "physical_robot_commands_permitted": False,
            "source": source_document.get("source"),
            "action_audit": source_document.get("action_audit"),
            "camera_matched_appearance": appearance_evidence,
            "camera_profiles": {
                "path": camera_profiles.source_path,
                "sha256": camera_profiles.source_sha256,
                "curation_path": camera_profiles.curation.source_path,
                "curation_sha256": camera_profiles.curation.source_sha256,
            },
            "default_task_objects_visible": bool(args.show_default_task_objects),
            "wrist_hose_stress_enabled": bool(args.wrist_hose_stress),
            "maximum_allowed_set_pose_error_deg": args.maximum_set_pose_error_deg,
            "frame_count": len(frame_reports),
            "frames": frame_reports,
            "limitations": [
                "Camera rows have no independent exposure timestamps.",
                "Flexible hose pose and real scene objects are not reconstructed.",
                "This sets measured state directly and does not model controller dynamics.",
                "Deployment previews approximate YUV420 but are not AV1 bitstream matches.",
            ],
        }
        _atomic_json(run / "replay_report.json", report)
        print(
            f"rendered {len(frame_reports)} curated poses x {len(cameras)} cameras to {run}",
            flush=True,
        )
    finally:
        app.close()


if __name__ == "__main__":
    main()
