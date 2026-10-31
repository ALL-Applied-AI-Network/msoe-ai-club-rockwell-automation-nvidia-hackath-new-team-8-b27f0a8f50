#!/usr/bin/env python3
"""Run pretrained FoundationPose on an isolated RGB-D capture.

This script is executed inside NVIDIA's FoundationPose inference environment.
It deliberately has no argument for simulator truth or spawn configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from foundation_pose_nvidia import Estimator, EstimatorOptions, RgbdFrame, RuntimeConfig


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug-dir", type=Path, required=True)
    parser.add_argument("--refine-model-path", type=Path)
    parser.add_argument("--score-model-path", type=Path)
    parser.add_argument("--engine-cache-dir", type=Path, required=True)
    parser.add_argument(
        "--n-hypotheses",
        type=int,
        default=int(os.environ.get("FP_N_HYPOTHESES", "252")),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("FP_BATCH_SIZE", "0")),
        help=(
            "RefineNet micro-batch size; 0 uses n-hypotheses. Must divide "
            "n-hypotheses. Smaller values reduce peak VRAM use."
        ),
    )
    return parser.parse_args()


def _matrix(raw, rows: int, columns: int, name: str) -> np.ndarray:
    result = np.asarray(raw, dtype=np.float64)
    if result.shape != (rows, columns) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {rows}x{columns} matrix")
    return result


def _rotation_to_quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    """Convert a proper 3x3 rotation matrix to a canonical unit quaternion."""

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion.tolist()


def main() -> int:
    args = _arguments()
    if args.n_hypotheses <= 0:
        raise ValueError("--n-hypotheses must be positive")
    batch_size = args.batch_size or args.n_hypotheses
    if batch_size <= 0 or args.n_hypotheses % batch_size:
        raise ValueError("--batch-size must be positive and divide --n-hypotheses")

    capture = args.capture.expanduser().resolve()
    manifest = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("capture manifest schema_version must be 1")
    rgb_bgr = cv2.imread(str(capture / manifest["rgb_file"]), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError("capture RGB image could not be read")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    depth = np.load(capture / manifest["depth_file"]).astype(np.float32)
    if depth.shape != rgb.shape[:2]:
        raise ValueError("RGB and depth dimensions do not match")
    camera = manifest["camera"]
    intrinsics = _matrix(camera["intrinsics"], 3, 3, "camera intrinsics")
    camera_to_base = _matrix(
        camera["camera_to_base_column_matrix"], 4, 4, "camera_to_base"
    )
    if not np.allclose(camera_to_base[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("camera_to_base must use the column-vector convention")

    args.debug_dir.mkdir(parents=True, exist_ok=True)
    args.engine_cache_dir.mkdir(parents=True, exist_ok=True)
    runtime = RuntimeConfig(
        max_image_width=rgb.shape[1],
        max_image_height=rgb.shape[0],
        n_hypotheses=args.n_hypotheses,
        batch_size=batch_size,
        capture_cuda_graph=False,
    )
    objects: dict[str, dict] = {}
    for object_spec in manifest["objects"]:
        object_id = str(object_spec["object_id"])
        mask = cv2.imread(
            str(capture / object_spec["mask_file"]), cv2.IMREAD_GRAYSCALE
        )
        if mask is None or mask.shape != depth.shape or not np.any(mask):
            raise ValueError(f"{object_id} mask is missing, empty, or mis-sized")
        mesh_path = Path(object_spec["mesh_file"])
        if not mesh_path.is_absolute():
            mesh_path = (capture / mesh_path).resolve()
        mesh_units = object_spec.get("mesh_units", "m")
        if mesh_units == "mm":
            mesh_unit_scale = 0.001
        elif mesh_units == "m":
            mesh_unit_scale = 1.0
        else:
            raise ValueError(f"unsupported mesh units for {object_id}")
        options = EstimatorOptions.from_env(
            str(mesh_path),
            refine_model_path=(
                None if args.refine_model_path is None else str(args.refine_model_path)
            ),
            score_model_path=(
                None if args.score_model_path is None else str(args.score_model_path)
            ),
            engine_cache_dir=str(args.engine_cache_dir),
            mesh_unit_scale=mesh_unit_scale,
        )
        frame = RgbdFrame(
            rgb=rgb,
            depth_m=depth,
            intrinsics=intrinsics.astype(np.float32),
            mask=mask,
        )
        with Estimator(
            options,
            runtime,
            prepare_batch=batch_size,
        ) as estimator:
            result = estimator.register(frame)
        object_to_base = camera_to_base @ np.asarray(result.pose)
        raw_score = float(result.score)
        if not math.isfinite(raw_score):
            raise RuntimeError(
                f"FoundationPose returned a non-finite ScoreNet score for {object_id}"
            )
        # NVIDIA ScoreNet emits an unbounded ranking score, not a probability.
        # Keep it losslessly and expose a stable monotonic [0, 1] display value.
        # Downstream hardware deployments should calibrate raw_score against the
        # accumulated pose errors rather than interpret this value as probability.
        confidence = 0.5 + math.atan(raw_score) / math.pi
        objects[object_id] = {
            "class_id": object_spec["class_id"],
            "sort_label": object_spec.get("sort_label"),
            "position_base_m": object_to_base[:3, 3].tolist(),
            "orientation_xyzw": _rotation_to_quaternion_xyzw(
                object_to_base[:3, :3]
            ),
            "confidence": confidence,
            "confidence_source": "foundationpose_scorenet_arctan_normalized",
            "foundationpose_raw_score": raw_score,
            "confidence_is_calibrated_probability": False,
            "mask_source": object_spec.get("mask_source", "unknown"),
            "inference_time_s": float(result.elapsed_s),
        }

    output = {
        "schema_version": 1,
        "coordinate_frame": "robot_base",
        "provider": "nvidia_foundationpose_inference_library",
        "provider_mode": "model_based_rgbd",
        "capture_id": manifest.get("capture_id"),
        "objects": objects,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "objects": len(objects)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
