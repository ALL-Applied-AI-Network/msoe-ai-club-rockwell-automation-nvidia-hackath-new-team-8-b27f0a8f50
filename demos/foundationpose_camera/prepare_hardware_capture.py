#!/usr/bin/env python3
"""Build a validated FoundationPose capture from hardware RGB-D and masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgbd", type=Path, required=True)
    parser.add_argument("--camera-to-base", type=Path, required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-id", required=True)
    return parser.parse_args()


def matrix(path: Path, shape: tuple[int, int], label: str) -> np.ndarray:
    value = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{label} must be a finite {shape[0]}x{shape[1]} JSON matrix")
    return value


def main() -> int:
    args = arguments()
    rgbd = args.rgbd.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to mix files in non-empty directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "masks").mkdir()
    (output / "meshes").mkdir()

    rgb = cv2.imread(str(rgbd / "rgb.png"), cv2.IMREAD_COLOR)
    depth = np.load(rgbd / "depth_m.npy").astype(np.float32)
    if rgb is None or depth.shape != rgb.shape[:2]:
        raise ValueError("RGB and metre-depth dimensions do not match")
    intrinsics = matrix(rgbd / "intrinsics.json", (3, 3), "intrinsics")
    camera_to_base = matrix(
        args.camera_to_base.expanduser().resolve(), (4, 4), "camera-to-base"
    )
    if not np.allclose(camera_to_base[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("camera-to-base must use column-vector homogeneous form")
    object_specs = json.loads(
        args.objects.expanduser().resolve().read_text(encoding="utf-8")
    )
    if not isinstance(object_specs, list) or not object_specs:
        raise ValueError("objects JSON must be a nonempty list")

    manifest_objects = []
    for spec in object_specs:
        object_id = str(spec["object_id"])
        if re.fullmatch(r"[A-Za-z0-9_.-]+", object_id) is None:
            raise ValueError(f"unsafe object_id: {object_id}")
        mask_source = Path(spec["mask_file"]).expanduser().resolve()
        mesh_source = Path(spec["mesh_file"]).expanduser().resolve()
        mask = cv2.imread(str(mask_source), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != depth.shape or not np.any(mask):
            raise ValueError(f"mask for {object_id} is missing, empty, or mis-sized")
        if mesh_source.suffix.lower() not in (".obj", ".ply"):
            raise ValueError(f"mesh for {object_id} must be OBJ or PLY")
        mask_name = f"masks/{object_id}.png"
        mesh_name = f"meshes/{object_id}{mesh_source.suffix.lower()}"
        if not cv2.imwrite(str(output / mask_name), mask):
            raise RuntimeError(f"could not write mask for {object_id}")
        shutil.copy2(mesh_source, output / mesh_name)
        manifest_objects.append(
            {
                "object_id": object_id,
                "class_id": str(spec["class_id"]),
                "sort_label": spec.get("sort_label"),
                "mask_file": mask_name,
                "mask_source": str(spec.get("mask_source", "hardware_segmentation")),
                "mesh_file": mesh_name,
                "mesh_units": str(spec.get("mesh_units", "m")),
            }
        )

    shutil.copy2(rgbd / "rgb.png", output / "rgb.png")
    np.save(output / "depth_m.npy", depth)
    manifest = {
        "schema_version": 1,
        "capture_id": args.capture_id,
        "rgb_file": "rgb.png",
        "depth_file": "depth_m.npy",
        "depth_definition": "distance_to_image_plane_metres",
        "camera": {
            "intrinsics": intrinsics.tolist(),
            "camera_axes": "opencv_x_right_y_down_z_forward",
            "camera_to_base_column_matrix": camera_to_base.tolist(),
        },
        "objects": manifest_objects,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"capture": str(output), "objects": len(manifest_objects)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
