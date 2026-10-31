#!/usr/bin/env python3
"""Capture one aligned hardware RGB-D frame from an Intel RealSense camera."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise SystemExit(
            "pyrealsense2 is required: python -m pip install pyrealsense2"
        ) from error
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to mix files in non-empty directory {output}")
    output.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.depth, args.width, args.height, rs.format.z16, args.fps
    )
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    try:
        frames = None
        for _ in range(args.warmup_frames + 1):
            frames = align.process(pipeline.wait_for_frames())
        assert frames is not None
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense did not return aligned color and depth")
        color_bgr = np.asanyarray(color_frame.get_data())
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        camera_matrix = [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ]
        if not cv2.imwrite(str(output / "rgb.png"), color_bgr):
            raise RuntimeError("could not write rgb.png")
        np.save(output / "depth_m.npy", depth_m)
        (output / "intrinsics.json").write_text(
            json.dumps(camera_matrix, indent=2) + "\n", encoding="utf-8"
        )
        metadata = {
            "depth_units": "metres",
            "depth_aligned_to": "rgb",
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "camera_model": intrinsics.model.name,
            "distortion_coefficients": list(intrinsics.coeffs),
        }
        (output / "capture_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        pipeline.stop()
    print(json.dumps({"output": str(output), "depth_units": "metres"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
