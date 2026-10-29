#!/usr/bin/env python3
"""Compare broad table-camera appearance between real video and Isaac images.

This tool deliberately measures only low-frequency colour and exposure from
compressed rollout video.  It downsamples synthetic images to the real video
resolution and excludes saturated objects plus the central robot corridor.
It must not be used to claim a calibrated noise, blur, or camera-response
model; that requires native, lossless real-camera frames.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-video", required=True, type=Path)
    parser.add_argument("--sim-dir", required=True, type=Path)
    parser.add_argument("--sim-pattern", default="*.png")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument(
        "--sim-domain",
        choices=("raw", "sensor"),
        required=True,
        help="Records whether input images are raw Isaac RGB or derived sensor RGB.",
    )
    return parser.parse_args()


def _uniform_indices(count: int, requested: int) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("source contains no frames")
    if requested < 1:
        raise ValueError("--samples must be at least 1")
    return tuple(
        int(value)
        for value in np.unique(
            np.rint(np.linspace(0, count - 1, min(count, requested)))
        )
    )


def _read_video(path: Path, samples: int) -> tuple[list[np.ndarray], dict]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open real video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    indices = _uniform_indices(count, samples)
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise RuntimeError(f"cannot decode real-video frame {index}: {path}")
        frames.append(frame)
    capture.release()
    return frames, {
        "path": str(path.resolve()),
        "frame_count": count,
        "sampled_indices": list(indices),
        "fps": fps,
        "resolution": [width, height],
    }


def _read_images(
    directory: Path,
    pattern: str,
    samples: int,
    resolution: tuple[int, int],
) -> tuple[list[np.ndarray], dict]:
    paths = sorted(directory.glob(pattern))
    indices = _uniform_indices(len(paths), samples)
    frames = []
    original_sizes = set()
    for index in indices:
        image = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot decode synthetic image: {paths[index]}")
        original_sizes.add((int(image.shape[1]), int(image.shape[0])))
        frames.append(cv2.resize(image, resolution, interpolation=cv2.INTER_AREA))
    return frames, {
        "directory": str(directory.resolve()),
        "pattern": pattern,
        "image_count": len(paths),
        "sampled_files": [paths[index].name for index in indices],
        "original_resolutions": [list(value) for value in sorted(original_sizes)],
        "comparison_resolution": list(resolution),
        "resize_interpolation": "cv2.INTER_AREA",
    }


def _region_masks(width: int, height: int) -> dict[str, np.ndarray]:
    table = np.zeros((height, width), dtype=np.uint8)
    table_points = np.asarray(
        [
            (0.20 * width, 0.24 * height),
            (0.80 * width, 0.24 * height),
            (0.97 * width, 0.93 * height),
            (0.03 * width, 0.93 * height),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(table, table_points, 255)
    # Robot, gripper and hoses move through this corridor.  Their colours must
    # not determine the table's exposure or white-balance estimate.
    table[: int(0.62 * height), int(0.36 * width) : int(0.64 * width)] = 0

    background = np.zeros((height, width), dtype=np.uint8)
    background[: int(0.17 * height), : int(0.34 * width)] = 255
    background[: int(0.17 * height), int(0.66 * width) :] = 255
    return {"neutral_table": table, "neutral_background": background}


def _neutral_pixels(frames: Iterable[np.ndarray], mask: np.ndarray) -> np.ndarray:
    chunks = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        neutral = (hsv[..., 1] <= 48) & (mask != 0)
        pixels = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[neutral]
        if pixels.size:
            chunks.append(pixels)
    if not chunks:
        raise RuntimeError("comparison mask retained no neutral pixels")
    pixels = np.concatenate(chunks, axis=0)
    if len(pixels) > 1_000_000:
        rng = np.random.default_rng(20261029)
        pixels = pixels[rng.choice(len(pixels), 1_000_000, replace=False)]
    return pixels


def _statistics(rgb: np.ndarray) -> dict:
    rgb_f = rgb.astype(np.float32)
    sample = rgb.reshape((-1, 1, 3))
    lab = cv2.cvtColor(sample, cv2.COLOR_RGB2LAB).reshape((-1, 3)).astype(float)
    luminance = (
        0.2126 * rgb_f[:, 0] + 0.7152 * rgb_f[:, 1] + 0.0722 * rgb_f[:, 2]
    )
    return {
        "pixel_count": int(len(rgb)),
        "rgb_median": np.median(rgb_f, axis=0).tolist(),
        "rgb_p05": np.percentile(rgb_f, 5, axis=0).tolist(),
        "rgb_p95": np.percentile(rgb_f, 95, axis=0).tolist(),
        "lab_median_opencv": np.median(lab, axis=0).tolist(),
        "lab_iqr_opencv": (
            np.percentile(lab, 75, axis=0) - np.percentile(lab, 25, axis=0)
        ).tolist(),
        "luminance_median_8bit": float(np.median(luminance)),
        "luminance_p05_8bit": float(np.percentile(luminance, 5)),
        "luminance_p95_8bit": float(np.percentile(luminance, 95)),
        "near_black_fraction": float(np.mean(np.max(rgb_f, axis=1) <= 5.0)),
        "near_white_fraction": float(np.mean(np.min(rgb_f, axis=1) >= 250.0)),
    }


def _compare(real: dict, sim: dict) -> dict:
    real_rgb = np.asarray(real["rgb_median"], dtype=float)
    sim_rgb = np.asarray(sim["rgb_median"], dtype=float)
    real_luma = float(real["luminance_median_8bit"])
    sim_luma = float(sim["luminance_median_8bit"])
    exposure = math.log2(max(real_luma, 1.0) / max(sim_luma, 1.0))
    real_relative = real_rgb / max(real_rgb[1], 1.0)
    sim_relative = sim_rgb / max(sim_rgb[1], 1.0)
    wb = real_relative / np.maximum(sim_relative, 1e-6)
    return {
        "rgb_median_real_minus_sim": (real_rgb - sim_rgb).tolist(),
        "absolute_rgb_median_error": float(np.mean(np.abs(real_rgb - sim_rgb))),
        "broad_exposure_hint_stops": exposure,
        "broad_white_balance_hint_rgb_relative_to_green": wb.tolist(),
        "warning": (
            "Exposure and white-balance hints confound illumination and material "
            "reflectance. Use them only as bounded starting values, then validate "
            "on held-out real frames."
        ),
    }


def _write_contact_sheet(
    path: Path, real_frames: list[np.ndarray], sim_frames: list[np.ndarray]
) -> None:
    real = real_frames[len(real_frames) // 2].copy()
    sim = sim_frames[len(sim_frames) // 2].copy()
    for image, label in ((real, "REAL TABLE VIDEO"), (sim, "SYNTHETIC")):
        cv2.rectangle(image, (0, 0), (260, 32), (15, 15, 15), -1)
        cv2.putText(
            image,
            label,
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.concatenate((real, sim), axis=1)):
        raise RuntimeError(f"cannot write contact sheet: {path}")


def main() -> None:
    args = _parse_args()
    real_frames, real_source = _read_video(args.real_video, args.samples)
    height, width = real_frames[0].shape[:2]
    sim_frames, sim_source = _read_images(
        args.sim_dir,
        args.sim_pattern,
        args.samples,
        (width, height),
    )
    masks = _region_masks(width, height)
    regions = {}
    for name, mask in masks.items():
        real_stats = _statistics(_neutral_pixels(real_frames, mask))
        sim_stats = _statistics(_neutral_pixels(sim_frames, mask))
        regions[name] = {
            "real": real_stats,
            "sim": sim_stats,
            "comparison": _compare(real_stats, sim_stats),
        }
    report = {
        "schema_version": 1,
        "real": real_source,
        "sim": {**sim_source, "domain": args.sim_domain},
        "regions": regions,
        "method": {
            "colour_space": "RGB plus OpenCV 8-bit Lab",
            "neutral_pixel_rule": "HSV saturation <= 48",
            "geometry": (
                "fixed normalized table/background masks; central moving robot "
                "corridor excluded"
            ),
            "limitation": (
                "Real input is compressed 640x360 rollout video. Results support "
                "broad colour/exposure matching only, not native sensor noise, "
                "blur, small-object edge quality, or a photometric calibration claim."
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.contact_sheet is not None:
        _write_contact_sheet(args.contact_sheet, real_frames, sim_frames)
    print(json.dumps(report["regions"], indent=2))


if __name__ == "__main__":
    main()
