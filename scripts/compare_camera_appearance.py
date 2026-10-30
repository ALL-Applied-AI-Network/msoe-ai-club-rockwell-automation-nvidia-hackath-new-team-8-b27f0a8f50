#!/usr/bin/env python3
"""Compare broad per-camera appearance between reviewed real and Isaac images.

This tool deliberately measures only low-frequency colour and exposure from
compressed rollout video.  It downsamples synthetic images to the real video
resolution and excludes saturated objects plus camera-specific robot regions.
It must not be used to claim a calibrated noise, blur, or camera-response
model; that requires native, lossless real-camera frames. Raw rollout videos
require a default-deny curation manifest unless the caller explicitly asserts
that the entire input was independently privacy-reviewed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sorting_core.camera_appearance import camera_neutral_region_masks  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-video", required=True, type=Path)
    parser.add_argument("--sim-dir", required=True, type=Path)
    parser.add_argument("--sim-pattern", default="*.png")
    parser.add_argument(
        "--camera",
        choices=("table", "side", "wrist"),
        default="table",
        help="Selects camera-specific neutral regions and report metadata.",
    )
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--contact-sheet", type=Path)
    privacy_source = parser.add_mutually_exclusive_group(required=True)
    privacy_source.add_argument(
        "--curation-manifest",
        type=Path,
        help=(
            "Privacy/default-deny frame manifest. When supplied, --dataset-id, "
            "--episode and --curation-split are required."
        ),
    )
    privacy_source.add_argument(
        "--assume-precurated-input",
        action="store_true",
        help=(
            "Explicitly assert that the entire input video was independently "
            "privacy-reviewed. Never use this for the downloaded raw rollouts."
        ),
    )
    parser.add_argument("--dataset-id")
    parser.add_argument("--episode", type=int)
    parser.add_argument(
        "--curation-split",
        default="appearance_fit",
        help="Named split in the curation manifest (default: appearance_fit).",
    )
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


def _curated_ranges(
    manifest_path: Path | None,
    *,
    dataset_id: str | None,
    episode: int | None,
    split: str,
    assume_precurated_input: bool = False,
) -> tuple[tuple[int, int], ...] | None:
    """Resolve reviewed frame ranges without ever loading excluded imagery.

    The checked-in curation contract is intentionally default-deny.  Supplying
    a manifest without an exact dataset/episode match is an error rather than a
    silent fallback to the complete video.
    """

    if manifest_path is None:
        if dataset_id is not None or episode is not None:
            raise ValueError("dataset/episode require --curation-manifest")
        if not assume_precurated_input:
            raise ValueError(
                "real video requires --curation-manifest or the explicit "
                "--assume-precurated-input acknowledgement"
            )
        return None
    if assume_precurated_input:
        raise ValueError(
            "--curation-manifest and --assume-precurated-input are mutually exclusive"
        )
    if not dataset_id or episode is None:
        raise ValueError(
            "--curation-manifest requires --dataset-id and --episode"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("default_policy") != "deny":
            raise ValueError("curation manifest must be default-deny")
        splits = manifest["splits"]
        entries = splits[split]
        excluded_entries = splits["excluded_human_or_setup"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot read curation split {split!r}: {exc}") from exc
    matches = [
        entry
        for entry in entries
        if entry.get("dataset") == dataset_id
        and int(entry.get("episode", -1)) == episode
    ]
    if len(matches) != 1:
        raise ValueError(
            f"curation split {split!r} has no unique {dataset_id} episode {episode}"
        )
    result = []
    for index, value in enumerate(matches[0].get("ranges", ())):
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"curation range {index} must be [start, end)")
        start, end = (int(item) for item in value)
        if start < 0 or end <= start:
            raise ValueError(f"curation range {index} is invalid: {value!r}")
        result.append((start, end))
    if not result:
        raise ValueError("curation entry contains no accepted ranges")
    excluded_ranges = []
    for excluded in excluded_entries:
        if (
            excluded.get("dataset") != dataset_id
            or int(excluded.get("episode", -1)) != episode
        ):
            continue
        for value in excluded.get("ranges", ()):
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError("privacy exclusion range must be [start, end)")
            start, end = (int(item) for item in value)
            if start < 0 or end <= start:
                raise ValueError(f"privacy exclusion range is invalid: {value!r}")
            excluded_ranges.append((start, end))
    for accepted_start, accepted_end in result:
        for excluded_start, excluded_end in excluded_ranges:
            if accepted_start < excluded_end and excluded_start < accepted_end:
                raise ValueError(
                    "curated frame range overlaps excluded human/setup imagery"
                )
    return tuple(result)


def _indices_in_ranges(
    count: int,
    requested: int,
    accepted_ranges: tuple[tuple[int, int], ...] | None,
) -> tuple[int, ...]:
    if accepted_ranges is None:
        return _uniform_indices(count, requested)
    chunks = [
        np.arange(max(0, start), min(count, end), dtype=np.int64)
        for start, end in accepted_ranges
        if min(count, end) > max(0, start)
    ]
    if not chunks:
        raise ValueError("curated frame ranges retain no frames from this video")
    eligible = np.concatenate(chunks)
    if eligible.size == 0:
        raise ValueError("curated frame ranges retain no frames from this video")
    positions = _uniform_indices(len(eligible), requested)
    return tuple(int(eligible[position]) for position in positions)


def _read_video(
    path: Path,
    samples: int,
    *,
    accepted_ranges: tuple[tuple[int, int], ...] | None = None,
) -> tuple[list[np.ndarray], dict]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open real video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    indices = _indices_in_ranges(count, samples, accepted_ranges)
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
        "accepted_frame_ranges": (
            None
            if accepted_ranges is None
            else [list(value) for value in accepted_ranges]
        ),
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


def _region_masks(
    width: int, height: int, camera: str = "table"
) -> dict[str, np.ndarray]:
    return dict(camera_neutral_region_masks(width, height, camera))


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
    path: Path,
    real_frames: list[np.ndarray],
    sim_frames: list[np.ndarray],
    *,
    camera: str = "table",
) -> None:
    real = real_frames[len(real_frames) // 2].copy()
    sim = sim_frames[len(sim_frames) // 2].copy()
    camera_label = camera.upper()
    for image, label in (
        (real, f"REAL {camera_label} VIDEO"),
        (sim, f"SYNTHETIC {camera_label}"),
    ):
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
    try:
        accepted_ranges = _curated_ranges(
            args.curation_manifest,
            dataset_id=args.dataset_id,
            episode=args.episode,
            split=args.curation_split,
            assume_precurated_input=args.assume_precurated_input,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    real_frames, real_source = _read_video(
        args.real_video,
        args.samples,
        accepted_ranges=accepted_ranges,
    )
    height, width = real_frames[0].shape[:2]
    sim_frames, sim_source = _read_images(
        args.sim_dir,
        args.sim_pattern,
        args.samples,
        (width, height),
    )
    masks = _region_masks(width, height, args.camera)
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
        "camera": args.camera,
        "real": real_source,
        "sim": {**sim_source, "domain": args.sim_domain},
        "regions": regions,
        "method": {
            "colour_space": "RGB plus OpenCV 8-bit Lab",
            "neutral_pixel_rule": "HSV saturation <= 48",
            "geometry": (
                f"fixed normalized {args.camera}-camera neutral masks; saturated "
                "objects and hoses excluded by the neutral-pixel rule"
            ),
            "curation": (
                {
                    "manifest": None,
                    "assume_precurated_input": True,
                    "warning": (
                        "The caller explicitly asserted that the entire video was "
                        "independently privacy-reviewed."
                    ),
                }
                if args.curation_manifest is None
                else {
                    "manifest": str(args.curation_manifest.resolve()),
                    "dataset_id": args.dataset_id,
                    "episode": args.episode,
                    "split": args.curation_split,
                    "default_deny": True,
                }
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
        _write_contact_sheet(
            args.contact_sheet,
            real_frames,
            sim_frames,
            camera=args.camera,
        )
    print(json.dumps(report["regions"], indent=2))


if __name__ == "__main__":
    main()
