"""Run a hash-pinned, offline YOLOE visual-prompt segmentation experiment.

This command never constructs a hardware ``VisionProvider`` and never outputs
robot coordinates.  It tests only whether YOLOE can provide useful instance
masks on native-resolution ROI tiles.  A separate calibrated geometry stage is
required before a mask can become a grasp-team ``SceneObservation``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping

import cv2
import numpy as np

from .yoloe_experiment import (
    ImagePrediction,
    evaluate_boxes,
    extract_tile_predictions,
    generate_tiles,
    merge_tile_predictions,
)


EXPECTED_ULTRALYTICS_VERSION = "8.4.194"
ALLOWED_LICENSE_CONTEXTS = frozenset(
    {"evaluation-only-agpl-review", "enterprise-authorized"}
)


class ManifestError(ValueError):
    """Raised when a manifest is ambiguous or cannot be reproduced."""


def _reject_constant(value: str) -> None:
    raise ManifestError(f"non-standard JSON number {value!r} is not allowed")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_object(
    value: Any,
    name: str,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{name} must be an object")
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - set(optional))
    if missing:
        raise ManifestError(f"{name} is missing keys: {missing}")
    if unknown:
        raise ManifestError(f"{name} has unknown keys: {unknown}")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(path: Path, expected: Any, name: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64 or any(
        char not in "0123456789abcdefABCDEF" for char in expected
    ):
        raise ManifestError(f"{name}.sha256 must be 64 hexadecimal characters")
    actual = _sha256(path)
    if actual != expected.lower():
        raise ManifestError(
            f"{name} SHA-256 mismatch: expected {expected.lower()}, got {actual}"
        )
    return actual


def _resolve_file(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{name} must be a non-empty path string")
    path = Path(value).expanduser()
    resolved = path if path.is_absolute() else base / path
    try:
        resolved = resolved.resolve(strict=True)
    except OSError as exc:
        raise ManifestError(f"cannot resolve {name}: {exc}") from exc
    if not resolved.is_file():
        raise ManifestError(f"{name} is not a file: {resolved}")
    return resolved


def _pair(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)):
        raise ManifestError(f"{name} must have two integers")
    try:
        source = tuple(value)
        result = tuple(int(item) for item in source)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{name} must have two integers") from exc
    if (
        len(result) != 2
        or min(result) <= 0
        or any(isinstance(item, bool) or result[index] != item for index, item in enumerate(source))
    ):
        raise ManifestError(f"{name} must have two positive integers")
    return result


def _box(value: Any, name: str) -> tuple[float, float, float, float]:
    if isinstance(value, (str, bytes)):
        raise ManifestError(f"{name} must have four finite numbers")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{name} must have four finite numbers") from exc
    if len(result) != 4 or not all(np.isfinite(result)):
        raise ManifestError(f"{name} must have four finite numbers")
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ManifestError(f"{name} must be a non-empty XYXY box")
    return result


def _git_value(repository: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _load_annotation(path: Path, expected_image_sha256: str) -> tuple[dict[str, Any], ...]:
    raw = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    document = _strict_object(
        raw,
        "annotation",
        required={"schema_version", "image_sha256", "instances"},
    )
    if document["schema_version"] != 1:
        raise ManifestError("annotation.schema_version must be the integer 1")
    if document["image_sha256"] != expected_image_sha256:
        raise ManifestError("annotation image_sha256 does not match its input image")
    if not isinstance(document["instances"], list):
        raise ManifestError("annotation.instances must be a list")
    instances = []
    for index, value in enumerate(document["instances"]):
        instance = _strict_object(
            value,
            f"annotation.instances[{index}]",
            required={"class_id", "bbox_xyxy"},
        )
        if not isinstance(instance["class_id"], str) or not instance["class_id"].strip():
            raise ManifestError("annotation class_id must be a non-empty string")
        instances.append(
            {
                "class_id": instance["class_id"].strip(),
                "bbox_xyxy": list(_box(instance["bbox_xyxy"], "annotation bbox")),
            }
        )
    return tuple(instances)


def load_manifest(path: Path) -> dict[str, Any]:
    """Load strict JSON and resolve all external artifacts without importing AI code."""

    path = path.expanduser().resolve(strict=True)
    raw = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    root = _strict_object(
        raw,
        "manifest",
        required={
            "schema_version",
            "experiment_id",
            "license_context",
            "model",
            "expected_image_size",
            "roi_xyxy",
            "tile_size",
            "overlap_fraction",
            "merge_iou_threshold",
            "prompts",
            "inputs",
            "inference",
        },
    )
    if root["schema_version"] != 1:
        raise ManifestError("manifest.schema_version must be the integer 1")
    if not isinstance(root["experiment_id"], str) or not root["experiment_id"].strip():
        raise ManifestError("experiment_id must be a non-empty string")
    if root["license_context"] not in ALLOWED_LICENSE_CONTEXTS:
        raise ManifestError(
            "license_context must explicitly be evaluation-only-agpl-review or "
            "enterprise-authorized"
        )
    base = path.parent
    model = _strict_object(
        root["model"],
        "model",
        required={"checkpoint", "sha256", "ultralytics_version", "model_id"},
    )
    checkpoint = _resolve_file(base, model["checkpoint"], "model.checkpoint")
    checkpoint_sha256 = _verify_sha256(checkpoint, model["sha256"], "model")
    if model["ultralytics_version"] != EXPECTED_ULTRALYTICS_VERSION:
        raise ManifestError(
            f"ultralytics_version must be pinned to {EXPECTED_ULTRALYTICS_VERSION}"
        )
    if not isinstance(model["model_id"], str) or not model["model_id"].strip():
        raise ManifestError("model.model_id must be a non-empty string")
    image_size = _pair(root["expected_image_size"], "expected_image_size")
    roi = _box(root["roi_xyxy"], "roi_xyxy")
    roi_integer = tuple(int(value) for value in roi)
    if tuple(float(value) for value in roi_integer) != roi:
        raise ManifestError("roi_xyxy must use integer native-image pixels")
    tile_size = _pair(root["tile_size"], "tile_size")
    overlap_fraction = float(root["overlap_fraction"])
    merge_iou_threshold = float(root["merge_iou_threshold"])
    if not np.isfinite(merge_iou_threshold) or not 0.0 <= merge_iou_threshold <= 1.0:
        raise ManifestError("merge_iou_threshold must be in [0, 1]")
    # generate_tiles validates the ROI and tiling numbers before any model load.
    tiles = generate_tiles(
        image_size=image_size,
        roi_xyxy=roi_integer,
        tile_size=tile_size,
        overlap_fraction=overlap_fraction,
    )
    prompts_raw = root["prompts"]
    if not isinstance(prompts_raw, list) or not prompts_raw:
        raise ManifestError("prompts must be a non-empty list")
    prompts = []
    reference_path = None
    reference_sha256 = None
    class_ids = set()
    for index, value in enumerate(prompts_raw):
        prompt = _strict_object(
            value,
            f"prompts[{index}]",
            required={"class_id", "reference_image", "reference_sha256", "bbox_xyxy"},
        )
        class_id = prompt["class_id"]
        if not isinstance(class_id, str) or not class_id.strip():
            raise ManifestError("prompt class_id must be a non-empty string")
        class_id = class_id.strip()
        if class_id in class_ids:
            raise ManifestError(f"duplicate prompt class_id {class_id!r}")
        class_ids.add(class_id)
        prompt_reference = _resolve_file(
            base, prompt["reference_image"], f"prompts[{index}].reference_image"
        )
        prompt_sha = _verify_sha256(
            prompt_reference,
            prompt["reference_sha256"],
            f"prompts[{index}].reference_image",
        )
        if reference_path is None:
            reference_path = prompt_reference
            reference_sha256 = prompt_sha
        elif prompt_reference != reference_path or prompt_sha != reference_sha256:
            raise ManifestError(
                "this bounded harness requires all visual prompts in one reference image"
            )
        prompt_box = _box(prompt["bbox_xyxy"], f"prompts[{index}].bbox_xyxy")
        if not (
            0 <= prompt_box[0] < prompt_box[2] <= image_size[0]
            and 0 <= prompt_box[1] < prompt_box[3] <= image_size[1]
        ):
            raise ManifestError("prompt bbox lies outside expected_image_size")
        prompts.append({"class_id": class_id, "bbox_xyxy": prompt_box})
    reference_image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if reference_image is None:
        raise ManifestError("unable to decode visual-prompt reference image")
    if (reference_image.shape[1], reference_image.shape[0]) != image_size:
        raise ManifestError("visual-prompt image does not match expected_image_size")
    inputs_raw = root["inputs"]
    if not isinstance(inputs_raw, list) or not inputs_raw:
        raise ManifestError("inputs must be a non-empty list")
    inputs = []
    frame_ids: set[str] = set()
    for index, value in enumerate(inputs_raw):
        item = _strict_object(
            value,
            f"inputs[{index}]",
            required={"image", "sha256", "split", "frame_id"},
            optional={"annotation", "annotation_sha256"},
        )
        image_path = _resolve_file(base, item["image"], f"inputs[{index}].image")
        image_sha256 = _verify_sha256(
            image_path, item["sha256"], f"inputs[{index}].image"
        )
        if item["split"] not in {"pilot", "validation", "test"}:
            raise ManifestError("input split must be pilot, validation, or test")
        if not isinstance(item["frame_id"], str) or not item["frame_id"].strip():
            raise ManifestError("input frame_id must be a non-empty string")
        frame_id = item["frame_id"].strip()
        if any(not (char.isalnum() or char in "-_.") for char in frame_id):
            raise ManifestError("input frame_id may contain only letters, numbers, -_.")
        if frame_id in frame_ids:
            raise ManifestError(f"duplicate input frame_id {frame_id!r}")
        frame_ids.add(frame_id)
        decoded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ManifestError(f"unable to decode inputs[{index}].image")
        if (decoded.shape[1], decoded.shape[0]) != image_size:
            raise ManifestError(
                f"inputs[{index}].image size {(decoded.shape[1], decoded.shape[0])} "
                f"does not match expected_image_size {image_size}"
            )
        annotation_path = None
        annotation_sha256 = None
        references = None
        if "annotation" in item or "annotation_sha256" in item:
            if set(item) & {"annotation", "annotation_sha256"} != {
                "annotation",
                "annotation_sha256",
            }:
                raise ManifestError("annotation and annotation_sha256 must appear together")
            annotation_path = _resolve_file(
                base, item["annotation"], f"inputs[{index}].annotation"
            )
            annotation_sha256 = _verify_sha256(
                annotation_path,
                item["annotation_sha256"],
                f"inputs[{index}].annotation",
            )
            references = _load_annotation(annotation_path, image_sha256)
        inputs.append(
            {
                "path": image_path,
                "sha256": image_sha256,
                "split": item["split"],
                "frame_id": frame_id,
                "annotation_path": annotation_path,
                "annotation_sha256": annotation_sha256,
                "references": references,
            }
        )
    inference = _strict_object(
        root["inference"],
        "inference",
        required={"imgsz", "confidence", "model_iou", "device"},
    )
    imgsz = int(inference["imgsz"])
    confidence = float(inference["confidence"])
    model_iou = float(inference["model_iou"])
    device = inference["device"]
    if (
        isinstance(inference["imgsz"], bool)
        or imgsz != inference["imgsz"]
        or imgsz <= 0
        or not np.isfinite(confidence)
        or not np.isfinite(model_iou)
        or not 0.0 <= confidence <= 1.0
        or not 0.0 <= model_iou <= 1.0
    ):
        raise ManifestError("inference numeric options are invalid")
    if not isinstance(device, str) or not device.strip():
        raise ManifestError("inference.device must be a non-empty string")
    return {
        "manifest_path": path,
        "manifest_sha256": _sha256(path),
        "experiment_id": root["experiment_id"].strip(),
        "license_context": root["license_context"],
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "model_id": model["model_id"].strip(),
        "ultralytics_version": model["ultralytics_version"],
        "image_size": image_size,
        "roi_xyxy": roi_integer,
        "tile_size": tile_size,
        "overlap_fraction": overlap_fraction,
        "merge_iou_threshold": merge_iou_threshold,
        "tiles": tiles,
        "reference_path": reference_path,
        "reference_sha256": reference_sha256,
        "prompts": tuple(prompts),
        "inputs": tuple(inputs),
        "inference": {
            "imgsz": imgsz,
            "confidence": confidence,
            "model_iou": model_iou,
            "device": device.strip(),
        },
    }


def _draw_predictions(image: np.ndarray, predictions: tuple[ImagePrediction, ...]) -> np.ndarray:
    overlay = image.copy()
    colours = {
        "cube": (0, 255, 255),
        "screw_iso10642_m6x30": (0, 255, 0),
    }
    for prediction in predictions:
        colour = colours.get(prediction.class_id, (255, 255, 255))
        polygon = np.rint(np.asarray(prediction.polygon_xy)).astype(np.int32)
        cv2.polylines(overlay, (polygon,), True, colour, 2, cv2.LINE_AA)
        x0, y0, x1, y1 = np.rint(prediction.bbox_xyxy).astype(int)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), colour, 2)
        cv2.putText(
            overlay,
            f"{prediction.class_id} {prediction.confidence:.2f}",
            (x0, max(18, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )
    return overlay


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_manifest(args.manifest)
    installed_version = metadata.version("ultralytics")
    if installed_version != config["ultralytics_version"]:
        raise RuntimeError(
            f"installed ultralytics {installed_version}, expected "
            f"{config['ultralytics_version']}"
        )
    repository = Path(__file__).resolve().parents[2]
    try:
        config["checkpoint"].relative_to(repository)
    except ValueError:
        pass
    else:
        raise RuntimeError("model checkpoints must remain outside the repository")
    output_dir = args.output_dir.expanduser().resolve()
    try:
        output_dir.relative_to(repository)
    except ValueError:
        pass
    else:
        raise RuntimeError("generated experiment outputs must remain outside the repository")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Heavy import occurs only after every path/hash/shape/license/output gate passes.
    from ultralytics import YOLOE
    import torch

    model_load_start = time.perf_counter()
    model = YOLOE(str(config["checkpoint"]))
    model_load_ms = (time.perf_counter() - model_load_start) * 1000.0
    prompt_boxes = np.asarray(
        [item["bbox_xyxy"] for item in config["prompts"]], dtype=np.float32
    )
    prompt_classes = np.arange(len(config["prompts"]), dtype=np.int32)
    visual_prompts = {"bboxes": prompt_boxes, "cls": prompt_classes}
    class_ids = tuple(item["class_id"] for item in config["prompts"])
    records = []
    total_tile_latencies_ms: list[float] = []
    for item in config["inputs"]:
        image = cv2.imread(str(item["path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"unable to decode input {item['frame_id']}")
        if (image.shape[1], image.shape[0]) != config["image_size"]:
            raise RuntimeError(
                f"input {item['frame_id']} size {(image.shape[1], image.shape[0])} "
                f"does not match {config['image_size']}"
            )
        raw_predictions = []
        tile_latencies_ms = []
        for tile in config["tiles"]:
            crop = image[tile.y0 : tile.y1, tile.x0 : tile.x1]
            started = time.perf_counter()
            results = model.predict(
                crop,
                refer_image=str(config["reference_path"]),
                visual_prompts=visual_prompts,
                imgsz=config["inference"]["imgsz"],
                conf=config["inference"]["confidence"],
                iou=config["inference"]["model_iou"],
                device=config["inference"]["device"],
                verbose=False,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            tile_latencies_ms.append(elapsed_ms)
            total_tile_latencies_ms.append(elapsed_ms)
            if len(results) != 1:
                raise RuntimeError("YOLOE returned an unexpected result count")
            raw_predictions.extend(
                extract_tile_predictions(
                    results[0], tile=tile, class_ids=class_ids
                )
            )
        predictions = merge_tile_predictions(
            raw_predictions, iou_threshold=config["merge_iou_threshold"]
        )
        metrics = None
        if item["references"] is not None:
            metrics = evaluate_boxes(
                predictions, item["references"], iou_threshold=0.5
            )
        overlay = _draw_predictions(image, predictions)
        overlay_name = f"{item['frame_id']}-overlay.png"
        if not cv2.imwrite(str(output_dir / overlay_name), overlay):
            raise RuntimeError(f"unable to write {overlay_name}")
        records.append(
            {
                "frame_id": item["frame_id"],
                "split": item["split"],
                "image_name": item["path"].name,
                "image_sha256": item["sha256"],
                "annotation_sha256": item["annotation_sha256"],
                "tile_latencies_ms": tile_latencies_ms,
                "raw_tile_prediction_count": len(raw_predictions),
                "merged_prediction_count": len(predictions),
                "predictions": [value.to_dict() for value in predictions],
                "metrics": metrics,
                "overlay": overlay_name,
            }
        )
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "scope": "offline_image_space_segmentation_experiment_only",
        "not_provided": [
            "robot_base_position",
            "object_orientation",
            "pose_quality",
            "hardware_motion_readiness",
        ],
        "license_context": config["license_context"],
        "provenance": {
            "manifest_sha256": config["manifest_sha256"],
            "project_commit": _git_value(repository, "rev-parse", "HEAD"),
            "branch": _git_value(repository, "branch", "--show-current"),
            "worktree_dirty": _git_value(
                repository, "status", "--porcelain=v1", "--untracked-files=normal"
            )
            is not None,
            "python": sys.version,
            "platform": platform.platform(),
            "ultralytics_version": installed_version,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "model_id": config["model_id"],
            "checkpoint_name": config["checkpoint"].name,
            "checkpoint_sha256": config["checkpoint_sha256"],
            "reference_image_name": config["reference_path"].name,
            "reference_image_sha256": config["reference_sha256"],
            "dependency_lock_sha256": _sha256(
                Path(__file__).with_name("yoloe_env") / "uv.lock"
            ),
            "experiment_core_sha256": _sha256(
                Path(__file__).with_name("yoloe_experiment.py")
            ),
            "experiment_runner_sha256": _sha256(Path(__file__)),
        },
        "preprocessing": {
            "expected_image_size": config["image_size"],
            "roi_xyxy": config["roi_xyxy"],
            "tile_size": config["tile_size"],
            "overlap_fraction": config["overlap_fraction"],
            "tile_count": len(config["tiles"]),
            "tiles": [tile.xyxy for tile in config["tiles"]],
            "merge_iou_threshold": config["merge_iou_threshold"],
            "no_pre_inference_resize": True,
        },
        "prompt_classes": list(class_ids),
        "inference": config["inference"],
        "model_load_ms": model_load_ms,
        "tile_latency_ms": total_tile_latencies_ms,
        "records": records,
        "evidence_limit": (
            "Image-space boxes/masks do not establish robot-base millimetre "
            "accuracy, symmetry-aware orientation, or grasp success."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "predictions.jsonl").write_text(
        "".join(
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "frames": len(records),
        "predictions": sum(value["merged_prediction_count"] for value in records),
        "model_load_ms": model_load_ms,
        "tile_latency_ms_mean": (
            sum(total_tile_latencies_ms) / len(total_tile_latencies_ms)
            if total_tile_latencies_ms
            else None
        ),
        "output": str(output_dir / "report.json"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
