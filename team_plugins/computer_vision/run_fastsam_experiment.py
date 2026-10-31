"""Run a hash-pinned, offline FastSAM class-agnostic mask experiment.

This command never constructs a hardware ``VisionProvider`` and never emits
robot coordinates.  It tests whether FastSAM supplies useful native-image mask
proposals for a later calibrated geometry/classification stage.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .fastsam_experiment import (
    evaluate_proposal_coverage,
    extract_proposals,
    generate_tiles,
    merge_proposals,
)


EXPECTED_ULTRALYTICS_VERSION = "8.4.194"
ALLOWED_LICENSE_CONTEXTS = frozenset(
    {"evaluation-only-agpl-review", "enterprise-authorized"}
)
REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
RUNTIME_CODE_PATHS = (
    Path(__file__).resolve(),
    Path(__file__).with_name("fastsam_experiment.py").resolve(),
    Path(__file__).with_name("fastsam_env").joinpath("uv.lock").resolve(),
)


class ManifestError(ValueError):
    """Raised when an experiment manifest is ambiguous or not reproducible."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise ManifestError(f"{name} missing keys: {missing}")
    if unknown:
        raise ManifestError(f"{name} unknown keys: {unknown}")
    return dict(value)


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ManifestError(f"non-standard JSON number {value!r}")

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ManifestError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read strict JSON {path}: {exc}") from exc


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ManifestError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ManifestError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ManifestError(f"{name} must be an integer >= {minimum}")
    return value


def _numbers(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ManifestError(f"{name} must contain {length} numbers")
    return tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))


def _integers(value: Any, length: int, name: str, *, minimum: int = 0) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ManifestError(f"{name} must contain {length} integers")
    return tuple(
        _integer(item, f"{name}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    )


def _safe_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
        raise ManifestError(
            f"{name} must be 1-128 filename-safe ASCII letters, digits, '.', '_', or '-'"
        )
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{name} must be a SHA-256 string")
    result = value.lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ManifestError(f"{name} must be 64 hexadecimal characters")
    return result


def _resolve_file(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{name} must be a non-empty path")
    candidate = Path(value).expanduser()
    try:
        path = (candidate if candidate.is_absolute() else base / candidate).resolve(strict=True)
    except OSError as exc:
        raise ManifestError(f"cannot resolve {name}: {exc}") from exc
    if not path.is_file():
        raise ManifestError(f"{name} is not a file")
    return path


def _verify_file(base: Path, raw: Mapping[str, Any], name: str) -> tuple[Path, str]:
    path = _resolve_file(base, raw.get("path"), f"{name}.path")
    expected = _sha(raw.get("sha256"), f"{name}.sha256")
    actual = _sha256(path)
    if actual != expected:
        raise ManifestError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
    return path, actual


def _load_annotation(
    path: Path,
    expected_image_hash: str,
    expected_image_size: tuple[int, int],
) -> tuple[tuple[dict[str, Any], ...], str]:
    document = _strict_object(
        _strict_json(path),
        "annotation",
        required={"schema_version", "image_sha256", "instances"},
    )
    if document["schema_version"] != 1 or isinstance(document["schema_version"], bool):
        raise ManifestError("annotation.schema_version must be 1")
    if _sha(document["image_sha256"], "annotation.image_sha256") != expected_image_hash:
        raise ManifestError("annotation image hash does not match the input")
    if not isinstance(document["instances"], list):
        raise ManifestError("annotation.instances must be a list")
    instances = []
    ids = set()
    for index, value in enumerate(document["instances"]):
        item = _strict_object(
            value,
            f"annotation.instances[{index}]",
            required={"instance_id", "class_id", "bbox_xyxy"},
        )
        instance_id = _safe_id(
            item["instance_id"], f"annotation.instances[{index}].instance_id"
        )
        class_id = _safe_id(
            item["class_id"], f"annotation.instances[{index}].class_id"
        )
        if instance_id in ids:
            raise ManifestError("annotation instance IDs must be unique")
        box = _numbers(item["bbox_xyxy"], 4, "annotation.bbox_xyxy")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ManifestError("annotation bbox must be non-empty")
        width, height = expected_image_size
        if not (0.0 <= box[0] < box[2] <= width and 0.0 <= box[1] < box[3] <= height):
            raise ManifestError("annotation bbox must lie inside the expected image")
        ids.add(instance_id)
        instances.append(
            {"instance_id": instance_id, "class_id": class_id, "bbox_xyxy": box}
        )
    return tuple(instances), _sha256(path)


def load_manifest(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    root = _strict_object(
        _strict_json(path),
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
            "minimum_reference_bbox_iou",
            "inference",
            "inputs",
        },
    )
    if root["schema_version"] != 1 or isinstance(root["schema_version"], bool):
        raise ManifestError("schema_version must be 1")
    experiment_id = _safe_id(root["experiment_id"], "experiment_id")
    if root["license_context"] not in ALLOWED_LICENSE_CONTEXTS:
        raise ManifestError(f"license_context must be one of {sorted(ALLOWED_LICENSE_CONTEXTS)}")
    model = _strict_object(
        root["model"],
        "model",
        required={"path", "sha256", "ultralytics_version"},
    )
    model_path, model_hash = _verify_file(path.parent, model, "model")
    if model_path.is_relative_to(REPO_ROOT):
        raise ManifestError("model weights must remain outside the repository")
    if model["ultralytics_version"] != EXPECTED_ULTRALYTICS_VERSION:
        raise ManifestError(
            f"ultralytics_version must be {EXPECTED_ULTRALYTICS_VERSION}"
        )
    expected_size = _integers(root["expected_image_size"], 2, "expected_image_size", minimum=1)
    roi = _integers(root["roi_xyxy"], 4, "roi_xyxy", minimum=0)
    tile_size = _integers(root["tile_size"], 2, "tile_size", minimum=1)
    overlap = _finite(root["overlap_fraction"], "overlap_fraction")
    merge_iou = _finite(root["merge_iou_threshold"], "merge_iou_threshold")
    minimum_reference_iou = _finite(
        root["minimum_reference_bbox_iou"], "minimum_reference_bbox_iou"
    )
    generate_tiles(
        image_size=expected_size,
        roi_xyxy=roi,
        tile_size=tile_size,
        overlap_fraction=overlap,
    )
    if not 0.0 <= merge_iou <= 1.0:
        raise ManifestError("merge_iou_threshold must be in [0, 1]")
    if not 0.0 < minimum_reference_iou <= 1.0:
        raise ManifestError("minimum_reference_bbox_iou must be in (0, 1]")
    inference = _strict_object(
        root["inference"],
        "inference",
        required={"device", "imgsz", "confidence", "iou", "minimum_mask_area_px"},
    )
    if not isinstance(inference["device"], str) or not inference["device"].strip():
        raise ManifestError("inference.device must be non-empty")
    imgsz = _integer(inference["imgsz"], "inference.imgsz", minimum=32)
    confidence = _finite(inference["confidence"], "inference.confidence")
    iou = _finite(inference["iou"], "inference.iou")
    minimum_area = _integer(
        inference["minimum_mask_area_px"],
        "inference.minimum_mask_area_px",
        minimum=1,
    )
    if not 0.0 <= confidence <= 1.0 or not 0.0 <= iou <= 1.0:
        raise ManifestError("inference confidence/iou must be in [0, 1]")
    if not isinstance(root["inputs"], list) or not root["inputs"]:
        raise ManifestError("inputs must be a non-empty list")
    inputs = []
    input_ids = set()
    for index, value in enumerate(root["inputs"]):
        item = _strict_object(
            value,
            f"inputs[{index}]",
            required={"input_id", "image", "annotation_path"},
        )
        input_id = _safe_id(item["input_id"], f"inputs[{index}].input_id")
        if input_id in input_ids:
            raise ManifestError("input_id values must be unique")
        image_object = _strict_object(
            item["image"], f"inputs[{index}].image", required={"path", "sha256"}
        )
        image_path, image_hash = _verify_file(path.parent, image_object, f"inputs[{index}].image")
        annotation = None
        annotation_path = None
        annotation_hash = None
        if item["annotation_path"] is not None:
            annotation_path = _resolve_file(
                path.parent, item["annotation_path"], f"inputs[{index}].annotation_path"
            )
            annotation, annotation_hash = _load_annotation(
                annotation_path, image_hash, expected_size
            )
        inputs.append(
            {
                "input_id": input_id,
                "image_path": image_path,
                "image_sha256": image_hash,
                "annotation": annotation,
                "annotation_path": annotation_path,
                "annotation_sha256": annotation_hash,
            }
        )
        input_ids.add(input_id)
    return {
        "manifest_path": path,
        "manifest_sha256": _sha256(path),
        "experiment_id": experiment_id,
        "license_context": root["license_context"],
        "model_path": model_path,
        "model_sha256": model_hash,
        "expected_image_size": expected_size,
        "roi_xyxy": roi,
        "tile_size": tile_size,
        "overlap_fraction": overlap,
        "merge_iou_threshold": merge_iou,
        "minimum_reference_bbox_iou": minimum_reference_iou,
        "inference": {
            "device": inference["device"],
            "imgsz": imgsz,
            "confidence": confidence,
            "iou": iou,
            "minimum_mask_area_px": minimum_area,
        },
        "inputs": tuple(inputs),
    }


def _git_value(*arguments: str) -> str | None:
    result = subprocess.run(
        ("git", *arguments),
        cwd=REPO_ROOT,
        env=_git_environment(),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _worktree_dirty() -> bool:
    result = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=normal"),
        cwd=REPO_ROOT,
        env=_git_environment(),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git status failed")
    return bool(result.stdout.strip())


def _git_environment() -> dict[str, str]:
    """Return Git overrides for cross-OS worktree metadata when required.

    A worktree created with Windows Git stores a Windows absolute `gitdir` in
    its `.git` pointer.  Linux Git under WSL cannot resolve that spelling even
    though the directory is mounted under `/mnt/<drive>`.  Explicit, verified
    GIT_DIR/GIT_WORK_TREE values preserve provenance checks in either runtime.
    """

    environment = os.environ.copy()
    pointer = REPO_ROOT / ".git"
    if pointer.is_dir():
        return environment
    try:
        text = pointer.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"cannot read worktree Git pointer: {exc}") from exc
    prefix = "gitdir: "
    if not text.startswith(prefix):
        raise RuntimeError("worktree .git file does not contain a gitdir pointer")
    raw = text[len(prefix) :]
    windows_absolute = re.fullmatch(r"([A-Za-z]):[\\/](.+)", raw)
    if os.name != "nt" and windows_absolute is not None:
        drive, remainder = windows_absolute.groups()
        git_directory = Path("/mnt") / drive.lower() / Path(
            remainder.replace("\\", "/")
        )
    else:
        candidate = Path(raw)
        git_directory = candidate if candidate.is_absolute() else pointer.parent / candidate
    git_directory = git_directory.resolve(strict=False)
    if not git_directory.is_dir():
        raise RuntimeError(f"worktree gitdir does not exist: {git_directory}")
    environment["GIT_DIR"] = str(git_directory)
    environment["GIT_WORK_TREE"] = str(REPO_ROOT)
    return environment


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _runtime_code_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)).replace("\\", "/"): _sha256(path)
        for path in RUNTIME_CODE_PATHS
    }


def _runtime_inputs_still_match(manifest: Mapping[str, Any]) -> bool:
    try:
        if _sha256(manifest["manifest_path"]) != manifest["manifest_sha256"]:
            return False
        if _sha256(manifest["model_path"]) != manifest["model_sha256"]:
            return False
        for item in manifest["inputs"]:
            if _sha256(item["image_path"]) != item["image_sha256"]:
                return False
            if item["annotation_path"] is not None and _sha256(
                item["annotation_path"]
            ) != item["annotation_sha256"]:
                return False
    except OSError:
        return False
    return True


def _draw_overlay(image: np.ndarray, proposals) -> np.ndarray:
    overlay = image.copy()
    for proposal in proposals:
        polygon = np.rint(np.asarray(proposal.polygon_xy)).astype(np.int32)
        cv2.polylines(overlay, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)
        x0, y0, _, _ = proposal.bbox_xyxy
        cv2.putText(
            overlay,
            f"{proposal.proposal_id} {proposal.confidence:.3f}",
            (int(x0), max(12, int(y0) - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return overlay


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = load_manifest(args.manifest)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.is_relative_to(REPO_ROOT):
        raise ManifestError("output directory must remain outside the repository")
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise ManifestError("output directory must be new or empty")
    else:
        output_dir.mkdir(parents=True)
    dirty_at_start = _worktree_dirty()
    if dirty_at_start and not args.allow_dirty:
        raise ManifestError("clean committed worktree required; use --allow-dirty only for development")
    commit_at_start = _git_value("rev-parse", "HEAD")
    branch_at_start = _git_value("branch", "--show-current")
    code_hashes_at_start = _runtime_code_hashes()
    installed_version = metadata.version("ultralytics")
    if installed_version != EXPECTED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"expected ultralytics {EXPECTED_ULTRALYTICS_VERSION}, got {installed_version}"
        )
    from ultralytics import FastSAM  # Heavy dependency remains lazy.

    if not _runtime_inputs_still_match(manifest):
        raise ManifestError("manifest, model, image, or annotation changed before inference")
    model = FastSAM(str(manifest["model_path"]))
    tiles = generate_tiles(
        image_size=manifest["expected_image_size"],
        roi_xyxy=manifest["roi_xyxy"],
        tile_size=manifest["tile_size"],
        overlap_fraction=manifest["overlap_fraction"],
    )
    reports = []
    for item in manifest["inputs"]:
        bgr = cv2.imread(str(item["image_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ManifestError(f"cannot decode {item['image_path']}")
        actual_size = (int(bgr.shape[1]), int(bgr.shape[0]))
        if actual_size != manifest["expected_image_size"]:
            raise ManifestError(
                f"input {item['input_id']} size {actual_size} != expected {manifest['expected_image_size']}"
            )
        proposals = []
        tile_latencies_ms = []
        for tile in tiles:
            crop = bgr[tile.y0 : tile.y1, tile.x0 : tile.x1]
            started = time.perf_counter()
            result = model(
                crop,
                device=manifest["inference"]["device"],
                retina_masks=True,
                imgsz=manifest["inference"]["imgsz"],
                conf=manifest["inference"]["confidence"],
                iou=manifest["inference"]["iou"],
                verbose=False,
            )[0]
            tile_latencies_ms.append((time.perf_counter() - started) * 1000.0)
            if result.masks is None or result.boxes is None:
                continue
            proposals.extend(
                extract_proposals(
                    masks=result.masks.data.cpu().numpy(),
                    confidences=result.boxes.conf.cpu().numpy(),
                    tile=tile,
                    minimum_mask_area_px=manifest["inference"]["minimum_mask_area_px"],
                )
            )
        merged = merge_proposals(
            proposals, iou_threshold=manifest["merge_iou_threshold"]
        )
        predictions_path = output_dir / f"{item['input_id']}.predictions.json"
        predictions = {
            "schema_version": 1,
            "input_id": item["input_id"],
            "image_sha256": item["image_sha256"],
            "proposals": [proposal.to_dict() for proposal in merged],
        }
        _write_json(predictions_path, predictions)
        overlay_path = output_dir / f"{item['input_id']}.overlay.png"
        if not cv2.imwrite(str(overlay_path), _draw_overlay(bgr, merged)):
            raise RuntimeError(f"unable to write {overlay_path}")
        coverage = (
            None
            if item["annotation"] is None
            else evaluate_proposal_coverage(
                item["annotation"],
                merged,
                minimum_bbox_iou=manifest["minimum_reference_bbox_iou"],
            )
        )
        reports.append(
            {
                "input_id": item["input_id"],
                "image_sha256": item["image_sha256"],
                "annotation_sha256": item["annotation_sha256"],
                "tile_count": len(tiles),
                "raw_proposals": len(proposals),
                "merged_proposals": len(merged),
                "tile_latency_ms": tile_latencies_ms,
                "coverage": coverage,
                "predictions_file": predictions_path.name,
                "predictions_sha256": _sha256(predictions_path),
                "overlay_file": overlay_path.name,
                "overlay_sha256": _sha256(overlay_path),
            }
        )
    commit_at_end = _git_value("rev-parse", "HEAD")
    branch_at_end = _git_value("branch", "--show-current")
    code_hashes_at_end = _runtime_code_hashes()
    dirty_at_end = _worktree_dirty()
    input_files_stable = _runtime_inputs_still_match(manifest)
    provenance_stable = (
        commit_at_start == commit_at_end
        and branch_at_start == branch_at_end
        and code_hashes_at_start == code_hashes_at_end
        and dirty_at_start == dirty_at_end
        and input_files_stable
    )
    report = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "evidence_scope": (
            "Offline class-agnostic image-mask proposals only; no class identity, "
            "robot-base pose, physical accuracy, or grasp success is established."
        ),
        "license_context": manifest["license_context"],
        "manifest_sha256": manifest["manifest_sha256"],
        "project_commit": commit_at_start,
        "branch": branch_at_start,
        "runtime_code_sha256": code_hashes_at_start,
        "model": {
            "name": "FastSAM-s",
            "sha256": manifest["model_sha256"],
            "ultralytics_version": installed_version,
            "environment_versions": {
                package: metadata.version(package)
                for package in ("torch", "torchvision", "numpy", "opencv-python")
            },
        },
        "settings": {
            "expected_image_size": manifest["expected_image_size"],
            "roi_xyxy": manifest["roi_xyxy"],
            "tile_size": manifest["tile_size"],
            "overlap_fraction": manifest["overlap_fraction"],
            "merge_iou_threshold": manifest["merge_iou_threshold"],
            "minimum_reference_bbox_iou": manifest["minimum_reference_bbox_iou"],
            "inference": manifest["inference"],
        },
        "inputs": reports,
        "worktree_dirty_at_start": dirty_at_start,
        "worktree_dirty_at_end": dirty_at_end,
        "input_files_stable_during_run": input_files_stable,
        "provenance_stable_during_run": provenance_stable,
    }
    _write_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if provenance_stable else 2


if __name__ == "__main__":
    raise SystemExit(main())
