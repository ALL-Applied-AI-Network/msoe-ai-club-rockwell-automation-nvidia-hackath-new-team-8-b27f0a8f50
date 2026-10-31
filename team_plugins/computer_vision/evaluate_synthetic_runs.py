"""Score the selected vision plugin on realistic Isaac sensor frames.

This tool is deliberately offline and vision-only.  It gives the provider one
native table-camera ``sensor_frames/f00000.png`` image using the fixed physical
table calibration.  Only after inference returns does it open the matching
``frame_labels/f00000.json`` evaluator-only truth.  Simulator poses therefore
cannot repair or otherwise influence a detection.

Only the first captured frame is scored.  Later frames currently have no
measured visibility/occlusion label and must not be counted as misses merely
because a robot or held object may cover them.  The realistic cube scene has
no screws, so screw outputs are false-positive evidence only; screw recall is
not measurable with this dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from hardware.fanuc_cell.calibration.geometry import (
    load as load_calibration,
    pixel_to_base,
)
from sorting_core import load_config, scene_observation_to_dict
from sorting_core.integration_plugins import (
    VisionFactoryContext,
    load_factory,
    load_integration_profile,
)
from sorting_core.perception import (
    CameraFrame,
    PerceptionRequest,
    SceneObservation,
    validate_scene_for_request,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = Path(__file__).with_name("profile.hybrid-geometry.v1.json")
DEFAULT_CALIBRATION = (
    REPO_ROOT / "hardware" / "fanuc_cell" / "calibration" / "cameras.json"
)
DEFAULT_FRAME_STEM = "f00000"
EXPECTED_IMAGE_SIZE = (1920, 1080)


class SyntheticEvaluationError(ValueError):
    """Raised when synthetic evidence cannot support an attributable score."""


@dataclass(frozen=True)
class BatchEntry:
    seed: int | None
    run_dir: Path | None
    status: str
    returncode: int | None = None
    error: str | None = None


def _reject_constant(value: str) -> None:
    raise SyntheticEvaluationError(f"invalid JSON number {value!r}")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SyntheticEvaluationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SyntheticEvaluationError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SyntheticEvaluationError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str | None:
    result = subprocess.run(
        ("git", *arguments),
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _git_dirty() -> bool:
    result = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=normal"),
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise SyntheticEvaluationError(
            f"cannot record Git worktree state: {result.stderr.strip()}"
        )
    return bool(result.stdout.strip())


def _integer(value: Any, name: str, *, nullable: bool = False) -> int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SyntheticEvaluationError(f"{name} must be an integer")
    return value


def load_batch_status(path: Path) -> tuple[BatchEntry, ...]:
    """Load a batch status manifest, including failed/missing Isaac runs.

    Format::

        {"schema_version": 1, "runs": [
          {"seed": 101, "status": "success", "run_dir": "..."},
          {"seed": 211, "status": "failed", "returncode": 1,
           "error": "Isaac exited before capture"}
        ]}
    """

    source = path.expanduser().resolve(strict=True)
    raw = _load_json(source)
    if raw.get("schema_version") != 1:
        raise SyntheticEvaluationError("batch status schema_version must be 1")
    rows = raw.get("runs")
    if not isinstance(rows, list) or not rows:
        raise SyntheticEvaluationError("batch status runs must be a non-empty array")
    entries: list[BatchEntry] = []
    seeds: set[int] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SyntheticEvaluationError(f"runs[{index}] must be an object")
        seed = _integer(row.get("seed"), f"runs[{index}].seed")
        assert seed is not None
        if seed in seeds:
            raise SyntheticEvaluationError(f"duplicate expected seed {seed}")
        seeds.add(seed)
        status = row.get("status")
        if status not in {"success", "failed"}:
            raise SyntheticEvaluationError(
                f"runs[{index}].status must be 'success' or 'failed'"
            )
        run_value = row.get("run_dir")
        if run_value is None:
            run_dir = None
        elif isinstance(run_value, str) and run_value.strip() == run_value and run_value:
            run_dir = Path(run_value).expanduser()
            if not run_dir.is_absolute():
                run_dir = source.parent / run_dir
            run_dir = run_dir.resolve()
        else:
            raise SyntheticEvaluationError(
                f"runs[{index}].run_dir must be a non-empty path string or null"
            )
        if status == "success" and run_dir is None:
            raise SyntheticEvaluationError(
                f"runs[{index}] success requires run_dir"
            )
        returncode = _integer(
            row.get("returncode"), f"runs[{index}].returncode", nullable=True
        )
        error = row.get("error")
        if error is not None and (not isinstance(error, str) or not error.strip()):
            raise SyntheticEvaluationError(
                f"runs[{index}].error must be a non-empty string or null"
            )
        entries.append(BatchEntry(seed, run_dir, status, returncode, error))
    return tuple(entries)


def _seed_from_manifest(run_dir: Path) -> int | None:
    path = run_dir / "randomization_manifest.json"
    if not path.is_file():
        return None
    raw = _load_json(path)
    seed = raw.get("episode_seed")
    if seed is None:
        seed = raw.get("seed")
    return _integer(seed, f"{path}.episode_seed", nullable=True)


def collect_batch_entries(
    run_dirs: Sequence[Path], batch_status: Path | None
) -> tuple[BatchEntry, ...]:
    entries = list(load_batch_status(batch_status)) if batch_status else []
    known_paths = {
        entry.run_dir.resolve()
        for entry in entries
        if entry.run_dir is not None
    }
    for raw_path in run_dirs:
        run_dir = raw_path.expanduser().resolve()
        if run_dir in known_paths:
            continue
        entries.append(
            BatchEntry(
                seed=_seed_from_manifest(run_dir) if run_dir.is_dir() else None,
                run_dir=run_dir,
                status="success",
            )
        )
        known_paths.add(run_dir)
    if not entries:
        raise SyntheticEvaluationError(
            "provide at least one run directory or --batch-status manifest"
        )
    return tuple(entries)


def _build_provider(profile_path: Path, calibration_path: Path):
    profile = load_integration_profile(profile_path)
    if profile.vision is None:
        raise SyntheticEvaluationError("profile does not select a vision provider")
    if profile.vision.camera_ids != ("table",) or profile.vision.depth_camera_ids:
        raise SyntheticEvaluationError(
            "synthetic evaluator requires the RGB-only table-camera profile"
        )
    cameras = load_calibration(calibration_path)
    table = cameras["table"]
    calibration_id = profile.vision.calibration_ids["table"]
    factory = load_factory(profile.vision.factory, "vision")
    provider = factory(
        VisionFactoryContext(
            config=load_config(),
            camera_projectors={
                "table": lambda uv, z: pixel_to_base(table, uv, z=z)
            },
            calibration_ids={"table": calibration_id},
            expected_cube_edge_m=0.010,
            checkpoint=None,
            detector=None,
            options=profile.vision.options,
            camera_ids=profile.vision.camera_ids,
            depth_camera_ids=profile.vision.depth_camera_ids,
            declared_calibration_ids=profile.vision.calibration_ids,
            declared_depth_calibration_ids=(
                profile.vision.depth_calibration_ids
            ),
            profile_id=profile.profile_id,
            profile_path=profile.source_path,
            profile_sha256=profile.source_sha256,
        )
    )
    return provider, calibration_id, profile


def _yaw_xyzw(quaternion: Sequence[float]) -> float:
    x, y, z, w = (float(value) for value in quaternion)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _yaw_wxyz(quaternion: Sequence[float]) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return _yaw_xyzw((x, y, z, w))


def _square_yaw_error_deg(first: float, second: float) -> float:
    delta_deg = math.degrees(first - second)
    return abs((delta_deg + 45.0) % 90.0 - 45.0)


def _truth_cubes(label: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    evaluator = label.get("evaluator_only")
    if not isinstance(evaluator, dict):
        raise SyntheticEvaluationError("frame label lacks evaluator_only object")
    objects = evaluator.get("objects")
    if not isinstance(objects, list):
        raise SyntheticEvaluationError("evaluator_only.objects must be an array")
    cubes: list[Mapping[str, Any]] = []
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise SyntheticEvaluationError(
                f"evaluator_only.objects[{index}] must be an object"
            )
        if item.get("class_name") != "cube":
            continue
        if item.get("derived_amodal_bbox_status") != "valid":
            raise SyntheticEvaluationError(
                "first-frame cube truth must have a valid derived amodal box"
            )
        colour = item.get("colour")
        position = item.get("position_base_m")
        quaternion = item.get("quaternion_base_wxyz")
        if colour not in {"red", "yellow", "blue"}:
            raise SyntheticEvaluationError(f"unsupported cube truth colour {colour!r}")
        if not isinstance(position, list) or len(position) != 3:
            raise SyntheticEvaluationError("cube truth position must contain 3 values")
        if not isinstance(quaternion, list) or len(quaternion) != 4:
            raise SyntheticEvaluationError("cube truth quaternion must contain 4 values")
        if not all(math.isfinite(float(value)) for value in (*position, *quaternion)):
            raise SyntheticEvaluationError("cube truth pose must be finite")
        cubes.append(item)
    if not cubes:
        raise SyntheticEvaluationError("first frame has no cube truth")
    return tuple(cubes)


def score_scene(
    scene: SceneObservation, truth_cubes: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    """Match cubes by logical colour and report detection plus pose metrics."""

    predicted_cubes = [item for item in scene.objects if item.class_id == "cube"]
    by_colour: dict[str, list[Any]] = {}
    for item in predicted_cubes:
        by_colour.setdefault(str(item.sort_label), []).append(item)
    truth_by_colour: dict[str, list[Mapping[str, Any]]] = {}
    for item in truth_cubes:
        truth_by_colour.setdefault(str(item["colour"]), []).append(item)

    matches: list[Mapping[str, Any]] = []
    false_positives = 0
    false_negatives = 0
    for colour in sorted(set(by_colour) | set(truth_by_colour)):
        predictions = list(by_colour.get(colour, ()))
        truths = list(truth_by_colour.get(colour, ()))
        while predictions and truths:
            candidates = []
            for prediction_index, prediction in enumerate(predictions):
                pred_xy = np.asarray(prediction.pose.position_base_m[:2], dtype=float)
                for truth_index, truth in enumerate(truths):
                    truth_xy = np.asarray(truth["position_base_m"][:2], dtype=float)
                    candidates.append(
                        (
                            float(np.linalg.norm(pred_xy - truth_xy)),
                            prediction_index,
                            truth_index,
                        )
                    )
            _, prediction_index, truth_index = min(candidates)
            prediction = predictions.pop(prediction_index)
            truth = truths.pop(truth_index)
            truth_xyz = np.asarray(truth["position_base_m"], dtype=float)
            predicted_xyz = np.asarray(prediction.pose.position_base_m, dtype=float)
            matches.append(
                {
                    "colour": colour,
                    "truth_track_id": truth.get("track_id"),
                    "predicted_track_id": prediction.track_id,
                    "confidence": prediction.confidence,
                    "xy_error_mm": float(
                        np.linalg.norm(predicted_xyz[:2] - truth_xyz[:2]) * 1000.0
                    ),
                    "position_3d_error_mm": float(
                        np.linalg.norm(predicted_xyz - truth_xyz) * 1000.0
                    ),
                    "square_yaw_error_deg": _square_yaw_error_deg(
                        _yaw_xyzw(prediction.pose.orientation_xyzw),
                        _yaw_wxyz(truth["quaternion_base_wxyz"]),
                    ),
                }
            )
        false_positives += len(predictions)
        false_negatives += len(truths)

    true_positives = len(matches)
    predicted_count = len(predicted_cubes)
    truth_count = len(truth_cubes)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else None
    )
    recall = true_positives / truth_count if truth_count else None
    output_classes = Counter(item.class_id for item in scene.objects)
    return {
        "truth_cube_count": truth_count,
        "predicted_cube_count": predicted_count,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "matches": matches,
        "non_cube_outputs": {
            key: value for key, value in sorted(output_classes.items()) if key != "cube"
        },
    }


def evaluate_run(
    entry: BatchEntry,
    *,
    profile_path: Path,
    calibration_path: Path,
    frame_stem: str = DEFAULT_FRAME_STEM,
    provider_builder: Callable[[Path, Path], Any] | None = None,
) -> Mapping[str, Any]:
    if entry.status != "success":
        return {
            "seed": entry.seed,
            "status": "generation_failed",
            "returncode": entry.returncode,
            "error": entry.error,
            "run_dir": None if entry.run_dir is None else str(entry.run_dir),
        }
    if entry.run_dir is None:
        raise SyntheticEvaluationError("successful batch entry has no run directory")
    run_dir = entry.run_dir.resolve()
    image_path = run_dir / "sensor_frames" / f"{frame_stem}.png"
    label_path = run_dir / "frame_labels" / f"{frame_stem}.json"
    if not run_dir.is_dir():
        raise SyntheticEvaluationError(f"run directory does not exist: {run_dir}")
    if not image_path.is_file():
        raise SyntheticEvaluationError(f"missing sensor frame: {image_path}")
    if not label_path.is_file():
        raise SyntheticEvaluationError(f"missing frame label: {label_path}")

    # No label/truth file is opened before provider inference completes.
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SyntheticEvaluationError(f"cannot decode sensor frame: {image_path}")
    if (bgr.shape[1], bgr.shape[0]) != EXPECTED_IMAGE_SIZE:
        raise SyntheticEvaluationError(
            f"sensor frame must be native {EXPECTED_IMAGE_SIZE}, got "
            f"{(bgr.shape[1], bgr.shape[0])}"
        )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    builder = _build_provider if provider_builder is None else provider_builder
    provider, calibration_id, profile = builder(profile_path, calibration_path)
    request = PerceptionRequest(
        scene_id=f"synthetic-eval:{entry.seed}:{run_dir.name}:{frame_stem}",
        capture_time_s=0.0,
        frames=(
            CameraFrame(
                camera_id="table",
                capture_time_s=0.0,
                calibration_id=calibration_id,
                rgb=rgb,
            ),
        ),
        metadata={
            "evaluator": "first-frame-realistic-sensor",
            "truth_available_to_provider": False,
        },
    )
    started = time.perf_counter()
    scene = validate_scene_for_request(request, provider.observe(request))
    latency_ms = (time.perf_counter() - started) * 1000.0

    # Inference is complete.  Evaluator-only simulator truth may now be read.
    label = _load_json(label_path)
    camera = label.get("camera")
    if not isinstance(camera, dict) or camera.get("derived_sensor_image_relpath") != (
        f"sensor_frames/{frame_stem}.png"
    ):
        raise SyntheticEvaluationError(
            "frame label does not identify the scored table sensor image"
        )
    truth = _truth_cubes(label)
    scored = score_scene(scene, truth)
    seed = entry.seed
    if seed is None:
        seed = _integer(label.get("randomization", {}).get("episode_seed"), "episode_seed")
    return {
        "seed": seed,
        "status": "evaluated",
        "run_dir": str(run_dir),
        "frame_stem": frame_stem,
        "sensor_frame_sha256": _sha256(image_path),
        "truth_label_sha256": _sha256(label_path),
        "truth_was_read_after_inference": True,
        "provider_name": scene.provider_name,
        "provider_version": scene.provider_version,
        "profile_id": profile.profile_id,
        "profile_sha256": profile.source_sha256,
        "calibration_id": calibration_id,
        "latency_ms": latency_ms,
        "scene": scene_observation_to_dict(scene),
        "vision": scored,
    }


def _distribution(values: Sequence[float]) -> Mapping[str, float] | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "maximum": ordered[-1],
    }


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    evaluated = [item for item in results if item.get("status") == "evaluated"]
    failed = [item for item in results if item.get("status") != "evaluated"]
    true_positives = sum(int(item["vision"]["true_positives"]) for item in evaluated)
    false_positives = sum(int(item["vision"]["false_positives"]) for item in evaluated)
    false_negatives = sum(int(item["vision"]["false_negatives"]) for item in evaluated)
    truth_count = sum(int(item["vision"]["truth_cube_count"]) for item in evaluated)
    matches = [match for item in evaluated for match in item["vision"]["matches"]]
    non_cube = Counter()
    for item in evaluated:
        non_cube.update(item["vision"]["non_cube_outputs"])
    return {
        "expected_runs": len(results),
        "evaluated_runs": len(evaluated),
        "failed_or_missing_runs": len(failed),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else None
        ),
        "recall": true_positives / truth_count if truth_count else None,
        "xy_error_mm": _distribution([item["xy_error_mm"] for item in matches]),
        "position_3d_error_mm": _distribution(
            [item["position_3d_error_mm"] for item in matches]
        ),
        "square_yaw_error_deg": _distribution(
            [item["square_yaw_error_deg"] for item in matches]
        ),
        "latency_ms": _distribution([item["latency_ms"] for item in evaluated]),
        "non_cube_outputs": dict(sorted(non_cube.items())),
    }


def evaluate_batch(
    entries: Sequence[BatchEntry],
    *,
    profile_path: Path,
    calibration_path: Path,
    frame_stem: str = DEFAULT_FRAME_STEM,
) -> Mapping[str, Any]:
    results: list[Mapping[str, Any]] = []
    for entry in entries:
        try:
            results.append(
                evaluate_run(
                    entry,
                    profile_path=profile_path,
                    calibration_path=calibration_path,
                    frame_stem=frame_stem,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "seed": entry.seed,
                    "status": "evaluation_failed",
                    "run_dir": (
                        None if entry.run_dir is None else str(entry.run_dir)
                    ),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "benchmark": {
            "input": "native table sensor_frames/f00000.png",
            "provider_input_contains_simulator_truth": False,
            "truth_read_order": "after provider inference",
            "camera_geometry": "fixed checked-in physical table calibration",
            "cube_edge_mm": 10.0,
            "visibility_scope": (
                "first captured frame only; later unknown-visibility frames excluded"
            ),
            "detection_match_rule": "same logical cube colour, nearest XY; no pose threshold",
            "screw_scope": "false-positive evidence only; realistic scene contains no screws",
        },
        "provenance": {
            "project_commit": _git_value("rev-parse", "HEAD"),
            "branch": _git_value("branch", "--show-current"),
            "worktree_dirty": _git_dirty(),
            "profile_path": str(profile_path.resolve()),
            "profile_sha256": _sha256(profile_path.resolve()),
            "calibration_path": str(calibration_path.resolve()),
            "calibration_sha256": _sha256(calibration_path.resolve()),
            "evaluator_sha256": _sha256(Path(__file__).resolve()),
        },
        "runs": results,
        "aggregate": aggregate_results(results),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="*", type=Path)
    parser.add_argument("--batch-status", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--frame-stem",
        default=DEFAULT_FRAME_STEM,
        help="first-frame stem; later frames are rejected by default scope",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.frame_stem != DEFAULT_FRAME_STEM:
        raise SyntheticEvaluationError(
            "only f00000 is supported until visibility/occlusion labels exist"
        )
    profile_path = args.profile.expanduser().resolve(strict=True)
    calibration_path = args.calibration.expanduser().resolve(strict=True)
    entries = collect_batch_entries(args.run_dirs, args.batch_status)
    report = evaluate_batch(
        entries,
        profile_path=profile_path,
        calibration_path=calibration_path,
        frame_stem=args.frame_stem,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SyntheticEvaluationError(f"output already exists: {output}")
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True, allow_nan=False))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
