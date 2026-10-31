from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from team_plugins.computer_vision.yoloe_experiment import (
    ExperimentError,
    ImagePrediction,
    Tile,
    box_iou,
    evaluate_boxes,
    extract_tile_predictions,
    generate_tiles,
    merge_tile_predictions,
)
from team_plugins.computer_vision.yoloe_visual_prompt_experiment import (
    ManifestError,
    _git_environment,
    _git_value,
    _worktree_dirty,
    load_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction(
    class_id: str,
    confidence: float,
    box: tuple[float, float, float, float],
    tile_id: str,
) -> ImagePrediction:
    x0, y0, x1, y1 = box
    return ImagePrediction(
        class_id=class_id,
        confidence=confidence,
        bbox_xyxy=box,
        polygon_xy=((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
        tile_id=tile_id,
    )


class FakeBoxes:
    def __init__(self, xyxy, confidence, classes):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.conf = np.asarray(confidence, dtype=np.float32)
        self.cls = np.asarray(classes, dtype=np.float32)

    def __len__(self):
        return len(self.xyxy)


class FakeMasks:
    def __init__(self, polygons):
        self.xy = tuple(np.asarray(value, dtype=np.float32) for value in polygons)


class FakeResult:
    def __init__(self, boxes, masks):
        self.boxes = boxes
        self.masks = masks


class TilingTests(unittest.TestCase):
    def test_tiles_cover_native_roi_without_resize_or_gaps(self):
        roi = (300, 160, 1720, 920)
        tiles = generate_tiles(
            image_size=(1920, 1080),
            roi_xyxy=roi,
            tile_size=(640, 640),
            overlap_fraction=0.25,
        )
        coverage = np.zeros((roi[3] - roi[1], roi[2] - roi[0]), dtype=np.uint8)
        for tile in tiles:
            self.assertGreaterEqual(tile.x0, roi[0])
            self.assertGreaterEqual(tile.y0, roi[1])
            self.assertLessEqual(tile.x1, roi[2])
            self.assertLessEqual(tile.y1, roi[3])
            coverage[
                tile.y0 - roi[1] : tile.y1 - roi[1],
                tile.x0 - roi[0] : tile.x1 - roi[0],
            ] += 1
        self.assertTrue(np.all(coverage >= 1))
        self.assertTrue(any(value > 1 for value in coverage.ravel()))
        self.assertEqual(len({tile.tile_id for tile in tiles}), len(tiles))

    def test_small_roi_becomes_one_exact_tile(self):
        tiles = generate_tiles(
            image_size=(640, 360),
            roi_xyxy=(10, 20, 110, 80),
            tile_size=(640, 640),
            overlap_fraction=0.5,
        )
        self.assertEqual(tuple(tile.xyxy for tile in tiles), ((10, 20, 110, 80),))

    def test_invalid_roi_and_overlap_fail_before_model_load(self):
        with self.assertRaisesRegex(ValueError, "inside the image"):
            generate_tiles(
                image_size=(640, 360),
                roi_xyxy=(-1, 0, 100, 100),
                tile_size=(100, 100),
                overlap_fraction=0.1,
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
            generate_tiles(
                image_size=(640, 360),
                roi_xyxy=(0, 0, 100, 100),
                tile_size=(100, 100),
                overlap_fraction=1.0,
            )


class PredictionTests(unittest.TestCase):
    def test_tile_boxes_and_masks_map_back_to_full_frame(self):
        tile = Tile("tile-test", 300, 200, 940, 840)
        result = FakeResult(
            FakeBoxes([[10, 20, 30, 50]], [0.8], [1]),
            FakeMasks([[[10, 20], [30, 20], [30, 50], [10, 50]]]),
        )
        predictions = extract_tile_predictions(
            result,
            tile=tile,
            class_ids=("cube", "screw_iso10642_m6x30"),
        )
        self.assertEqual(len(predictions), 1)
        value = predictions[0]
        self.assertEqual(value.class_id, "screw_iso10642_m6x30")
        self.assertEqual(value.bbox_xyxy, (310.0, 220.0, 330.0, 250.0))
        self.assertEqual(value.polygon_xy[0], (310.0, 220.0))

    def test_boxes_without_segmentation_fail_closed(self):
        result = FakeResult(FakeBoxes([[1, 2, 3, 4]], [0.8], [0]), None)
        with self.assertRaisesRegex(ExperimentError, "without one polygon"):
            extract_tile_predictions(
                result,
                tile=Tile("tile-test", 0, 0, 10, 10),
                class_ids=("cube",),
            )

    def test_overlap_merge_is_class_aware_and_deterministic(self):
        predictions = (
            _prediction("cube", 0.7, (10, 10, 30, 30), "b"),
            _prediction("cube", 0.9, (11, 11, 31, 31), "a"),
            _prediction("screw_iso10642_m6x30", 0.8, (11, 11, 31, 31), "c"),
        )
        merged = merge_tile_predictions(predictions, iou_threshold=0.5)
        self.assertEqual(len(merged), 2)
        cube = next(item for item in merged if item.class_id == "cube")
        self.assertEqual(cube.confidence, 0.9)
        self.assertAlmostEqual(box_iou((10, 10, 30, 30), (11, 11, 31, 31)), 361 / 439)

    def test_merge_does_not_suppress_distinct_instances_from_one_tile(self):
        predictions = (
            _prediction("cube", 0.9, (10, 10, 30, 30), "same-tile"),
            _prediction("cube", 0.8, (11, 11, 31, 31), "same-tile"),
        )
        self.assertEqual(
            len(merge_tile_predictions(predictions, iou_threshold=0.5)),
            2,
        )

    def test_box_evaluation_keeps_missing_and_extra_instances_visible(self):
        predictions = (
            _prediction("cube", 0.9, (0, 0, 10, 10), "a"),
            _prediction("cube", 0.8, (50, 50, 60, 60), "b"),
        )
        result = evaluate_boxes(
            predictions,
            (
                {"class_id": "cube", "bbox_xyxy": (0, 0, 10, 10)},
                {"class_id": "cube", "bbox_xyxy": (20, 20, 30, 30)},
            ),
            iou_threshold=0.5,
        )
        row = result["per_class"][0]
        self.assertEqual(row["true_positives"], 1)
        self.assertEqual(row["false_positives"], 1)
        self.assertEqual(row["dropouts"], 1)
        self.assertEqual(row["precision"], 0.5)
        self.assertEqual(row["recall"], 0.5)


class ManifestTests(unittest.TestCase):
    def make_manifest(self, root: Path, *, license_context="evaluation-only-agpl-review") -> Path:
        model = root / "model.pt"
        model.write_bytes(b"not-a-real-model")
        image = np.full((48, 64, 3), 220, dtype=np.uint8)
        reference = root / "reference.png"
        target = root / "target.png"
        self.assertTrue(cv2.imwrite(str(reference), image))
        self.assertTrue(cv2.imwrite(str(target), image))
        annotation = root / "target.annotation.json"
        annotation.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "image_sha256": _sha256(target),
                    "instances": [
                        {"class_id": "cube", "bbox_xyxy": [10, 10, 20, 20]}
                    ],
                }
            ),
            encoding="utf-8",
        )
        manifest = root / "experiment.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "experiment_id": "unit-test",
                    "license_context": license_context,
                    "model": {
                        "checkpoint": model.name,
                        "sha256": _sha256(model),
                        "ultralytics_version": "8.4.194",
                        "model_id": "yoloe-26s-seg",
                    },
                    "expected_image_size": [64, 48],
                    "roi_xyxy": [0, 0, 64, 48],
                    "tile_size": [32, 32],
                    "overlap_fraction": 0.25,
                    "merge_iou_threshold": 0.5,
                    "prompts": [
                        {
                            "class_id": "cube",
                            "reference_image": reference.name,
                            "reference_sha256": _sha256(reference),
                            "bbox_xyxy": [10, 10, 20, 20],
                        }
                    ],
                    "inputs": [
                        {
                            "image": target.name,
                            "sha256": _sha256(target),
                            "split": "pilot",
                            "frame_id": "frame-1",
                            "annotation": annotation.name,
                            "annotation_sha256": _sha256(annotation),
                        }
                    ],
                    "inference": {
                        "imgsz": 640,
                        "confidence": 0.1,
                        "model_iou": 0.7,
                        "device": "cpu",
                    },
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_manifest_resolves_hashes_tiles_prompts_and_annotations(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.make_manifest(Path(temporary))
            result = load_manifest(manifest)
        self.assertEqual(result["experiment_id"], "unit-test")
        self.assertEqual(result["prompts"][0]["class_id"], "cube")
        self.assertEqual(result["inputs"][0]["references"][0]["class_id"], "cube")
        self.assertGreater(len(result["tiles"]), 1)

    def test_manifest_requires_explicit_license_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.make_manifest(Path(temporary), license_context="unspecified")
            with self.assertRaisesRegex(ManifestError, "license_context"):
                load_manifest(manifest)

    def test_manifest_rejects_artifact_hash_mismatch_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.make_manifest(Path(temporary))
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            raw["model"]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "SHA-256 mismatch"):
                load_manifest(manifest)

    def test_manifest_rejects_wrong_input_size_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.make_manifest(Path(temporary))
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            raw["expected_image_size"] = [65, 48]
            manifest.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "does not match expected_image_size"):
                load_manifest(manifest)

    def test_manifest_rejects_unsafe_or_duplicate_frame_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.make_manifest(Path(temporary))
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            raw["inputs"][0]["frame_id"] = "../escape"
            manifest.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "letters, numbers"):
                load_manifest(manifest)


class GitProvenanceTests(unittest.TestCase):
    def test_current_cross_os_worktree_provenance_is_readable(self):
        repository = Path(__file__).resolve().parents[3]
        environment = _git_environment(repository)
        self.assertIsInstance(environment, dict)
        self.assertRegex(
            _git_value(repository, "rev-parse", "HEAD"), r"^[0-9a-f]{40}$"
        )
        self.assertTrue(_git_value(repository, "branch", "--show-current"))
        self.assertIsInstance(_worktree_dirty(repository), bool)

    def test_git_failures_are_not_misreported_as_clean(self):
        repository = Path(__file__).resolve().parents[3]
        failed = mock.Mock(returncode=1, stdout="", stderr="synthetic git failure")
        with mock.patch(
            "team_plugins.computer_vision.yoloe_visual_prompt_experiment._git_environment",
            return_value={},
        ), mock.patch(
            "team_plugins.computer_vision.yoloe_visual_prompt_experiment.subprocess.run",
            return_value=failed,
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic git failure"):
                _git_value(repository, "rev-parse", "HEAD")
            with self.assertRaisesRegex(RuntimeError, "synthetic git failure"):
                _worktree_dirty(repository)


if __name__ == "__main__":
    unittest.main()
