from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from team_plugins.computer_vision.fastsam_experiment import (
    MaskProposal,
    Tile,
    evaluate_proposal_coverage,
    extract_proposals,
    generate_tiles,
    merge_proposals,
)
from team_plugins.computer_vision.run_fastsam_experiment import (
    ManifestError,
    load_manifest,
)


def proposal(
    proposal_id: str,
    bbox,
    *,
    confidence: float = 0.8,
    tile_id: str = "tile-a",
) -> MaskProposal:
    x0, y0, x1, y1 = bbox
    return MaskProposal(
        proposal_id=proposal_id,
        confidence=confidence,
        bbox_xyxy=bbox,
        polygon_xy=((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
        centroid_uv=((x0 + x1) / 2, (y0 + y1) / 2),
        mask_area_px=max(1, int((x1 - x0) * (y1 - y0))),
        tile_id=tile_id,
    )


class FastSamExperimentTests(unittest.TestCase):
    def test_tiles_cover_roi_edges_without_rescaling(self):
        tiles = generate_tiles(
            image_size=(1920, 1080),
            roi_xyxy=(200, 100, 1720, 980),
            tile_size=(640, 640),
            overlap_fraction=0.25,
        )
        self.assertEqual(min(tile.x0 for tile in tiles), 200)
        self.assertEqual(max(tile.x1 for tile in tiles), 1720)
        self.assertEqual(min(tile.y0 for tile in tiles), 100)
        self.assertEqual(max(tile.y1 for tile in tiles), 980)
        self.assertTrue(all(tile.width == 640 and tile.height == 640 for tile in tiles))

    def test_extract_proposals_maps_masks_to_full_frame_and_filters_tiny(self):
        masks = np.zeros((2, 20, 30), dtype=np.float32)
        masks[0, 5:10, 7:19] = 1.0
        masks[1, 0:1, 0:2] = 1.0
        result = extract_proposals(
            masks=masks,
            confidences=(0.9, 0.8),
            tile=Tile("tile", 100, 200, 130, 220),
            minimum_mask_area_px=5,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].bbox_xyxy, (107.0, 205.0, 119.0, 210.0))
        self.assertAlmostEqual(result[0].centroid_uv[0], 112.5)
        self.assertAlmostEqual(result[0].centroid_uv[1], 207.0)
        self.assertEqual(result[0].mask_area_px, 60)

    def test_extract_proposals_keeps_largest_connected_component_consistent(self):
        masks = np.zeros((1, 20, 30), dtype=np.float32)
        masks[0, 1:4, 1:4] = 1.0
        masks[0, 10:15, 20:25] = 1.0
        result = extract_proposals(
            masks=masks,
            confidences=(0.9,),
            tile=Tile("tile", 100, 200, 130, 220),
            minimum_mask_area_px=1,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].bbox_xyxy, (120.0, 210.0, 125.0, 215.0))
        self.assertEqual(result[0].centroid_uv, (122.0, 212.0))
        self.assertEqual(result[0].mask_area_px, 25)

    def test_merge_suppresses_only_cross_tile_overlap(self):
        kept = merge_proposals(
            (
                proposal("a", (10, 10, 30, 30), confidence=0.9, tile_id="left"),
                proposal("b", (11, 11, 31, 31), confidence=0.8, tile_id="right"),
                proposal("c", (11, 11, 31, 31), confidence=0.7, tile_id="left"),
            ),
            iou_threshold=0.5,
        )
        self.assertEqual({item.proposal_id for item in kept}, {"a", "c"})

    def test_coverage_uses_maximum_cardinality_and_does_not_call_unmatched_false(self):
        references = (
            {"instance_id": "left", "bbox_xyxy": (0, 0, 10, 10)},
            {"instance_id": "right", "bbox_xyxy": (8, 0, 18, 10)},
        )
        proposals = (
            proposal("bridge", (5, 0, 13, 10)),
            proposal("left-only", (0, 0, 6, 10)),
            proposal("unrelated", (30, 30, 40, 40)),
        )
        result = evaluate_proposal_coverage(
            references, proposals, minimum_bbox_iou=0.1
        )
        self.assertEqual(result["covered_references"], 2)
        self.assertEqual(result["dropouts"], 0)
        self.assertEqual(result["unmatched_proposals"], 1)
        self.assertEqual(result["by_class"]["unlabeled"]["covered_references"], 2)
        self.assertNotIn("false_positive", json.dumps(result))

    def test_coverage_rejects_scene_sized_mask_at_explicit_iou_gate(self):
        result = evaluate_proposal_coverage(
            ({"instance_id": "tiny", "bbox_xyxy": (0, 0, 10, 10)},),
            (proposal("scene", (0, 0, 100, 100)),),
            minimum_bbox_iou=0.1,
        )
        self.assertEqual(result["covered_references"], 0)
        self.assertEqual(result["dropouts"], 1)
        self.assertEqual(result["unmatched_proposals"], 1)

    def test_manifest_is_hash_pinned_and_requires_external_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "FastSAM-s.pt"
            model.write_bytes(b"test-model")
            image = root / "image.png"
            cv2.imwrite(str(image), np.zeros((20, 30, 3), dtype=np.uint8))
            annotation = root / "annotation.json"
            image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
            annotation.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "image_sha256": image_hash,
                        "instances": [
                            {
                                "instance_id": "screw-1",
                                "class_id": "screw_iso10642_m6x30",
                                "bbox_xyxy": [5, 5, 15, 12],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiment_id": "test",
                        "license_context": "evaluation-only-agpl-review",
                        "model": {
                            "path": str(model),
                            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                            "ultralytics_version": "8.4.194",
                        },
                        "expected_image_size": [30, 20],
                        "roi_xyxy": [0, 0, 30, 20],
                        "tile_size": [30, 20],
                        "overlap_fraction": 0.0,
                        "merge_iou_threshold": 0.5,
                        "minimum_reference_bbox_iou": 0.1,
                        "inference": {
                            "device": "cpu",
                            "imgsz": 1024,
                            "confidence": 0.25,
                            "iou": 0.9,
                            "minimum_mask_area_px": 4,
                        },
                        "inputs": [
                            {
                                "input_id": "frame",
                                "image": {"path": str(image), "sha256": image_hash},
                                "annotation_path": str(annotation),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_manifest(manifest)
            self.assertEqual(loaded["inputs"][0]["annotation"][0]["instance_id"], "screw-1")
            raw = json.loads(manifest.read_text(encoding="utf-8"))

            raw["expected_image_size"] = [30.5, 20]
            manifest.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "must be an integer"):
                load_manifest(manifest)

            raw["expected_image_size"] = [30, 20]
            raw["inputs"][0]["input_id"] = "../../escape"
            manifest.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "filename-safe"):
                load_manifest(manifest)

            raw["inputs"][0]["input_id"] = "frame"
            raw["model"]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "SHA-256 mismatch"):
                load_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
