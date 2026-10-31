from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from sorting_core.perception import DetectedObject, Pose3D, SceneObservation
from team_plugins.computer_vision import evaluate_synthetic_runs as evaluator


def _object(
    track_id: str,
    colour: str,
    xy: tuple[float, float],
    *,
    yaw_deg: float = 0.0,
    class_id: str = "cube",
) -> DetectedObject:
    half = math.radians(yaw_deg) / 2.0
    return DetectedObject(
        track_id=track_id,
        class_id=class_id,
        sort_label=colour if class_id == "cube" else None,
        capture_time_s=0.0,
        pose=Pose3D(
            position_base_m=(xy[0], xy[1], 0.0),
            orientation_xyzw=(0.0, 0.0, math.sin(half), math.cos(half)),
        ),
        dimensions_m=(0.01, 0.01, 0.01),
        confidence=0.9,
        source_camera_ids=("table",),
    )


def _truth(track_id: str, colour: str, xy: tuple[float, float], yaw_deg: float = 0.0):
    half = math.radians(yaw_deg) / 2.0
    return {
        "track_id": track_id,
        "class_name": "cube",
        "colour": colour,
        "position_base_m": [xy[0], xy[1], 0.0],
        "quaternion_base_wxyz": [math.cos(half), 0.0, 0.0, math.sin(half)],
        "derived_amodal_bbox_status": "valid",
    }


class ScoreSceneTests(unittest.TestCase):
    def test_matches_by_colour_and_keeps_pose_error_separate(self):
        scene = SceneObservation(
            scene_id="scene",
            capture_time_s=0.0,
            objects=(
                _object("red-good", "red", (0.401, -0.101), yaw_deg=88.0),
                _object("red-extra", "red", (0.60, 0.20)),
                _object("blue-good", "blue", (0.45, 0.12), yaw_deg=47.0),
                _object("unknown", "none", (0.30, 0.30), class_id="unknown_obstacle"),
            ),
            provider_name="test",
            provider_version="1",
            calibration_ids=("table",),
        )
        result = evaluator.score_scene(
            scene,
            (
                _truth("truth-red", "red", (0.4, -0.1), yaw_deg=0.0),
                _truth("truth-blue", "blue", (0.45, 0.12), yaw_deg=2.0),
                _truth("truth-yellow", "yellow", (0.5, 0.0)),
            ),
        )
        self.assertEqual(result["true_positives"], 2)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["false_negatives"], 1)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertEqual(result["non_cube_outputs"], {"unknown_obstacle": 1})
        matches = {item["colour"]: item for item in result["matches"]}
        self.assertAlmostEqual(matches["red"]["xy_error_mm"], math.sqrt(2), places=6)
        self.assertAlmostEqual(matches["red"]["square_yaw_error_deg"], 2.0, places=6)
        self.assertAlmostEqual(matches["blue"]["square_yaw_error_deg"], 45.0, places=6)


class BatchStatusTests(unittest.TestCase):
    def test_failed_seed_survives_without_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "batch.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "runs": [
                            {
                                "seed": 211,
                                "status": "failed",
                                "returncode": 1,
                                "error": "Isaac exited before capture",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            entries = evaluator.load_batch_status(manifest)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].seed, 211)
        self.assertEqual(entries[0].status, "failed")
        self.assertIsNone(entries[0].run_dir)
        result = evaluator.evaluate_run(
            entries[0],
            profile_path=Path("profile.json"),
            calibration_path=Path("cameras.json"),
        )
        self.assertEqual(result["status"], "generation_failed")


class RunOrderingTests(unittest.TestCase):
    def test_truth_is_opened_only_after_provider_returns(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            (run_dir / "sensor_frames").mkdir(parents=True)
            (run_dir / "frame_labels").mkdir()
            image_path = run_dir / "sensor_frames" / "f00000.png"
            self.assertTrue(
                cv2.imwrite(
                    str(image_path),
                    np.full((1080, 1920, 3), 255, dtype=np.uint8),
                )
            )
            label_path = run_dir / "frame_labels" / "f00000.json"
            label_path.write_text(
                json.dumps(
                    {
                        "camera": {
                            "derived_sensor_image_relpath": "sensor_frames/f00000.png"
                        },
                        "randomization": {"episode_seed": 101},
                        "evaluator_only": {
                            "objects": [_truth("truth-red", "red", (0.4, -0.1))]
                        },
                    }
                ),
                encoding="utf-8",
            )
            events: list[str] = []

            class Provider:
                def observe(self, request):
                    events.append("provider")
                    return SceneObservation(
                        scene_id=request.scene_id,
                        capture_time_s=request.capture_time_s,
                        objects=(_object("red", "red", (0.4, -0.1)),),
                        provider_name="test-provider",
                        provider_version="1",
                        calibration_ids=("table-calibration",),
                    )

            def builder(profile_path, calibration_path):
                return (
                    Provider(),
                    "table-calibration",
                    SimpleNamespace(profile_id="test-profile", source_sha256="a" * 64),
                )

            original_load_json = evaluator._load_json

            def ordered_load_json(path):
                if Path(path) == label_path:
                    self.assertEqual(events, ["provider"])
                    events.append("truth")
                return original_load_json(path)

            with mock.patch.object(evaluator, "_load_json", side_effect=ordered_load_json):
                result = evaluator.evaluate_run(
                    evaluator.BatchEntry(101, run_dir, "success"),
                    profile_path=Path("profile.json"),
                    calibration_path=Path("cameras.json"),
                    provider_builder=builder,
                )
        self.assertEqual(events, ["provider", "truth"])
        self.assertTrue(result["truth_was_read_after_inference"])
        self.assertEqual(result["vision"]["true_positives"], 1)


if __name__ == "__main__":
    unittest.main()
