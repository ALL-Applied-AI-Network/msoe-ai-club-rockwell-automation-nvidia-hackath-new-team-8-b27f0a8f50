from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from team_plugins.computer_vision import prepare_annotation_bundle as bundle_module
from sorting_core import (
    DetectedObject,
    Pose3D,
    SceneObservation,
    scene_observation_to_dict,
)
from team_plugins.computer_vision.evaluate_annotations import validate_manifest
from team_plugins.computer_vision.prepare_annotation_bundle import (
    AnnotationBundleError,
    prepare_bundle,
    read_frame_indices,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AnnotationBundleTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        capture = root / "capture.png"
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        image[:, :] = (20, 40, 60)
        self.assertTrue(cv2.imwrite(str(capture), image))

        calibration = root / "cameras.json"
        calibration.write_text(
            json.dumps(
                {
                    "table": {
                        "T_base_cam": np.eye(4).tolist(),
                        "fx": 100.0,
                        "size": [100, 80],
                        "k1": 0.0,
                    }
                }
            ),
            encoding="utf-8",
        )
        frame_indices = root / "frames.txt"
        frame_indices.write_text("# pilot selection\n0\n", encoding="utf-8")
        replay = root / "replay"
        replay.mkdir()

        cube = DetectedObject(
            track_id="cube-track",
            class_id="cube",
            sort_label="red",
            capture_time_s=0.0,
            pose=Pose3D(
                position_base_m=(0.0, 0.0, 1.0),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
            dimensions_m=(0.016, 0.016, 0.016),
            confidence=0.9,
            source_camera_ids=("table",),
            attributes={"pixel_center_uv": [49.0, 39.0]},
        )
        screw = DetectedObject(
            track_id="screw-track",
            class_id="screw_iso10642_m6x30",
            sort_label="red",
            capture_time_s=0.0,
            pose=Pose3D(
                position_base_m=(-0.1, 0.0, 1.0),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
            dimensions_m=(0.012, 0.012, 0.03),
            confidence=0.8,
            pose_quality=0.7,
            source_camera_ids=("table",),
            attributes={
                "pixel_center_uv": [50.0, 40.0],
                "head_xy_base_m": [-0.1, 0.0],
                "tip_xy_base_m": [0.1, 0.0],
            },
        )
        withheld_screw = DetectedObject(
            track_id="withheld-screw-track",
            class_id="unknown_obstacle",
            sort_label=None,
            capture_time_s=0.0,
            pose=Pose3D(
                position_base_m=(-0.2, 0.0, 1.0),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
            dimensions_m=(0.03, 0.03, 0.013),
            confidence=0.49,
            pose_quality=0.0,
            source_camera_ids=("table",),
            attributes={
                "pixel_center_uv": [30.0, 40.0],
                "candidate_class": "screw_iso10642_m6x30",
                "non_executable": True,
                "ambiguity_reason": (
                    "footprint_outside_pickup_region,not_isolated_on_white_table"
                ),
            },
        )
        scene = SceneObservation(
            scene_id="capture:0",
            capture_time_s=0.0,
            objects=(cube, screw, withheld_screw),
            provider_name="test-provider",
            provider_version="1",
            calibration_ids=("cameras.json:table",),
        )
        record = {
            "frame_index": 0,
            "timestamp_s": 0.0,
            "latency_ms": 12.5,
            "input_rgb_sha256": hashlib.sha256(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes(order="C")
            ).hexdigest(),
            "input_rgb_shape_hwc": [80, 100, 3],
            "input_rgb_dtype": "uint8",
            "input_channel_order": "RGB",
            "scene": scene_observation_to_dict(scene),
        }
        scenes_path = replay / "scenes.jsonl"
        scenes_path.write_text(
            json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
        )
        profile_sha = "b" * 64
        provider_sha = "c" * 64
        report = {
            "schema_version": 1,
            "tool": "team_plugins.computer_vision.replay_camera",
            "capture_sha256": _sha256(capture),
            "calibration_id": "cameras.json:table",
            "calibration_sha256": _sha256(calibration),
            "provider_name": "test-provider",
            "provider_version": "1",
            "provider_artifact_sha256": provider_sha,
            "replay_tool_sha256": "d" * 64,
            "scenes_jsonl_sha256": _sha256(scenes_path),
            "project_commit": "a" * 40,
            "branch": "computer-vision",
            "worktree_dirty": False,
            "worktree_dirty_at_start": False,
            "worktree_dirty_at_end": False,
            "provenance_stable_during_replay": True,
            "provenance_fields_changed_during_replay": [],
            "integration_profile": {
                "profile_id": "test-profile",
                "profile_path": "/not/used/in/test.json",
                "profile_sha256": profile_sha,
            },
            "processed_frame_indices": [0],
            "processed_frames": 1,
            "source_size": [100, 80],
            "native_calibration_size": [100, 80],
            "decoded_source_fps": 1.0,
            "object_observations_by_class": {
                "cube": 1,
                "screw_iso10642_m6x30": 1,
                "unknown_obstacle": 1,
            },
            "evaluation_result": "evaluation-result.v1.json",
            "evaluation_result_skipped_reason": None,
        }
        (replay / "report.json").write_text(
            json.dumps(report, sort_keys=True), encoding="utf-8"
        )
        evaluation = {
            "schema_version": 1,
            "provenance": {
                "project_commit": "a" * 40,
                "branch": "computer-vision",
                "integration_profile": {
                    "profile_id": "test-profile",
                    "profile_sha256": profile_sha,
                },
                "object_catalog_version": "sha256:" + "e" * 64,
                "calibration_ids": ["cameras.json:table"],
                "components": [
                    {
                        "component": "vision",
                        "name": "test-provider",
                        "version": "1",
                        "artifact_sha256": provider_sha,
                    }
                ],
                "source_run_ids": [f"capture-sha256:{_sha256(capture)}"],
            },
        }
        (replay / "evaluation-result.v1.json").write_text(
            json.dumps(evaluation, sort_keys=True), encoding="utf-8"
        )
        return {
            "capture": capture,
            "calibration": calibration,
            "frame_indices": frame_indices,
            "replay": replay,
            "image": image,
        }

    @staticmethod
    def _refresh_scenes_hash(fixture: dict[str, Path]) -> None:
        report_path = fixture["replay"] / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["scenes_jsonl_sha256"] = _sha256(
            fixture["replay"] / "scenes.jsonl"
        )
        report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    def _prepare(self, fixture: dict[str, Path], output: Path):
        return prepare_bundle(
            capture=fixture["capture"],
            replay_dir=fixture["replay"],
            frame_indices_file=fixture["frame_indices"],
            output_dir=output,
            annotation_id="pilot-annotation-1",
            dataset_id="synthetic-test",
            source_run_id="synthetic-run",
            layout_id="layout-1",
            annotator_id="annotator-a",
            calibration_path=fixture["calibration"],
            profile_path=fixture["calibration"],
            provider_path=fixture["calibration"],
            replay_tool_path=fixture["calibration"],
            catalog_path=fixture["calibration"],
            verify_repository=False,
            verify_local_artifacts=False,
        )

    def test_bundle_is_schema_valid_and_preserves_lossless_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            output = root / "bundle"
            bundle_report = self._prepare(fixture, output)

            manifest = json.loads(
                (output / "annotation-manifest.pilot.json").read_text(
                    encoding="utf-8"
                )
            )
            validate_manifest(manifest, source="generated")
            self.assertEqual(manifest["split"], "pilot")
            self.assertEqual(manifest["provenance"]["review"]["review_status"], "unreviewed")
            self.assertEqual(manifest["frames"][0]["references"], [])
            self.assertEqual(manifest["frames"][0]["regions"], [])
            predictions = {
                item["class_id"]: item for item in manifest["frames"][0]["predictions"]
            }
            # Preserve the detector's actual mask centroid. Reprojecting the
            # lower 3D cube center would move this to [50, 40].
            self.assertEqual(predictions["cube"]["center_uv"], [49.0, 39.0])
            self.assertEqual(
                predictions["screw_iso10642_m6x30"]["head_uv"],
                [40.0, 40.0],
            )
            self.assertEqual(
                predictions["screw_iso10642_m6x30"]["tip_uv"],
                [60.0, 40.0],
            )
            self.assertEqual(
                predictions["screw_iso10642_m6x30"]["axis_observability"],
                "directed",
            )
            self.assertEqual(
                predictions["cube"]["provider_output_eligibility"],
                "published_supported",
            )
            self.assertEqual(
                predictions["unknown_obstacle"]["semantic_candidate_class_id"],
                "screw_iso10642_m6x30",
            )
            self.assertEqual(
                predictions["unknown_obstacle"]["provider_output_eligibility"],
                "withheld_non_executable",
            )
            self.assertEqual(
                predictions["unknown_obstacle"]["eligibility_reasons"],
                [
                    "footprint_outside_pickup_region",
                    "not_isolated_on_white_table",
                ],
            )
            source = cv2.imread(str(output / "source" / "frame-000000.png"))
            annotation = cv2.imread(
                str(output / "annotation" / "frame-000000.png")
            )
            preview = cv2.imread(str(output / "preview" / "frame-000000.png"))
            np.testing.assert_array_equal(source, fixture["image"])
            np.testing.assert_array_equal(annotation, fixture["image"])
            self.assertEqual(preview.shape, annotation.shape)
            self.assertFalse(np.array_equal(preview, annotation))
            self.assertFalse(
                bundle_report["frames"][0]["review_overlay_is_ground_truth"]
            )
            self.assertFalse(bundle_report["ready_for_scoring"])
            self.assertEqual(bundle_report["selected_frame_indices"], [0])
            self.assertEqual(
                bundle_report["bundle_tool_sha256"],
                _sha256(Path(bundle_module.__file__).resolve()),
            )
            self.assertEqual(len(bundle_report["bundle_tool_project_commit"]), 40)
            self.assertTrue(bundle_report["bundle_tool_branch"])

    def test_frame_index_file_requires_strictly_increasing_unique_integers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.txt"
            path.write_text("0\n0\n", encoding="utf-8")
            with self.assertRaisesRegex(AnnotationBundleError, "strictly increasing"):
                read_frame_indices(path)
            path.write_text("2\n1\n", encoding="utf-8")
            with self.assertRaisesRegex(AnnotationBundleError, "strictly increasing"):
                read_frame_indices(path)
            path.write_text("zero\n", encoding="utf-8")
            with self.assertRaisesRegex(AnnotationBundleError, "non-negative integer"):
                read_frame_indices(path)

    def test_capture_hash_mismatch_fails_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            report_path = fixture["replay"] / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["capture_sha256"] = "f" * 64
            report_path.write_text(json.dumps(report), encoding="utf-8")
            output = root / "bundle"
            with self.assertRaisesRegex(AnnotationBundleError, "capture hash"):
                self._prepare(fixture, output)
            self.assertFalse(output.exists())

    def test_redecoded_pixels_must_match_the_replay_input_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            scenes_path = fixture["replay"] / "scenes.jsonl"
            record = json.loads(scenes_path.read_text(encoding="utf-8"))
            record["input_rgb_sha256"] = "f" * 64
            scenes_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self._refresh_scenes_hash(fixture)

            with self.assertRaisesRegex(
                AnnotationBundleError, "pixels differ from replay input_rgb_sha256"
            ):
                self._prepare(fixture, root / "bundle")

    def test_scenes_hash_mismatch_fails_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            scenes_path = fixture["replay"] / "scenes.jsonl"
            record = json.loads(scenes_path.read_text(encoding="utf-8"))
            # This edit keeps frame indices, RGB provenance, classes, counts,
            # and provider identity plausible. Only the file-level hash can
            # prove that it was not the replay's original prediction.
            record["scene"]["objects"][0]["confidence"] = 0.8
            scenes_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            output = root / "bundle"
            with self.assertRaisesRegex(AnnotationBundleError, "scenes.jsonl hash"):
                self._prepare(fixture, output)
            self.assertFalse(output.exists())

    def test_missing_scenes_hash_fails_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            report_path = fixture["replay"] / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report.pop("scenes_jsonl_sha256")
            report_path.write_text(json.dumps(report), encoding="utf-8")
            output = root / "bundle"
            with self.assertRaisesRegex(
                AnnotationBundleError, "valid scenes.jsonl SHA-256"
            ):
                self._prepare(fixture, output)
            self.assertFalse(output.exists())

    def test_report_and_scene_indices_must_match_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            report_path = fixture["replay"] / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["processed_frame_indices"] = [0, 1]
            report["processed_frames"] = 2
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(AnnotationBundleError, "exactly match"):
                self._prepare(fixture, root / "bundle")

    def test_supported_prediction_requires_detector_pixel_center(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            scenes_path = fixture["replay"] / "scenes.jsonl"
            record = json.loads(scenes_path.read_text(encoding="utf-8"))
            record["scene"]["objects"][0]["attributes"].pop("pixel_center_uv")
            scenes_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self._refresh_scenes_hash(fixture)

            with self.assertRaisesRegex(
                AnnotationBundleError, "lacks detector pixel_center_uv"
            ):
                self._prepare(fixture, root / "bundle")

    def test_semantic_candidate_requires_explicit_non_executable_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            scenes_path = fixture["replay"] / "scenes.jsonl"
            record = json.loads(scenes_path.read_text(encoding="utf-8"))
            candidate = record["scene"]["objects"][2]
            candidate["attributes"]["non_executable"] = False
            scenes_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self._refresh_scenes_hash(fixture)

            with self.assertRaisesRegex(
                AnnotationBundleError, "without explicit non_executable=true"
            ):
                self._prepare(fixture, root / "bundle")

    def test_semantic_candidate_rejects_empty_reason_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            scenes_path = fixture["replay"] / "scenes.jsonl"
            record = json.loads(scenes_path.read_text(encoding="utf-8"))
            record["scene"]["objects"][2]["attributes"]["ambiguity_reason"] = (
                "footprint_outside_pickup_region,,not_isolated_on_white_table"
            )
            scenes_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self._refresh_scenes_hash(fixture)

            with self.assertRaisesRegex(
                AnnotationBundleError, "unique atomic ambiguity reasons"
            ):
                self._prepare(fixture, root / "bundle")

    def test_published_supported_class_rejects_non_executable_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            scenes_path = fixture["replay"] / "scenes.jsonl"
            record = json.loads(scenes_path.read_text(encoding="utf-8"))
            record["scene"]["objects"][0]["attributes"].update(
                {
                    "candidate_class": "cube",
                    "non_executable": True,
                    "ambiguity_reason": "occluded",
                }
            )
            scenes_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self._refresh_scenes_hash(fixture)

            with self.assertRaisesRegex(
                AnnotationBundleError,
                "publishes supported class with contradictory non-executable candidate metadata",
            ):
                self._prepare(fixture, root / "bundle")

    def test_nonempty_output_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            output = root / "bundle"
            output.mkdir()
            (output / "old.txt").write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(AnnotationBundleError, "new or empty"):
                self._prepare(fixture, output)


if __name__ == "__main__":
    unittest.main()
