from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from sorting_core.perception import (
    DetectedObject,
    Pose3D,
    SceneObservation,
    scene_observation_to_dict,
)
from team_plugins.computer_vision import compare_replays


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_GIT_SHA = "c" * 40


def _object(
    track_id: str,
    *,
    class_id: str = "cube",
    sort_label: str | None = "red",
    xyz: tuple[float, float, float] = (0.4, 0.0, 0.01),
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> DetectedObject:
    return DetectedObject(
        track_id=track_id,
        class_id=class_id,
        sort_label=sort_label,
        capture_time_s=0.0,
        pose=Pose3D(position_base_m=xyz, orientation_xyzw=quaternion),
        dimensions_m=(0.016, 0.016, 0.016),
        confidence=0.9,
        pose_quality=0.8,
        visibility=1.0,
        source_camera_ids=("table",),
        attributes={"symmetry_id": "upright_square_yaw_modulo_90_deg"},
    )


def _scene(
    frame_index: int,
    provider: str,
    objects: tuple[DetectedObject, ...],
) -> dict:
    timestamp = frame_index / 10.0
    rewritten = tuple(
        DetectedObject(
            track_id=item.track_id,
            class_id=item.class_id,
            sort_label=item.sort_label,
            capture_time_s=timestamp,
            pose=item.pose,
            dimensions_m=item.dimensions_m,
            confidence=item.confidence,
            pose_quality=item.pose_quality,
            visibility=item.visibility,
            source_camera_ids=item.source_camera_ids,
            attributes=item.attributes,
        )
        for item in objects
    )
    return scene_observation_to_dict(
        SceneObservation(
            scene_id=f"capture:{frame_index}",
            capture_time_s=timestamp,
            objects=rewritten,
            provider_name=provider,
            provider_version=f"{provider}-v1",
            calibration_ids=("cameras.json:table",),
        )
    )


def _scene_record(frame_index: int, provider: str, objects: tuple[DetectedObject, ...]) -> dict:
    return {
        "frame_index": frame_index,
        "timestamp_s": frame_index / 10.0,
        "latency_ms": 5.0,
        "input_rgb_sha256": hashlib.sha256(f"rgb-{frame_index}".encode()).hexdigest(),
        "input_rgb_shape_hwc": [1080, 1920, 3],
        "input_rgb_dtype": "uint8",
        "input_channel_order": "RGB",
        "scene": _scene(frame_index, provider, objects),
    }


def _error_record(frame_index: int, message: str = "whole-scene disagreement") -> dict:
    return {
        "frame_index": frame_index,
        "timestamp_s": frame_index / 10.0,
        "latency_ms": 7.0,
        "input_rgb_sha256": hashlib.sha256(f"rgb-{frame_index}".encode()).hexdigest(),
        "input_rgb_shape_hwc": [1080, 1920, 3],
        "input_rgb_dtype": "uint8",
        "input_channel_order": "RGB",
        "explicit_abstention": True,
        "vision_error": {"error_type": "VisionError", "message": message},
    }


def _write_bundle(
    root: Path,
    name: str,
    records: list[dict],
    *,
    provider: str | None,
    dirty: bool = False,
    report_overrides: dict | None = None,
) -> Path:
    directory = root / name
    directory.mkdir()
    scenes_payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    scenes_path = directory / "scenes.jsonl"
    scenes_path.write_bytes(scenes_payload.encode("utf-8"))
    object_counts: dict[str, int] = {}
    for record in records:
        for item in record.get("scene", {}).get("objects", []):
            class_id = item["class_id"]
            object_counts[class_id] = object_counts.get(class_id, 0) + 1
    validated = sum("scene" in record for record in records)
    errors = sum("vision_error" in record for record in records)
    report = {
        "schema_version": 1,
        "tool": "team_plugins.computer_vision.replay_camera",
        "capture_sha256": _SHA_A,
        "calibration_id": "cameras.json:table",
        "calibration_sha256": _SHA_B,
        "provider_name": provider,
        "provider_version": None if provider is None else f"{provider}-v1",
        "vision_factory": f"example.{name}:create",
        "provider_factory_module": f"example.{name}",
        "provider_artifact_sha256": "d" * 64,
        "vision_checkpoint": (
            None
            if provider == "baseline"
            else {"file_name": "model.pt", "sha256": "e" * 64}
        ),
        "replay_tool_sha256": "f" * 64,
        "scenes_jsonl_sha256": hashlib.sha256(scenes_payload.encode()).hexdigest(),
        "project_commit": _GIT_SHA,
        "branch": "hardware/smooth-motion-yolo11n-v2",
        "worktree_dirty": dirty,
        "worktree_dirty_at_start": dirty,
        "worktree_dirty_at_end": dirty,
        "provenance_stable_during_replay": True,
        "provenance_fields_changed_during_replay": [],
        "integration_profile": {
            "profile_id": f"profile-{name}",
            "profile_path": f"/external/{name}.json",
            "profile_sha256": "1" * 64,
        },
        "processed_frame_indices": [record["frame_index"] for record in records],
        "source_size": [640, 360],
        "native_calibration_size": [1920, 1080],
        "qualitative_upscale": True,
        "cube_edge_mm_profile": 16.0,
        "sample_every_n_frames": 10,
        "decoded_source_fps": 10.0,
        "start_frame": 0,
        "maximum_frames": None,
        "processed_frames": len(records),
        "validated_scene_frames": validated,
        "vision_error_frames": errors,
        "object_observations_by_class": object_counts,
    }
    if report_overrides:
        report.update(report_overrides)
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return directory


class ReplayComparisonTests(unittest.TestCase):
    def test_acceptance_error_streak_and_pose_disagreement_are_reported(self) -> None:
        baseline_cube = _object("cube-red")
        candidate_cube = _object(
            "cube-red",
            xyz=(0.401, 0.0, 0.01),
            quaternion=(0.0, 0.0, 0.0, -1.0),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_path = _write_bundle(
                root,
                "baseline",
                [
                    _scene_record(0, "baseline", (baseline_cube,)),
                    _scene_record(10, "baseline", (baseline_cube,)),
                ],
                provider="baseline",
            )
            candidate_path = _write_bundle(
                root,
                "candidate",
                [
                    _scene_record(0, "candidate", (candidate_cube,)),
                    _error_record(10),
                ],
                provider="candidate",
            )

            baseline = compare_replays.load_replay_bundle(
                baseline_path, name="baseline", baseline=True
            )
            candidate = compare_replays.load_replay_bundle(
                candidate_path, name="candidate", baseline=False
            )
            report = compare_replays.compare_replays(baseline, candidate)

        availability = report["availability"]
        self.assertEqual(availability["candidate_validated_scene_frames"], 1)
        self.assertEqual(availability["candidate_explicit_abstention_frames"], 1)
        self.assertEqual(availability["candidate_acceptance_fraction"], 0.5)
        self.assertEqual(
            availability["error_streaks"]["maximum_length_processed_frames"], 1
        )
        self.assertEqual(
            availability["error_streaks"]["streaks"][0]["start_source_frame_index"],
            10,
        )
        disagreement = report["disagreement"]
        self.assertAlmostEqual(disagreement["position_delta_mm"]["maximum"], 1.0)
        self.assertEqual(disagreement["matched_pairs_with_raw_orientation_difference"], 0)
        self.assertRegex(report["comparison_tool_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotIn("profile_path", report["baseline"]["integration_profile"])
        self.assertIn("not ground truth", report["evidence_limit"])

    def test_variable_per_frame_errors_are_compacted_by_semantic_category(self) -> None:
        cube = _object("cube-red")
        baseline_records = [
            _scene_record(frame_index, "baseline", (cube,))
            for frame_index in range(10)
        ]
        candidate_records = [_scene_record(0, "candidate", (cube,))]
        for frame_index in range(1, 10):
            candidate_records.append(
                _error_record(
                    frame_index,
                    (
                        "YOLO11n and calibrated geometry disagree inside the "
                        "inference crop: unmatched_yolo_boxes=0, "
                        "unmatched_geometry_detections=1, unmatched_yolo=[], "
                        "unmatched_geometry=[{'track_id': "
                        f"'frame-{frame_index}', 'pixel_center_uv': "
                        f"({700 + frame_index}.0, 500.0)}}]"
                    ),
                )
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_path = _write_bundle(
                root,
                "baseline",
                baseline_records,
                provider="baseline",
            )
            candidate_path = _write_bundle(
                root,
                "candidate",
                candidate_records,
                provider="candidate",
            )
            baseline = compare_replays.load_replay_bundle(
                baseline_path, name="baseline", baseline=True
            )
            candidate = compare_replays.load_replay_bundle(
                candidate_path, name="candidate", baseline=False
            )
            report = compare_replays.compare_replays(baseline, candidate)

        self.assertEqual(report["schema_version"], 2)
        categories = report["availability"]["error_categories"]
        self.assertEqual(len(categories), 1)
        category = categories[0]
        self.assertEqual(category["category"], "yolo_geometry_disagreement")
        self.assertEqual(category["frames"], 9)
        self.assertEqual(category["distinct_message_sha256_count"], 9)
        self.assertEqual(len(category["representative_examples"]), 3)
        self.assertTrue(
            all(
                "message" not in example
                for example in category["representative_examples"]
            )
        )
        self.assertEqual(category["first_source_frame_index"], 1)
        self.assertEqual(category["last_source_frame_index"], 9)
        abstention_row = report["frames"][1]["vision_error"]
        self.assertNotIn("message", abstention_row)
        self.assertRegex(abstention_row["message_sha256"], r"^[0-9a-f]{64}$")
        streak_reason = report["availability"]["error_streaks"]["streaks"][0][
            "reasons"
        ][0]
        self.assertNotIn("message", streak_reason)
        self.assertEqual(streak_reason["processed_frames"], 9)
        self.assertNotIn("frame-9", json.dumps(report))
        self.assertEqual(
            report["availability"]["full_error_detail_source"]["sha256"],
            candidate.scenes_fingerprint["sha256"],
        )

    def test_absolute_path_like_error_is_absent_from_report_and_receipt(self) -> None:
        cube = _object("cube-red")
        private_message = (
            "provider failed while reading /home/alice/private/checkpoint.pt"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_path = _write_bundle(
                root,
                "baseline",
                [_scene_record(0, "baseline", (cube,))],
                provider="baseline",
            )
            candidate_path = _write_bundle(
                root,
                "candidate",
                [_error_record(0, private_message)],
                provider=None,
            )
            output = root / "comparison.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    compare_replays.main(
                        [
                            str(baseline_path),
                            str(candidate_path),
                            "--output",
                            str(output),
                        ]
                    ),
                    0,
                )
            report_text = output.read_text(encoding="utf-8")
            receipt_text = stdout.getvalue()

        self.assertNotIn(private_message, report_text)
        self.assertNotIn("/home/alice", report_text)
        self.assertNotIn("checkpoint.pt", report_text)
        self.assertNotIn(private_message, receipt_text)
        self.assertNotIn("/home/alice", receipt_text)
        report = json.loads(report_text)
        category = report["availability"]["error_categories"][0]
        self.assertRegex(
            category["attributes"]["bounded_prefix_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertEqual(
            category["representative_examples"][0]["source_frame_index"],
            0,
        )
        self.assertRegex(
            category["representative_examples"][0]["message_sha256"],
            r"^[0-9a-f]{64}$",
        )

    def test_every_per_frame_rgb_hash_must_match(self) -> None:
        cube = _object("cube-red")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_path = _write_bundle(
                root,
                "baseline",
                [_scene_record(0, "baseline", (cube,))],
                provider="baseline",
            )
            candidate_record = _scene_record(0, "candidate", (cube,))
            candidate_record["input_rgb_sha256"] = "9" * 64
            candidate_path = _write_bundle(
                root,
                "candidate",
                [candidate_record],
                provider="candidate",
            )
            baseline = compare_replays.load_replay_bundle(
                baseline_path, name="baseline", baseline=True
            )
            candidate = compare_replays.load_replay_bundle(
                candidate_path, name="candidate", baseline=False
            )

            with self.assertRaisesRegex(ValueError, "input RGB hashes differ"):
                compare_replays.compare_replays(baseline, candidate)

    def test_equal_rgb_hash_with_different_shape_is_rejected(self) -> None:
        cube = _object("cube-red")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_path = _write_bundle(
                root,
                "baseline",
                [_scene_record(0, "baseline", (cube,))],
                provider="baseline",
            )
            candidate_record = _scene_record(0, "candidate", (cube,))
            candidate_record["input_rgb_shape_hwc"] = [1920, 1080, 3]
            candidate_path = _write_bundle(
                root,
                "candidate",
                [candidate_record],
                provider="candidate",
            )
            baseline = compare_replays.load_replay_bundle(
                baseline_path, name="baseline", baseline=True
            )
            with self.assertRaisesRegex(ValueError, "native calibration size"):
                compare_replays.load_replay_bundle(
                    candidate_path, name="candidate", baseline=False
                )

            self.assertEqual(baseline.records[0].input_rgb_shape_hwc, (1080, 1920, 3))

    def test_baseline_cannot_contain_an_abstention(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_bundle(
                Path(temporary),
                "baseline",
                [_error_record(0)],
                provider=None,
            )
            with self.assertRaisesRegex(ValueError, "baseline record"):
                compare_replays.load_replay_bundle(path, name="baseline", baseline=True)

    def test_candidate_must_contain_exactly_one_scene_or_error(self) -> None:
        cube = _object("cube-red")
        record = _scene_record(0, "candidate", (cube,))
        record["vision_error"] = {
            "error_type": "VisionError",
            "message": "also failed",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_bundle(
                Path(temporary),
                "candidate",
                [record],
                provider="candidate",
            )
            with self.assertRaisesRegex(ValueError, "exactly one"):
                compare_replays.load_replay_bundle(path, name="candidate", baseline=False)

    def test_scene_record_cannot_also_claim_explicit_abstention(self) -> None:
        cube = _object("cube-red")
        record = _scene_record(0, "candidate", (cube,))
        record["explicit_abstention"] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_bundle(
                Path(temporary),
                "candidate",
                [record],
                provider="candidate",
            )
            with self.assertRaisesRegex(ValueError, "must not claim"):
                compare_replays.load_replay_bundle(path, name="candidate", baseline=False)

    def test_scene_hash_and_clean_provenance_are_required(self) -> None:
        cube = _object("cube-red")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dirty_path = _write_bundle(
                root,
                "dirty",
                [_scene_record(0, "candidate", (cube,))],
                provider="candidate",
                dirty=True,
            )
            with self.assertRaisesRegex(ValueError, "worktree_dirty"):
                compare_replays.load_replay_bundle(
                    dirty_path, name="candidate", baseline=False
                )

            tampered_path = _write_bundle(
                root,
                "tampered",
                [_scene_record(0, "candidate", (cube,))],
                provider="candidate",
            )
            with (tampered_path / "scenes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                compare_replays.load_replay_bundle(
                    tampered_path, name="candidate", baseline=False
                )

    def test_same_track_label_change_is_not_silently_spatially_matched(self) -> None:
        baseline_cube = _object("cube-one", sort_label="red")
        candidate_cube = _object("cube-one", sort_label="blue")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_path = _write_bundle(
                root,
                "baseline",
                [_scene_record(0, "baseline", (baseline_cube,))],
                provider="baseline",
            )
            candidate_path = _write_bundle(
                root,
                "candidate",
                [_scene_record(0, "candidate", (candidate_cube,))],
                provider="candidate",
            )
            baseline = compare_replays.load_replay_bundle(
                baseline_path, name="baseline", baseline=True
            )
            candidate = compare_replays.load_replay_bundle(
                candidate_path, name="candidate", baseline=False
            )
            report = compare_replays.compare_replays(baseline, candidate)

        self.assertEqual(
            report["disagreement"]["accepted_frames_with_same_track_label_difference"],
            1,
        )
        self.assertEqual(report["disagreement"]["unmatched_objects_across_accepted_frames"], 2)
        self.assertEqual(report["disagreement"]["matched_object_pairs"], 0)

    def test_replay_sampling_provenance_must_match(self) -> None:
        cube = _object("cube-red")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_path = _write_bundle(
                root,
                "baseline",
                [_scene_record(0, "baseline", (cube,))],
                provider="baseline",
            )
            candidate_path = _write_bundle(
                root,
                "candidate",
                [_scene_record(0, "candidate", (cube,))],
                provider="candidate",
                report_overrides={"cube_edge_mm_profile": 10.0},
            )
            baseline = compare_replays.load_replay_bundle(
                baseline_path, name="baseline", baseline=True
            )
            candidate = compare_replays.load_replay_bundle(
                candidate_path, name="candidate", baseline=False
            )
            with self.assertRaisesRegex(ValueError, "cube_edge_mm_profile"):
                compare_replays.compare_replays(baseline, candidate)

    def test_cli_exclusively_writes_a_new_output(self) -> None:
        cube = _object("cube-red")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_path = _write_bundle(
                root,
                "baseline",
                [_scene_record(0, "baseline", (cube,))],
                provider="baseline",
            )
            candidate_path = _write_bundle(
                root,
                "candidate",
                [_scene_record(0, "candidate", (cube,))],
                provider="candidate",
            )
            output = root / "comparison.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = compare_replays.main(
                    [str(baseline_path), str(candidate_path), "--output", str(output)]
                )
            self.assertEqual(result, 0)
            receipt = json.loads(stdout.getvalue())
            full_report = json.loads(output.read_text())
            self.assertEqual(
                receipt["tool"],
                "team_plugins.computer_vision.compare_replays.output_receipt",
            )
            self.assertEqual(receipt["output_file"], "comparison.json")
            self.assertNotIn(str(root), stdout.getvalue())
            self.assertEqual(
                receipt["output_sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            self.assertEqual(receipt["processed_frames"], 1)
            self.assertEqual(receipt["candidate_validated_scene_frames"], 1)
            self.assertEqual(receipt["candidate_explicit_abstention_frames"], 0)
            self.assertEqual(receipt["maximum_error_streak_processed_frames"], 0)
            self.assertEqual(receipt["error_category_count"], 0)
            self.assertEqual(receipt["evidence_limit"], full_report["evidence_limit"])

            full_stdout = io.StringIO()
            with contextlib.redirect_stdout(full_stdout):
                self.assertEqual(
                    compare_replays.main([str(baseline_path), str(candidate_path)]),
                    0,
                )
            self.assertEqual(json.loads(full_stdout.getvalue()), full_report)
            with self.assertRaisesRegex(ValueError, "new path"):
                compare_replays.main(
                    [str(baseline_path), str(candidate_path), "--output", str(output)]
                )


if __name__ == "__main__":
    unittest.main()
