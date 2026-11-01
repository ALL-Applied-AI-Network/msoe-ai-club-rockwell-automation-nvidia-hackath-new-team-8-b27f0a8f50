from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from sorting_core.perception import (
    DetectedObject,
    Pose3D,
    SceneObservation,
    scene_observation_to_dict,
)
from sorting_core.vision import VisionError
from team_plugins.computer_vision import replay_camera


def _fourcc(value: str) -> int:
    return sum(ord(character) << (8 * index) for index, character in enumerate(value))


class _FakeCapture:
    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        fps: float,
        advertised_frames: int,
        fourcc: str = "mp4v",
    ) -> None:
        self._frames = list(frames)
        self._fps = fps
        self._advertised_frames = advertised_frames
        self._fourcc = _fourcc(fourcc)
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FPS:
            return self._fps
        if property_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._advertised_frames)
        if property_id == cv2.CAP_PROP_FOURCC:
            return float(self._fourcc)
        return 0.0

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self) -> None:
        self.released = True


class _FakeWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.released = True


class ReplayFrameTests(unittest.TestCase):
    def test_av1_fourcc_uses_pyav_without_attempting_opencv_decode(self):
        capture = _FakeCapture([], fps=10.0, advertised_frames=50, fourcc="av01")
        expected_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        expected = [(0, 0.0, expected_frame, 12.5)]

        with (
            patch.object(replay_camera.cv2, "imread", return_value=None),
            patch.object(replay_camera.cv2, "VideoCapture", return_value=capture),
            patch.object(replay_camera, "_pyav_frames", return_value=iter(expected)) as pyav,
        ):
            actual = list(replay_camera._frames(Path("archive-av1.mp4")))

        self.assertEqual(len(actual), 1)
        self.assertIs(actual[0][2], expected_frame)
        self.assertEqual(actual[0][3], 12.5)
        self.assertTrue(capture.released)
        pyav.assert_called_once_with(Path("archive-av1.mp4"))

    def test_partial_opencv_decode_is_rejected(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        capture = _FakeCapture(
            [frame, frame],
            fps=30.0,
            advertised_frames=10,
        )

        with (
            patch.object(replay_camera.cv2, "imread", return_value=None),
            patch.object(replay_camera.cv2, "VideoCapture", return_value=capture),
            self.assertRaisesRegex(ValueError, "decoded only 2/10 advertised frames"),
        ):
            list(replay_camera._frames(Path("partial.mp4")))

        self.assertTrue(capture.released)

    def test_opencv_frames_retain_actual_decoded_fps(self):
        frames = [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2, 3), dtype=np.uint8),
        ]
        capture = _FakeCapture(frames, fps=25.0, advertised_frames=2)

        with (
            patch.object(replay_camera.cv2, "imread", return_value=None),
            patch.object(replay_camera.cv2, "VideoCapture", return_value=capture),
        ):
            decoded = list(replay_camera._frames(Path("25fps.mp4")))

        self.assertEqual([item[1] for item in decoded], [0.0, 1.0 / 25.0])
        self.assertEqual([item[3] for item in decoded], [25.0, 25.0])


class ReplayOverlayTests(unittest.TestCase):
    @staticmethod
    def _detected(
        track_id: str,
        class_id: str,
        *,
        sort_label: str | None,
        attributes: dict,
        confidence: float,
        pose_quality: float | None,
    ) -> DetectedObject:
        return DetectedObject(
            track_id=track_id,
            class_id=class_id,
            sort_label=sort_label,
            capture_time_s=1.0,
            pose=Pose3D(
                position_base_m=(0.4, 0.0, 0.01),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
            dimensions_m=(0.01, 0.01, 0.01),
            confidence=confidence,
            pose_quality=pose_quality,
            source_camera_ids=("table",),
            attributes=attributes,
        )

    def test_overlay_is_deterministic_and_does_not_mutate_inputs(self):
        cube = self._detected(
            "cube-track-0001",
            "cube",
            sort_label="red",
            attributes={"pixel_center_uv": [70.0, 140.0], "in_pickup_region": True},
            confidence=0.91,
            pose_quality=None,
        )
        screw = self._detected(
            "screw-track-0002",
            "screw_iso10642_m6x30",
            sort_label="red",
            attributes={
                "head_xy_base_m": [0.2, 0.2],
                "tip_xy_base_m": [0.35, 0.2],
                "footprint_inside_pickup_region": True,
            },
            confidence=0.84,
            pose_quality=0.77,
        )
        unknown = self._detected(
            "unknown-track-0003",
            "unknown_obstacle",
            sort_label=None,
            attributes={
                "pixel_center_uv": [250.0, 160.0],
                "non_executable": True,
                "candidate_class": "screw_iso10642_m6x30",
            },
            confidence=0.49,
            pose_quality=0.0,
        )
        scene = SceneObservation(
            scene_id="overlay-scene",
            capture_time_s=1.0,
            objects=(unknown, screw, cube),
            provider_name="test-provider",
            provider_version="1",
            calibration_ids=("table-calibration",),
        )
        source = np.full((240, 320, 3), 210, dtype=np.uint8)
        source_before = source.copy()
        scene_before = scene_observation_to_dict(scene)

        def fake_project(unused_camera, points):
            return np.asarray(
                [[point[0] * 300.0, point[1] * 300.0] for point in points],
                dtype=float,
            )

        with patch.object(replay_camera, "project", side_effect=fake_project):
            first = replay_camera._draw_overlay(source, scene, {})
            second = replay_camera._draw_overlay(source, scene, {})

        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.array_equal(source, source_before))
        self.assertEqual(scene_observation_to_dict(scene), scene_before)
        self.assertEqual(first.shape, source.shape)
        self.assertFalse(np.array_equal(first, source))
        self.assertEqual(replay_camera._overlay_status(screw), ("candidate", True))

    def test_overlay_status_never_promotes_missing_or_non_executable_metadata(self):
        cube_in_region = self._detected(
            "cube-in-region",
            "cube",
            sort_label="red",
            attributes={"pixel_center_uv": [12.0, 12.0], "in_pickup_region": True},
            confidence=0.9,
            pose_quality=None,
        )
        cube_unknown_region = self._detected(
            "cube",
            "cube",
            sort_label="blue",
            attributes={"pixel_center_uv": [10.0, 10.0]},
            confidence=0.5,
            pose_quality=None,
        )
        screw_outside = self._detected(
            "screw",
            "screw_iso10642_m6x30",
            sort_label="red",
            attributes={
                "head_xy_base_m": [0.2, 0.2],
                "tip_xy_base_m": [0.3, 0.2],
                "footprint_inside_pickup_region": False,
            },
            confidence=0.5,
            pose_quality=0.5,
        )
        unknown = self._detected(
            "unknown",
            "unknown_obstacle",
            sort_label=None,
            attributes={"pixel_center_uv": [20.0, 20.0]},
            confidence=0.49,
            pose_quality=0.0,
        )

        self.assertEqual(
            replay_camera._overlay_status(cube_in_region), ("in-region", True)
        )
        self.assertEqual(
            replay_camera._overlay_status(cube_unknown_region), ("region?", False)
        )
        self.assertEqual(replay_camera._overlay_status(screw_outside), ("out", False))
        self.assertEqual(replay_camera._overlay_status(unknown), ("noexec", False))
        self.assertEqual(
            replay_camera._overlay_colour(screw_outside, False),
            replay_camera._OVERLAY_INELIGIBLE_COLOUR,
        )
        self.assertEqual(
            replay_camera._overlay_colour(unknown, False),
            replay_camera._OVERLAY_CLASS_STYLES["unknown_obstacle"][1],
        )


class ReplayProviderLoadingTests(unittest.TestCase):
    def test_factory_artifact_follows_selected_yolo_module(self):
        reference = "team_plugins.computer_vision.yolo11n_provider:create"
        factory = replay_camera.load_factory(reference, "vision")

        module_name, artifact = replay_camera._factory_artifact(factory, reference)

        self.assertEqual(
            module_name,
            "team_plugins.computer_vision.yolo11n_provider",
        )
        self.assertEqual(artifact.name, "yolo11n_provider.py")
        self.assertNotEqual(artifact.name, "hybrid_geometry_provider.py")

    def test_checkpoint_inside_repository_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "repository"
            repository.mkdir()
            checkpoint = repository / "model.pt"
            checkpoint.write_bytes(b"weights")

            with (
                patch.object(replay_camera, "REPO_ROOT", repository),
                self.assertRaisesRegex(ValueError, "outside the Git repository"),
            ):
                replay_camera._resolve_vision_checkpoint(checkpoint)

    def test_factory_failure_without_checkpoint_names_replay_flag(self):
        def factory(unused_context):
            raise ValueError("weights are required")

        context = SimpleNamespace(checkpoint=None)
        with self.assertRaisesRegex(RuntimeError, "--vision-checkpoint PATH"):
            replay_camera._create_provider(factory, context, "example.provider:create")

    def test_vision_error_recording_is_opt_in_and_does_not_fabricate_scene(self):
        class Provider:
            def observe(self, unused_request):
                raise VisionError("whole-scene agreement failed")

        provider = Provider()
        with self.assertRaisesRegex(VisionError, "whole-scene agreement failed"):
            replay_camera._observe_frame(
                provider,
                object(),
                record_vision_errors=False,
            )

        scene, error = replay_camera._observe_frame(
            provider,
            object(),
            record_vision_errors=True,
        )
        self.assertIsNone(scene)
        self.assertEqual(
            error,
            {
                "error_type": "VisionError",
                "message": "whole-scene agreement failed",
            },
        )

    def test_vision_error_streaks_use_processed_order_and_keep_source_indices(self):
        error_a = {"error_type": "VisionError", "message": "agreement failed"}
        error_b = {"error_type": "VisionError", "message": "screw abstention"}
        records = [
            {"frame_index": 10, "vision_error": error_a},
            {"frame_index": 20, "vision_error": error_a},
            {"frame_index": 30, "scene": {}},
            {"frame_index": 40, "vision_error": error_b},
        ]

        summary = replay_camera._summarize_vision_error_streaks(records)

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["maximum_length_processed_frames"], 2)
        self.assertEqual(
            summary["unit"],
            "consecutive processed replay frames (1-based processed ordinals)",
        )
        self.assertEqual(
            summary["streaks"][0],
            {
                "start_processed_ordinal": 1,
                "end_processed_ordinal": 2,
                "length_processed_frames": 2,
                "start_source_frame_index": 10,
                "end_source_frame_index": 20,
                "reasons": [
                    {
                        "error_type": "VisionError",
                        "message": "agreement failed",
                        "processed_frames": 2,
                    }
                ],
            },
        )
        self.assertEqual(summary["streaks"][1]["start_source_frame_index"], 40)
        self.assertEqual(summary["streaks"][1]["end_source_frame_index"], 40)


class ReplayMainGuardTests(unittest.TestCase):
    @staticmethod
    def _arguments(root: Path, output: Path) -> argparse.Namespace:
        capture = root / "capture.mp4"
        calibration = root / "calibration.json"
        profile = root / "profile.json"
        for path in (capture, calibration, profile):
            path.write_bytes(b"fixture")
        return argparse.Namespace(
            capture=capture,
            output_dir=output,
            calibration=calibration,
            profile=profile,
            vision_checkpoint=None,
            cube_edge_mm=16.0,
            every=2,
            start_frame=0,
            maximum_frames=1,
            allow_upscale_compressed=False,
            save_source_frames=False,
            record_vision_errors=False,
        )

    def test_main_rejects_nonempty_output_directory_before_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "existing-output"
            output.mkdir()
            (output / "evaluation-result.v1.json").write_text("stale", encoding="utf-8")
            arguments = self._arguments(root, output)

            with (
                patch.object(replay_camera, "_parse_args", return_value=arguments),
                self.assertRaisesRegex(ValueError, "new or empty"),
            ):
                replay_camera.main()

    def test_changed_provenance_suppresses_evaluation_and_returns_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "new-output"
            arguments = self._arguments(root, output)
            digest = "c" * 64
            profile = SimpleNamespace(
                source_sha256=digest,
                source_path=str(arguments.profile),
                profile_id="test-profile",
                vision=SimpleNamespace(
                    camera_ids=("table",),
                    depth_camera_ids=(),
                    factory=(
                        "team_plugins.computer_vision."
                        "hybrid_geometry_provider:create"
                    ),
                    options={},
                    calibration_ids={"table": "declared-table"},
                    depth_calibration_ids={},
                ),
            )

            class Provider:
                PROVIDER_NAME = "test-provider"
                PROVIDER_VERSION = "1"

                def observe(self, request):
                    return SceneObservation(
                        scene_id=request.scene_id,
                        capture_time_s=request.capture_time_s,
                        objects=(),
                        provider_name=self.PROVIDER_NAME,
                        provider_version=self.PROVIDER_VERSION,
                        calibration_ids=("calibration.json:table",),
                    )

            writer = _FakeWriter()
            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            git_values = iter(("a" * 40, "vision/test", "b" * 40, "vision/test"))
            writer_arguments: list[tuple] = []

            def video_writer(*args):
                writer_arguments.append(args)
                return writer

            with (
                patch.object(replay_camera, "_parse_args", return_value=arguments),
                patch.object(replay_camera, "_git_value", side_effect=lambda *unused: next(git_values)),
                patch.object(replay_camera, "_git_worktree_dirty", side_effect=(False, False)),
                patch.object(replay_camera, "_sha256", return_value=digest),
                patch.object(
                    replay_camera,
                    "load_calibration",
                    return_value={"table": {"size": [2, 2]}},
                ),
                patch.object(replay_camera, "load_integration_profile", return_value=profile),
                patch.object(replay_camera, "load_factory", return_value=lambda unused: Provider()),
                patch.object(replay_camera, "_frames", return_value=iter(((0, 0.0, frame, 24.0),))),
                patch.object(replay_camera, "_draw_overlay", return_value=frame),
                patch.object(replay_camera.cv2, "resize", return_value=np.zeros((540, 960, 3), dtype=np.uint8)),
                patch.object(replay_camera.cv2, "VideoWriter", side_effect=video_writer),
                patch.object(replay_camera.cv2, "imwrite", return_value=True),
                patch("builtins.print"),
            ):
                result = replay_camera.main()

            self.assertEqual(result, 2)
            self.assertTrue(writer.released)
            self.assertEqual(writer_arguments[0][2], 12.0)
            self.assertFalse((output / "evaluation-result.v1.json").exists())
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["decoded_source_fps"], 24.0)
            self.assertFalse(report["provenance_stable_during_replay"])
            self.assertEqual(
                report["provenance_fields_changed_during_replay"],
                ["project_commit"],
            )
            self.assertEqual(
                report["evaluation_result_skipped_reason"],
                "repository/input provenance changed during replay",
            )
            self.assertEqual(
                report["scenes_jsonl_sha256"],
                hashlib.sha256((output / "scenes.jsonl").read_bytes()).hexdigest(),
            )

    def test_main_replays_other_table_rgb_factory_with_external_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "new-output"
            arguments = self._arguments(root, output)
            checkpoint = root / "seed55-best.pt"
            checkpoint.write_bytes(b"external-model")
            arguments.vision_checkpoint = checkpoint
            arguments.maximum_frames = 2
            digest = "c" * 64
            profile = SimpleNamespace(
                source_sha256=digest,
                source_path=str(arguments.profile),
                profile_id="yolo-replay-profile",
                vision=SimpleNamespace(
                    camera_ids=("table",),
                    depth_camera_ids=(),
                    factory="team_plugins.computer_vision.yolo11n_provider:create",
                    options={},
                    calibration_ids={"table": "declared-table"},
                    depth_calibration_ids={},
                ),
            )

            class Provider:
                def observe(self, request):
                    return SceneObservation(
                        scene_id=request.scene_id,
                        capture_time_s=request.capture_time_s,
                        objects=(),
                        provider_name="model-backed-provider",
                        provider_version=f"1+sha256:{digest}",
                        calibration_ids=("calibration.json:table",),
                    )

            factory_contexts = []

            def factory(context):
                factory_contexts.append(context)
                return Provider()

            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            writer = _FakeWriter()
            with (
                patch.object(replay_camera, "_parse_args", return_value=arguments),
                patch.object(
                    replay_camera,
                    "_git_value",
                    side_effect=("a" * 40, "vision/test", "a" * 40, "vision/test"),
                ),
                patch.object(
                    replay_camera,
                    "_git_worktree_dirty",
                    side_effect=(False, False),
                ),
                patch.object(replay_camera, "_sha256", return_value=digest),
                patch.object(
                    replay_camera,
                    "load_calibration",
                    return_value={"table": {"size": [2, 2]}},
                ),
                patch.object(
                    replay_camera,
                    "load_integration_profile",
                    return_value=profile,
                ),
                patch.object(replay_camera, "load_factory", return_value=factory),
                patch.object(
                    replay_camera,
                    "_frames",
                    return_value=iter(
                        (
                            (0, 0.0, frame, 10.0),
                            (2, 0.2, frame, 10.0),
                        )
                    ),
                ),
                patch.object(replay_camera, "_draw_overlay", return_value=frame),
                patch.object(
                    replay_camera.cv2,
                    "resize",
                    return_value=np.zeros((540, 960, 3), dtype=np.uint8),
                ),
                patch.object(replay_camera.cv2, "VideoWriter", return_value=writer),
                patch.object(replay_camera.cv2, "imwrite", return_value=True),
                patch("builtins.print"),
            ):
                result = replay_camera.main()

            self.assertEqual(result, 0)
            self.assertEqual(len(factory_contexts), 1)
            self.assertEqual(factory_contexts[0].checkpoint, str(checkpoint.resolve()))
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(
                report["vision_factory"],
                "team_plugins.computer_vision.yolo11n_provider:create",
            )
            self.assertEqual(report["provider_factory_module"], __name__)
            self.assertEqual(report["provider_artifact_file"], Path(__file__).name)
            self.assertEqual(
                report["vision_checkpoint"],
                {"file_name": checkpoint.name, "sha256": digest},
            )
            self.assertEqual(report["provider_name"], "model-backed-provider")
            self.assertEqual(report["validated_scene_frames"], 2)
            self.assertEqual(report["vision_error_frames"], 0)
            self.assertEqual(
                report["experiment_identity"]["provider_name"],
                "model-backed-provider",
            )
            self.assertEqual(
                report["experiment_identity"]["provider_version"],
                f"1+sha256:{digest}",
            )
            self.assertEqual(
                report["experiment_identity"]["calibration_sha256"],
                digest,
            )
            evaluation = json.loads(
                (output / "evaluation-result.v1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                evaluation["provenance"]["source_run_ids"],
                [
                    f"capture-sha256:{digest}",
                    f"calibration-sha256:{digest}",
                    f"vision-checkpoint-sha256:{digest}",
                ],
            )
            self.assertEqual(
                evaluation["provenance"]["components"][0]["name"],
                "model-backed-provider",
            )
            self.assertEqual(
                evaluation["provenance"]["components"][0]["version"],
                f"1+sha256:{digest}",
            )
            self.assertNotIn(str(root), json.dumps(evaluation, sort_keys=True))

    def test_partial_recorded_vision_errors_skip_standardized_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "new-output"
            arguments = self._arguments(root, output)
            arguments.maximum_frames = 2
            arguments.record_vision_errors = True
            digest = "d" * 64
            profile = SimpleNamespace(
                source_sha256=digest,
                source_path=str(arguments.profile),
                profile_id="strict-profile",
                vision=SimpleNamespace(
                    camera_ids=("table",),
                    depth_camera_ids=(),
                    factory="team_plugins.computer_vision.strict_provider:create",
                    options={},
                    calibration_ids={"table": "declared-table"},
                    depth_calibration_ids={},
                ),
            )

            class Provider:
                def __init__(self):
                    self.calls = 0

                def observe(self, request):
                    self.calls += 1
                    if self.calls == 2:
                        raise VisionError("no complete-scene agreement")
                    return SceneObservation(
                        scene_id=request.scene_id,
                        capture_time_s=request.capture_time_s,
                        objects=(),
                        provider_name="strict-provider",
                        provider_version="2",
                        calibration_ids=("calibration.json:table",),
                    )

            def factory(unused_context):
                return Provider()

            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            with (
                patch.object(replay_camera, "_parse_args", return_value=arguments),
                patch.object(
                    replay_camera,
                    "_git_value",
                    side_effect=("a" * 40, "vision/test", "a" * 40, "vision/test"),
                ),
                patch.object(
                    replay_camera,
                    "_git_worktree_dirty",
                    side_effect=(False, False),
                ),
                patch.object(replay_camera, "_sha256", return_value=digest),
                patch.object(
                    replay_camera,
                    "load_calibration",
                    return_value={"table": {"size": [2, 2]}},
                ),
                patch.object(
                    replay_camera,
                    "load_integration_profile",
                    return_value=profile,
                ),
                patch.object(replay_camera, "load_factory", return_value=factory),
                patch.object(
                    replay_camera,
                    "_frames",
                    return_value=iter(
                        (
                            (0, 0.0, frame, 10.0),
                            (2, 0.2, frame, 10.0),
                        )
                    ),
                ),
                patch.object(
                    replay_camera.cv2,
                    "resize",
                    return_value=np.zeros((540, 960, 3), dtype=np.uint8),
                ),
                patch.object(
                    replay_camera.cv2,
                    "VideoWriter",
                    return_value=_FakeWriter(),
                ),
                patch.object(replay_camera.cv2, "imwrite", return_value=True),
                patch("builtins.print"),
            ):
                result = replay_camera.main()

            self.assertEqual(result, 0)
            self.assertFalse((output / "evaluation-result.v1.json").exists())
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["validated_scene_frames"], 1)
            self.assertEqual(report["vision_error_frames"], 1)
            self.assertEqual(report["provider_name"], "strict-provider")
            self.assertEqual(
                report["vision_error_streaks"]["maximum_length_processed_frames"],
                1,
            )
            self.assertEqual(
                report["evaluation_result_skipped_reason"],
                "recorded vision abstentions make evaluation-result.v1 schema incomplete",
            )
            self.assertEqual(
                report["standardized_evaluation_skip_reason"],
                "recorded vision abstentions make evaluation-result.v1 schema incomplete",
            )

    def test_clean_to_dirty_transition_is_provenance_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "new-output"
            arguments = self._arguments(root, output)
            digest = "c" * 64
            profile = SimpleNamespace(
                source_sha256=digest,
                source_path=str(arguments.profile),
                profile_id="test-profile",
                vision=SimpleNamespace(
                    camera_ids=("table",),
                    depth_camera_ids=(),
                    factory=(
                        "team_plugins.computer_vision."
                        "hybrid_geometry_provider:create"
                    ),
                    options={},
                    calibration_ids={"table": "declared-table"},
                    depth_calibration_ids={},
                ),
            )

            class Provider:
                PROVIDER_NAME = "test-provider"
                PROVIDER_VERSION = "1"

                def observe(self, request):
                    return SceneObservation(
                        scene_id=request.scene_id,
                        capture_time_s=request.capture_time_s,
                        objects=(),
                        provider_name=self.PROVIDER_NAME,
                        provider_version=self.PROVIDER_VERSION,
                        calibration_ids=("calibration.json:table",),
                    )

            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            with (
                patch.object(replay_camera, "_parse_args", return_value=arguments),
                patch.object(
                    replay_camera,
                    "_git_value",
                    side_effect=("a" * 40, "vision/test", "a" * 40, "vision/test"),
                ),
                patch.object(
                    replay_camera,
                    "_git_worktree_dirty",
                    side_effect=(False, True),
                ),
                patch.object(replay_camera, "_sha256", return_value=digest),
                patch.object(
                    replay_camera,
                    "load_calibration",
                    return_value={"table": {"size": [2, 2]}},
                ),
                patch.object(replay_camera, "load_integration_profile", return_value=profile),
                patch.object(replay_camera, "load_factory", return_value=lambda unused: Provider()),
                patch.object(
                    replay_camera,
                    "_frames",
                    return_value=iter(((0, 0.0, frame, 10.0),)),
                ),
                patch.object(replay_camera, "_draw_overlay", return_value=frame),
                patch.object(
                    replay_camera.cv2,
                    "resize",
                    return_value=np.zeros((540, 960, 3), dtype=np.uint8),
                ),
                patch.object(replay_camera.cv2, "VideoWriter", return_value=_FakeWriter()),
                patch.object(replay_camera.cv2, "imwrite", return_value=True),
                patch("builtins.print"),
            ):
                result = replay_camera.main()

            self.assertEqual(result, 2)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(
                report["provenance_fields_changed_during_replay"],
                ["worktree_dirty"],
            )
            self.assertFalse((output / "evaluation-result.v1.json").exists())


if __name__ == "__main__":
    unittest.main()
