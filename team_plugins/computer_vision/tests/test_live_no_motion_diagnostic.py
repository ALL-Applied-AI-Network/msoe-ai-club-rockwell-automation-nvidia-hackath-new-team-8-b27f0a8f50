from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from sorting_core.perception import SceneObservation
from team_plugins.computer_vision import live_no_motion_diagnostic as diagnostic


class _FakePolicy:
    ACTION = "tcp"
    CAMERAS = ("table",)
    IMAGE_FORMAT = "png"
    QUEUE_POLL_CAMERAS = ("table",)
    QUEUE_POLL_IMAGE_SIZE = (160, 90)

    def __init__(self, *, fail_perception: bool = False, motion: bool = False):
        self._motion_enabled = motion
        self._descent_enabled = False
        self.fail_perception = fail_perception
        self.perception_calls = 0
        self.hold_action = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0]]

    def act(self, obs):
        return self.hold_action

    def reset(self):
        return None

    def _detect_scene_views(self, obs):
        self.perception_calls += 1
        if self.fail_perception:
            raise ValueError("synthetic provider failure")
        return SimpleNamespace(
            observation=SceneObservation(
                scene_id="diagnostic-scene",
                capture_time_s=float(obs["t"]),
                provider_name="test-provider",
                provider_version="1",
                calibration_ids=("test-calibration",),
            )
        )


class _ProviderThenPartitionFailurePolicy(_FakePolicy):
    class _Provider:
        def observe(self, request):
            return SceneObservation(
                scene_id="provider-scene-before-task-partition",
                capture_time_s=12.5,
                provider_name="captured-provider",
                provider_version="1",
                calibration_ids=("test-calibration",),
            )

    def __init__(self):
        super().__init__()
        self._perception_provider = self._Provider()

    def _detect_scene_views(self, obs):
        self.perception_calls += 1
        # The recorder wraps this provider call and validates/captures its
        # output before this synthetic downstream task failure occurs.
        self._perception_provider.observe(object())
        raise ValueError("synthetic cube-task partition failure")


def _observation(camera_id: str = "table") -> dict:
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    return {
        "t": 12.5,
        "trial": {"trial_id": "test-trial"},
        "images": {camera_id: rgb},
        "joints": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25],
        "tcp_pose": [400.0, 10.0, -200.0, 180.0, 0.0, 90.0],
    }


def _recorder(path: Path, maximum: int = 2) -> diagnostic.SnapshotRecorder:
    return diagnostic.SnapshotRecorder(
        path,
        maximum_snapshots=maximum,
        cube_edge_mm=16,
        provenance={
            "commit": "a" * 40,
            "branch": "vision/test",
            "worktree_dirty": False,
            "profile_name": "profile.hybrid-geometry.v1.json",
            "profile_sha256": "b" * 64,
            "diagnostic_tool_sha256": "c" * 64,
        },
    )


class LiveNoMotionDiagnosticTests(unittest.TestCase):
    def test_camera_specs_use_each_pinned_native_rgb_size(self):
        table = diagnostic._capture_camera_spec("table")
        side = diagnostic._capture_camera_spec("side")
        wrist = diagnostic._capture_camera_spec("wrist")

        self.assertEqual(table["image_size_wh"], (1920, 1080))
        self.assertEqual(side["image_size_wh"], (1920, 1080))
        self.assertEqual(wrist["image_size_wh"], (1280, 720))
        self.assertTrue(table["provider_inference_enabled"])
        self.assertFalse(side["provider_inference_enabled"])
        self.assertFalse(wrist["provider_inference_enabled"])
        self.assertTrue(wrist["moving_camera"])
        self.assertTrue(table["trusted_for_motion_coordinates"])
        self.assertFalse(side["trusted_for_motion_coordinates"])

    def test_paired_capture_accepts_table_side_and_rejects_wrist(self):
        specs = diagnostic._capture_camera_specs(("table", "side"))

        self.assertEqual(tuple(specs), ("table", "side"))
        self.assertEqual(specs["table"]["image_size_wh"], (1920, 1080))
        self.assertEqual(specs["side"]["image_size_wh"], (1920, 1080))
        self.assertTrue(specs["table"]["provider_inference_enabled"])
        self.assertFalse(specs["side"]["provider_inference_enabled"])
        with self.assertRaisesRegex(ValueError, "table\+side"):
            diagnostic._capture_camera_specs(("table", "wrist"))

    def test_loaded_profile_validation_rejects_cached_wrong_profile(self):
        wrong_profile = SimpleNamespace(
            source_path=str(diagnostic.REPOSITORY_ROOT / "wrong-profile.json"),
            source_sha256="0" * 64,
        )
        team_integrations = SimpleNamespace(ACTIVE_INTEGRATION_PROFILE=wrong_profile)
        official_run = SimpleNamespace(Policy=SimpleNamespace(CAMERAS=("table",)))

        with self.assertRaisesRegex(RuntimeError, "different integration profile"):
            diagnostic._validate_loaded_profile(team_integrations, official_run)

    def test_repository_provenance_pins_camera_calibration_bytes(self):
        provenance = diagnostic._repository_provenance()
        self.assertEqual(
            provenance["calibration_path"],
            "hardware/fanuc_cell/calibration/cameras.json",
        )
        self.assertEqual(
            provenance["calibration_sha256"],
            diagnostic._sha256_file(diagnostic.CALIBRATION_PATH),
        )
        self.assertEqual(len(provenance["calibration_sha256"]), 64)

    def test_environment_selection_clears_every_motion_enabler(self):
        inherited = {
            diagnostic.MOTION_ENV: "1",
            diagnostic.DESCENT_ENV: "1",
            diagnostic.GRASP_FIT_ENV: "1",
        }
        with patch.dict(os.environ, inherited, clear=True):
            summary = diagnostic.configure_no_motion_environment(16)

            for name in inherited:
                self.assertNotIn(name, os.environ)
            self.assertEqual(os.environ[diagnostic.CUBE_EDGE_ENV], "16")
            self.assertEqual(
                Path(os.environ[diagnostic.PROFILE_ENV]).resolve(),
                diagnostic.PROFILE_PATH.resolve(),
            )
            self.assertFalse(summary["physical_motion_enabled"])
            self.assertFalse(summary["physical_descent_enabled"])

    def test_yolo_environment_selects_profile_and_component_checkpoint(self):
        with patch.dict(
            os.environ,
            {diagnostic.VISION_CHECKPOINT_ENV: "stale.pt"},
            clear=True,
        ):
            summary = diagnostic.configure_no_motion_environment(
                16,
                profile_path=diagnostic.YOLO11N_PROFILE_PATH,
                vision_checkpoint="seed55-best.pt",
            )
            self.assertEqual(
                Path(os.environ[diagnostic.PROFILE_ENV]).resolve(),
                diagnostic.YOLO11N_PROFILE_PATH.resolve(),
            )
            self.assertEqual(
                os.environ[diagnostic.VISION_CHECKPOINT_ENV], "seed55-best.pt"
            )
            self.assertTrue(summary["vision_checkpoint_configured"])

    def test_yolo_diagnostic_requires_checkpoint_before_loading_client(self):
        with tempfile.TemporaryDirectory() as temporary:
            loader_called = False

            def forbidden_loader():
                nonlocal loader_called
                loader_called = True
                raise AssertionError("client must not load")

            with self.assertRaises(SystemExit):
                diagnostic.main(
                    [
                        "--output-dir",
                        temporary,
                        "--cube-mm",
                        "16",
                        "--vision",
                        "yolo11n",
                    ],
                    official_run_loader=forbidden_loader,
                )
            self.assertFalse(loader_called)

    def test_yolo_diagnostic_dry_run_selects_profile_without_loading_client(self):
        with tempfile.TemporaryDirectory() as temporary:
            loader_called = False

            def forbidden_loader():
                nonlocal loader_called
                loader_called = True
                raise AssertionError("client must not load")

            output = io.StringIO()
            with patch.dict(os.environ, {}, clear=True):
                with contextlib.redirect_stdout(output):
                    result = diagnostic.main(
                        [
                            "--output-dir",
                            temporary,
                            "--cube-mm",
                            "16",
                            "--vision",
                            "yolo11n",
                            "--vision-checkpoint",
                            "seed55-best.pt",
                            "--dry-run",
                        ],
                        official_run_loader=forbidden_loader,
                    )
                self.assertEqual(
                    Path(os.environ[diagnostic.PROFILE_ENV]).resolve(),
                    diagnostic.YOLO11N_PROFILE_PATH.resolve(),
                )
                self.assertEqual(
                    os.environ[diagnostic.VISION_CHECKPOINT_ENV],
                    "seed55-best.pt",
                )

            self.assertEqual(result, 0)
            self.assertFalse(loader_called)
            summary = json.loads(output.getvalue())
            self.assertEqual(summary["vision"], "yolo11n")
            self.assertTrue(summary["vision_checkpoint_configured"])
            self.assertNotIn("seed55-best.pt", output.getvalue())

    def test_policy_returns_exact_hold_and_saves_native_scene_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = _recorder(Path(temporary))
            policy_class = diagnostic.diagnostic_policy_class(_FakePolicy, recorder)
            policy = policy_class()

            result = policy.act(_observation())

            self.assertIs(result, policy.hold_action)
            self.assertEqual(policy.perception_calls, 1)
            scene_path = Path(temporary) / "snapshot-000-scene.json"
            record_path = Path(temporary) / "snapshot-000-record.json"
            image_path = Path(temporary) / "snapshot-000-table.png"
            self.assertTrue(scene_path.is_file())
            self.assertTrue(record_path.is_file())
            self.assertTrue(image_path.is_file())
            scene = json.loads(scene_path.read_text())
            record = json.loads(record_path.read_text())
            decoded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            self.assertEqual(scene["provider_name"], "test-provider")
            self.assertEqual(record["status"], "ok")
            self.assertFalse(record["motion_planning_called"])
            self.assertEqual(
                record["robot_feedback"]["joints_deg_and_gripper_0_to_1"],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25],
            )
            self.assertEqual(
                record["robot_feedback"]["tcp_pose_uf0_mm_wpr_deg"],
                [400.0, 10.0, -200.0, 180.0, 0.0, 90.0],
            )
            self.assertEqual(
                record["robot_feedback"]["source_timestamp_s"], 12.5
            )
            self.assertEqual(
                record["robot_feedback"]["timestamp_semantics"],
                "backend_observation_time_not_camera_exposure_time",
            )
            self.assertEqual(record["frames"][0]["channel_order"], "RGB")
            self.assertEqual(
                record["frames"][0]["pixel_provenance"],
                "decoded_backend_observation_at_requested_resolution",
            )
            self.assertFalse(record["frames"][0]["sensor_native_unmodified"])
            self.assertEqual(record["frames"][0]["requested_wire_format"], "png")
            self.assertGreaterEqual(record["timing"]["hold_construction_ms"], 0.0)
            self.assertGreaterEqual(
                record["timing"]["action_return_delay_lower_bound_ms"],
                record["timing"]["perception_and_task_boundary_ms"],
            )
            self.assertEqual(decoded.shape, (6, 8, 3))
            # Source was pure RGB red; OpenCV decodes the PNG as BGR.
            self.assertEqual(decoded[0, 0].tolist(), [0, 0, 255])

    def test_auxiliary_camera_capture_skips_provider_and_records_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = {
                "camera_id": "wrist",
                "image_size_wh": (8, 6),
                "calibration_id": "sidclab-cameras.json:wrist",
                "calibration_solved": "2026-10-07",
                "calibration_rms_px": 0.9,
                "transform_semantics": "T_flange_cam: camera -> flange",
                "moving_camera": True,
                "trusted_for_motion_coordinates": False,
                "provider_inference_enabled": False,
            }
            recorder = diagnostic.SnapshotRecorder(
                Path(temporary),
                maximum_snapshots=1,
                cube_edge_mm=16,
                provenance={},
                capture_camera_spec=spec,
            )
            policy_class = diagnostic.diagnostic_policy_class(
                _FakePolicy,
                recorder,
                capture_camera_id="wrist",
                capture_image_size=(8, 6),
            )
            policy = policy_class()

            result = policy.act(_observation("wrist"))

            self.assertIs(result, policy.hold_action)
            self.assertEqual(policy.perception_calls, 0)
            self.assertEqual(policy.CAMERAS, ("wrist",))
            self.assertEqual(policy.IMAGE_SIZE, (8, 6))
            self.assertEqual(policy.QUEUE_POLL_CAMERAS, ("wrist",))
            self.assertFalse(
                (Path(temporary) / "snapshot-000-scene.json").exists()
            )
            record = json.loads(
                (Path(temporary) / "snapshot-000-record.json").read_text()
            )
            self.assertEqual(record["status"], "capture_only")
            self.assertFalse(record["provider_inference_ran"])
            self.assertEqual(
                record["frames"][0]["calibration"]["calibration_id"],
                "sidclab-cameras.json:wrist",
            )
            self.assertTrue(
                record["frames"][0]["calibration"]["moving_camera"]
            )
            self.assertFalse(
                record["frames"][0]["calibration"][
                    "trusted_for_motion_coordinates"
                ]
            )

    def test_paired_table_side_capture_runs_table_inference_only(self):
        class _CameraRecordingPolicy(_FakePolicy):
            def __init__(self):
                super().__init__()
                self.inference_camera_sets = []

            def _detect_scene_views(self, obs):
                self.inference_camera_sets.append(tuple(self.CAMERAS))
                return super()._detect_scene_views(obs)

        def spec(camera_id: str, *, inference: bool, trusted: bool) -> dict:
            return {
                "camera_id": camera_id,
                "image_size_wh": (8, 6),
                "calibration_id": f"sidclab-cameras.json:{camera_id}",
                "calibration_solved": "2026-10-07",
                "calibration_rms_px": 0.5 if camera_id == "table" else 3.99,
                "transform_semantics": "T_base_cam: camera -> base",
                "moving_camera": False,
                "trusted_for_motion_coordinates": trusted,
                "provider_inference_enabled": inference,
            }

        with tempfile.TemporaryDirectory() as temporary:
            recorder = diagnostic.SnapshotRecorder(
                Path(temporary),
                maximum_snapshots=1,
                cube_edge_mm=16,
                provenance={},
                capture_camera_specs={
                    "table": spec("table", inference=True, trusted=True),
                    "side": spec("side", inference=False, trusted=False),
                },
            )
            policy_class = diagnostic.diagnostic_policy_class(
                _CameraRecordingPolicy,
                recorder,
                capture_camera_ids=("table", "side"),
                capture_image_size=(8, 6),
            )
            policy = policy_class()
            obs = _observation()
            obs["images"]["side"] = np.full((6, 8, 3), 127, dtype=np.uint8)

            result = policy.act(obs)

            self.assertIs(result, policy.hold_action)
            self.assertEqual(policy.CAMERAS, ("table", "side"))
            self.assertEqual(policy.QUEUE_POLL_CAMERAS, ("table",))
            self.assertEqual(policy.inference_camera_sets, [("table",)])
            self.assertEqual(policy.perception_calls, 1)
            record = json.loads(
                (Path(temporary) / "snapshot-000-record.json").read_text()
            )
            self.assertEqual(record["status"], "ok")
            self.assertTrue(record["provider_inference_ran"])
            self.assertEqual(
                [frame["camera_id"] for frame in record["frames"]],
                ["table", "side"],
            )
            self.assertTrue(
                record["frames"][0]["calibration"][
                    "trusted_for_motion_coordinates"
                ]
            )
            self.assertFalse(
                record["frames"][1]["calibration"][
                    "trusted_for_motion_coordinates"
                ]
            )
            self.assertTrue(
                record["frames"][0]["calibration"]["provider_inference_enabled"]
            )
            self.assertFalse(
                record["frames"][1]["calibration"]["provider_inference_enabled"]
            )
            self.assertTrue((Path(temporary) / "snapshot-000-table.png").is_file())
            self.assertTrue((Path(temporary) / "snapshot-000-side.png").is_file())

    def test_paired_capture_requires_both_frames_without_partial_files(self):
        table_spec = {
            "camera_id": "table",
            "image_size_wh": (8, 6),
            "calibration_id": "sidclab-cameras.json:table",
            "calibration_solved": "test",
            "calibration_rms_px": 0.5,
            "transform_semantics": "T_base_cam: camera -> base",
            "moving_camera": False,
            "trusted_for_motion_coordinates": True,
            "provider_inference_enabled": True,
        }
        side_spec = {
            **table_spec,
            "camera_id": "side",
            "calibration_id": "sidclab-cameras.json:side",
            "trusted_for_motion_coordinates": False,
            "provider_inference_enabled": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            recorder = diagnostic.SnapshotRecorder(
                Path(temporary),
                maximum_snapshots=1,
                cube_edge_mm=16,
                provenance={},
                capture_camera_specs={"table": table_spec, "side": side_spec},
            )
            policy_class = diagnostic.diagnostic_policy_class(
                _FakePolicy,
                recorder,
                capture_camera_ids=("table", "side"),
                capture_image_size=(8, 6),
            )
            policy = policy_class()

            result = policy.act(_observation())

            self.assertIs(result, policy.hold_action)
            self.assertEqual(recorder.failures, 1)
            self.assertFalse((Path(temporary) / "snapshot-000-table.png").exists())
            self.assertFalse((Path(temporary) / "snapshot-000-side.png").exists())

    def test_provider_failure_is_recorded_and_hold_is_still_returned(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = _recorder(Path(temporary))
            policy_class = diagnostic.diagnostic_policy_class(_FakePolicy, recorder)
            policy = policy_class(fail_perception=True)

            result = policy.act(_observation())

            self.assertIs(result, policy.hold_action)
            self.assertEqual(recorder.failures, 1)
            self.assertFalse(
                (Path(temporary) / "snapshot-000-scene.json").exists()
            )
            record = json.loads(
                (Path(temporary) / "snapshot-000-record.json").read_text()
            )
            self.assertEqual(record["status"], "error")
            self.assertEqual(record["error"]["type"], "ValueError")
            self.assertIn("synthetic provider failure", record["error"]["message"])
            self.assertEqual(
                record["error"]["stage"], "provider_or_request_boundary"
            )

    def test_valid_provider_scene_survives_downstream_task_partition_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = _recorder(Path(temporary))
            policy_class = diagnostic.diagnostic_policy_class(
                _ProviderThenPartitionFailurePolicy, recorder
            )
            policy = policy_class()

            with patch.object(
                diagnostic,
                "validate_scene_for_request",
                side_effect=lambda _request, result: result,
            ):
                result = policy.act(_observation())

            self.assertIs(result, policy.hold_action)
            self.assertIsInstance(policy._perception_provider, policy._Provider)
            self.assertEqual(recorder.failures, 1)
            scene = json.loads(
                (Path(temporary) / "snapshot-000-scene.json").read_text()
            )
            record = json.loads(
                (Path(temporary) / "snapshot-000-record.json").read_text()
            )
            self.assertEqual(
                scene["scene_id"], "provider-scene-before-task-partition"
            )
            self.assertEqual(
                record["error"]["stage"],
                "task_partition_after_valid_provider_scene",
            )
            self.assertEqual(record["provider_name"], "captured-provider")

    def test_recording_failure_preserves_already_constructed_hold(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = _recorder(Path(temporary))
            policy_class = diagnostic.diagnostic_policy_class(_FakePolicy, recorder)
            policy = policy_class()

            with patch.object(
                recorder, "capture", side_effect=OSError("synthetic disk failure")
            ):
                result = policy.act(_observation())

            self.assertIs(result, policy.hold_action)
            self.assertEqual(recorder.failures, 1)

    def test_snapshot_limit_bounds_provider_calls_and_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = _recorder(Path(temporary), maximum=1)
            policy_class = diagnostic.diagnostic_policy_class(_FakePolicy, recorder)
            policy = policy_class()

            first = policy.act(_observation())
            second = policy.act(_observation())

            self.assertIs(first, policy.hold_action)
            self.assertIs(second, policy.hold_action)
            self.assertEqual(policy.perception_calls, 1)
            self.assertEqual(recorder.attempts, 1)
            self.assertFalse(
                (Path(temporary) / "snapshot-001-record.json").exists()
            )

    def test_adapter_refuses_policy_that_reports_motion_enabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = _recorder(Path(temporary))
            policy_class = diagnostic.diagnostic_policy_class(_FakePolicy, recorder)
            with self.assertRaisesRegex(RuntimeError, "motion/descent enabled"):
                policy_class(motion=True)

    def test_output_directory_inside_repository_is_refused(self):
        forbidden = diagnostic.REPOSITORY_ROOT / ".forbidden-live-capture-test"
        self.assertFalse(forbidden.exists())
        with self.assertRaisesRegex(ValueError, "outside the repository"):
            diagnostic.SnapshotRecorder(
                forbidden,
                maximum_snapshots=1,
                cube_edge_mm=16,
                provenance={},
            )
        self.assertFalse(forbidden.exists())

    def test_reset_rechecks_closed_interlocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = _recorder(Path(temporary))
            policy_class = diagnostic.diagnostic_policy_class(_FakePolicy, recorder)
            policy = policy_class()
            policy._motion_enabled = True

            with self.assertRaisesRegex(RuntimeError, "motion/descent enabled"):
                policy.reset()

    def test_act_rechecks_closed_interlocks_before_base_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = _recorder(Path(temporary))
            policy_class = diagnostic.diagnostic_policy_class(_FakePolicy, recorder)
            policy = policy_class()
            policy._descent_enabled = True

            with self.assertRaisesRegex(RuntimeError, "motion/descent enabled"):
                policy.act(_observation())
            self.assertEqual(policy.perception_calls, 0)
            self.assertEqual(recorder.attempts, 0)

    def test_dry_run_does_not_import_or_contact_official_client(self):
        with tempfile.TemporaryDirectory() as temporary:
            loader_called = False

            def forbidden_loader():
                nonlocal loader_called
                loader_called = True
                raise AssertionError("official client must not load during dry-run")

            with patch.dict(os.environ, {}, clear=True):
                result = diagnostic.main(
                    [
                        "--output-dir",
                        temporary,
                        "--cube-mm",
                        "10",
                        "--dry-run",
                    ],
                    official_run_loader=forbidden_loader,
                )

            self.assertEqual(result, 0)
            self.assertFalse(loader_called)

    def test_wrist_dry_run_reports_native_capture_without_provider_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = diagnostic.main(
                    [
                        "--output-dir",
                        temporary,
                        "--cube-mm",
                        "10",
                        "--capture-camera",
                        "wrist",
                        "--dry-run",
                    ],
                    official_run_loader=lambda: self.fail(
                        "dry-run must not load the official client"
                    ),
                )

            self.assertEqual(result, 0)
            summary = json.loads(output.getvalue())
            self.assertEqual(summary["capture_camera"], "wrist")
            self.assertEqual(summary["capture_image_size_wh"], [1280, 720])
            self.assertFalse(summary["provider_inference_enabled"])
            self.assertFalse(summary["trusted_for_motion_coordinates"])

    def test_table_side_dry_run_reports_table_only_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = diagnostic.main(
                    [
                        "--output-dir",
                        temporary,
                        "--cube-mm",
                        "16",
                        "--capture-camera",
                        "table",
                        "--aux-camera",
                        "side",
                        "--dry-run",
                    ],
                    official_run_loader=lambda: self.fail(
                        "dry-run must not load the official client"
                    ),
                )

            self.assertEqual(result, 0)
            summary = json.loads(output.getvalue())
            self.assertEqual(summary["capture_cameras"], ["table", "side"])
            self.assertEqual(summary["provider_inference_cameras"], ["table"])
            self.assertEqual(summary["auxiliary_capture_only_cameras"], ["side"])
            self.assertEqual(
                summary["trusted_for_motion_coordinate_cameras"], ["table"]
            )
            self.assertFalse(summary["trusted_for_motion_coordinates"])
            self.assertEqual(summary["capture_image_size_wh"], [1920, 1080])

    def test_table_wrist_pair_is_rejected_before_loading_client(self):
        with tempfile.TemporaryDirectory() as temporary:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    diagnostic.main(
                        [
                            "--output-dir",
                            temporary,
                            "--cube-mm",
                            "16",
                            "--capture-camera",
                            "table",
                            "--aux-camera",
                            "wrist",
                            "--dry-run",
                        ],
                        official_run_loader=lambda: self.fail(
                            "invalid camera pairing must not load the client"
                        ),
                    )

    def test_main_selects_profile_before_loader_and_restores_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            fake_run = SimpleNamespace(Policy=_FakePolicy)
            original_policy = fake_run.Policy
            observed: dict[str, object] = {}

            def fake_main():
                observed["profile"] = os.environ.get(diagnostic.PROFILE_ENV)
                observed["motion"] = os.environ.get(diagnostic.MOTION_ENV)
                policy = fake_run.Policy()
                observed["action"] = policy.act(_observation())

            fake_run.main = fake_main
            with patch.dict(
                os.environ,
                {
                    diagnostic.MOTION_ENV: "1",
                    diagnostic.DESCENT_ENV: "1",
                },
                clear=True,
            ):
                with patch.object(
                    diagnostic,
                    "_capture_camera_specs",
                    return_value={
                        "table": {
                            "camera_id": "table",
                            "image_size_wh": (8, 6),
                            "calibration_id": "sidclab-cameras.json:table",
                            "calibration_solved": "test",
                            "calibration_rms_px": 0.5,
                            "transform_semantics": "T_base_cam: camera -> base",
                            "moving_camera": False,
                            "trusted_for_motion_coordinates": True,
                            "provider_inference_enabled": True,
                        }
                    },
                ):
                    result = diagnostic.main(
                        [
                            "--output-dir",
                            temporary,
                            "--cube-mm",
                            "16",
                            "--max-snapshots",
                            "1",
                        ],
                        official_run_loader=lambda: fake_run,
                    )

            self.assertEqual(result, 0)
            self.assertEqual(
                Path(str(observed["profile"])).resolve(),
                diagnostic.PROFILE_PATH.resolve(),
            )
            self.assertIsNone(observed["motion"])
            self.assertEqual(observed["action"], fake_run.Policy().hold_action)
            self.assertIs(fake_run.Policy, original_policy)


if __name__ == "__main__":
    unittest.main()
