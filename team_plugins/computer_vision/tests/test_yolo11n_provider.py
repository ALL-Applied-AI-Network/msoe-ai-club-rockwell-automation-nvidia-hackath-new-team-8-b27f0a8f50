from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sorting_core import load_config
from sorting_core.contracts import CubeDetection, SceneDetections
from sorting_core.integration_plugins import (
    VisionFactoryContext,
    load_integration_profile,
)
from sorting_core.perception import CameraFrame, PerceptionRequest
from sorting_core.vision import VisionError
from team_plugins.computer_vision import yolo11n_provider
from team_plugins.computer_vision.yolo11n_provider import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CLASS_NAMES,
    EXPECTED_CONFIDENCE_THRESHOLD,
    EXPECTED_CROP_XYXY,
    EXPECTED_IMAGE_SIZE,
    EXPECTED_INFERENCE_IMAGE_SIZE,
    EXPECTED_WARMUP_PASSES,
    Yolo11nCubeVisionProvider,
    create,
)


class FakeBoxes:
    def __init__(self, xyxy=(), confidence=(), classes=()):
        if len(xyxy):
            self.xyxy = np.asarray(xyxy, dtype=np.float32)
        else:
            self.xyxy = np.empty((0, 4), dtype=np.float32)
        self.conf = np.asarray(confidence, dtype=np.float32)
        self.cls = np.asarray(classes, dtype=np.float32)


class FakeResult:
    def __init__(self, boxes: FakeBoxes | None):
        self.boxes = boxes


class FakeModel:
    def __init__(self, boxes: FakeBoxes | None = None, *, names=None):
        self.names = dict(EXPECTED_CLASS_NAMES if names is None else names)
        self.boxes = FakeBoxes() if boxes is None else boxes
        self.calls: list[dict] = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [FakeResult(self.boxes)]


def projector(uv, z_m):
    u, v = uv
    return np.asarray((0.2 + float(u) * 0.00025, float(v) * 0.00025, z_m))


def request(
    image: np.ndarray,
    *,
    scene_id: str = "scene-yolo",
    capture_time_s: float = 4.0,
    frame_capture_time_s: float | None = None,
    calibration_id: str = "test-table-v1",
) -> PerceptionRequest:
    if frame_capture_time_s is None:
        frame_capture_time_s = capture_time_s
    return PerceptionRequest(
        scene_id=scene_id,
        capture_time_s=capture_time_s,
        frames=(
            CameraFrame(
                camera_id="table",
                capture_time_s=frame_capture_time_s,
                calibration_id=calibration_id,
                rgb=image,
            ),
        ),
    )


class Yolo11nProviderTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def provider(
        self,
        model: FakeModel,
        detector=None,
        *,
        geometry_agreement_scope: str = "all_inference_crop",
    ) -> Yolo11nCubeVisionProvider:
        return Yolo11nCubeVisionProvider(
            config=self.config,
            camera_projectors={"table": projector},
            calibration_ids={"table": "test-table-v1"},
            model=model,
            checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
            expected_cube_edge_m=0.010,
            detector=detector,
            geometry_agreement_scope=geometry_agreement_scope,
        )

    def test_cube_box_corroborates_full_frame_geometry_and_converts_rgb_to_bgr(self):
        model = FakeModel(
            FakeBoxes(
                xyxy=((100.0, 100.0, 180.0, 180.0),),
                confidence=(0.9,),
                classes=(0,),
            )
        )
        detector_calls = []

        def detector(rgb, injected_projector, config, **kwargs):
            detector_calls.append((rgb, injected_projector, config, kwargs))
            np.testing.assert_array_equal(rgb[540, 640], (10, 20, 30))
            np.testing.assert_array_equal(rgb[100, 100], (10, 20, 30))
            return SceneDetections(
                scene_id=kwargs["scene_id"],
                capture_time_s=kwargs["capture_time_s"],
                cubes=(
                    CubeDetection(
                        track_id="inside-red",
                        colour="red",
                        capture_time_s=kwargs["capture_time_s"],
                        position_base_m=(0.36, 0.02, 0.005),
                        yaw_rad=0.2,
                        confidence=0.8,
                        pixel_center_uv=(640.0, 540.0),
                        mask_area_px=150.0,
                    ),
                ),
            )

        image = np.empty((1080, 1920, 3), dtype=np.uint8)
        image[:] = (10, 20, 30)
        scene = self.provider(model, detector).observe(request(image))

        self.assertEqual(len(model.calls), 1)
        call = model.calls[0]
        self.assertEqual(call["source"].shape, (650, 1000, 3))
        np.testing.assert_array_equal(call["source"][0, 0], (30, 20, 10))
        self.assertEqual(call["imgsz"], EXPECTED_INFERENCE_IMAGE_SIZE)
        self.assertEqual(call["conf"], EXPECTED_CONFIDENCE_THRESHOLD)
        self.assertEqual(call["iou"], 0.7)
        self.assertFalse(call["verbose"])
        self.assertEqual(len(detector_calls), 1)
        self.assertEqual(len(scene.objects), 1)
        detected = scene.objects[0]
        self.assertEqual(detected.track_id, "inside-red")
        self.assertEqual(detected.class_id, "cube")
        self.assertEqual(detected.sort_label, "red")
        self.assertAlmostEqual(detected.confidence, 0.8)
        self.assertEqual(
            detected.attributes["yolo_bbox_xyxy_full"],
            (600.0, 500.0, 680.0, 580.0),
        )
        self.assertAlmostEqual(detected.attributes["yolo_confidence"], 0.9, places=6)
        self.assertEqual(detected.attributes["yolo_status"], "corroborated")
        self.assertEqual(
            detected.attributes["checkpoint_sha256"], EXPECTED_CHECKPOINT_SHA256
        )
        self.assertIn(EXPECTED_CHECKPOINT_SHA256, scene.provider_version)

    def test_screw_prediction_abstains_without_calling_geometry(self):
        model = FakeModel(
            FakeBoxes(
                xyxy=((50.0, 50.0, 100.0, 100.0),),
                confidence=(0.95,),
                classes=(1,),
            )
        )

        def detector(*args, **kwargs):
            self.fail("cube geometry must not run after an unvalidated screw result")

        with self.assertRaisesRegex(VisionError, "unvalidated screw"):
            self.provider(model, detector).observe(
                request(np.zeros((1080, 1920, 3), dtype=np.uint8))
            )

    def test_empty_prediction_returns_valid_empty_scene(self):
        model = FakeModel()

        def detector(rgb, projector, config, **kwargs):
            return SceneDetections(
                scene_id=kwargs["scene_id"],
                capture_time_s=kwargs["capture_time_s"],
                cubes=(),
            )

        scene = self.provider(model, detector).observe(
            request(np.zeros((1080, 1920, 3), dtype=np.uint8))
        )
        self.assertEqual(scene.objects, ())
        self.assertEqual(scene.scene_id, "scene-yolo")
        self.assertEqual(scene.calibration_ids, ("test-table-v1",))

    def test_unmatched_yolo_box_fails_visible_abstention(self):
        model = FakeModel(
            FakeBoxes(
                xyxy=((100.0, 100.0, 180.0, 180.0),),
                confidence=(0.9,),
                classes=(0,),
            )
        )

        def detector(rgb, projector, config, **kwargs):
            return SceneDetections(
                scene_id=kwargs["scene_id"],
                capture_time_s=kwargs["capture_time_s"],
                cubes=(),
            )

        with self.assertRaisesRegex(VisionError, "unmatched_yolo_boxes=1"):
            self.provider(model, detector).observe(
                request(np.zeros((1080, 1920, 3), dtype=np.uint8))
            )

    def test_unmatched_in_crop_geometry_fails_visible_abstention(self):
        model = FakeModel()

        def detector(rgb, projector, config, **kwargs):
            return SceneDetections(
                scene_id=kwargs["scene_id"],
                capture_time_s=kwargs["capture_time_s"],
                cubes=(
                    CubeDetection(
                        track_id="missed-red",
                        colour="red",
                        capture_time_s=kwargs["capture_time_s"],
                        position_base_m=(0.36, 0.02, 0.005),
                        yaw_rad=0.0,
                        pixel_center_uv=(640.0, 540.0),
                    ),
                ),
            )

        with self.assertRaisesRegex(VisionError, "unmatched_geometry_detections=1"):
            self.provider(model, detector).observe(
                request(np.zeros((1080, 1920, 3), dtype=np.uint8))
            )

    def test_strict_scope_rejects_unmatched_off_pickup_geometry_in_crop(self):
        model = FakeModel()

        def detector(rgb, projector, config, **kwargs):
            return SceneDetections(
                scene_id=kwargs["scene_id"],
                capture_time_s=kwargs["capture_time_s"],
                cubes=(
                    CubeDetection(
                        track_id="off-pickup-artifact",
                        colour="yellow",
                        capture_time_s=kwargs["capture_time_s"],
                        position_base_m=(0.20, 0.02, 0.005),
                        yaw_rad=0.0,
                        pixel_center_uv=(640.0, 540.0),
                    ),
                ),
            )

        with self.assertRaisesRegex(VisionError, "unmatched_geometry_detections=1"):
            self.provider(model, detector).observe(
                request(np.zeros((1080, 1920, 3), dtype=np.uint8))
            )

    def test_pickup_scope_preserves_unmatched_off_pickup_geometry_in_crop(self):
        model = FakeModel()

        def detector(rgb, projector, config, **kwargs):
            return SceneDetections(
                scene_id=kwargs["scene_id"],
                capture_time_s=kwargs["capture_time_s"],
                cubes=(
                    CubeDetection(
                        track_id="off-pickup-artifact",
                        colour="yellow",
                        capture_time_s=kwargs["capture_time_s"],
                        position_base_m=(0.20, 0.02, 0.005),
                        yaw_rad=0.0,
                        confidence=0.7,
                        pixel_center_uv=(640.0, 540.0),
                    ),
                ),
            )

        scene = self.provider(
            model,
            detector,
            geometry_agreement_scope="pickup_region",
        ).observe(request(np.zeros((1080, 1920, 3), dtype=np.uint8)))

        self.assertEqual(len(scene.objects), 1)
        detected = scene.objects[0]
        self.assertEqual(
            detected.attributes["yolo_status"],
            "inside_inference_crop_outside_pickup_geometry_only",
        )
        self.assertFalse(detected.attributes["in_pickup_region"])
        self.assertEqual(
            detected.attributes["geometry_agreement_scope"], "pickup_region"
        )
        self.assertNotIn("yolo_confidence", detected.attributes)

    def test_pickup_scope_still_rejects_unmatched_pickup_geometry(self):
        model = FakeModel()

        def detector(rgb, projector, config, **kwargs):
            return SceneDetections(
                scene_id=kwargs["scene_id"],
                capture_time_s=kwargs["capture_time_s"],
                cubes=(
                    CubeDetection(
                        track_id="missed-pickup-red",
                        colour="red",
                        capture_time_s=kwargs["capture_time_s"],
                        position_base_m=(0.36, 0.02, 0.005),
                        yaw_rad=0.0,
                        pixel_center_uv=(640.0, 540.0),
                    ),
                ),
            )

        with self.assertRaisesRegex(VisionError, "unmatched_geometry_detections=1"):
            self.provider(
                model,
                detector,
                geometry_agreement_scope="pickup_region",
            ).observe(request(np.zeros((1080, 1920, 3), dtype=np.uint8)))

    def test_pickup_scope_still_rejects_unmatched_yolo_box(self):
        model = FakeModel(
            FakeBoxes(
                xyxy=((100.0, 100.0, 180.0, 180.0),),
                confidence=(0.9,),
                classes=(0,),
            )
        )

        def detector(rgb, projector, config, **kwargs):
            return SceneDetections(
                scene_id=kwargs["scene_id"],
                capture_time_s=kwargs["capture_time_s"],
                cubes=(),
            )

        with self.assertRaisesRegex(VisionError, "unmatched_yolo_boxes=1"):
            self.provider(
                model,
                detector,
                geometry_agreement_scope="pickup_region",
            ).observe(request(np.zeros((1080, 1920, 3), dtype=np.uint8)))

    def test_pickup_geometry_wins_box_assignment_over_closer_optional_geometry(self):
        model = FakeModel(
            FakeBoxes(
                xyxy=((100.0, 100.0, 180.0, 180.0),),
                confidence=(0.9,),
                classes=(0,),
            )
        )

        def detector(rgb, projector, config, **kwargs):
            common = {
                "colour": "red",
                "capture_time_s": kwargs["capture_time_s"],
                "yaw_rad": 0.0,
            }
            return SceneDetections(
                scene_id=kwargs["scene_id"],
                capture_time_s=kwargs["capture_time_s"],
                cubes=(
                    CubeDetection(
                        track_id="closer-off-pickup",
                        position_base_m=(0.20, 0.02, 0.005),
                        pixel_center_uv=(640.0, 540.0),
                        **common,
                    ),
                    CubeDetection(
                        track_id="required-pickup",
                        position_base_m=(0.36, 0.02, 0.005),
                        pixel_center_uv=(645.0, 540.0),
                        **common,
                    ),
                ),
            )

        scene = self.provider(
            model,
            detector,
            geometry_agreement_scope="pickup_region",
        ).observe(request(np.zeros((1080, 1920, 3), dtype=np.uint8)))
        by_id = {detected.track_id: detected for detected in scene.objects}
        self.assertEqual(
            by_id["closer-off-pickup"].attributes["yolo_status"],
            "inside_inference_crop_outside_pickup_geometry_only",
        )
        self.assertEqual(
            by_id["required-pickup"].attributes["yolo_status"], "corroborated"
        )
        self.assertIn(
            "yolo_confidence", by_id["required-pickup"].attributes
        )

    def test_outside_crop_destination_geometry_is_preserved_without_yolo(self):
        model = FakeModel()

        def detector(rgb, projector, config, **kwargs):
            return SceneDetections(
                scene_id=kwargs["scene_id"],
                capture_time_s=kwargs["capture_time_s"],
                cubes=(
                    CubeDetection(
                        track_id="destination-red",
                        colour="red",
                        capture_time_s=kwargs["capture_time_s"],
                        position_base_m=(0.20, 0.02, 0.005),
                        yaw_rad=0.0,
                        confidence=0.75,
                        pixel_center_uv=(1600.0, 900.0),
                    ),
                ),
            )

        scene = self.provider(model, detector).observe(
            request(np.zeros((1080, 1920, 3), dtype=np.uint8))
        )
        self.assertEqual(len(scene.objects), 1)
        detected = scene.objects[0]
        self.assertEqual(
            detected.attributes["yolo_status"],
            "outside_inference_crop_geometry_only",
        )
        self.assertNotIn("yolo_confidence", detected.attributes)

    def test_pick_eligible_geometry_outside_crop_fails_closed(self):
        model = FakeModel()

        def detector(rgb, projector, config, **kwargs):
            return SceneDetections(
                scene_id=kwargs["scene_id"],
                capture_time_s=kwargs["capture_time_s"],
                cubes=(
                    CubeDetection(
                        track_id="uncovered-blue",
                        colour="blue",
                        capture_time_s=kwargs["capture_time_s"],
                        position_base_m=(0.40, 0.02, 0.005),
                        yaw_rad=0.0,
                        pixel_center_uv=(1600.0, 900.0),
                    ),
                ),
            )

        with self.assertRaisesRegex(VisionError, "pick-eligible cube outside"):
            self.provider(model, detector).observe(
                request(np.zeros((1080, 1920, 3), dtype=np.uint8))
            )

    def test_rejects_wrong_class_map(self):
        model = FakeModel(names={0: "cube"})
        with self.assertRaisesRegex(ValueError, "class map"):
            self.provider(model)

    def test_rejects_invalid_geometry_agreement_scope(self):
        for invalid in ("whole_scene", True, None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "geometry_agreement_scope"):
                    self.provider(
                        FakeModel(),
                        geometry_agreement_scope=invalid,
                    )

    def test_rejects_bad_image_calibration_and_timestamp(self):
        model = FakeModel()
        provider = self.provider(model)
        with self.subTest("resolution"):
            with self.assertRaisesRegex(VisionError, "1920x1080"):
                provider.observe(request(np.zeros((720, 1280, 3), dtype=np.uint8)))
        with self.subTest("calibration"):
            with self.assertRaisesRegex(VisionError, "calibration mismatch"):
                provider.observe(
                    request(
                        np.zeros((1080, 1920, 3), dtype=np.uint8),
                        calibration_id="wrong-calibration",
                    )
                )
        with self.subTest("stale"):
            with self.assertRaisesRegex(VisionError, "frame is stale"):
                provider.observe(
                    request(
                        np.zeros((1080, 1920, 3), dtype=np.uint8),
                        capture_time_s=5.0,
                        frame_capture_time_s=4.0,
                    )
                )

    def test_rejects_malformed_model_box(self):
        model = FakeModel(
            FakeBoxes(
                xyxy=((-1.0, 0.0, 20.0, 20.0),),
                confidence=(0.9,),
                classes=(0,),
            )
        )
        with self.assertRaisesRegex(VisionError, "outside the pinned crop"):
            self.provider(model).observe(
                request(np.zeros((1080, 1920, 3), dtype=np.uint8))
            )

    def factory_context(self, checkpoint: Path) -> VisionFactoryContext:
        return VisionFactoryContext(
            config=self.config,
            camera_projectors={"table": projector},
            calibration_ids={"table": "test-table-v1"},
            expected_cube_edge_m=0.010,
            checkpoint=str(checkpoint),
            detector=None,
            options={
                "camera_id": "table",
                "maximum_frame_age_s": 0.15,
                "expected_image_size": list(EXPECTED_IMAGE_SIZE),
                "crop_xyxy": list(EXPECTED_CROP_XYXY),
                "inference_image_size": EXPECTED_INFERENCE_IMAGE_SIZE,
                "confidence_threshold": EXPECTED_CONFIDENCE_THRESHOLD,
                "iou_threshold": 0.7,
                "roi_margin_px": 4,
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            },
            camera_ids=("table",),
            depth_camera_ids=(),
            declared_calibration_ids={"table": "test-table-v1"},
            declared_depth_calibration_ids={},
            profile_id="test-yolo-profile",
            profile_path="/external/test-profile.json",
            profile_sha256="a" * 64,
        )

    def test_factory_verifies_hash_before_lazy_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.write_bytes(b"fake-checkpoint")
            model = FakeModel()
            with (
                patch.object(
                    yolo11n_provider,
                    "_sha256_file",
                    return_value=EXPECTED_CHECKPOINT_SHA256,
                ) as hash_file,
                patch.object(
                    yolo11n_provider, "_load_yolo_model", return_value=model
                ) as load_model,
            ):
                provider = create(self.factory_context(checkpoint))
            hash_file.assert_called_once_with(checkpoint.resolve())
            load_model.assert_called_once_with(checkpoint.resolve())
            self.assertIsInstance(provider, Yolo11nCubeVisionProvider)
            self.assertEqual(len(model.calls), EXPECTED_WARMUP_PASSES)
            self.assertTrue(
                all(
                    call["source"].shape == (650, 1000, 3)
                    for call in model.calls
                )
            )
            self.assertGreaterEqual(provider._warmup_latency_ms, 0.0)
            self.assertEqual(
                len(provider._warmup_pass_latencies_ms), EXPECTED_WARMUP_PASSES
            )

    def test_factory_warmup_failure_is_visible_before_live_observation(self):
        class FailingModel(FakeModel):
            def predict(self, **kwargs):
                raise RuntimeError("synthetic backend initialization failure")

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.write_bytes(b"fake-checkpoint")
            with (
                patch.object(
                    yolo11n_provider,
                    "_sha256_file",
                    return_value=EXPECTED_CHECKPOINT_SHA256,
                ),
                patch.object(
                    yolo11n_provider,
                    "_load_yolo_model",
                    return_value=FailingModel(),
                ),
            ):
                with self.assertRaisesRegex(VisionError, "inference failed"):
                    create(self.factory_context(checkpoint))

    def test_factory_propagates_pickup_region_agreement_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.write_bytes(b"fake-checkpoint")
            context = self.factory_context(checkpoint)
            context = replace(
                context,
                options={
                    **dict(context.options),
                    "geometry_agreement_scope": "pickup_region",
                },
            )
            with (
                patch.object(
                    yolo11n_provider,
                    "_sha256_file",
                    return_value=EXPECTED_CHECKPOINT_SHA256,
                ),
                patch.object(
                    yolo11n_provider,
                    "_load_yolo_model",
                    return_value=FakeModel(),
                ),
            ):
                provider = create(context)
            self.assertEqual(provider._geometry_agreement_scope, "pickup_region")

    def test_factory_rejects_hash_mismatch_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.write_bytes(b"wrong-checkpoint")
            with (
                patch.object(
                    yolo11n_provider, "_sha256_file", return_value="b" * 64
                ),
                patch.object(yolo11n_provider, "_load_yolo_model") as load_model,
            ):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    create(self.factory_context(checkpoint))
            load_model.assert_not_called()

    def test_checked_in_profile_pins_the_evaluated_artifact(self):
        profile_path = Path(yolo11n_provider.__file__).with_name(
            "profile.yolo11n-cube.v1.json"
        )
        profile = load_integration_profile(profile_path)
        self.assertEqual(profile.profile_id, "vision-yolo11n-seed55-cube-geometry-v1")
        self.assertIsNotNone(profile.vision)
        options = dict(profile.vision.options)
        self.assertEqual(options["checkpoint_sha256"], EXPECTED_CHECKPOINT_SHA256)
        self.assertEqual(tuple(options["expected_image_size"]), EXPECTED_IMAGE_SIZE)
        self.assertEqual(tuple(options["crop_xyxy"]), EXPECTED_CROP_XYXY)
        self.assertEqual(
            options["confidence_threshold"], EXPECTED_CONFIDENCE_THRESHOLD
        )
        self.assertNotIn("geometry_agreement_scope", options)

    def test_pickup_gate_profile_is_separate_and_pins_same_artifact(self):
        profile_path = Path(yolo11n_provider.__file__).with_name(
            "profile.yolo11n-cube-pickup-gate.v1.json"
        )
        profile = load_integration_profile(profile_path)
        self.assertEqual(
            profile.profile_id,
            "vision-yolo11n-seed55-cube-pickup-gate-v1",
        )
        self.assertIsNotNone(profile.vision)
        options = dict(profile.vision.options)
        self.assertEqual(options["checkpoint_sha256"], EXPECTED_CHECKPOINT_SHA256)
        self.assertEqual(options["geometry_agreement_scope"], "pickup_region")


if __name__ == "__main__":
    unittest.main()
