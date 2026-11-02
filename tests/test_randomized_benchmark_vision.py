from __future__ import annotations

import hashlib
import unittest

import numpy as np

from experiments.randomized_sort_benchmark.vision import (
    BenchmarkVisionAbstention,
    BenchmarkVisionConfig,
    BenchmarkVisionProvider,
    DETERMINISTIC_HYBRID,
    PINNED_YOLO11N_SHA256,
    YOLO_ASSISTED_HYBRID,
    load_benchmark_vision_profile,
)
from sorting_core.perception import (
    CameraFrame,
    DetectedObject,
    PerceptionRequest,
    Pose3D,
    SceneObservation,
)
from sorting_core.vision import VisionError


class _GeometryProvider:
    PROVIDER_NAME = "fake-hybrid-geometry"
    PROVIDER_VERSION = "test-1"

    def __init__(self, objects: tuple[DetectedObject, ...]) -> None:
        self._objects = objects
        self.calls = 0

    def observe(self, request: PerceptionRequest) -> SceneObservation:
        self.calls += 1
        objects = tuple(
            DetectedObject(
                track_id=value.track_id,
                class_id=value.class_id,
                sort_label=value.sort_label,
                capture_time_s=request.capture_time_s,
                pose=value.pose,
                dimensions_m=value.dimensions_m,
                confidence=value.confidence,
                pose_quality=value.pose_quality,
                visibility=value.visibility,
                source_camera_ids=value.source_camera_ids,
                attributes=dict(value.attributes),
            )
            for value in self._objects
        )
        return SceneObservation(
            scene_id=request.scene_id,
            capture_time_s=request.capture_time_s,
            objects=objects,
            provider_name=self.PROVIDER_NAME,
            provider_version=self.PROVIDER_VERSION,
            calibration_ids=("cal-table",),
        )


class _Boxes:
    def __init__(self, values: tuple[tuple[int, float, tuple[float, ...]], ...]):
        self.cls = np.asarray([value[0] for value in values], dtype=float)
        self.conf = np.asarray([value[1] for value in values], dtype=float)
        self.xyxy = np.asarray([value[2] for value in values], dtype=float)
        if not values:
            self.xyxy = np.empty((0, 4), dtype=float)


class _Result:
    def __init__(self, values: tuple[tuple[int, float, tuple[float, ...]], ...]):
        self.boxes = _Boxes(values)


class _Model:
    names = {0: "cube", 1: "screw_iso10642_m6x30"}

    def __init__(self, values: tuple[tuple[int, float, tuple[float, ...]], ...]):
        self.values = values
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [_Result(self.values)]


def _object(
    track_id: str,
    class_id: str,
    centre: tuple[float, float],
    *,
    confidence: float = 0.9,
) -> DetectedObject:
    return DetectedObject(
        track_id=track_id,
        class_id=class_id,
        sort_label=("red" if class_id != "cube" else "blue"),
        capture_time_s=12.0,
        pose=Pose3D.from_yaw((0.4, -0.1, 0.02), 0.2),
        dimensions_m=(0.01, 0.01, 0.01),
        confidence=confidence,
        source_camera_ids=("table",),
        attributes={"pixel_center_uv": centre, "original_evidence": "kept"},
    )


class RandomizedBenchmarkVisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.image[500, 600] = (10, 20, 30)
        self.request = PerceptionRequest(
            scene_id="condition-17:observation-0",
            capture_time_s=12.0,
            frames=(
                CameraFrame(
                    camera_id="table",
                    capture_time_s=12.0,
                    calibration_id="cal-table",
                    rgb=self.image,
                ),
            ),
        )
        self.cube = _object("cube-0", "cube", (600.0, 500.0), confidence=0.92)
        self.screw = _object(
            "screw-0",
            "screw_iso10642_m6x30",
            (800.0, 600.0),
            confidence=0.88,
        )
        self.unknown = _object(
            "unknown-0", "unknown_obstacle", (1000.0, 700.0), confidence=0.4
        )

    def test_deterministic_mode_preserves_geometry_and_records_input_digest(self):
        geometry = _GeometryProvider((self.cube, self.screw, self.unknown))
        provider = BenchmarkVisionProvider(
            geometry_provider=geometry,
            config=BenchmarkVisionConfig(mode=DETERMINISTIC_HYBRID),
        )

        result = provider.observe_with_report(self.request)

        self.assertEqual(geometry.calls, 1)
        self.assertEqual(result.report.status, "accepted")
        self.assertEqual(result.report.geometry_supported_count, 2)
        self.assertEqual(result.report.geometry_unknown_count, 1)
        self.assertEqual(result.report.yolo_box_count, 0)
        self.assertEqual(
            result.report.input_rgb_sha256,
            hashlib.sha256(np.ascontiguousarray(self.image).tobytes()).hexdigest(),
        )
        self.assertEqual(
            tuple(value.pose for value in result.observation.objects),
            (self.cube.pose, self.screw.pose, self.unknown.pose),
        )
        self.assertTrue(
            all(
                value.attributes["neural_gate_status"] == "not_requested"
                for value in result.observation.objects
            )
        )
        self.assertEqual(
            result.observation.objects[0].attributes["original_evidence"], "kept"
        )

    def test_yolo_mode_corroborates_cube_and_screw_without_changing_pose(self):
        # Boxes are crop-relative.  Crop origin is full-frame (500, 400).
        model = _Model(
            (
                (0, 0.86, (80.0, 80.0, 120.0, 120.0)),
                (1, 0.81, (280.0, 180.0, 320.0, 220.0)),
            )
        )
        geometry = _GeometryProvider((self.cube, self.screw, self.unknown))
        provider = BenchmarkVisionProvider(
            geometry_provider=geometry,
            yolo_model=model,
            config=BenchmarkVisionConfig(
                mode=YOLO_ASSISTED_HYBRID,
                checkpoint_sha256=PINNED_YOLO11N_SHA256,
            ),
        )

        result = provider.observe_with_report(self.request)

        self.assertEqual(result.report.status, "accepted")
        self.assertEqual(result.report.yolo_box_count, 2)
        self.assertEqual(result.report.matched_count, 2)
        self.assertEqual(len(model.calls), 1)
        source = model.calls[0]["source"]
        self.assertEqual(source.shape, (650, 1000, 3))
        self.assertEqual(result.observation.objects[0].pose, self.cube.pose)
        self.assertEqual(result.observation.objects[1].pose, self.screw.pose)
        self.assertAlmostEqual(result.observation.objects[0].confidence, 0.86)
        self.assertAlmostEqual(result.observation.objects[1].confidence, 0.81)
        self.assertEqual(
            result.observation.objects[1].attributes["metric_pose_source"],
            "calibrated_hybrid_geometry",
        )
        self.assertEqual(
            result.observation.objects[2].attributes["neural_gate_status"],
            "geometry_only_non_target_obstacle",
        )

    def test_missing_screw_box_abstains_with_structured_report(self):
        model = _Model(((0, 0.86, (80.0, 80.0, 120.0, 120.0)),))
        provider = BenchmarkVisionProvider(
            geometry_provider=_GeometryProvider((self.cube, self.screw)),
            yolo_model=model,
            config=BenchmarkVisionConfig(
                mode=YOLO_ASSISTED_HYBRID,
                checkpoint_sha256=PINNED_YOLO11N_SHA256,
            ),
        )

        with self.assertRaises(BenchmarkVisionAbstention) as caught:
            provider.observe(self.request)

        self.assertEqual(caught.exception.report.status, "abstained")
        self.assertEqual(caught.exception.report.geometry_supported_count, 2)
        self.assertEqual(caught.exception.report.yolo_box_count, 1)
        self.assertIn("screw-0", caught.exception.report.failure_reason)
        self.assertIs(provider.last_report, caught.exception.report)

    def test_unmatched_yolo_false_positive_abstains(self):
        model = _Model(
            (
                (0, 0.86, (80.0, 80.0, 120.0, 120.0)),
                (1, 0.81, (400.0, 300.0, 440.0, 340.0)),
            )
        )
        provider = BenchmarkVisionProvider(
            geometry_provider=_GeometryProvider((self.cube,)),
            yolo_model=model,
            config=BenchmarkVisionConfig(
                mode=YOLO_ASSISTED_HYBRID,
                checkpoint_sha256=PINNED_YOLO11N_SHA256,
            ),
        )

        with self.assertRaises(BenchmarkVisionAbstention) as caught:
            provider.observe(self.request)

        self.assertIn("unmatched_yolo", str(caught.exception))

    def test_modes_record_identical_digest_for_same_request(self):
        deterministic = BenchmarkVisionProvider(
            geometry_provider=_GeometryProvider((self.cube, self.screw)),
            config=BenchmarkVisionConfig(mode=DETERMINISTIC_HYBRID),
        ).observe_with_report(self.request)
        yolo = BenchmarkVisionProvider(
            geometry_provider=_GeometryProvider((self.cube, self.screw)),
            yolo_model=_Model(
                (
                    (0, 0.86, (80.0, 80.0, 120.0, 120.0)),
                    (1, 0.81, (280.0, 180.0, 320.0, 220.0)),
                )
            ),
            config=BenchmarkVisionConfig(
                mode=YOLO_ASSISTED_HYBRID,
                checkpoint_sha256=PINNED_YOLO11N_SHA256,
            ),
        ).observe_with_report(self.request)

        self.assertEqual(
            deterministic.report.input_rgb_sha256,
            yolo.report.input_rgb_sha256,
        )

    def test_wrong_shape_is_rejected_before_geometry_or_model(self):
        geometry = _GeometryProvider((self.cube,))
        request = PerceptionRequest(
            scene_id="wrong-shape",
            capture_time_s=12.0,
            frames=(
                CameraFrame(
                    camera_id="table",
                    capture_time_s=12.0,
                    calibration_id="cal-table",
                    rgb=np.zeros((720, 1280, 3), dtype=np.uint8),
                ),
            ),
        )
        provider = BenchmarkVisionProvider(
            geometry_provider=geometry,
            config=BenchmarkVisionConfig(mode=DETERMINISTIC_HYBRID),
        )

        with self.assertRaises(VisionError):
            provider.observe(request)

        self.assertEqual(geometry.calls, 0)

    def test_profile_rejects_unknown_options(self):
        with self.assertRaisesRegex(ValueError, "unknown benchmark vision options"):
            BenchmarkVisionConfig.from_mapping(
                {"mode": DETERMINISTIC_HYBRID, "typo": True}
            )

    def test_checked_in_yolo_profile_loads(self):
        profile = load_benchmark_vision_profile(
            "experiments/randomized_sort_benchmark/profiles/"
            "vision.yolo11n-assisted-hybrid.v1.json"
        )

        self.assertEqual(profile.schema_version, 1)
        self.assertEqual(profile.vision.mode, YOLO_ASSISTED_HYBRID)
        self.assertEqual(
            profile.vision.checkpoint_sha256, PINNED_YOLO11N_SHA256
        )


if __name__ == "__main__":
    unittest.main()
