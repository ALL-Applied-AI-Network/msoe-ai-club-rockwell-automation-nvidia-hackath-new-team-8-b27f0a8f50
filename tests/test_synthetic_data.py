from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:  # Sensor tests run in the pinned CV/Isaac environment.
    np = None


HAS_SENSOR_DEPS = np is not None and importlib.util.find_spec("cv2") is not None


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sorting_core.synthetic_data import (  # noqa: E402
    SyntheticDataError,
    augment_sensor_rgb,
    build_synchronized_frame_record,
    canonical_json_sha256,
    distort_points_uv,
    finalize_application_status,
    load_camera_calibration,
    load_capture_calibration,
    load_synthetic_data_spec,
    sample_randomization,
    write_json_atomic,
)


class SyntheticDataConfigTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_synthetic_data_spec()

    def test_native_raw_capture_matches_physical_calibration(self):
        capture = self.spec.capture
        self.assertEqual(capture.camera_name, "table")
        self.assertEqual(capture.resolution, (1920, 1080))
        self.assertEqual(capture.colour_space, "RGB")
        self.assertEqual(capture.image_encoding, "png")
        self.assertTrue(capture.save_raw_frames)
        self.assertFalse(capture.burn_in_overlay)
        self.assertTrue(capture.apply_calibrated_radial_distortion)
        calibration = load_capture_calibration(self.spec)
        self.assertEqual(tuple(calibration["size"]), capture.resolution)
        self.assertAlmostEqual(calibration["fx"], 1024.8627910268606)
        self.assertAlmostEqual(calibration["k1"], 0.008859283169230157)

    def test_all_colour_camera_calibrations_are_loadable_at_native_resolution(self):
        expected = {
            "table": (1920, 1080),
            "side": (1920, 1080),
            "wrist": (1280, 720),
        }
        for camera_id, resolution in expected.items():
            with self.subTest(camera_id=camera_id):
                calibration = load_camera_calibration(
                    self.spec, calibration_entry=camera_id
                )
                self.assertEqual(tuple(calibration["size"]), resolution)
                self.assertGreater(float(calibration["fx"]), 0.0)
                self.assertIn("k1", calibration)

    def test_ranges_are_bounded_provisional_and_seeded(self):
        first = sample_randomization(
            self.spec, episode_seed=20261025, episode_id="pilot-1"
        )
        again = sample_randomization(
            self.spec, episode_seed=20261025, episode_id="pilot-1"
        )
        other = sample_randomization(
            self.spec, episode_seed=20261026, episode_id="pilot-2"
        )
        self.assertEqual(first.as_dict(), again.as_dict())
        self.assertEqual(first.sha256, again.sha256)
        self.assertNotEqual(first.parameters, other.parameters)
        self.assertEqual(len(first.sha256), 64)
        self.assertEqual(self.spec.range_status, "provisional_unmeasured")
        for name, entry in first.parameters.items():
            self.assertLessEqual(entry["minimum"], entry["requested_value"], name)
            self.assertLessEqual(entry["requested_value"], entry["maximum"], name)
            self.assertEqual(entry["status"], "provisional_unmeasured")
            self.assertEqual(entry["application_status"], "pending")

    def test_dynamic_friction_never_exceeds_static_friction(self):
        static = self.spec.parameters["physics.object_static_friction"]
        dynamic = self.spec.parameters["physics.object_dynamic_friction"]
        self.assertLessEqual(dynamic.maximum, static.minimum)
        for seed in range(1000):
            manifest = sample_randomization(
                self.spec, episode_seed=seed, episode_id=f"friction-{seed}"
            )
            self.assertLessEqual(
                manifest.value("physics.object_dynamic_friction"),
                manifest.value("physics.object_static_friction"),
            )

    def test_application_manifest_cannot_claim_partial_randomization(self):
        manifest = sample_randomization(
            self.spec, episode_seed=7, episode_id="episode-7"
        )
        names = set(manifest.parameters)
        with self.assertRaisesRegex(SyntheticDataError, "unclassified"):
            finalize_application_status(manifest, applied={next(iter(names))}, skipped={})

        sensor = {
            name
            for name, entry in manifest.parameters.items()
            if entry["consumer"] == "sensor_postprocess"
        }
        skipped = {
            name: "Isaac stage adapter has not been implemented"
            for name in names - sensor
        }
        final = finalize_application_status(
            manifest,
            applied=sensor,
            skipped=skipped,
        )
        self.assertTrue(final["all_parameters_classified"])
        self.assertTrue(
            all(final["parameters"][name]["application_status"] == "applied" for name in sensor)
        )
        self.assertTrue(
            all(final["parameters"][name]["application_status"] == "skipped" for name in skipped)
        )


@unittest.skipUnless(HAS_SENSOR_DEPS, "NumPy/OpenCV sensor dependencies unavailable")
class SensorModelTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_synthetic_data_spec()
        manifest = sample_randomization(
            self.spec, episode_seed=99, episode_id="sensor-test"
        )
        capture = dict(manifest.capture)
        capture["resolution"] = [64, 48]
        self.manifest = replace(manifest, capture=capture)
        x = np.linspace(0, 255, 64, dtype=np.uint8)
        row = np.stack((x, np.flip(x), np.full_like(x, 127)), axis=1)
        self.image = np.repeat(row[None, :, :], 48, axis=0)
        self.calibration = {"size": [64, 48], "fx": 34.0, "k1": 0.01}

    def test_sensor_augmentation_is_deterministic_and_preserves_contract(self):
        a = augment_sensor_rgb(
            self.image,
            self.manifest,
            frame_index=12,
            camera_calibration=self.calibration,
        )
        b = augment_sensor_rgb(
            self.image,
            self.manifest,
            frame_index=12,
            camera_calibration=self.calibration,
        )
        np.testing.assert_array_equal(a.image_rgb, b.image_rgb)
        self.assertEqual(a.image_rgb.shape, self.image.shape)
        self.assertEqual(a.image_rgb.dtype, np.uint8)
        self.assertFalse(np.array_equal(a.image_rgb, self.image))
        self.assertTrue(a.processing["calibrated_radial_distortion_applied"])
        self.assertEqual(a.processing["frame_index"], 12)
        self.assertEqual(
            a.processing["sensor_noise_frame_seed"],
            b.processing["sensor_noise_frame_seed"],
        )
        self.assertEqual(len(a.applied_parameters), 9)

        adjacent = augment_sensor_rgb(
            self.image,
            self.manifest,
            frame_index=13,
            camera_calibration=self.calibration,
        )
        self.assertNotEqual(
            a.processing["sensor_noise_frame_seed"],
            adjacent.processing["sensor_noise_frame_seed"],
        )
        self.assertGreater(
            self.manifest.value("sensor.gaussian_noise_std_dn"), 0.0
        )
        self.assertFalse(np.array_equal(a.image_rgb, adjacent.image_rgb))

    def test_distorted_point_labels_keep_optical_centre_fixed(self):
        points = distort_points_uv(
            [(32, 24), (60, 44)],
            width=64,
            height=48,
            fx=34.0,
            k1=0.01,
        )
        self.assertEqual(points[0], (32.0, 24.0))
        self.assertGreater(points[1][0], 60.0)
        self.assertGreater(points[1][1], 44.0)

    def test_wrong_resolution_is_rejected_instead_of_silently_resized(self):
        with self.assertRaisesRegex(SyntheticDataError, "native capture resolution"):
            augment_sensor_rgb(
                self.image[:, :-1],
                self.manifest,
                frame_index=0,
                camera_calibration=self.calibration,
            )

    def test_explicit_camera_seed_is_deterministic_and_camera_specific(self):
        table = augment_sensor_rgb(
            self.image,
            self.manifest,
            frame_index=12,
            camera_calibration=self.calibration,
            camera_id="table",
        )
        repeated = augment_sensor_rgb(
            self.image,
            self.manifest,
            frame_index=12,
            camera_calibration=self.calibration,
            camera_id="table",
        )
        side = augment_sensor_rgb(
            self.image,
            self.manifest,
            frame_index=12,
            camera_calibration=self.calibration,
            camera_id="side",
        )
        np.testing.assert_array_equal(table.image_rgb, repeated.image_rgb)
        self.assertEqual(
            table.processing["sensor_noise_frame_seed"],
            repeated.processing["sensor_noise_frame_seed"],
        )
        self.assertNotEqual(
            table.processing["sensor_noise_frame_seed"],
            side.processing["sensor_noise_frame_seed"],
        )
        self.assertEqual(table.processing["camera_id"], "table")
        self.assertEqual(
            table.processing["sensor_seed_namespace"],
            "rockwell-sensor-noise-frame-v2-camera",
        )
        self.assertFalse(
            table.processing["compression_domain_approximation"][
                "deployment_yuv420_or_archive_codec_reproduced"
            ]
        )

    def test_explicit_camera_uses_its_calibrated_resolution(self):
        image = np.full((30, 40, 3), 127, dtype=np.uint8)
        result = augment_sensor_rgb(
            image,
            self.manifest,
            frame_index=0,
            camera_calibration={"size": [40, 30], "fx": 22.0, "k1": 0.0},
            camera_id="wrist",
        )
        self.assertEqual(result.image_rgb.shape, image.shape)
        self.assertEqual(result.processing["input_resolution"], [40, 30])
        with self.assertRaisesRegex(
            SyntheticDataError, "calibration size does not match"
        ):
            augment_sensor_rgb(
                image,
                self.manifest,
                frame_index=0,
                camera_calibration={"size": [40, 30], "fx": 22.0, "k1": 0.0},
                camera_id="wrist",
                expected_resolution=(42, 30),
            )


class LabelContractTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_synthetic_data_spec()
        self.manifest = sample_randomization(
            self.spec, episode_seed=123, episode_id="labels-123"
        )
        self.objects = [
            {
                "track_id": "red-0",
                "class_name": "cube",
                "colour": "red",
                "position_base_m": [0.44, -0.19, -0.0001],
                "quaternion_base_wxyz": [1.0, 0.0, 0.0, 0.0],
                "dimensions_m": [0.01, 0.01, 0.01],
                "visible_fraction": 0.8,
                "occluded": True,
                "amodal_bbox_type": "projected_3d_cuboid_clipped_to_image",
                "raw_amodal_bbox_xyxy_px": [900, 500, 920, 520],
                "raw_amodal_bbox_status": "valid",
                "derived_amodal_bbox_xyxy_px": [899, 499, 921, 521],
                "derived_amodal_bbox_status": "valid",
                "instance_mask_relpath": "masks/000001-red-0.png",
            }
        ]

    def _record(self, **overrides):
        values = {
            "frame_index": 1,
            "raw_image_relpath": "raw/000001.png",
            "capture_time_s": 10.0,
            "robot_state_time_s": 10.005,
            "scene_state_time_s": 9.995,
            "measured_arm_joints_deg": [0.0] * 6,
            "measured_gripper_half_opening_m": 0.005,
            "measured_gripper_closure_fraction": 1.0 / 6.0,
            "measured_tcp_pose_uf0_mm_deg": [300, 0, -90, -180, 0, 0],
            "commanded_arm_joints_deg": [0.1] * 6,
            "commanded_gripper_half_opening_m": 0.0049,
            "commanded_gripper_closure_fraction": 0.18,
            "commanded_tcp_pose_uf0_mm_deg": [301, 0, -90, -180, 0, 0],
            "scene_version": "scene-4",
            "phase": "approach_pick",
            "objects": self.objects,
            "derived_image_relpath": "sensor/000001.png",
        }
        values.update(overrides)
        return build_synchronized_frame_record(self.spec, self.manifest, **values)

    def test_record_separates_policy_inputs_from_simulator_truth(self):
        record = self._record()
        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(record["camera"]["resolution"], [1920, 1080])
        self.assertTrue(record["synchronization"]["within_tolerance"])
        measured = record["policy_input"]["measured_robot_state"]
        self.assertEqual(len(measured["arm_joint_positions_deg"]), 6)
        self.assertEqual(measured["gripper"]["jaw_half_opening_m"], 0.005)
        commanded = record["commanded_robot_state"]
        self.assertEqual(len(commanded["arm_joint_targets_deg"]), 6)
        self.assertEqual(commanded["gripper"]["closure_fraction"], 0.18)
        self.assertNotIn("objects", record["policy_input"])
        label = record["evaluator_only"]["objects"][0]
        self.assertEqual(label["track_id"], "red-0")
        self.assertNotIn("bbox_xyxy_px", label)
        self.assertEqual(
            label["amodal_bbox_type"],
            "projected_3d_cuboid_clipped_to_image",
        )
        self.assertEqual(
            record["randomization"]["manifest_sha256"], self.manifest.sha256
        )

    def test_stale_state_and_unsafe_paths_are_rejected(self):
        with self.assertRaisesRegex(SyntheticDataError, "state/image skew"):
            self._record(robot_state_time_s=10.1)
        with self.assertRaisesRegex(SyntheticDataError, "safe run-relative"):
            self._record(raw_image_relpath="../leak.png")
        with self.assertRaisesRegex(SyntheticDataError, "must be PNG"):
            self._record(raw_image_relpath="raw/frame.jpg")

    def test_gripper_and_amodal_box_status_fail_closed(self):
        with self.assertRaisesRegex(SyntheticDataError, "closure_fraction"):
            self._record(measured_gripper_closure_fraction=1.1)
        invalid_objects = [dict(self.objects[0])]
        invalid_objects[0]["raw_amodal_bbox_status"] = "invalid_outside_image"
        with self.assertRaisesRegex(SyntheticDataError, "status/value disagree"):
            self._record(objects=invalid_objects)
        outside_objects = [dict(self.objects[0])]
        outside_objects[0]["raw_amodal_bbox_xyxy_px"] = [-1, 500, 920, 520]
        with self.assertRaisesRegex(SyntheticDataError, "clipped to the image"):
            self._record(objects=outside_objects)

    def test_atomic_json_is_canonical_and_round_trips(self):
        record = self._record()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "frame.json"
            write_json_atomic(path, record)
            loaded = __import__("json").loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded, record)
            self.assertEqual(canonical_json_sha256(loaded), canonical_json_sha256(record))
            self.assertFalse(path.with_name(path.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
