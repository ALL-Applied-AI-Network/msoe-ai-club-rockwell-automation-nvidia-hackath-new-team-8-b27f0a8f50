from __future__ import annotations

import math
from pathlib import Path
import importlib.util
import sys
import unittest


# Load this deliberately stdlib-only module without importing isaac_bridge's
# NumPy-backed perception facade.  That proves the validation gate remains
# usable in the minimal host-Python checks as well as inside Isaac.
MODULE_PATH = Path(__file__).resolve().parents[1] / "isaac_bridge" / "sim_validation.py"
SPEC = importlib.util.spec_from_file_location("_sim_validation_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATION
SPEC.loader.exec_module(VALIDATION)

CameraCalibrationExpectation = VALIDATION.CameraCalibrationExpectation
GeometryExpectation = VALIDATION.GeometryExpectation
SimulationValidationError = VALIDATION.SimulationValidationError
isaac_camera_config_from_calibration = (
    VALIDATION.isaac_camera_config_from_calibration
)
measurements_from_bounds = VALIDATION.measurements_from_bounds
validate_camera_readback = VALIDATION.validate_camera_readback
validate_geometry_readback = VALIDATION.validate_geometry_readback


PROVENANCE = "hardware/fanuc_cell/calibration/cameras.json#table"
CALIBRATION = {
    "size": [1920, 1080],
    "fx": 1024.8627910268606,
    "k1": 0.008859283169230157,
    "T_base_cam": [
        [1.0, 0.0, 0.0, 0.775],
        [0.0, 1.0, 0.0, -0.043],
        [0.0, 0.0, 1.0, 0.801],
        [0.0, 0.0, 0.0, 1.0],
    ],
}


class CameraValidationTests(unittest.TestCase):
    def setUp(self):
        self.expected = CameraCalibrationExpectation.from_calibration_mapping(
            "table", CALIBRATION, provenance_id=PROVENANCE
        )
        horizontal_aperture_mm = 20.955
        self.readback = {
            "resolution": [1920, 1080],
            "focal_length_mm": (
                CALIBRATION["fx"] * horizontal_aperture_mm / 1920.0
            ),
            "horizontal_aperture_mm": horizontal_aperture_mm,
            "projection_provenance_id": PROVENANCE,
            "distortion": {
                "k1": CALIBRATION["k1"],
                "application": "postprocess",
                "provenance_id": PROVENANCE,
            },
            "extrinsic_frame": "base",
            "camera_to_parent_row_major": CALIBRATION["T_base_cam"],
        }

    def test_matching_camera_metadata_and_extrinsic_validate(self):
        result = validate_camera_readback(
            self.expected, self.readback, require_extrinsics=True
        )
        self.assertEqual(result.camera_id, "table")
        self.assertLess(result.focal_error_px, 1e-9)
        self.assertEqual(result.distortion_application, "postprocess")
        self.assertEqual(result.translation_error_m, 0.0)
        self.assertEqual(result.rotation_error_deg, 0.0)
        self.assertEqual(result.as_dict()["resolution"], [1920, 1080])

    def test_calibration_converts_to_exact_isaac_axis_convention(self):
        converted = isaac_camera_config_from_calibration(
            CALIBRATION,
            parent="/World",
            horizontal_aperture_mm=20.955,
            parent_origin_offset_m=(0.0, 0.0, 0.9),
        )
        self.assertEqual(converted["resolution"], (1920, 1080))
        self.assertAlmostEqual(converted["focal_length"], 11.185416555, places=8)
        self.assertEqual(converted["eye"], (0.775, -0.043, 1.701))
        self.assertEqual(converted["look_at"], (0.775, -0.043, 2.701))
        self.assertEqual(converted["up"], (-0.0, -1.0, -0.0))

    def test_focal_mismatch_is_rejected(self):
        self.readback["focal_length_mm"] *= 1.003
        with self.assertRaisesRegex(SimulationValidationError, "focal error"):
            validate_camera_readback(self.expected, self.readback)

    def test_nonzero_distortion_cannot_be_declared_ideal(self):
        self.readback["distortion"]["application"] = "ideal_pinhole"
        with self.assertRaisesRegex(
            SimulationValidationError, "non-zero calibrated distortion"
        ):
            validate_camera_readback(self.expected, self.readback)

    def test_projection_and_distortion_provenance_are_independent_gates(self):
        self.readback["distortion"]["provenance_id"] = "guessed-profile"
        with self.assertRaisesRegex(SimulationValidationError, "distortion provenance"):
            validate_camera_readback(self.expected, self.readback)

    def test_extrinsic_parent_frame_and_pose_are_checked(self):
        self.readback["extrinsic_frame"] = "world"
        with self.assertRaisesRegex(SimulationValidationError, "extrinsic frame"):
            validate_camera_readback(
                self.expected, self.readback, require_extrinsics=True
            )
        self.readback["extrinsic_frame"] = "base"
        moved = [row[:] for row in CALIBRATION["T_base_cam"]]
        moved[0][3] += 0.001
        self.readback["camera_to_parent_row_major"] = moved
        with self.assertRaisesRegex(
            SimulationValidationError, "extrinsic translation error"
        ):
            validate_camera_readback(
                self.expected, self.readback, require_extrinsics=True
            )

    def test_rotation_difference_is_reported_in_degrees(self):
        angle = math.radians(0.1)
        rotated = [row[:] for row in CALIBRATION["T_base_cam"]]
        rotated[0][:3] = [math.cos(angle), -math.sin(angle), 0.0]
        rotated[1][:3] = [math.sin(angle), math.cos(angle), 0.0]
        self.readback["camera_to_parent_row_major"] = rotated
        with self.assertRaisesRegex(
            SimulationValidationError, "extrinsic rotation error"
        ):
            validate_camera_readback(
                self.expected, self.readback, require_extrinsics=True
            )


class GeometryValidationTests(unittest.TestCase):
    def test_bounds_conversion_and_cube_provenance_validate(self):
        measurements = measurements_from_bounds(
            (-0.005, -0.005, 0.0),
            (0.005, 0.005, 0.010),
        )
        expected = GeometryExpectation.from_measurements(
            "cube_10mm_visual",
            measurements,
            geometry_provenance_id="task-config#cube_10mm",
            silhouette_provenance_id="sha256:official-cube-10mm-mesh",
        )
        result = validate_geometry_readback(
            expected,
            {
                "measurements_m": {
                    "size_x": 0.01001,
                    "size_y": 0.00999,
                    "size_z": 0.01000,
                },
                "geometry_provenance_id": "task-config#cube_10mm",
                "silhouette_provenance_id": "sha256:official-cube-10mm-mesh",
            },
        )
        self.assertEqual(result.geometry_id, "cube_10mm_visual")
        self.assertLess(result.as_dict()["absolute_errors_m"]["size_x"], 0.00005)

    def test_bin_internal_measurements_can_be_validated(self):
        expected = GeometryExpectation.from_measurements(
            "bin_50x50x55",
            {
                "outer_x": 0.050,
                "outer_y": 0.050,
                "height": 0.055,
                "pocket_x": 0.035,
                "pocket_y": 0.035,
                "floor_z": 0.030,
            },
            geometry_provenance_id="task-config#bin",
            silhouette_provenance_id="sha256:official-bin-mesh",
        )
        result = validate_geometry_readback(
            expected,
            {
                "measurements_m": dict(expected.measurements_m),
                "geometry_provenance_id": "task-config#bin",
                "silhouette_provenance_id": "sha256:official-bin-mesh",
            },
            tolerance_m={name: 0.00005 for name, _ in expected.measurements_m},
        )
        self.assertEqual(len(result.measurements_m), 6)

    def test_dimension_and_silhouette_mismatches_are_rejected(self):
        expected = GeometryExpectation.from_measurements(
            "cube",
            {"edge": 0.010},
            geometry_provenance_id="cube-profile-10mm",
            silhouette_provenance_id="sha256:10mm",
        )
        readback = {
            "measurements_m": {"edge": 0.016},
            "geometry_provenance_id": "cube-profile-10mm",
            "silhouette_provenance_id": "sha256:10mm",
        }
        with self.assertRaisesRegex(SimulationValidationError, "edge error"):
            validate_geometry_readback(expected, readback)
        readback["measurements_m"]["edge"] = 0.010
        readback["silhouette_provenance_id"] = "sha256:scaled-box"
        with self.assertRaisesRegex(SimulationValidationError, "silhouette provenance"):
            validate_geometry_readback(expected, readback)


if __name__ == "__main__":
    unittest.main()
