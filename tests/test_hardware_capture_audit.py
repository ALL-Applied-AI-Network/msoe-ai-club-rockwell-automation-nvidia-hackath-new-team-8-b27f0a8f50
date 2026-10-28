from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "hardware" / "fanuc_cell" / "audit_table_capture.py"
SPEC = importlib.util.spec_from_file_location("audit_table_capture", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


class HardwareCaptureAuditTests(unittest.TestCase):
    def test_resized_whole_frame_maps_pixel_centres_to_native_calibration(self):
        self.assertEqual(
            audit.scale_uv_to_native((319.5, 179.5), (640, 360), (1920, 1080)),
            (959.5, 539.5),
        )

    def test_aspect_mismatch_is_rejected_instead_of_projected(self):
        cropped = np.zeros((360, 600, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            audit.analyse_frame(cropped)

    def test_large_real_like_yellow_patch_reports_size_gate_not_hsv_failure(self):
        # This is deliberately a resized recording-like frame.  The yellow
        # square is inside the configured pickup region and its HSV is well
        # inside the checked-in yellow range, but its calibrated footprint is
        # too large for the current 10 mm + 40% size gate.
        hsv = np.zeros((360, 640, 3), dtype=np.uint8)
        hsv[..., 2] = 180
        # Seven contour pixels project to about 17 mm here: above the 10 mm
        # profile's 14 mm maximum, but inside the 16 mm profile's gate.
        cv2.rectangle(hsv, (326, 265), (333, 272), (31, 220, 155), thickness=-1)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        frame = audit.analyse_frame(bgr)
        yellow = [
            component
            for component in frame["components"]
            if component["colour"] == "yellow" and component["in_pickup_region"]
        ]
        self.assertEqual(len(yellow), 1)
        self.assertIn("yellow", frame["replayed_detector_pickable_counts"])
        self.assertIn("physical_edge_size", yellow[0]["rejection_reasons"])
        self.assertNotIn("outside_pickup_region", yellow[0]["rejection_reasons"])
        self.assertGreater(yellow[0]["hsv_selected_pixels"]["median"][1], 45)

    def test_overlay_does_not_mutate_source_frame(self):
        hsv = np.zeros((360, 640, 3), dtype=np.uint8)
        hsv[..., 2] = 180
        cv2.rectangle(hsv, (254, 257), (264, 267), (172, 220, 110), thickness=-1)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        original = bgr.copy()
        frame = audit.analyse_frame(bgr)
        overlay = audit.draw_overlay(bgr, frame)
        np.testing.assert_array_equal(bgr, original)
        self.assertEqual(overlay.shape, bgr.shape)
        self.assertFalse(np.array_equal(overlay, bgr))

    def test_explicit_16_mm_profile_changes_size_gate_without_changing_hsv(self):
        hsv = np.zeros((360, 640, 3), dtype=np.uint8)
        hsv[..., 2] = 180
        cv2.rectangle(hsv, (326, 265), (333, 272), (31, 220, 155), thickness=-1)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        ten = audit.analyse_frame(bgr, expected_cube_edge_m=0.010)
        sixteen = audit.analyse_frame(bgr, expected_cube_edge_m=0.016)
        ten_yellow = [
            component
            for component in ten["components"]
            if component["colour"] == "yellow" and component["in_pickup_region"]
        ][0]
        sixteen_yellow = [
            component
            for component in sixteen["components"]
            if component["colour"] == "yellow" and component["in_pickup_region"]
        ][0]
        self.assertIn("physical_edge_size", ten_yellow["rejection_reasons"])
        self.assertNotIn("physical_edge_size", sixteen_yellow["rejection_reasons"])
        self.assertEqual(sixteen["expected_cube_edge_mm"], 16.0)


if __name__ == "__main__":
    unittest.main()
