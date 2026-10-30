from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sorting_core.camera_appearance import (  # noqa: E402
    CameraAppearanceError,
    create_deployment_preview,
    load_camera_appearance_profiles,
    load_frame_curation,
)


class FrameCurationTests(unittest.TestCase):
    def test_human_and_unknown_frames_are_default_denied(self):
        curation = load_frame_curation()
        self.assertTrue(
            curation.approved(
                dataset="r20261029-88a87b",
                episode=0,
                frame_index=3399,
            )
        )
        self.assertFalse(
            curation.approved(
                dataset="r20261029-88a87b",
                episode=0,
                frame_index=3400,
            )
        )
        self.assertFalse(
            curation.approved(
                dataset="r20261027-4a4b7d",
                episode=4,
                frame_index=100,
            )
        )
        self.assertFalse(
            curation.approved(dataset="unknown", episode=0, frame_index=0)
        )

    def test_fit_and_excluded_overlap_is_rejected(self):
        source = REPO / "config" / "hardware_frame_curation.json"
        value = json.loads(source.read_text(encoding="utf-8"))
        value = copy.deepcopy(value)
        value["splits"]["appearance_fit"].append(
            {
                "dataset": "r20261029-88a87b",
                "episode": 0,
                "ranges": [[3500, 3501]],
                "tags": ["clean_cell"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curation.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(
                CameraAppearanceError, "overlaps privacy exclusion"
            ):
                load_frame_curation(path)


class CameraProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = load_camera_appearance_profiles()

    def test_profile_matches_checked_in_calibration_and_stream(self):
        self.assertEqual(set(self.profiles.cameras), {"table", "side", "wrist"})
        self.assertEqual(self.profiles.deployment_resolution, (640, 360))
        self.assertEqual(self.profiles.deployment_fps, 10)
        self.assertEqual(self.profiles.camera("table").native_resolution, (1920, 1080))
        self.assertEqual(self.profiles.camera("side").native_resolution, (1920, 1080))
        self.assertEqual(self.profiles.camera("wrist").native_resolution, (1280, 720))
        self.assertFalse(self.profiles.camera("table").hose_occlusion.enabled)
        self.assertFalse(self.profiles.camera("side").hose_occlusion.enabled)
        self.assertTrue(self.profiles.camera("wrist").hose_occlusion.enabled)
        self.assertEqual(
            self.profiles.camera("table").fit_neutral_rgb_median_8bit,
            (141, 140, 139),
        )
        self.assertEqual(
            self.profiles.camera("side").heldout_neutral_rgb_median_8bit,
            (166, 157, 152),
        )

    @staticmethod
    def _pattern(width: int, height: int) -> np.ndarray:
        yy, xx = np.indices((height, width), dtype=np.uint16)
        return np.stack(
            (
                (xx % 256).astype(np.uint8),
                (yy % 256).astype(np.uint8),
                ((xx + yy) % 256).astype(np.uint8),
            ),
            axis=2,
        )

    def test_table_preview_is_deterministic_and_never_gets_hose(self):
        source = self._pattern(1920, 1080)
        preserved = source.copy()
        first = create_deployment_preview(
            source,
            self.profiles,
            camera_id="table",
            episode_seed=7,
            frame_index=12,
        )
        second = create_deployment_preview(
            source,
            self.profiles,
            camera_id="table",
            episode_seed=7,
            frame_index=12,
        )
        np.testing.assert_array_equal(source, preserved)
        np.testing.assert_array_equal(first.image_rgb, second.image_rgb)
        np.testing.assert_array_equal(first.occluder_mask, second.occluder_mask)
        self.assertEqual(first.image_rgb.shape, (360, 640, 3))
        self.assertEqual(first.image_rgb.dtype, np.uint8)
        self.assertEqual(np.count_nonzero(first.occluder_mask), 0)
        self.assertFalse(first.processing["synthetic_wrist_hose"]["applied"])
        self.assertFalse(first.processing["av1_bitstream_reproduced"])

    def test_wrist_hose_is_deterministic_and_returns_mask(self):
        source = np.full((720, 1280, 3), 180, dtype=np.uint8)
        selected = None
        selected_frame = None
        # The configured probability remains evidence-driven/provisional.  Find
        # one deterministic applied sample without weakening it for the test.
        for frame_index in range(32):
            candidate = create_deployment_preview(
                source,
                self.profiles,
                camera_id="wrist",
                episode_seed=20261029,
                frame_index=frame_index,
                add_wrist_occlusion=True,
            )
            if candidate.processing["synthetic_wrist_hose"]["applied"]:
                selected = candidate
                selected_frame = frame_index
                break
        self.assertIsNotNone(selected)
        self.assertIsNotNone(selected_frame)
        self.assertGreater(np.count_nonzero(selected.occluder_mask), 0)
        self.assertGreater(
            selected.processing["synthetic_wrist_hose"]["mask_nonzero_fraction"],
            0.0,
        )
        repeated = create_deployment_preview(
            source,
            self.profiles,
            camera_id="wrist",
            episode_seed=20261029,
            frame_index=selected_frame,
            add_wrist_occlusion=True,
        )
        np.testing.assert_array_equal(selected.image_rgb, repeated.image_rgb)
        np.testing.assert_array_equal(selected.occluder_mask, repeated.occluder_mask)

    def test_wrist_occlusion_can_be_disabled_without_changing_shape(self):
        source = np.full((720, 1280, 3), 180, dtype=np.uint8)
        result = create_deployment_preview(
            source,
            self.profiles,
            camera_id="wrist",
            episode_seed=1,
            frame_index=0,
            add_wrist_occlusion=False,
        )
        self.assertEqual(result.image_rgb.shape, (360, 640, 3))
        self.assertEqual(np.count_nonzero(result.occluder_mask), 0)

    def test_native_resolution_mismatch_is_rejected(self):
        with self.assertRaisesRegex(CameraAppearanceError, "native 1920x1080"):
            create_deployment_preview(
                np.zeros((360, 640, 3), dtype=np.uint8),
                self.profiles,
                camera_id="side",
                episode_seed=1,
                frame_index=0,
            )

    def test_i420_round_trip_reduces_chroma_resolution(self):
        source = np.zeros((1080, 1920, 3), dtype=np.uint8)
        source[:, 0::2] = (255, 0, 0)
        source[:, 1::2] = (0, 0, 255)
        result = create_deployment_preview(
            source,
            self.profiles,
            camera_id="table",
            episode_seed=3,
            frame_index=4,
        )
        resized = cv2.resize(source, (640, 360), interpolation=cv2.INTER_AREA)
        self.assertFalse(np.array_equal(result.image_rgb, resized))

    def test_curated_neutral_median_is_applied_only_to_preview(self):
        source = np.full((1080, 1920, 3), (90, 100, 110), dtype=np.uint8)
        preserved = source.copy()
        result = create_deployment_preview(
            source,
            self.profiles,
            camera_id="table",
            episode_seed=9,
            frame_index=4,
        )
        np.testing.assert_array_equal(source, preserved)
        evidence = result.processing["curated_neutral_median_match"]
        self.assertTrue(evidence["applied"])
        self.assertEqual(evidence["target_neutral_rgb_median_8bit"], [141.0, 140.0, 139.0])
        self.assertTrue(all(2.0 / 3.0 <= value <= 1.5 for value in evidence["applied_rgb_gain"]))


if __name__ == "__main__":
    unittest.main()
