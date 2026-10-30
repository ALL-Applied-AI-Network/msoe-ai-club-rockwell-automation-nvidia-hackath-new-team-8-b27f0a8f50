from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "compare_camera_appearance.py"
SPEC = importlib.util.spec_from_file_location("compare_camera_appearance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CameraAppearanceCurationTests(unittest.TestCase):
    def test_uncurated_video_requires_an_explicit_acknowledgement(self):
        with self.assertRaisesRegex(ValueError, "requires --curation-manifest"):
            MODULE._curated_ranges(
                None,
                dataset_id=None,
                episode=None,
                split="appearance_fit",
            )
        self.assertIsNone(
            MODULE._curated_ranges(
                None,
                dataset_id=None,
                episode=None,
                split="appearance_fit",
                assume_precurated_input=True,
            )
        )

    def test_curation_is_default_deny_for_an_unlisted_episode(self):
        manifest = {
            "default_policy": "deny",
            "splits": {
                "appearance_fit": [
                    {
                        "dataset": "run-a",
                        "episode": 2,
                        "ranges": [[10, 20], [30, 40]],
                    }
                ],
                "excluded_human_or_setup": [],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curation.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            accepted = MODULE._curated_ranges(
                path,
                dataset_id="run-a",
                episode=2,
                split="appearance_fit",
            )
            self.assertEqual(accepted, ((10, 20), (30, 40)))
            with self.assertRaisesRegex(ValueError, "no unique"):
                MODULE._curated_ranges(
                    path,
                    dataset_id="run-a",
                    episode=3,
                    split="appearance_fit",
                )

    def test_overlap_with_privacy_exclusion_is_rejected(self):
        manifest = {
            "default_policy": "deny",
            "splits": {
                "appearance_fit": [
                    {"dataset": "run-a", "episode": 2, "ranges": [[10, 20]]}
                ],
                "excluded_human_or_setup": [
                    {"dataset": "run-a", "episode": 2, "ranges": [[15, 18]]}
                ],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curation.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlaps excluded"):
                MODULE._curated_ranges(
                    path,
                    dataset_id="run-a",
                    episode=2,
                    split="appearance_fit",
                )

    def test_sampled_indices_never_leave_curated_ranges(self):
        indices = MODULE._indices_in_ranges(100, 9, ((10, 20), (70, 80)))
        self.assertEqual(len(indices), 9)
        self.assertTrue(
            all(10 <= value < 20 or 70 <= value < 80 for value in indices)
        )

    def test_each_camera_uses_a_nonempty_camera_specific_mask(self):
        expected = {
            "table": {"neutral_table", "neutral_background"},
            "side": {"neutral_wall", "neutral_table"},
            "wrist": {"neutral_workspace"},
        }
        for camera, names in expected.items():
            with self.subTest(camera=camera):
                masks = MODULE._region_masks(640, 360, camera)
                self.assertEqual(set(masks), names)
                self.assertTrue(all(mask.dtype == np.uint8 for mask in masks.values()))
                self.assertTrue(all(np.count_nonzero(mask) > 0 for mask in masks.values()))


if __name__ == "__main__":
    unittest.main()
