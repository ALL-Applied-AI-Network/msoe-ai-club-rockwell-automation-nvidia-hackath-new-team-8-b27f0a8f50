from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import unittest


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sorting_core.synthetic_data import (  # noqa: E402
    load_synthetic_data_spec,
    sample_randomization,
)


HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAS_NUMPY, "NumPy is available in the pinned Isaac environment")
class ReadbackMetadataTests(unittest.TestCase):
    def test_effective_readback_hash_is_stable_and_content_sensitive(self):
        from isaac_bridge.training_episode import _with_sha256

        first = _with_sha256({"b": [2, 3], "a": 1})
        reordered = _with_sha256({"a": 1, "b": [2, 3]})
        changed = _with_sha256({"a": 1, "b": [2, 4]})

        self.assertEqual(first["sha256"], reordered["sha256"])
        self.assertNotEqual(first["sha256"], changed["sha256"])
        self.assertEqual(len(first["sha256"]), 64)


@unittest.skipUnless(HAS_NUMPY, "NumPy is available in the pinned Isaac environment")
class TrainingEpisodeTests(unittest.TestCase):
    def test_prebuild_values_are_pending_until_usd_readback(self):
        from isaac_bridge.training_episode import prepare_prebuild_randomization

        camera = {
            "eye": (0.77, -0.04, 1.70),
            "look_at": (0.16, -0.02, 0.90),
            "up": (0.0, 0.0, 1.0),
            "resolution": (1920, 1080),
            "focal_length": 11.2158,
            "horiz_aperture": 20.955,
        }
        scene = SimpleNamespace(CUBE_MASS=0.004, TABLE_CAM=dict(camera))
        materials = {
            "plastic_red": ((0.64, 0.055, 0.045), 0.0, 0.28),
            "plastic_yellow": ((0.78, 0.62, 0.04), 0.0, 0.28),
            "plastic_blue": ((0.045, 0.165, 0.62), 0.0, 0.28),
            "table_white": ((0.88, 0.88, 0.875), 0.0, 0.55),
        }
        cell = SimpleNamespace(_MATERIAL_SPEC=materials)
        base = {
            "red": (0.44, -0.19, 0.0),
            "yellow": (0.44, 0.0, 22.0),
            "blue": (0.44, 0.19, -28.0),
        }
        manifest = sample_randomization(
            load_synthetic_data_spec(),
            episode_seed=20261025,
            episode_id="prebuild-test",
        )

        spawns, applied = prepare_prebuild_randomization(scene, cell, manifest, base)

        self.assertNotEqual(spawns, base)
        self.assertNotEqual(scene.CUBE_MASS, 0.004)
        self.assertNotEqual(scene.TABLE_CAM["eye"], camera["eye"])
        self.assertNotEqual(
            cell._MATERIAL_SPEC["plastic_red"][2], 0.28
        )
        self.assertEqual(applied, set())

if __name__ == "__main__":
    unittest.main()
