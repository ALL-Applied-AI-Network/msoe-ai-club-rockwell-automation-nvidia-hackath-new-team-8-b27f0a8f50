from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import copy
import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest import mock


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
    def test_default_module_import_does_not_read_training_colour_manifest(self):
        import isaac_bridge.training_episode as training_episode

        with mock.patch.object(
            Path,
            "read_text",
            side_effect=AssertionError("training manifest read at import time"),
        ):
            reloaded = importlib.reload(training_episode)
        self.assertTrue(callable(reloaded.load_colour_patch_evidence))

    def test_effective_readback_hash_is_stable_and_content_sensitive(self):
        from isaac_bridge.training_episode import _with_sha256

        first = _with_sha256({"b": [2, 3], "a": 1})
        reordered = _with_sha256({"a": 1, "b": [2, 3]})
        changed = _with_sha256({"a": 1, "b": [2, 4]})

        self.assertEqual(first["sha256"], reordered["sha256"])
        self.assertNotEqual(first["sha256"], changed["sha256"])
        self.assertEqual(len(first["sha256"]), 64)

    def test_float32_usd_round_trip_is_accepted_without_hiding_real_error(self):
        import numpy as np

        from isaac_bridge.training_episode import _assert_close
        from sorting_core.synthetic_data import SyntheticDataError

        requested = 1257.9247767124546
        usd_readback = float(np.float32(requested))
        _assert_close(usd_readback, requested, "dome intensity")

        with self.assertRaisesRegex(SyntheticDataError, "readback mismatch"):
            _assert_close(usd_readback + 0.01, requested, "dome intensity")


@unittest.skipUnless(HAS_NUMPY, "NumPy is available in the pinned Isaac environment")
class TrainingEpisodeTests(unittest.TestCase):
    def test_colour_fit_rejects_changed_aggregation_or_formula(self):
        import isaac_bridge.training_episode as training_episode
        from sorting_core.synthetic_data import SyntheticDataError

        original_path = training_episode.COLOUR_PATCH_EVIDENCE_PATH
        original = json.loads(original_path.read_text(encoding="utf-8"))
        cases = {
            "rounded physical-source median": (
                "physical_display_rgb_target_8bit",
                [96, 14, 39],
            ),
            "recorded channel-ratio fit": (
                "fitted_nominal_diffuse_rgb",
                [0.400, 0.020, 0.050],
            ),
        }
        try:
            with tempfile.TemporaryDirectory() as directory:
                for expected, (field, value) in cases.items():
                    with self.subTest(field=field):
                        payload = copy.deepcopy(original)
                        payload["targets"]["cubes"]["red"][field] = value
                        path = Path(directory) / f"{field}.json"
                        path.write_text(json.dumps(payload), encoding="utf-8")
                        training_episode.COLOUR_PATCH_EVIDENCE_PATH = path
                        training_episode.load_colour_patch_evidence.cache_clear()
                        training_episode._colour_patch_context.cache_clear()
                        with self.assertRaisesRegex(SyntheticDataError, expected):
                            training_episode.load_colour_patch_evidence()
        finally:
            training_episode.COLOUR_PATCH_EVIDENCE_PATH = original_path
            training_episode.load_colour_patch_evidence.cache_clear()
            training_episode._colour_patch_context.cache_clear()

    def test_paired_patch_fit_keeps_semantics_and_records_provenance(self):
        from isaac_bridge.training_episode import (
            DESTINATION_MATERIAL_PARAMETER_NAMES,
            MATERIAL_PARAMETER_NAMES,
            colour_patch_appearance_targets,
            load_colour_patch_evidence,
        )

        colour_patch_evidence, colour_patch_sha256 = load_colour_patch_evidence()
        cube_appearance_targets, destination_appearance_targets = (
            colour_patch_appearance_targets()
        )
        self.assertEqual(
            set(cube_appearance_targets), {"red", "yellow", "blue"}
        )
        self.assertEqual(
            set(destination_appearance_targets), {"red", "yellow", "blue"}
        )
        self.assertEqual(
            {
                label: target["appearance_name"]
                for label, target in destination_appearance_targets.items()
            },
            {"red": "orange", "yellow": "yellow", "blue": "light_blue"},
        )
        for group in (cube_appearance_targets, destination_appearance_targets):
            for target in group.values():
                self.assertEqual(len(target["nominal_diffuse_rgb"]), 3)
                self.assertTrue(
                    all(0.0 <= value <= 1.0 for value in target["nominal_diffuse_rgb"])
                )
                self.assertTrue(
                    all(
                        0.0 <= value <= 255.0
                        for value in target["physical_display_rgb_target_8bit"]
                    )
                )
        self.assertEqual(len(colour_patch_sha256), 64)
        self.assertEqual(
            colour_patch_evidence["sources"]["physical_fit_frame_0"]["sha256"],
            "a8ebeecc9569d27d03faba86bb6071ec168ac8e11148d48bf76b115d0e556949",
        )
        self.assertIn(
            "10 mm versus 16 mm",
            " ".join(colour_patch_evidence["limitations"]),
        )
        self.assertTrue(
            DESTINATION_MATERIAL_PARAMETER_NAMES.isdisjoint(
                MATERIAL_PARAMETER_NAMES
            )
        )

    def test_material_and_light_variations_are_independently_sampled(self):
        from isaac_bridge.training_episode import LIGHT_COLOR_PARAMETER_NAMES

        spec = load_synthetic_data_spec()
        destination_names = {
            "materials.destination_plastic_roughness",
            "materials.destination_albedo_scale",
            "materials.destination_hue_jitter_deg",
            "materials.destination_saturation_scale",
        }
        cube_names = {
            "materials.coloured_plastic_roughness",
            "materials.coloured_plastic_albedo_scale",
            "materials.cube_hue_jitter_deg",
            "materials.cube_saturation_scale",
        }
        self.assertTrue(destination_names <= set(spec.parameters))
        self.assertTrue(cube_names <= set(spec.parameters))
        self.assertTrue(LIGHT_COLOR_PARAMETER_NAMES <= set(spec.parameters))
        self.assertTrue(destination_names.isdisjoint(cube_names))

    def test_camera_matched_table_profile_is_training_only_and_auditable(self):
        from isaac_bridge.training_episode import apply_training_appearance_profile

        scene = SimpleNamespace(MAT_MARKINGS=False, SHOW_BACKDROP=True)
        evidence = apply_training_appearance_profile(scene)

        self.assertTrue(scene.MAT_MARKINGS)
        self.assertFalse(scene.SHOW_BACKDROP)
        self.assertEqual(evidence["name"], "real-table-video-broad-fit-v1")
        self.assertEqual(
            evidence["previous"],
            {"mat_markings": False, "clean_paper_backdrop": True},
        )
        self.assertIn("Compressed 640x360", evidence["limitation"])

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
        starter_diffuse = {
            name: value[0]
            for name, value in materials.items()
            if name.startswith("plastic_")
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
        fitted_base = scene._TRAINING_RANDOMIZATION_BASE["materials"]
        starter_base = scene._TRAINING_RANDOMIZATION_BASE[
            "starter_cube_materials"
        ]
        from isaac_bridge.training_episode import colour_patch_appearance_targets

        cube_appearance_targets, _ = colour_patch_appearance_targets()

        for colour in ("red", "yellow", "blue"):
            name = f"plastic_{colour}"
            self.assertEqual(
                fitted_base[name][0],
                cube_appearance_targets[colour]["nominal_diffuse_rgb"],
            )
            self.assertEqual(starter_base[name][0], starter_diffuse[name])
        self.assertEqual(
            scene._TRAINING_RANDOMIZATION_BASE["colour_patch_manifest"]["path"],
            "config/real_camera_colour_patches.json",
        )
        self.assertEqual(applied, set())

if __name__ == "__main__":
    unittest.main()
