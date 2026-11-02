from __future__ import annotations

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[1]
CELL = REPO / "hardware" / "fanuc_cell"
for path in (CELL, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calibration.geometry import load as load_calibration  # noqa: E402
from calibration.geometry import project as project_camera_points  # noqa: E402
from sorting_core import (  # noqa: E402
    DestinationMaskError,
    build_configured_bucket_pixel_mask,
    ignore_configured_bucket_pixels,
    load_config,
)


class ConfiguredDestinationMaskTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.table_camera = load_calibration()["table"]
        self.project = lambda point: project_camera_points(
            self.table_camera, np.asarray([point], dtype=float)
        )[0]

    @staticmethod
    def _bounds(polygon):
        points = np.asarray(polygon, dtype=int)
        return (
            int(points[:, 0].min()),
            int(points[:, 1].min()),
            int(points[:, 0].max()),
            int(points[:, 1].max()),
        )

    def test_calibrated_projection_matches_reviewed_isaac_bin_silhouettes(self):
        _, polygons = build_configured_bucket_pixel_mask(
            (1080, 1920), self.project, self.config, margin_px=0
        )
        # Reviewed Isaac sensor frame f00000 showed visible bucket bounds within
        # 0--2 pixels of these calibration-derived cuboid projections.
        expected = {
            "red": (740, 543, 808, 624),
            "yellow": (953, 552, 1012, 632),
            "blue": (1152, 560, 1225, 640),
        }
        for colour, bounds in expected.items():
            measured = self._bounds(polygons[colour])
            np.testing.assert_allclose(measured, bounds, atol=1)

    def test_mask_covers_destinations_but_not_pickup_cube_centres(self):
        mask, polygons = build_configured_bucket_pixel_mask(
            (1080, 1920), self.project, self.config
        )
        self.assertEqual(set(polygons), {"red", "yellow", "blue"})
        for colour, centre in self.config.task.bin_centres_base_m.items():
            uv = np.rint(
                self.project(
                    (
                        centre[0],
                        centre[1],
                        self.config.frames.surface_base_z_m
                        + self.config.bin.height_m / 2.0,
                    )
                )
            ).astype(int)
            self.assertEqual(mask[uv[1], uv[0]], 255, colour)
        for colour, spawn in self.config.task.cube_spawns_base_xy_yaw_deg.items():
            uv = np.rint(
                self.project(
                    (
                        spawn[0],
                        spawn[1],
                        self.config.cube_top_base_z_m,
                    )
                )
            ).astype(int)
            self.assertEqual(mask[uv[1], uv[0]], 0, colour)

    def test_ignore_returns_copy_and_preserves_every_unmasked_pixel(self):
        image = np.full((1080, 1920, 3), 205, dtype=np.uint8)
        mask, polygons = build_configured_bucket_pixel_mask(
            image.shape[:2], self.project, self.config
        )
        bucket_colours = {
            "red": (190, 80, 20),
            "yellow": (185, 175, 45),
            "blue": (55, 115, 165),
        }
        for colour, polygon in polygons.items():
            cv2.fillConvexPoly(
                image,
                np.asarray(polygon, dtype=np.int32),
                bucket_colours[colour],
            )
        source = image.copy()
        result = ignore_configured_bucket_pixels(
            image, self.project, self.config
        )
        np.testing.assert_array_equal(image, source)
        np.testing.assert_array_equal(result.rgb[result.mask == 0], source[result.mask == 0])
        self.assertEqual(result.masked_pixel_count, int(np.count_nonzero(mask)))
        self.assertFalse(np.array_equal(result.rgb[result.mask != 0], source[result.mask != 0]))
        self.assertTrue(result.metadata()["applied"])

    def test_invalid_projection_fails_instead_of_silently_skipping_mask(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with self.assertRaisesRegex(DestinationMaskError, "behind the camera"):
            ignore_configured_bucket_pixels(
                image, lambda point: None, self.config
            )


if __name__ == "__main__":
    unittest.main()
