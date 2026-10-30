from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from demos.scripted_colour_bucket_sort.colour_bucket_demo import (  # noqa: E402
    TASK_CONFIG,
    generate_background_distractors,
)


class TrainingClutterTests(unittest.TestCase):
    def test_seeded_background_clutter_is_reproducible_and_outside_pickup(self):
        first = generate_background_distractors(42, 4)
        second = generate_background_distractors(42, 4)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        pickup_x_max = TASK_CONFIG.task.pickup_region_base_m["x"][1]
        for row in first:
            x = row["centre_base_xy_m"][0]
            half_x = row["dimensions_m"][0] / 2.0
            self.assertGreater(x - half_x, pickup_x_max + 0.025)
            self.assertEqual(row["class_name"], "background_distractor")
            self.assertEqual(row["physics"], "visual-only; no collider")

    def test_clutter_count_is_bounded(self):
        self.assertEqual(generate_background_distractors(1, 0), ())
        for invalid in (-1, 5, True):
            with self.assertRaises(ValueError):
                generate_background_distractors(1, invalid)


if __name__ == "__main__":
    unittest.main()
