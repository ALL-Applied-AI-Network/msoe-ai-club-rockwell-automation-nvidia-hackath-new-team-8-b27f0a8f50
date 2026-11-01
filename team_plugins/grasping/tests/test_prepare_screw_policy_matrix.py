import json
import math
from pathlib import Path
import tempfile
import unittest

from demos.scripted_cube_screw_bucket_sort.scenario import (
    build_sort_items,
    load_scenario_config,
)

from team_plugins.grasping.prepare_screw_policy_matrix import (
    ROOT,
    prepare,
)


class ScrewPolicyMatrixPreparerTests(unittest.TestCase):
    def test_preparer_builds_twelve_matched_external_cases(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "pilot"
            manifest_path = prepare(output, require_clean=False)
            manifest = json.loads(manifest_path.read_text("utf-8"))

            self.assertEqual(manifest["layout_count"], 4)
            self.assertEqual(manifest["attempt_count"], 12)
            self.assertEqual(len(manifest["cases"]), 12)
            self.assertEqual(
                set(manifest["candidate_offsets_mm"]),
                {12, 16, 20},
            )
            self.assertEqual(
                len(manifest["split_manifest_sha256"]),
                64,
            )

            by_group = {}
            for case in manifest["cases"]:
                by_group.setdefault(case["group_id"], []).append(case)
                self.assertTrue(
                    case["configured_pre_settle_planning"]["accepted"]
                )
                scenario_path = Path(case["scenario_path"])
                self.assertTrue(scenario_path.is_file())
                validated = load_scenario_config(scenario_path)
                target = next(
                    item
                    for item in build_sort_items(validated)
                    if item.item_id == manifest["target_item_id"]
                )
                self.assertEqual(
                    list(target.placement_reference_base_m),
                    case["placement_head_base_m"],
                )

            self.assertEqual(len(by_group), 4)
            for cases in by_group.values():
                self.assertEqual(len(cases), 3)
                self.assertEqual(
                    {case["pinch_from_head_mm"] for case in cases},
                    {12, 16, 20},
                )
                self.assertEqual(
                    len(
                        {
                            case["layout_fingerprint_sha256"]
                            for case in cases
                        }
                    ),
                    1,
                )
                self.assertEqual(
                    len(
                        {
                            tuple(case["target_head_base_m"])
                            for case in cases
                        }
                    ),
                    1,
                )
                reference_head = cases[0]["placement_head_base_m"]
                self.assertLess(
                    max(
                        math.dist(reference_head, case["placement_head_base_m"])
                        for case in cases
                    ),
                    1e-12,
                )

            for case in manifest["cases"]:
                if case["tip_obstacle_gap_m"] is not None:
                    self.assertGreaterEqual(
                        case["tip_obstacle_gap_m"],
                        0.008,
                    )

    def test_preparer_refuses_output_inside_checkout(self):
        with self.assertRaisesRegex(
            ValueError, "outside the Git checkout"
        ):
            prepare(
                ROOT / "generated-pilot-must-not-exist",
                require_clean=False,
            )


if __name__ == "__main__":
    unittest.main()
