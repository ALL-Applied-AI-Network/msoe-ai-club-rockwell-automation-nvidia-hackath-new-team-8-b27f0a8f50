from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]

from sorting_core.table_framing import (  # noqa: E402
    TableFramingError,
    audit_table_framing,
    load_table_framing_evidence,
    verify_evidence_sources,
)


class TableFramingTests(unittest.TestCase):
    def test_checked_in_evidence_detects_stable_real_and_mismatched_sim(self):
        evidence = load_table_framing_evidence()
        report = audit_table_framing(evidence)
        rows = {row["name"]: row for row in report["comparisons"]}
        self.assertEqual(rows["real_fit_repeatability"]["status"], "within_budget")
        mismatch = rows["real_fit_vs_seed7_sim"]
        self.assertEqual(mismatch["status"], "mismatch_requires_investigation")
        self.assertGreater(mismatch["maximum_residual_px"], 50.0)
        self.assertAlmostEqual(
            mismatch["segments"][0]["length_error_px"], -43.0
        )
        self.assertFalse(report["automatic_correction_permitted"])

    def test_source_identity_verification_is_read_only_and_fail_closed(self):
        evidence = load_table_framing_evidence()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = next(iter(evidence["sources"].values()))
            path = root / first["workspace_relative_path"]
            path.parent.mkdir(parents=True)
            path.write_bytes(b"wrong evidence")
            result = verify_evidence_sources(evidence, root)
            self.assertFalse(result["all_sources_verified"])
            statuses = {row["status"] for row in result["sources"].values()}
            self.assertIn("identity_mismatch", statuses)
            self.assertIn("missing", statuses)

    def test_out_of_bounds_landmark_is_rejected(self):
        source = REPO / "config" / "table_framing_landmarks.json"
        value = json.loads(source.read_text(encoding="utf-8"))
        value = copy.deepcopy(value)
        value["frames"]["real_fit"]["landmarks_px"][
            "rear_left_table_corner"
        ] = [-1, 70]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(TableFramingError, "outside"):
                load_table_framing_evidence(path)


if __name__ == "__main__":
    unittest.main()
