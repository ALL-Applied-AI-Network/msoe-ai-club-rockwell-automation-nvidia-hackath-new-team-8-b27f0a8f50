from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from experiments.randomized_sort_benchmark.grasp_selection import (
    BenchmarkGraspPlanner,
)
from sorting_core import DetectedObject, Pose3D, SceneObservation, TipPoseBase, load_config
from sorting_core.grasping import GraspPlanningError
from sorting_core.integration_plugins import GraspFactoryContext
from team_plugins.grasping.ai_selection import RESPONSE_SCHEMA_VERSION
from team_plugins.grasping.multigrasp_pipeline import SCREW_CLASS_ID


class _RankingProvider:
    provider_id = "fixture-pretrained-ranker"
    model_version = "fixture-model@abc123"
    prompt_version = "fixture-prompt-v1"

    def __init__(self, preferred_suffix: str) -> None:
        self.preferred_suffix = preferred_suffix
        self.calls: list[list[str]] = []

    def select(self, request_json: str, *, timeout_s: float):
        request = json.loads(request_json)
        candidate_ids = [item["candidate_id"] for item in request["candidates"]]
        self.calls.append(candidate_ids)
        selected = next(
            item for item in candidate_ids if item.startswith(self.preferred_suffix)
        )
        ordered = [selected] + sorted(item for item in candidate_ids if item != selected)
        return {
            "schema_version": RESPONSE_SCHEMA_VERSION,
            "request_sha256": hashlib.sha256(request_json.encode("utf-8")).hexdigest(),
            "model_version": self.model_version,
            "selected_candidate_id": selected,
            "confidence": 0.95,
            "ranking": [
                {"candidate_id": item, "score": 0.95 - index * 0.25}
                for index, item in enumerate(ordered)
            ],
            "rationale": "deterministic fixture response",
        }


class _FailingProvider:
    provider_id = "fixture-failing-ranker"
    model_version = "fixture-model@failure"
    prompt_version = "fixture-prompt-v1"

    def select(self, request_json: str, *, timeout_s: float):
        raise TimeoutError("fixture timeout")


class BenchmarkGraspSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.capture_time_s = 12.5

    def _context(self) -> GraspFactoryContext:
        return GraspFactoryContext(
            config=self.config,
            checkpoint=None,
            options={
                "scorer": "geometric_prior",
                "yaw_offsets_deg": [0.0, 45.0, 90.0],
                "required_clearance_m": 0.001,
                "minimum_confidence": 0.2,
                "minimum_pose_quality": 0.2,
                "simulation_only": True,
            },
            profile_id="randomized-four-combo-test",
            profile_path="experiments/randomized_sort_benchmark/test-profile.json",
            profile_sha256="0" * 64,
        )

    def _cube(self, track_id="cube-red-0", *, x=0.43, y=0.0) -> DetectedObject:
        return DetectedObject(
            track_id=track_id,
            class_id="cube",
            sort_label="red",
            capture_time_s=self.capture_time_s,
            pose=Pose3D.from_yaw((x, y, -0.0001), 0.15),
            dimensions_m=(0.01, 0.01, 0.01),
            confidence=0.95,
            pose_quality=0.9,
            visibility=0.95,
            source_camera_ids=("table",),
        )

    def _screw(self, *, sort_label="red") -> DetectedObject:
        half = math.sqrt(0.5)
        return DetectedObject(
            track_id="screw-0",
            class_id=SCREW_CLASS_ID,
            sort_label=sort_label,
            capture_time_s=self.capture_time_s,
            pose=Pose3D(
                position_base_m=(0.40, -0.03, 0.0014),
                orientation_xyzw=(0.0, -half, 0.0, half),
            ),
            dimensions_m=(0.012, 0.01199627, 0.030),
            confidence=0.95,
            pose_quality=0.9,
            visibility=0.9,
            source_camera_ids=("table",),
        )

    def _scene(self, *objects: DetectedObject) -> SceneObservation:
        return SceneObservation(
            scene_id="scene-0001",
            capture_time_s=self.capture_time_s,
            objects=tuple(objects),
            provider_name="camera-fixture",
            provider_version="1",
            calibration_ids=("table-cal-v1",),
        )

    def _release(self) -> TipPoseBase:
        x, y = self.config.task.bin_centres_base_m["red"]
        return TipPoseBase((x, y, self.config.bin_release_tip_base_z_m), 0.0)

    def test_deterministic_cube_is_explicitly_uncalibrated_baseline(self):
        planner = BenchmarkGraspPlanner(self._context(), mode="deterministic")
        decision = planner.decide(
            self._scene(self._cube()), "cube-red-0", release_tip_pose=self._release()
        )

        self.assertEqual(decision.mode, "deterministic")
        self.assertEqual(decision.selected.candidate.candidate_id, "cube-yaw-000-0")
        self.assertEqual(len(decision.evaluations), 3)
        self.assertFalse(decision.ai_bound)
        self.assertFalse(decision.used_fallback)
        self.assertFalse(decision.selection_changed_from_baseline)
        self.assertIn("uncalibrated", decision.to_record()["baseline_score_semantics"])

    def test_ai_cube_binds_only_a_feasible_existing_candidate(self):
        provider = _RankingProvider("cube-yaw-090")
        with tempfile.TemporaryDirectory() as directory:
            planner = BenchmarkGraspPlanner(
                self._context(),
                mode="ai",
                ai_provider=provider,
                receipt_directory=directory,
            )
            decision = planner.decide(
                self._scene(self._cube()),
                "cube-red-0",
                release_tip_pose=self._release(),
            )
            receipt_files = sorted(Path(directory).glob("*.jsonl"))

        self.assertEqual(decision.selected.candidate.candidate_id, "cube-yaw-090-2")
        self.assertTrue(decision.ai_bound)
        self.assertFalse(decision.used_fallback)
        self.assertTrue(decision.selection_changed_from_baseline)
        self.assertEqual(set(provider.calls[0]), set(decision.to_record()["feasible_candidate_ids"]))
        self.assertEqual(len(receipt_files), 2)

    def test_ai_provider_failure_is_visible_deterministic_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            planner = BenchmarkGraspPlanner(
                self._context(),
                mode="ai",
                ai_provider=_FailingProvider(),
                receipt_directory=directory,
            )
            decision = planner.decide(
                self._scene(self._cube()),
                "cube-red-0",
                release_tip_pose=self._release(),
            )

        self.assertFalse(decision.ai_bound)
        self.assertTrue(decision.used_fallback)
        self.assertEqual(decision.selection_status, "fallback_provider_error")
        self.assertEqual(
            decision.selected.candidate.candidate_id, decision.baseline_candidate_id
        )

    def test_ai_screw_uses_same_feasible_catalog_and_reports_executor_gap(self):
        provider = _RankingProvider("screw-shaft-012mm")
        with tempfile.TemporaryDirectory() as directory:
            planner = BenchmarkGraspPlanner(
                self._context(),
                mode="ai",
                ai_provider=provider,
                receipt_directory=directory,
            )
            decision = planner.decide(
                self._scene(self._screw()),
                "screw-0",
                release_tip_pose=self._release(),
            )

        self.assertEqual(
            decision.selected.candidate.candidate_id, "screw-shaft-012mm-0"
        )
        self.assertTrue(decision.ai_bound)
        self.assertTrue(decision.selection_changed_from_baseline)
        self.assertFalse(decision.legacy_executor_supported)
        self.assertEqual(
            provider.calls[0],
            ["screw-shaft-012mm-0", "screw-shaft-016mm-1", "screw-shaft-020mm-2"],
        )

    def test_unknown_scene_object_is_not_silently_treated_as_clear_space(self):
        unknown = DetectedObject(
            track_id="unknown-0",
            class_id="unknown_obstacle",
            sort_label=None,
            capture_time_s=self.capture_time_s,
            pose=Pose3D.from_yaw((0.46, 0.0, 0.01), 0.0),
            dimensions_m=(0.02, 0.02, 0.02),
            confidence=0.8,
            pose_quality=0.5,
            visibility=0.7,
            source_camera_ids=("table",),
        )
        planner = BenchmarkGraspPlanner(self._context(), mode="deterministic")
        with self.assertRaisesRegex(GraspPlanningError, "unsupported scene class"):
            planner.decide(
                self._scene(self._cube(), unknown),
                "cube-red-0",
                release_tip_pose=self._release(),
            )

    def test_screw_destination_must_be_red(self):
        planner = BenchmarkGraspPlanner(self._context(), mode="deterministic")
        with self.assertRaisesRegex(GraspPlanningError, "red destination"):
            planner.decide(
                self._scene(self._screw(sort_label="blue")),
                "screw-0",
                release_tip_pose=self._release(),
            )


if __name__ == "__main__":
    unittest.main()
