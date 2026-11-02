"""Benchmark-only cube/screw grasp selection with auditable AI binding.

This module deliberately performs no actuation.  It bridges a generic,
camera-derived :class:`~sorting_core.perception.SceneObservation` to the
existing catalog candidate generators and deterministic envelope checks.  The
selected :class:`~sorting_core.grasping.GraspCandidate` is intended for an
experiment executor that separately validates and executes its motion.

Two modes are available:

``deterministic``
    The existing untrained geometric ranking prior selects among candidates
    that passed deterministic preflight.  It is a reproducible heuristic, not
    AI and not a calibrated grasp-success probability.

``ai``
    An external pretrained ranker may bind only one of the same feasible
    catalog candidates.  The existing fail-closed policy validates its reply,
    writes a durable receipt, and falls back to the deterministic choice when
    the provider abstains, fails, or lacks sufficient confidence/margin.

Simulator truth is neither accepted nor imported by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

from sorting_core import (
    BinTarget,
    SceneObservation,
    TipPoseBase,
    cube_scene_from_observation,
)
from sorting_core.grasping import (
    EvaluatedGrasp,
    GraspPlanningContext,
    GraspPlanningError,
    RejectedGrasp,
)
from sorting_core.integration_plugins import GraspFactoryContext
from team_plugins.grasping.ai_selected_sim_runner import (
    build_ai_selected_offline_screw_planner,
    build_ai_selected_simulation_planner,
)
from team_plugins.grasping.ai_selection import AISelectionProvider
from team_plugins.grasping.multigrasp_pipeline import (
    SCREW_CLASS_ID,
    build_offline_screw_planner,
    create_simulation,
)


GraspMode = Literal["deterministic", "ai"]
_CUBE_CLASS_IDS = ("cube",)
_VALID_MODES = frozenset({"deterministic", "ai"})
_SCORE_WARNING = (
    "baseline score is an uncalibrated geometric ranking value; it is not a "
    "measured complete-grasp success probability"
)


def _last_json_line(path: Path) -> Mapping[str, Any]:
    """Read the most recent durable AI receipt without weakening its writer."""

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise GraspPlanningError("AI receipt file is empty after selection")
    payload = json.loads(lines[-1])
    if not isinstance(payload, dict):
        raise GraspPlanningError("AI receipt is not a JSON object")
    return MappingProxyType(payload)


@dataclass(frozen=True)
class BenchmarkGraspDecision:
    """One target-bound selection result for the randomized benchmark."""

    scene_id: str
    target_track_id: str
    target_class_id: str
    mode: GraspMode
    selected: EvaluatedGrasp
    evaluations: tuple[EvaluatedGrasp, ...]
    rejections: tuple[RejectedGrasp, ...]
    baseline_candidate_id: str
    selection_status: str
    ai_bound: bool
    used_fallback: bool
    provider_id: str | None
    model_version: str | None
    proposed_candidate_id: str | None
    receipt: Mapping[str, Any] | None
    legacy_executor_supported: bool

    def __post_init__(self) -> None:
        evaluations = tuple(self.evaluations)
        rejections = tuple(self.rejections)
        if self.selected not in evaluations:
            raise ValueError("selected grasp must belong to evaluations")
        if self.selected.candidate.scene_id != self.scene_id:
            raise ValueError("selected grasp belongs to a different scene")
        if self.selected.candidate.target_track_id != self.target_track_id:
            raise ValueError("selected grasp belongs to a different target")
        if self.mode not in _VALID_MODES:
            raise ValueError("mode must be deterministic or ai")
        if self.mode == "deterministic" and (self.ai_bound or self.used_fallback):
            raise ValueError("deterministic decisions cannot report AI state")
        if self.ai_bound and self.used_fallback:
            raise ValueError("a bound AI decision cannot also use fallback")
        object.__setattr__(self, "evaluations", evaluations)
        object.__setattr__(self, "rejections", rejections)
        if self.receipt is not None:
            object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))

    @property
    def selection_changed_from_baseline(self) -> bool:
        return self.selected.candidate.candidate_id != self.baseline_candidate_id

    def to_record(self) -> dict[str, Any]:
        """Return a compact JSON-compatible experiment record."""

        return {
            "schema_version": "randomized-sort-grasp-decision-v1",
            "scene_id": self.scene_id,
            "target_track_id": self.target_track_id,
            "target_class_id": self.target_class_id,
            "mode": self.mode,
            "selected_candidate_id": self.selected.candidate.candidate_id,
            "baseline_candidate_id": self.baseline_candidate_id,
            "selection_changed_from_baseline": self.selection_changed_from_baseline,
            "selection_status": self.selection_status,
            "ai_bound": self.ai_bound,
            "used_fallback": self.used_fallback,
            "provider_id": self.provider_id,
            "model_version": self.model_version,
            "proposed_candidate_id": self.proposed_candidate_id,
            "ai_confidence": (
                None if self.receipt is None else self.receipt.get("confidence")
            ),
            "ai_score_margin": (
                None if self.receipt is None else self.receipt.get("score_margin")
            ),
            "ai_reason": (
                None if self.receipt is None else self.receipt.get("reason")
            ),
            # Model confidence/ranking score is intentionally separate from
            # complete-grasp success probability.  This benchmark must not
            # relabel one as the other in submission metrics.
            "ai_score_semantics": (
                None
                if self.receipt is None
                else "provider ranking confidence; not a calibrated grasp-success probability"
            ),
            "feasible_candidate_ids": [
                item.candidate.candidate_id for item in self.evaluations
            ],
            "rejected_candidates": [
                {
                    "candidate_id": item.candidate.candidate_id,
                    "reasons": list(item.feasibility.rejection_reasons),
                }
                for item in self.rejections
            ],
            "candidate_count": len(self.evaluations) + len(self.rejections),
            "feasible_candidate_count": len(self.evaluations),
            "rejected_candidate_count": len(self.rejections),
            "baseline_score_semantics": _SCORE_WARNING,
            "legacy_executor_supported": self.legacy_executor_supported,
            "candidate_motion_contract": "top_down_fingertip_xyz_plus_yaw",
            "receipt_record_sha256": (
                None if self.receipt is None else self.receipt.get("record_sha256")
            ),
        }


class BenchmarkGraspPlanner:
    """Plan cube or screw catalog grasps without importing an executor.

    ``factory_context.options`` must explicitly select the existing
    simulation-only geometric prior.  That prior is the deterministic
    baseline and the fail-closed fallback for AI mode.  It must never be
    described as a learned model or calibrated probability.
    """

    def __init__(
        self,
        factory_context: GraspFactoryContext,
        *,
        mode: GraspMode,
        ai_provider: AISelectionProvider | None = None,
        receipt_directory: str | Path | None = None,
        minimum_confidence: float = 0.7,
        minimum_score_margin: float = 0.1,
        timeout_s: float = 20.0,
    ) -> None:
        if not isinstance(factory_context, GraspFactoryContext):
            raise TypeError("factory_context must be a GraspFactoryContext")
        if mode not in _VALID_MODES:
            raise ValueError("mode must be deterministic or ai")
        if factory_context.options.get("simulation_only") is not True:
            raise ValueError("benchmark grasp selection requires simulation_only=true")
        if factory_context.options.get("scorer", "geometric_prior") != "geometric_prior":
            raise ValueError(
                "benchmark baseline must use the explicit geometric_prior scorer"
            )

        self.context = factory_context
        self.mode: GraspMode = mode
        self._cube_baseline = create_simulation(factory_context)
        self._screw_baseline = build_offline_screw_planner(
            factory_context.config,
            options=factory_context.options,
        )
        self._cube_ai = None
        self._screw_ai = None
        self._cube_receipt_path: Path | None = None
        self._screw_receipt_path: Path | None = None

        if mode == "ai":
            if ai_provider is None:
                raise ValueError("ai mode requires an external AISelectionProvider")
            if receipt_directory is None:
                raise ValueError("ai mode requires a durable receipt_directory")
            receipt_root = Path(receipt_directory).expanduser().resolve()
            receipt_root.mkdir(parents=True, exist_ok=True)
            self._cube_receipt_path = receipt_root / "cube-ai-selection.jsonl"
            self._screw_receipt_path = receipt_root / "screw-ai-selection.jsonl"
            self._cube_ai = build_ai_selected_simulation_planner(
                factory_context,
                ai_provider,
                receipt_path=self._cube_receipt_path,
                mode="simulation_binding",
                minimum_confidence=minimum_confidence,
                minimum_score_margin=minimum_score_margin,
                timeout_s=timeout_s,
            )
            self._screw_ai = build_ai_selected_offline_screw_planner(
                factory_context,
                ai_provider,
                receipt_path=self._screw_receipt_path,
                mode="simulation_binding",
                minimum_confidence=minimum_confidence,
                minimum_score_margin=minimum_score_margin,
                timeout_s=timeout_s,
            )

    def decide(
        self,
        scene: SceneObservation,
        target_track_id: str,
        *,
        release_tip_pose: TipPoseBase,
        pick_tip_z_m: float | None = None,
        approach_clearance_m: float | None = None,
    ) -> BenchmarkGraspDecision:
        """Select one feasible catalog candidate for a camera-observed target."""

        if not isinstance(scene, SceneObservation):
            raise TypeError("scene must be a SceneObservation")
        if not isinstance(release_tip_pose, TipPoseBase):
            raise TypeError("release_tip_pose must be a TipPoseBase")
        target = next(
            (item for item in scene.objects if item.track_id == target_track_id),
            None,
        )
        if target is None:
            raise GraspPlanningError("target does not belong to the scene")
        if target.class_id not in {*_CUBE_CLASS_IDS, SCREW_CLASS_ID}:
            raise GraspPlanningError(
                f"unsupported benchmark target class {target.class_id!r}"
            )
        unknown_classes = sorted(
            {
                item.class_id
                for item in scene.objects
                if item.class_id not in {*_CUBE_CLASS_IDS, SCREW_CLASS_ID}
            }
        )
        if unknown_classes:
            raise GraspPlanningError(
                "unsupported scene class cannot be assumed to be clear space: "
                + ", ".join(unknown_classes)
            )
        if target.sort_label not in {"red", "yellow", "blue"}:
            raise GraspPlanningError("target requires a supported destination label")
        if target.class_id == SCREW_CLASS_ID and target.sort_label != "red":
            raise GraspPlanningError("benchmark screws must use the red destination")

        config = self.context.config
        pick_z = config.pick_tip_base_z_m if pick_tip_z_m is None else pick_tip_z_m
        approach = (
            config.task.approach_clearance_m
            if approach_clearance_m is None
            else approach_clearance_m
        )

        if target.class_id in _CUBE_CLASS_IDS:
            baseline, selected, receipt = self._decide_cube(
                scene,
                target_track_id,
                release_tip_pose,
                pick_z,
                approach,
            )
            legacy_executor_supported = True
        else:
            baseline, selected, receipt = self._decide_screw(
                scene,
                target_track_id,
                release_tip_pose,
                pick_z,
                approach,
            )
            # The existing shared executor is cube-only.  A benchmark executor
            # must explicitly implement and validate screw motion before use.
            legacy_executor_supported = False

        if receipt is None:
            status = "deterministic_geometric_baseline"
            ai_bound = False
            used_fallback = False
            provider_id = None
            model_version = None
            proposed_candidate_id = None
        else:
            status = str(receipt["status"])
            ai_bound = status == "bound"
            used_fallback = bool(receipt["used_fallback"])
            provider_id = str(receipt["provider_id"])
            model_version = str(receipt["model_version"])
            proposed_candidate_id = receipt.get("proposed_candidate_id")
            if receipt.get("executed_candidate_id") != selected.selected.candidate.candidate_id:
                raise GraspPlanningError(
                    "AI receipt executed candidate disagrees with planner decision"
                )

        return BenchmarkGraspDecision(
            scene_id=scene.scene_id,
            target_track_id=target.track_id,
            target_class_id=target.class_id,
            mode=self.mode,
            selected=selected.selected,
            evaluations=selected.evaluations,
            rejections=selected.rejections,
            baseline_candidate_id=baseline.selected.candidate.candidate_id,
            selection_status=status,
            ai_bound=ai_bound,
            used_fallback=used_fallback,
            provider_id=provider_id,
            model_version=model_version,
            proposed_candidate_id=proposed_candidate_id,
            receipt=receipt,
            legacy_executor_supported=legacy_executor_supported,
        )

    def _decide_cube(
        self,
        scene: SceneObservation,
        target_track_id: str,
        release_tip_pose: TipPoseBase,
        pick_tip_z_m: float,
        approach_clearance_m: float,
    ):
        projection = cube_scene_from_observation(scene, class_ids=_CUBE_CLASS_IDS)
        target = next(
            (item for item in projection.scene.cubes if item.track_id == target_track_id),
            None,
        )
        if target is None:
            raise GraspPlanningError("cube target is absent from cube projection")
        context = GraspPlanningContext(
            destination=BinTarget(target.colour, release_tip_pose.xyz_m[:2]),
            release_tip_pose=release_tip_pose,
            pick_tip_z_m=pick_tip_z_m,
            approach_clearance_m=approach_clearance_m,
            generic_scene=scene,
            # Both catalog classes are modeled by SceneEnvelopeChecker.  An
            # actually unknown class is rejected before reaching this method.
            unsupported_objects=(),
        )
        baseline = self._cube_baseline.decide_for_target(
            projection.scene, target, context
        )
        if self.mode == "deterministic":
            return baseline, baseline, None
        selected = self._cube_ai.decide_for_target(projection.scene, target, context)
        return baseline, selected, _last_json_line(self._cube_receipt_path)

    def _decide_screw(
        self,
        scene: SceneObservation,
        target_track_id: str,
        release_tip_pose: TipPoseBase,
        pick_tip_z_m: float,
        approach_clearance_m: float,
    ):
        baseline = self._screw_baseline.decide(
            scene,
            target_track_id,
            pick_tip_z_m=pick_tip_z_m,
            approach_clearance_m=approach_clearance_m,
            release_tip_pose=release_tip_pose,
        )
        if self.mode == "deterministic":
            return baseline, baseline, None
        selected = self._screw_ai.decide(
            scene,
            target_track_id,
            pick_tip_z_m=pick_tip_z_m,
            approach_clearance_m=approach_clearance_m,
            release_tip_pose=release_tip_pose,
        )
        return baseline, selected, _last_json_line(self._screw_receipt_path)


__all__ = [
    "BenchmarkGraspDecision",
    "BenchmarkGraspPlanner",
    "GraspMode",
]
