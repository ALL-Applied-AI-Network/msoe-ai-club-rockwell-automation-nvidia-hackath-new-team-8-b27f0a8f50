"""Composable grasp generation, preflight, features, scoring, and selection.

This module is deliberately backend-neutral.  Components consume only the
camera-derived shared scene contract; simulator truth and robot-client objects
do not appear in the API.  The legacy implementations reproduce the current
single vertical cube grasp.  Their geometric checks are basic preflight, not
IK, collision, swept-path, or physical grasp validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from .contracts import (
    BinTarget,
    CubeDetection,
    GraspSelectionProvenance,
    PickPlacePlan,
    SceneDetections,
    TipPoseBase,
)
from .perception import DetectedObject, RobotState, SceneObservation


class GraspPlanningError(ValueError):
    """Raised when no valid grasp decision can be produced."""


def _nonempty(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_mapping(values: Mapping[str, Any], name: str) -> Mapping[str, float]:
    checked: dict[str, float] = {}
    for raw_key, raw_value in values.items():
        key = _nonempty(raw_key, f"{name} key")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{name}.{key} must be finite")
        if key in checked:
            raise ValueError(f"{name} contains duplicate key {key}")
        checked[key] = value
    return MappingProxyType(checked)


@dataclass(frozen=True)
class GraspPlanningContext:
    """Task and runtime context shared by all grasp-planning components."""

    destination: BinTarget
    release_tip_pose: TipPoseBase
    pick_tip_z_m: float
    approach_clearance_m: float
    generic_scene: SceneObservation | None = None
    unsupported_objects: tuple[DetectedObject, ...] | None = None
    robot_state: RobotState | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.destination, BinTarget):
            raise ValueError("destination must be a BinTarget")
        if not isinstance(self.release_tip_pose, TipPoseBase):
            raise ValueError("release_tip_pose must be a TipPoseBase")
        if self.generic_scene is not None and not isinstance(
            self.generic_scene, SceneObservation
        ):
            raise ValueError("generic_scene must be a SceneObservation or None")
        if self.generic_scene is not None and self.unsupported_objects is None:
            raise ValueError(
                "generic_scene requires an explicit unsupported_objects tuple"
            )
        unsupported = tuple(self.unsupported_objects or ())
        if not all(isinstance(value, DetectedObject) for value in unsupported):
            raise ValueError(
                "unsupported_objects must contain only DetectedObject values"
            )
        if unsupported and self.generic_scene is None:
            raise ValueError(
                "unsupported_objects require the complete generic_scene"
            )
        if self.generic_scene is not None:
            observed_ids = {value.track_id for value in self.generic_scene.objects}
            missing_ids = {
                value.track_id for value in unsupported
            } - observed_ids
            if missing_ids:
                raise ValueError(
                    "unsupported_objects must belong to generic_scene"
                )
        pick_z = float(self.pick_tip_z_m)
        approach = float(self.approach_clearance_m)
        if not math.isfinite(pick_z):
            raise ValueError("pick_tip_z_m must be finite")
        if not math.isfinite(approach) or approach <= 0.0:
            raise ValueError("approach_clearance_m must be positive and finite")
        if self.robot_state is not None and not isinstance(
            self.robot_state, RobotState
        ):
            raise ValueError("robot_state must be a RobotState or None")
        object.__setattr__(self, "pick_tip_z_m", pick_z)
        object.__setattr__(self, "approach_clearance_m", approach)
        object.__setattr__(self, "unsupported_objects", unsupported)


@dataclass(frozen=True)
class GraspCandidate:
    """One scene-bound top-down grasp proposed for a tracked target."""

    candidate_id: str
    scene_id: str
    target_track_id: str
    pick_tip_pose: TipPoseBase
    approach_tip_pose: TipPoseBase
    required_opening_m: float | None = None
    generator_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _nonempty(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "scene_id", _nonempty(self.scene_id, "scene_id"))
        object.__setattr__(
            self, "target_track_id", _nonempty(self.target_track_id, "target_track_id")
        )
        if not isinstance(self.pick_tip_pose, TipPoseBase):
            raise ValueError("pick_tip_pose must be a TipPoseBase")
        if not isinstance(self.approach_tip_pose, TipPoseBase):
            raise ValueError("approach_tip_pose must be a TipPoseBase")
        generator_id = str(self.generator_id).strip()
        object.__setattr__(self, "generator_id", generator_id)
        if self.required_opening_m is not None:
            object.__setattr__(
                self,
                "required_opening_m",
                _finite_nonnegative(self.required_opening_m, "required_opening_m"),
            )


@dataclass(frozen=True)
class FeasibilityResult:
    """Deterministic candidate preflight result and auditable measurements."""

    accepted: bool
    rejection_reasons: tuple[str, ...] = ()
    checker_id: str = ""
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise ValueError("accepted must be a boolean")
        accepted = self.accepted
        reasons = tuple(_nonempty(value, "rejection reason") for value in self.rejection_reasons)
        if accepted and reasons:
            raise ValueError("an accepted candidate cannot have rejection reasons")
        if not accepted and not reasons:
            raise ValueError("a rejected candidate must explain why")
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "checker_id", str(self.checker_id).strip())
        object.__setattr__(self, "metrics", _finite_mapping(self.metrics, "metrics"))


@dataclass(frozen=True)
class GraspFeatures:
    """Versioned numeric feature record shared by collection and inference."""

    schema_id: str
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_id", _nonempty(self.schema_id, "schema_id"))
        object.__setattr__(self, "values", _finite_mapping(self.values, "features"))


@dataclass(frozen=True)
class GraspScore:
    """Predicted complete-grasp outcome and optional time estimates.

    ``expected_total_episode_time_s`` is the unconditional expected duration
    of the team's bounded episode, including its fixed recovery/retry policy.
    It is the only time field used by the success/time selector.  Execution
    and recovery values are optional diagnostics and do not have enough
    semantics on their own to reconstruct total episode time.
    """

    success_probability: float
    expected_total_episode_time_s: float | None = None
    expected_execution_time_s: float | None = None
    expected_recovery_time_s: float | None = None
    scorer_id: str = ""
    model_version: str = ""

    def __post_init__(self) -> None:
        probability = float(self.success_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("success_probability must be finite and in [0, 1]")
        object.__setattr__(self, "success_probability", probability)
        for field_name in (
            "expected_total_episode_time_s",
            "expected_execution_time_s",
            "expected_recovery_time_s",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self, field_name, _finite_nonnegative(value, field_name)
                )
        object.__setattr__(self, "scorer_id", str(self.scorer_id).strip())
        object.__setattr__(self, "model_version", str(self.model_version).strip())


@dataclass(frozen=True)
class EvaluatedGrasp:
    candidate: GraspCandidate
    feasibility: FeasibilityResult
    features: GraspFeatures
    score: GraspScore

    def __post_init__(self) -> None:
        if not self.feasibility.accepted:
            raise ValueError("only accepted candidates may be evaluated and scored")


@dataclass(frozen=True)
class RejectedGrasp:
    """Candidate rejected by deterministic preflight before model scoring."""

    candidate: GraspCandidate
    feasibility: FeasibilityResult

    def __post_init__(self) -> None:
        if self.feasibility.accepted:
            raise ValueError("RejectedGrasp requires a rejected feasibility result")


@dataclass(frozen=True)
class GraspPlanningDecision:
    """Complete auditable result of one candidate-selection invocation."""

    plan: PickPlacePlan
    selected: EvaluatedGrasp
    evaluations: tuple[EvaluatedGrasp, ...]
    rejections: tuple[RejectedGrasp, ...]

    def __post_init__(self) -> None:
        evaluations = tuple(self.evaluations)
        rejections = tuple(self.rejections)
        if self.selected not in evaluations:
            raise ValueError("selected grasp must belong to evaluations")
        object.__setattr__(self, "evaluations", evaluations)
        object.__setattr__(self, "rejections", rejections)


@runtime_checkable
class GraspCandidateGenerator(Protocol):
    def generate(
        self,
        scene: SceneDetections,
        target: CubeDetection,
        context: GraspPlanningContext,
    ) -> tuple[GraspCandidate, ...]: ...


@runtime_checkable
class GraspFeasibilityChecker(Protocol):
    def check(
        self,
        scene: SceneDetections,
        candidate: GraspCandidate,
        context: GraspPlanningContext,
    ) -> FeasibilityResult: ...


@runtime_checkable
class GraspFeatureExtractor(Protocol):
    def extract(
        self,
        scene: SceneDetections,
        candidate: GraspCandidate,
        context: GraspPlanningContext,
    ) -> GraspFeatures: ...


@runtime_checkable
class GraspScorer(Protocol):
    def score(
        self, candidate: GraspCandidate, features: GraspFeatures
    ) -> GraspScore: ...


@runtime_checkable
class GraspSelector(Protocol):
    selector_id: str
    selection_objective: str

    def select(self, evaluations: tuple[EvaluatedGrasp, ...]) -> EvaluatedGrasp: ...


class LegacyTopDownCubeGenerator:
    """Generate the current single vertical grasp without changing its pose."""

    generator_id = "legacy-top-down-cube-v1"

    def generate(
        self,
        scene: SceneDetections,
        target: CubeDetection,
        context: GraspPlanningContext,
    ) -> tuple[GraspCandidate, ...]:
        pick = TipPoseBase(
            (
                target.position_base_m[0],
                target.position_base_m[1],
                context.pick_tip_z_m,
            ),
            target.yaw_rad,
        )
        approach = TipPoseBase(
            (
                pick.xyz_m[0],
                pick.xyz_m[1],
                pick.xyz_m[2] + context.approach_clearance_m,
            ),
            pick.yaw_rad,
        )
        return (
            GraspCandidate(
                candidate_id="legacy-top-down",
                scene_id=scene.scene_id,
                target_track_id=target.track_id,
                pick_tip_pose=pick,
                approach_tip_pose=approach,
                generator_id=self.generator_id,
            ),
        )


class LegacyGeometricChecker:
    """Basic reach/centre-clearance preflight, not collision validation."""

    checker_id = "legacy-radial-and-centre-clearance-v1"

    def __init__(
        self,
        *,
        maximum_radial_reach_m: float,
        minimum_centre_clearance_m: float,
    ) -> None:
        maximum = float(maximum_radial_reach_m)
        minimum = float(minimum_centre_clearance_m)
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum_radial_reach_m must be positive and finite")
        if not math.isfinite(minimum) or minimum < 0.0:
            raise ValueError(
                "minimum_centre_clearance_m must be finite and non-negative"
            )
        self.maximum_radial_reach_m = maximum
        self.minimum_centre_clearance_m = minimum

    def check(
        self,
        scene: SceneDetections,
        candidate: GraspCandidate,
        context: GraspPlanningContext,
    ) -> FeasibilityResult:
        pick_radius = math.hypot(*candidate.pick_tip_pose.xyz_m[:2])
        release_radius = math.hypot(*context.release_tip_pose.xyz_m[:2])
        neighbours = [
            math.dist(
                candidate.pick_tip_pose.xyz_m[:2], detection.position_base_m[:2]
            )
            for detection in scene.cubes
            if detection.track_id != candidate.target_track_id
        ]
        nearest = min(neighbours, default=math.inf)
        reasons: list[str] = []
        if pick_radius > self.maximum_radial_reach_m:
            reasons.append("pick pose exceeds radial reach")
        if release_radius > self.maximum_radial_reach_m:
            reasons.append("release pose exceeds radial reach")
        if nearest < self.minimum_centre_clearance_m:
            reasons.append("target centre is too close to another detection")
        unsupported_object_count = len(context.unsupported_objects)
        if unsupported_object_count:
            reasons.append(
                "generic scene contains unsupported objects; replace the "
                "legacy checker with one that models them"
            )
        metrics = {
            "pick_radius_m": pick_radius,
            "release_radius_m": release_radius,
            # Avoid infinity because all recorded numeric metrics are required
            # to be finite. The reach value is an explicit no-neighbour bound.
            "nearest_other_centre_m": (
                nearest if math.isfinite(nearest) else self.maximum_radial_reach_m
            ),
            "unsupported_object_count": float(unsupported_object_count),
        }
        return FeasibilityResult(
            accepted=not reasons,
            rejection_reasons=tuple(reasons),
            checker_id=self.checker_id,
            metrics=metrics,
        )


class LegacyFeatureExtractor:
    """Small stable feature schema suitable for a baseline scorer."""

    schema_id = "legacy-top-down-v2"

    def extract(
        self,
        scene: SceneDetections,
        candidate: GraspCandidate,
        context: GraspPlanningContext,
    ) -> GraspFeatures:
        neighbours = [
            math.dist(
                candidate.pick_tip_pose.xyz_m[:2], detection.position_base_m[:2]
            )
            for detection in scene.cubes
            if detection.track_id != candidate.target_track_id
        ]
        nearest = min(neighbours, default=0.0)
        opening_known = candidate.required_opening_m is not None
        values = {
            "pick_radius_m": math.hypot(*candidate.pick_tip_pose.xyz_m[:2]),
            "nearest_other_centre_m": nearest,
            "has_other_detection": float(bool(neighbours)),
            "absolute_yaw_rad": abs(candidate.pick_tip_pose.yaw_rad),
            "approach_distance_m": math.dist(
                candidate.pick_tip_pose.xyz_m, candidate.approach_tip_pose.xyz_m
            ),
            "required_opening_m": candidate.required_opening_m or 0.0,
            "required_opening_known": float(opening_known),
        }
        return GraspFeatures(self.schema_id, values)


class ConstantSuccessScorer:
    """Compatibility baseline: every preflight-accepted grasp scores equally."""

    def __init__(self, success_probability: float = 1.0) -> None:
        # Let the shared score validator own bounds and finite checks.
        self._score = GraspScore(
            success_probability,
            scorer_id="constant-success-v1",
            model_version="non-learned-baseline",
        )

    def score(
        self, candidate: GraspCandidate, features: GraspFeatures
    ) -> GraspScore:
        return self._score


class HighestSuccessSelector:
    """Choose maximum success probability with a deterministic ID tie-break."""

    selector_id = "highest-success-v1"
    selection_objective = "maximize_complete_grasp_success_probability"

    def select(self, evaluations: tuple[EvaluatedGrasp, ...]) -> EvaluatedGrasp:
        eligible = tuple(
            item
            for item in evaluations
            if item.score.success_probability > 0.0
        )
        if not eligible:
            raise GraspPlanningError(
                "no evaluated grasp has non-zero predicted success"
            )
        return min(
            eligible,
            key=lambda item: (-item.score.success_probability, item.candidate.candidate_id),
        )


class SuccessTimeSelector:
    """Minimize expected bounded-episode seconds per successful relocation.

    This implements the project's local throughput heuristic
    ``expected_total_episode_time_s / success_probability``.  It is not a
    whole-scene scheduling optimum.  Every non-zero-probability candidate must
    provide the same episode definition and recovery policy.
    """

    selector_id = "success-time-ratio-v1"
    selection_objective = "minimize_expected_episode_seconds_per_success"

    def select(self, evaluations: tuple[EvaluatedGrasp, ...]) -> EvaluatedGrasp:
        eligible = tuple(
            item
            for item in evaluations
            if item.score.success_probability > 0.0
        )
        if not eligible:
            raise GraspPlanningError(
                "no evaluated grasp has non-zero predicted success"
            )
        missing_time = tuple(
            item.candidate.candidate_id
            for item in eligible
            if item.score.expected_total_episode_time_s is None
        )
        if missing_time:
            raise GraspPlanningError(
                "success/time selection requires expected_total_episode_time_s "
                "for every eligible candidate: " + ", ".join(missing_time)
            )
        return min(
            eligible,
            key=lambda item: (
                item.score.expected_total_episode_time_s
                / item.score.success_probability,
                item.candidate.candidate_id,
            ),
        )


@dataclass(frozen=True)
class GraspPlanningPipeline:
    generator: GraspCandidateGenerator
    checker: GraspFeasibilityChecker
    feature_extractor: GraspFeatureExtractor
    scorer: GraspScorer
    selector: GraspSelector

    @classmethod
    def legacy_default(cls, config: Any) -> "GraspPlanningPipeline":
        """Build the compatibility pipeline from the shared task config."""

        return cls(
            LegacyTopDownCubeGenerator(),
            LegacyGeometricChecker(
                maximum_radial_reach_m=(
                    config.hardware.maximum_radial_reach_m
                ),
                minimum_centre_clearance_m=(
                    config.task.minimum_distinct_centres_m
                ),
            ),
            LegacyFeatureExtractor(),
            ConstantSuccessScorer(),
            HighestSuccessSelector(),
        )

    def plan_for_target(
        self,
        scene: SceneDetections,
        target: CubeDetection,
        context: GraspPlanningContext,
    ) -> PickPlacePlan:
        """Compatibility entry point returning only the executable plan."""

        return self.decide_for_target(scene, target, context).plan

    def decide_for_target(
        self,
        scene: SceneDetections,
        target: CubeDetection,
        context: GraspPlanningContext,
    ) -> GraspPlanningDecision:
        """Evaluate all candidates and retain evidence for logs/training."""

        if target not in scene.cubes:
            raise GraspPlanningError("target must belong to the supplied scene")
        if context.generic_scene is not None:
            if context.generic_scene.scene_id != scene.scene_id:
                raise GraspPlanningError(
                    "generic_scene and cube scene have different scene IDs"
                )
            if abs(
                context.generic_scene.capture_time_s - scene.capture_time_s
            ) > 1e-6:
                raise GraspPlanningError(
                    "generic_scene and cube scene have different timestamps"
                )
        try:
            candidates = tuple(self.generator.generate(scene, target, context))
        except GraspPlanningError:
            raise
        except Exception as exc:
            raise GraspPlanningError(
                f"candidate generator failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not candidates:
            raise GraspPlanningError("grasp generator returned no candidates")

        seen_ids: set[str] = set()
        evaluations: list[EvaluatedGrasp] = []
        rejected: list[RejectedGrasp] = []
        for candidate in candidates:
            if not isinstance(candidate, GraspCandidate):
                raise GraspPlanningError("generator returned a non-GraspCandidate")
            if candidate.candidate_id in seen_ids:
                raise GraspPlanningError(
                    f"duplicate grasp candidate ID: {candidate.candidate_id}"
                )
            seen_ids.add(candidate.candidate_id)
            if candidate.scene_id != scene.scene_id:
                raise GraspPlanningError(
                    f"candidate {candidate.candidate_id} belongs to a stale scene"
                )
            if candidate.target_track_id != target.track_id:
                raise GraspPlanningError(
                    f"candidate {candidate.candidate_id} targets another object"
                )

            try:
                feasibility = self.checker.check(scene, candidate, context)
            except GraspPlanningError:
                raise
            except Exception as exc:
                raise GraspPlanningError(
                    f"feasibility checker failed for {candidate.candidate_id}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(feasibility, FeasibilityResult):
                raise GraspPlanningError("checker returned an invalid result type")
            if not feasibility.accepted:
                rejected.append(RejectedGrasp(candidate, feasibility))
                continue

            try:
                features = self.feature_extractor.extract(
                    scene, candidate, context
                )
            except GraspPlanningError:
                raise
            except Exception as exc:
                raise GraspPlanningError(
                    f"feature extractor failed for {candidate.candidate_id}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(features, GraspFeatures):
                raise GraspPlanningError("feature extractor returned an invalid type")
            try:
                score = self.scorer.score(candidate, features)
            except GraspPlanningError:
                raise
            except Exception as exc:
                raise GraspPlanningError(
                    f"grasp scorer failed for {candidate.candidate_id}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(score, GraspScore):
                raise GraspPlanningError("scorer returned an invalid result type")
            # Reconstruct the result so even a custom scorer that bypassed the
            # frozen dataclass validator cannot smuggle NaN/inf into selection.
            try:
                score = GraspScore(
                    success_probability=score.success_probability,
                    expected_total_episode_time_s=(
                        score.expected_total_episode_time_s
                    ),
                    expected_execution_time_s=score.expected_execution_time_s,
                    expected_recovery_time_s=score.expected_recovery_time_s,
                    scorer_id=score.scorer_id,
                    model_version=score.model_version,
                )
            except (TypeError, ValueError) as exc:
                raise GraspPlanningError(
                    f"invalid score for candidate {candidate.candidate_id}: {exc}"
                ) from exc
            evaluations.append(
                EvaluatedGrasp(candidate, feasibility, features, score)
            )

        if not evaluations:
            detail = "; ".join(
                f"{item.candidate.candidate_id}: "
                + "; ".join(item.feasibility.rejection_reasons)
                for item in rejected
            ) or "all candidates were rejected"
            raise GraspPlanningError(f"no feasible grasp candidates: {detail}")

        try:
            selector_id = _nonempty(
                self.selector.selector_id, "selector.selector_id"
            )
            selection_objective = _nonempty(
                self.selector.selection_objective,
                "selector.selection_objective",
            )
            selected = self.selector.select(tuple(evaluations))
        except GraspPlanningError:
            raise
        except Exception as exc:
            raise GraspPlanningError(
                f"grasp selector failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(selected, EvaluatedGrasp) or selected not in evaluations:
            raise GraspPlanningError("selector returned an unknown grasp evaluation")
        candidate = selected.candidate
        plan = PickPlacePlan(
            plan_id=f"{scene.scene_id}:{target.track_id}:{target.colour}",
            scene_id=scene.scene_id,
            capture_time_s=scene.capture_time_s,
            target=target,
            destination=context.destination,
            pick_tip_pose=candidate.pick_tip_pose,
            release_tip_pose=context.release_tip_pose,
            approach_tip_pose=candidate.approach_tip_pose,
            grasp_selection=GraspSelectionProvenance(
                candidate_id=candidate.candidate_id,
                generator_id=(
                    candidate.generator_id
                    or type(self.generator).__name__
                ),
                checker_id=(
                    selected.feasibility.checker_id
                    or type(self.checker).__name__
                ),
                feature_schema_id=selected.features.schema_id,
                scorer_id=(
                    selected.score.scorer_id
                    or type(self.scorer).__name__
                ),
                model_version=(
                    selected.score.model_version or "unversioned"
                ),
                selector_id=selector_id,
                selection_objective=selection_objective,
                success_probability=selected.score.success_probability,
                expected_total_episode_time_s=(
                    selected.score.expected_total_episode_time_s
                ),
                expected_execution_time_s=(
                    selected.score.expected_execution_time_s
                ),
                expected_recovery_time_s=(
                    selected.score.expected_recovery_time_s
                ),
            ),
        )
        return GraspPlanningDecision(
            plan=plan,
            selected=selected,
            evaluations=tuple(evaluations),
            rejections=tuple(rejected),
        )


__all__ = [
    "ConstantSuccessScorer",
    "EvaluatedGrasp",
    "FeasibilityResult",
    "GraspCandidate",
    "GraspCandidateGenerator",
    "GraspFeatureExtractor",
    "GraspFeatures",
    "GraspFeasibilityChecker",
    "GraspPlanningContext",
    "GraspPlanningDecision",
    "GraspPlanningError",
    "GraspPlanningPipeline",
    "GraspScore",
    "GraspScorer",
    "GraspSelector",
    "HighestSuccessSelector",
    "LegacyFeatureExtractor",
    "LegacyGeometricChecker",
    "LegacyTopDownCubeGenerator",
    "RejectedGrasp",
    "SuccessTimeSelector",
]
