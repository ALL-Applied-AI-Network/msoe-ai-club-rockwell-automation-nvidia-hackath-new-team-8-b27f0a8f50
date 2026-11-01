"""Auditable cube multi-grasp selection and offline screw candidate planning.

The shared executable contract is currently cube-only. This module therefore
provides a normal GraspPlanningPipeline for cubes and a separate offline screw
planner that uses the same features and scorer without pretending that the
shared executor can already command a screw grasp.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from sorting_core import BinTarget, CubeDetection, SceneDetections, TipPoseBase
from sorting_core.grasping import (
    EvaluatedGrasp,
    FeasibilityResult,
    GraspCandidate,
    GraspFeatures,
    GraspPlanningContext,
    GraspPlanningError,
    GraspPlanningPipeline,
    GraspScore,
    HighestSuccessSelector,
    RejectedGrasp,
)
from sorting_core.perception import DetectedObject, Pose3D, SceneObservation

from team_plugins.grasping.ai_selection import (
    AIAssistedSelector,
    _AI_SIMULATION_CAPABILITY,
    _create_ai_selection_policy,
)

PLUGIN_API_VERSION = 1
FEATURE_SCHEMA_ID = "team8-preaction-grasp-features-v1"
SCREW_CLASS_ID = "screw_iso10642_m6x30"
CUBE_GENERATOR_ID = "team8-cube-yaw-family-v1"
SCREW_GENERATOR_ID = "team8-screw-shaft-regions-v1"
CHECKER_ID = "team8-scene-envelope-v1"
FEATURE_NAMES = (
    "object_is_screw",
    "axial_offset_m",
    "yaw_offset_abs_rad",
    "required_opening_m",
    "opening_fraction",
    "pick_radius_m",
    "release_radius_m",
    "nearest_object_centre_m",
    "minimum_footprint_clearance_m",
    "neighbour_count",
    "unknown_geometry_count",
    "confidence",
    "pose_quality",
    "visibility",
    "approach_distance_m",
)


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _unit_interval(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _quaternion_yaw(quaternion_xyzw: Sequence[float]) -> float:
    x, y, z, w = quaternion_xyzw
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _square_yaw_distance(first: float, second: float) -> float:
    return min(
        abs(_wrap_angle(first - second + quarter_turns * math.pi / 2.0))
        for quarter_turns in range(4)
    )


def _bounding_radius(item: DetectedObject) -> float | None:
    if item.dimensions_m is None:
        return None
    # Dimensions are object-local and supported poses may be tilted. The full
    # 3-D half-diagonal is an orientation-independent conservative table-plane
    # radius; a tighter bound would require projecting all rotated corners.
    return 0.5 * math.sqrt(sum(value * value for value in item.dimensions_m))


@dataclass(frozen=True)
class _EnvelopeSettings:
    maximum_radial_reach_m: float
    maximum_opening_m: float
    required_clearance_m: float
    finger_width_m: float
    finger_thickness_m: float
    cube_edge_m: float
    minimum_confidence: float
    minimum_pose_quality: float
    maximum_target_position_disagreement_m: float
    maximum_target_dimension_disagreement_m: float
    maximum_target_yaw_disagreement_rad: float

    def __post_init__(self) -> None:
        for name in (
            "maximum_radial_reach_m",
            "maximum_opening_m",
            "finger_width_m",
            "finger_thickness_m",
            "cube_edge_m",
            "maximum_target_position_disagreement_m",
            "maximum_target_dimension_disagreement_m",
            "maximum_target_yaw_disagreement_rad",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        clearance = _finite(self.required_clearance_m, "required_clearance_m")
        if clearance < 0.0:
            raise ValueError("required_clearance_m must be non-negative")
        object.__setattr__(self, "required_clearance_m", clearance)
        object.__setattr__(
            self,
            "minimum_confidence",
            _unit_interval(self.minimum_confidence, "minimum_confidence"),
        )
        object.__setattr__(
            self,
            "minimum_pose_quality",
            _unit_interval(self.minimum_pose_quality, "minimum_pose_quality"),
        )


def _scene_objects(
    scene: SceneDetections | None,
    context: GraspPlanningContext,
    *,
    cube_edge_m: float,
) -> tuple[DetectedObject, ...]:
    if context.generic_scene is not None:
        return context.generic_scene.objects
    if scene is None:
        return ()
    return tuple(
        DetectedObject(
            track_id=item.track_id,
            class_id="cube",
            sort_label=item.colour,
            capture_time_s=item.capture_time_s,
            pose=Pose3D.from_yaw(item.position_base_m, item.yaw_rad),
            dimensions_m=(cube_edge_m, cube_edge_m, cube_edge_m),
            confidence=item.confidence,
            pose_quality=item.confidence,
            visibility=1.0,
            source_camera_ids=("compatibility-cube-scene",),
        )
        for item in scene.cubes
    )


def _measure_candidate(
    scene: SceneDetections | None,
    candidate: GraspCandidate,
    context: GraspPlanningContext,
    settings: _EnvelopeSettings,
) -> dict[str, float]:
    objects = _scene_objects(scene, context, cube_edge_m=settings.cube_edge_m)
    target = next(
        (item for item in objects if item.track_id == candidate.target_track_id),
        None,
    )
    legacy_target = (
        None
        if scene is None
        else next(
            (
                item
                for item in scene.cubes
                if item.track_id == candidate.target_track_id
            ),
            None,
        )
    )
    target_missing = target is None
    expected_class = (
        SCREW_CLASS_ID if candidate.generator_id == SCREW_GENERATOR_ID else "cube"
    )
    target_class_mismatch = target is not None and target.class_id != expected_class
    target_geometry_missing = target is not None and target.dimensions_m is None
    legacy_adapter_evidence = (
        target is not None
        and legacy_target is not None
        and "legacy_square_yaw_rad" in target.attributes
    )
    target_evidence_missing = target is None or (
        (target.pose_quality is None or target.visibility is None)
        and not legacy_adapter_evidence
    )
    target_position_disagreement = 0.0
    target_dimension_disagreement = 0.0
    target_yaw_disagreement = 0.0
    if target is not None and candidate.generator_id == CUBE_GENERATOR_ID:
        reference_xy = (
            candidate.pick_tip_pose.xyz_m[:2]
            if legacy_target is None
            else legacy_target.position_base_m[:2]
        )
        target_position_disagreement = math.dist(
            target.pose.position_base_m[:2],
            reference_xy,
        )
        if target.dimensions_m is not None:
            target_dimension_disagreement = max(
                abs(value - settings.cube_edge_m) for value in target.dimensions_m
            )
        if legacy_target is not None:
            target_yaw_disagreement = _square_yaw_distance(
                _quaternion_yaw(target.pose.orientation_xyzw),
                legacy_target.yaw_rad,
            )
    target_x = settings.cube_edge_m
    target_y = settings.cube_edge_m
    if target is not None and target.dimensions_m is not None:
        target_x, target_y = target.dimensions_m[:2]

    opening = (
        settings.cube_edge_m
        if candidate.required_opening_m is None
        else candidate.required_opening_m
    )
    yaw = candidate.pick_tip_pose.yaw_rad
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    # TipPoseBase yaw follows the finger-length axis.  The twin converts it to
    # FINGER_SPREAD_AXIS=(-sin(yaw), cos(yaw)), the direction in which the jaws
    # close.  Keep the wide finger body on the yaw axis and the opening plus
    # finger thickness on the perpendicular spread axis.
    half_u = max(target_x, target_y) / 2.0 + settings.finger_width_m / 2.0
    half_v = opening / 2.0 + settings.finger_thickness_m / 2.0
    px, py = candidate.pick_tip_pose.xyz_m[:2]
    nearest_centre = settings.maximum_radial_reach_m
    minimum_clearance = settings.maximum_radial_reach_m
    neighbour_count = 0
    unknown_geometry_count = 0

    for item in objects:
        if item.track_id == candidate.target_track_id:
            continue
        neighbour_count += 1
        dx = item.pose.position_base_m[0] - px
        dy = item.pose.position_base_m[1] - py
        centre_distance = math.hypot(dx, dy)
        nearest_centre = min(nearest_centre, centre_distance)
        radius = _bounding_radius(item)
        if radius is None:
            unknown_geometry_count += 1
            continue
        along_u = abs(dx * cos_yaw + dy * sin_yaw)
        along_v = abs(-dx * sin_yaw + dy * cos_yaw)
        outside_u = max(along_u - half_u, 0.0)
        outside_v = max(along_v - half_v, 0.0)
        clearance = math.hypot(outside_u, outside_v) - radius
        minimum_clearance = min(minimum_clearance, clearance)

    confidence = 0.0 if target is None else target.confidence
    pose_quality = (
        confidence
        if legacy_adapter_evidence and target.pose_quality is None
        else (
            0.0
            if target is None or target.pose_quality is None
            else target.pose_quality
        )
    )
    visibility = (
        confidence
        if legacy_adapter_evidence and target.visibility is None
        else (0.0 if target is None or target.visibility is None else target.visibility)
    )
    return {
        "pick_radius_m": math.hypot(px, py),
        "release_radius_m": math.hypot(*context.release_tip_pose.xyz_m[:2]),
        "nearest_object_centre_m": nearest_centre,
        "minimum_footprint_clearance_m": minimum_clearance,
        "neighbour_count": float(neighbour_count),
        "unknown_geometry_count": float(unknown_geometry_count),
        "confidence": confidence,
        "pose_quality": pose_quality,
        "visibility": visibility,
        "required_opening_m": opening,
        "opening_fraction": opening / settings.maximum_opening_m,
        "approach_distance_m": math.dist(
            candidate.pick_tip_pose.xyz_m, candidate.approach_tip_pose.xyz_m
        ),
        "unsupported_object_count": float(len(context.unsupported_objects)),
        "target_observation_missing": float(target_missing),
        "target_class_mismatch": float(target_class_mismatch),
        "target_geometry_missing": float(target_geometry_missing),
        "target_evidence_missing": float(target_evidence_missing),
        "target_position_disagreement_m": target_position_disagreement,
        "target_dimension_disagreement_m": target_dimension_disagreement,
        "target_yaw_disagreement_rad": target_yaw_disagreement,
        "legacy_adapter_evidence": float(legacy_adapter_evidence),
    }


class CubeYawCandidateGenerator:
    """Generate a small, explicit family of supported top-down cube grasps."""

    generator_id = CUBE_GENERATOR_ID

    def __init__(
        self,
        *,
        cube_edge_m: float,
        seating_margin_m: float = 0.0003,
        yaw_offsets_rad: Sequence[float] = (0.0, math.pi / 4.0, math.pi / 2.0),
    ) -> None:
        self.cube_edge_m = _positive(cube_edge_m, "cube_edge_m")
        self.seating_margin_m = _finite(seating_margin_m, "seating_margin_m")
        if self.seating_margin_m < 0.0:
            raise ValueError("seating_margin_m must be non-negative")
        offsets = tuple(_finite(value, "yaw offset") for value in yaw_offsets_rad)
        if not offsets:
            raise ValueError("yaw_offsets_rad must not be empty")
        if len({_wrap_angle(value) for value in offsets}) != len(offsets):
            raise ValueError("yaw_offsets_rad must not contain duplicates")
        self.yaw_offsets_rad = offsets

    def generate(self, scene, target, context):
        candidates: list[GraspCandidate] = []
        for index, offset in enumerate(self.yaw_offsets_rad):
            yaw = _wrap_angle(target.yaw_rad + offset)
            projected_width = self.cube_edge_m * (
                abs(math.cos(offset)) + abs(math.sin(offset))
            )
            opening = projected_width + self.seating_margin_m
            pick = TipPoseBase(
                (
                    target.position_base_m[0],
                    target.position_base_m[1],
                    context.pick_tip_z_m,
                ),
                yaw,
            )
            approach = TipPoseBase(
                (
                    pick.xyz_m[0],
                    pick.xyz_m[1],
                    pick.xyz_m[2] + context.approach_clearance_m,
                ),
                yaw,
            )
            degrees = int(round(math.degrees(offset))) % 360
            candidates.append(
                GraspCandidate(
                    candidate_id=f"cube-yaw-{degrees:03d}-{index}",
                    scene_id=scene.scene_id,
                    target_track_id=target.track_id,
                    pick_tip_pose=pick,
                    approach_tip_pose=approach,
                    required_opening_m=opening,
                    generator_id=self.generator_id,
                )
            )
        return tuple(candidates)


class SceneEnvelopeChecker:
    """Conservative top-down reach, opening, evidence, and footprint preflight."""

    checker_id = CHECKER_ID

    def __init__(self, settings: _EnvelopeSettings) -> None:
        self.settings = settings

    def check(self, scene, candidate, context):
        metrics = _measure_candidate(scene, candidate, context, self.settings)
        reasons: list[str] = []
        if metrics["unsupported_object_count"] > 0.0:
            reasons.append("generic scene contains unsupported objects")
        if metrics["target_observation_missing"] > 0.0:
            reasons.append("candidate target is absent from the planning observation")
        if metrics["target_class_mismatch"] > 0.0:
            reasons.append("candidate target class disagrees with the grasp generator")
        if metrics["target_geometry_missing"] > 0.0:
            reasons.append("candidate target geometry is unknown")
        if metrics["target_evidence_missing"] > 0.0:
            reasons.append("candidate target evidence is incomplete")
        if (
            metrics["target_position_disagreement_m"]
            > self.settings.maximum_target_position_disagreement_m
        ):
            reasons.append(
                "cube target pose disagrees with the generic planning observation"
            )
        if (
            metrics["target_dimension_disagreement_m"]
            > self.settings.maximum_target_dimension_disagreement_m
        ):
            reasons.append(
                "cube target dimensions disagree with the configured cube size"
            )
        if (
            metrics["target_yaw_disagreement_rad"]
            > self.settings.maximum_target_yaw_disagreement_rad
        ):
            reasons.append(
                "cube target yaw disagrees with the generic planning observation"
            )
        if metrics["pick_radius_m"] > self.settings.maximum_radial_reach_m:
            reasons.append("pick pose exceeds radial reach")
        if metrics["release_radius_m"] > self.settings.maximum_radial_reach_m:
            reasons.append("release pose exceeds radial reach")
        if metrics["required_opening_m"] > self.settings.maximum_opening_m:
            reasons.append("required jaw opening exceeds gripper limit")
        if metrics["unknown_geometry_count"] > 0.0:
            reasons.append("nearby object geometry is unknown")
        if (
            metrics["minimum_footprint_clearance_m"]
            < self.settings.required_clearance_m
        ):
            reasons.append("top-down finger envelope is obstructed")
        if metrics["confidence"] < self.settings.minimum_confidence:
            reasons.append("target confidence is below threshold")
        if metrics["pose_quality"] < self.settings.minimum_pose_quality:
            reasons.append("target pose quality is below threshold")
        return FeasibilityResult(
            accepted=not reasons,
            rejection_reasons=tuple(reasons),
            checker_id=self.checker_id,
            metrics=metrics,
        )


class Team8FeatureExtractor:
    """Extract one stable pre-action schema for cube and screw candidates."""

    schema_id = FEATURE_SCHEMA_ID

    def __init__(self, settings: _EnvelopeSettings) -> None:
        self.settings = settings

    def extract(self, scene, candidate, context):
        metrics = _measure_candidate(scene, candidate, context, self.settings)
        target = next(
            (
                item
                for item in scene.cubes
                if item.track_id == candidate.target_track_id
            ),
            None,
        )
        if target is None:
            raise GraspPlanningError("cube candidate target is absent")
        values = {
            "object_is_screw": 0.0,
            "axial_offset_m": 0.0,
            "yaw_offset_abs_rad": abs(
                _wrap_angle(candidate.pick_tip_pose.yaw_rad - target.yaw_rad)
            ),
            **metrics,
        }
        return GraspFeatures(
            self.schema_id, {name: values[name] for name in FEATURE_NAMES}
        )


class GeometricPriorScorer:
    """Uncalibrated simulation-only ranker; replace before any hardware use."""

    scorer_id = "team8-simulation-ranking-prior-v1"
    model_version = "uncalibrated-simulation-ranking-only-v1"

    def score(self, candidate, features):
        values = features.values
        clearance = max(min(values["minimum_footprint_clearance_m"], 0.025), 0.0)
        evidence = (
            values["confidence"] + values["pose_quality"] + values["visibility"]
        ) / 3.0
        opening_penalty = max(values["opening_fraction"] - 0.65, 0.0)
        yaw_penalty = values["yaw_offset_abs_rad"] / (math.pi / 2.0)
        axial_penalty = 0.0
        if values["object_is_screw"] > 0.5:
            axial_penalty = abs(values["axial_offset_m"] - 0.016) / 0.008
        logit = (
            0.25
            + 1.4 * evidence
            + 1.2 * (clearance / 0.025)
            - 1.2 * opening_penalty
            - 0.35 * yaw_penalty
            - 0.20 * axial_penalty
        )
        probability = 1.0 / (1.0 + math.exp(-logit))
        return GraspScore(
            probability,
            scorer_id=self.scorer_id,
            model_version=self.model_version,
        )


def _reject_constant(value: str) -> None:
    raise ValueError(f"checkpoint contains invalid JSON constant {value}")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"checkpoint contains duplicate key {key!r}")
        result[key] = value
    return result


class JsonLogisticScorer:
    """Load a bounded, non-executable logistic model from strict JSON."""

    scorer_id = "team8-json-logistic-v1"
    _keys = frozenset(
        {
            "schema_version",
            "model_type",
            "feature_schema_id",
            "feature_order",
            "means",
            "scales",
            "coefficients",
            "intercept",
            "model_version",
            "training_data_sha256",
        }
    )

    def __init__(self, path: str | Path, *, expected_sha256: str) -> None:
        checkpoint = Path(path).expanduser().resolve()
        raw = checkpoint.read_bytes()
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        expected = str(expected_sha256).strip().lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("checkpoint_sha256 must be a hexadecimal SHA-256")
        if actual_sha256 != expected:
            raise ValueError(
                f"checkpoint SHA-256 mismatch: expected {expected}, got {actual_sha256}"
            )
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
        if not isinstance(payload, dict) or set(payload) != self._keys:
            raise ValueError("checkpoint keys do not match the v1 logistic schema")
        if payload["schema_version"] != 1:
            raise ValueError("unsupported checkpoint schema_version")
        if payload["model_type"] != "logistic_regression":
            raise ValueError("checkpoint model_type must be logistic_regression")
        if payload["feature_schema_id"] != FEATURE_SCHEMA_ID:
            raise ValueError("checkpoint feature schema does not match runtime")
        if tuple(payload["feature_order"]) != FEATURE_NAMES:
            raise ValueError("checkpoint feature order does not match runtime")
        self.means = self._vector(payload["means"], "means")
        self.scales = self._vector(payload["scales"], "scales")
        self.coefficients = self._vector(payload["coefficients"], "coefficients")
        if any(value <= 0.0 for value in self.scales):
            raise ValueError("checkpoint scales must be positive")
        self.intercept = _finite(payload["intercept"], "intercept")
        if type(payload["model_version"]) is not str:
            raise ValueError("checkpoint model_version must be a string")
        self.model_version = payload["model_version"].strip()
        if not self.model_version:
            raise ValueError("checkpoint model_version must not be empty")
        if type(payload["training_data_sha256"]) is not str:
            raise ValueError("training_data_sha256 must be a string")
        training_hash = payload["training_data_sha256"].strip().lower()
        if len(training_hash) != 64 or any(
            c not in "0123456789abcdef" for c in training_hash
        ):
            raise ValueError("training_data_sha256 must be a hexadecimal SHA-256")

    @staticmethod
    def _vector(value: Any, name: str) -> tuple[float, ...]:
        if not isinstance(value, list) or len(value) != len(FEATURE_NAMES):
            raise ValueError(f"{name} must contain {len(FEATURE_NAMES)} values")
        return tuple(_finite(item, name) for item in value)

    def score(self, candidate, features):
        if features.schema_id != FEATURE_SCHEMA_ID:
            raise ValueError("feature schema does not match logistic model")
        if tuple(features.values) != FEATURE_NAMES:
            raise ValueError("feature order does not match logistic model")
        standardized = tuple(
            (features.values[name] - mean) / scale
            for name, mean, scale in zip(FEATURE_NAMES, self.means, self.scales)
        )
        logit = self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, standardized)
        )
        if logit >= 0.0:
            exp_value = math.exp(-logit)
            probability = 1.0 / (1.0 + exp_value)
        else:
            exp_value = math.exp(logit)
            probability = exp_value / (1.0 + exp_value)
        return GraspScore(
            probability,
            scorer_id=self.scorer_id,
            model_version=self.model_version,
        )


def _settings_from_config(config, options: Mapping[str, Any]) -> _EnvelopeSettings:
    return _EnvelopeSettings(
        maximum_radial_reach_m=config.hardware.maximum_radial_reach_m,
        maximum_opening_m=config.gripper.hook_open_gap_m,
        required_clearance_m=options.get("required_clearance_m", 0.001),
        finger_width_m=config.gripper.finger_width_m,
        finger_thickness_m=config.gripper.finger_thickness_m,
        cube_edge_m=config.cube_edge_m,
        minimum_confidence=options.get(
            "minimum_confidence", config.vision.minimum_detection_confidence
        ),
        minimum_pose_quality=options.get("minimum_pose_quality", 0.2),
        maximum_target_position_disagreement_m=options.get(
            "maximum_target_position_disagreement_m", 0.003
        ),
        maximum_target_dimension_disagreement_m=options.get(
            "maximum_target_dimension_disagreement_m", 0.0002
        ),
        maximum_target_yaw_disagreement_rad=options.get(
            "maximum_target_yaw_disagreement_rad", math.radians(5.0)
        ),
    )


def _build_scorer(context, *, allow_uncalibrated: bool):
    default = "geometric_prior" if allow_uncalibrated else "checkpoint_logistic"
    scorer_name = str(context.options.get("scorer", default))
    if scorer_name == "geometric_prior":
        if not allow_uncalibrated:
            raise ValueError(
                "the uncalibrated ranker is unavailable through the executable "
                "integration factory; use create_simulation directly"
            )
        if context.options.get("simulation_only") is not True:
            raise ValueError(
                "the uncalibrated geometric prior requires simulation_only=true"
            )
        return GeometricPriorScorer()
    if scorer_name == "checkpoint_logistic":
        if not context.checkpoint:
            raise ValueError("checkpoint_logistic requires ROCKWELL_GRASP_CHECKPOINT")
        checkpoint_sha256 = context.options.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha256, str):
            raise ValueError("checkpoint_logistic requires options.checkpoint_sha256")
        return JsonLogisticScorer(context.checkpoint, expected_sha256=checkpoint_sha256)
    raise ValueError(f"unsupported scorer {scorer_name!r}")


def _create_pipeline(context, *, allow_uncalibrated: bool) -> GraspPlanningPipeline:
    allowed = {
        "scorer",
        "checkpoint_sha256",
        "yaw_offsets_deg",
        "required_clearance_m",
        "minimum_confidence",
        "minimum_pose_quality",
        "maximum_target_position_disagreement_m",
        "maximum_target_dimension_disagreement_m",
        "maximum_target_yaw_disagreement_rad",
        "simulation_only",
    }
    unknown = set(context.options) - allowed
    if unknown:
        raise ValueError("unknown grasp profile options: " + ", ".join(sorted(unknown)))
    raw_offsets = context.options.get("yaw_offsets_deg", (0.0, 45.0, 90.0))
    if not isinstance(raw_offsets, (tuple, list)):
        raise ValueError("yaw_offsets_deg must be an array")
    offsets = tuple(math.radians(_finite(value, "yaw offset")) for value in raw_offsets)
    settings = _settings_from_config(context.config, context.options)
    return GraspPlanningPipeline(
        generator=CubeYawCandidateGenerator(
            cube_edge_m=settings.cube_edge_m,
            yaw_offsets_rad=offsets,
        ),
        checker=SceneEnvelopeChecker(settings),
        feature_extractor=Team8FeatureExtractor(settings),
        scorer=_build_scorer(context, allow_uncalibrated=allow_uncalibrated),
        selector=HighestSuccessSelector(),
    )


def create(context) -> GraspPlanningPipeline:
    """Build a calibrated executable cube pipeline for standard integration."""

    return _create_pipeline(context, allow_uncalibrated=False)


def create_simulation(context) -> GraspPlanningPipeline:
    """Build the explicit uncalibrated cube ranker for direct Isaac use only."""

    return _create_pipeline(context, allow_uncalibrated=True)


def _create_ai_simulation_pipeline(
    context,
    provider,
    *,
    simulation_capability,
    mode="shadow",
    minimum_confidence=0.7,
    minimum_score_margin=0.1,
    timeout_s=20.0,
    decision_sink,
) -> GraspPlanningPipeline:
    """Build an explicitly simulation-only pretrained-model selector.

    The normal executable factory remains unchanged. The injected provider sees
    only candidates that have already passed SceneEnvelopeChecker.
    """

    if simulation_capability is not _AI_SIMULATION_CAPABILITY:
        raise ValueError("AI binding requires the isolated simulation capability")
    if context.options.get("simulation_only") is not True:
        raise ValueError("AI grasp selection requires simulation_only=true")
    if decision_sink is None or not callable(decision_sink):
        raise ValueError("AI grasp selection requires a callable durable receipt sink")
    baseline = create_simulation(context)
    policy = _create_ai_selection_policy(
        provider,
        simulation_capability=simulation_capability,
        mode=mode,
        minimum_confidence=minimum_confidence,
        minimum_score_margin=minimum_score_margin,
        timeout_s=timeout_s,
        request_context={
            "profile_id": str(context.profile_id),
            "profile_path": str(context.profile_path),
            "profile_sha256": str(context.profile_sha256),
            "simulation_only": True,
            "object_family": "10mm_block",
        },
    )
    return GraspPlanningPipeline(
        generator=baseline.generator,
        checker=baseline.checker,
        feature_extractor=baseline.feature_extractor,
        scorer=baseline.scorer,
        selector=AIAssistedSelector(policy, decision_sink=decision_sink),
    )


def _rotate_vector(
    quaternion_xyzw: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    x, y, z, w = quaternion_xyzw
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


class ScrewShaftCandidateGenerator:
    """Generate three catalog-relative shaft pinches for offline evaluation."""

    generator_id = SCREW_GENERATOR_ID

    def __init__(
        self,
        offsets_from_head_m: Sequence[float] = (0.012, 0.016, 0.020),
        *,
        length_m: float = 0.030,
        shank_start_m: float = 0.004,
        shank_diameter_m: float = 0.006,
        seating_margin_m: float = 0.0003,
        catalog_dimensions_m: Sequence[float] = (0.012, 0.01199627, 0.030),
        dimension_tolerance_m: float = 0.0002,
    ) -> None:
        self.offsets = tuple(
            _positive(value, "screw offset") for value in offsets_from_head_m
        )
        if not self.offsets or len(set(self.offsets)) != len(self.offsets):
            raise ValueError("screw offsets must be non-empty and unique")
        self.length_m = _positive(length_m, "length_m")
        self.shank_start_m = _positive(shank_start_m, "shank_start_m")
        self.shank_diameter_m = _positive(shank_diameter_m, "shank_diameter_m")
        self.seating_margin_m = _positive(seating_margin_m, "seating_margin_m")
        if len(catalog_dimensions_m) != 3:
            raise ValueError("catalog_dimensions_m must contain three extents")
        self.catalog_dimensions_m = tuple(
            _positive(value, "catalog dimension") for value in catalog_dimensions_m
        )
        self.dimension_tolerance_m = _positive(
            dimension_tolerance_m, "dimension_tolerance_m"
        )
        for offset in self.offsets:
            if not self.shank_start_m + self.seating_margin_m <= offset:
                raise ValueError("screw offset reaches the head region")
            if not offset <= self.length_m - self.seating_margin_m:
                raise ValueError("screw offset reaches the tip region")

    def generate(
        self,
        scene: SceneObservation,
        target: DetectedObject,
        *,
        pick_tip_z_m: float,
        approach_clearance_m: float,
    ) -> tuple[GraspCandidate, ...]:
        if target.class_id != SCREW_CLASS_ID:
            raise ValueError(f"target must have class_id {SCREW_CLASS_ID}")
        if target.dimensions_m is None:
            raise ValueError("screw target requires measured dimensions")
        if any(
            abs(measured - expected) > self.dimension_tolerance_m
            for measured, expected in zip(
                target.dimensions_m, self.catalog_dimensions_m
            )
        ):
            raise ValueError(
                "screw dimensions do not match the versioned M6x30 catalog geometry"
            )
        axis = _rotate_vector(target.pose.orientation_xyzw, (0.0, 0.0, -1.0))
        if abs(axis[2]) > math.sin(math.radians(10.0)):
            raise ValueError("offline screw planner supports only a lying screw")
        planar_norm = math.hypot(axis[0], axis[1])
        if planar_norm <= 1e-9:
            raise ValueError("screw axis has no horizontal direction")
        axis_x, axis_y = axis[0] / planar_norm, axis[1] / planar_norm
        yaw = math.atan2(axis_y, axis_x)
        head_x, head_y = target.pose.position_base_m[:2]
        candidates = []
        for index, offset in enumerate(self.offsets):
            pick = TipPoseBase(
                (
                    head_x + axis_x * offset,
                    head_y + axis_y * offset,
                    _finite(pick_tip_z_m, "pick_tip_z_m"),
                ),
                yaw,
            )
            approach = TipPoseBase(
                (
                    pick.xyz_m[0],
                    pick.xyz_m[1],
                    pick.xyz_m[2]
                    + _positive(approach_clearance_m, "approach_clearance_m"),
                ),
                yaw,
            )
            candidates.append(
                GraspCandidate(
                    candidate_id=f"screw-shaft-{int(round(offset * 1000)):03d}mm-{index}",
                    scene_id=scene.scene_id,
                    target_track_id=target.track_id,
                    pick_tip_pose=pick,
                    approach_tip_pose=approach,
                    required_opening_m=(
                        self.shank_diameter_m + 2.0 * self.seating_margin_m
                    ),
                    generator_id=self.generator_id,
                )
            )
        return tuple(candidates)


@dataclass(frozen=True)
class OfflineGraspDecision:
    """Auditable screw decision that is deliberately not an executable plan."""

    scene_id: str
    target_track_id: str
    selected: EvaluatedGrasp
    evaluations: tuple[EvaluatedGrasp, ...]
    rejections: tuple[RejectedGrasp, ...]
    execution_supported: bool = False


class OfflineScrewPlanner:
    """Score screw candidates without crossing the cube-only executor boundary."""

    def __init__(self, settings, scorer, generator=None, selector=None) -> None:
        self.settings = settings
        self.scorer = scorer
        self.generator = generator or ScrewShaftCandidateGenerator()
        self.checker = SceneEnvelopeChecker(settings)
        self.selector = HighestSuccessSelector() if selector is None else selector

    def decide(
        self,
        scene: SceneObservation,
        target_track_id: str,
        *,
        pick_tip_z_m: float,
        approach_clearance_m: float,
        release_tip_pose: TipPoseBase,
    ) -> OfflineGraspDecision:
        target = next(
            (item for item in scene.objects if item.track_id == target_track_id),
            None,
        )
        if target is None:
            raise GraspPlanningError("screw target does not belong to the scene")
        if target.sort_label not in {"red", "yellow", "blue"}:
            raise GraspPlanningError(
                "screw target requires a supported destination sort label"
            )
        candidates = self.generator.generate(
            scene,
            target,
            pick_tip_z_m=pick_tip_z_m,
            approach_clearance_m=approach_clearance_m,
        )
        context = GraspPlanningContext(
            destination=BinTarget(target.sort_label, release_tip_pose.xyz_m[:2]),
            release_tip_pose=release_tip_pose,
            pick_tip_z_m=pick_tip_z_m,
            approach_clearance_m=approach_clearance_m,
            generic_scene=scene,
            unsupported_objects=(),
        )
        offset_by_id = {
            candidate.candidate_id: offset
            for candidate, offset in zip(candidates, self.generator.offsets)
        }
        evaluations = []
        rejections = []
        for candidate in candidates:
            feasibility = self.checker.check(None, candidate, context)
            if not feasibility.accepted:
                rejections.append(RejectedGrasp(candidate, feasibility))
                continue
            metrics = _measure_candidate(None, candidate, context, self.settings)
            values = {
                "object_is_screw": 1.0,
                "axial_offset_m": offset_by_id[candidate.candidate_id],
                "yaw_offset_abs_rad": 0.0,
                **metrics,
            }
            features = GraspFeatures(
                FEATURE_SCHEMA_ID,
                {name: values[name] for name in FEATURE_NAMES},
            )
            score = self.scorer.score(candidate, features)
            evaluations.append(EvaluatedGrasp(candidate, feasibility, features, score))
        if not evaluations:
            details = "; ".join(
                f"{item.candidate.candidate_id}: "
                + ", ".join(item.feasibility.rejection_reasons)
                for item in rejections
            )
            raise GraspPlanningError("no feasible offline screw candidates: " + details)
        selected = self.selector.select(tuple(evaluations))
        return OfflineGraspDecision(
            scene_id=scene.scene_id,
            target_track_id=target.track_id,
            selected=selected,
            evaluations=tuple(evaluations),
            rejections=tuple(rejections),
        )


def build_offline_screw_planner(config, *, scorer=None, options=None):
    options = {} if options is None else dict(options)
    settings = _settings_from_config(config, options)
    return OfflineScrewPlanner(
        settings,
        scorer if scorer is not None else GeometricPriorScorer(),
    )


def _create_ai_offline_screw_planner(
    context,
    provider,
    *,
    simulation_capability,
    mode="shadow",
    minimum_confidence=0.7,
    minimum_score_margin=0.1,
    timeout_s=20.0,
    decision_sink,
) -> OfflineScrewPlanner:
    """Build a receipt-writing, non-executable AI screw planner."""

    if simulation_capability is not _AI_SIMULATION_CAPABILITY:
        raise ValueError("AI binding requires the isolated simulation capability")
    if context.options.get("simulation_only") is not True:
        raise ValueError("AI grasp selection requires simulation_only=true")
    if decision_sink is None or not callable(decision_sink):
        raise ValueError("AI grasp selection requires a callable durable receipt sink")
    policy = _create_ai_selection_policy(
        provider,
        simulation_capability=simulation_capability,
        mode=mode,
        minimum_confidence=minimum_confidence,
        minimum_score_margin=minimum_score_margin,
        timeout_s=timeout_s,
        request_context={
            "profile_id": str(context.profile_id),
            "profile_path": str(context.profile_path),
            "profile_sha256": str(context.profile_sha256),
            "simulation_only": True,
            "object_family": "iso10642_m6x30_screw",
            "execution_supported": False,
        },
    )
    settings = _settings_from_config(context.config, context.options)
    return OfflineScrewPlanner(
        settings,
        _build_scorer(context, allow_uncalibrated=True),
        selector=AIAssistedSelector(policy, decision_sink=decision_sink),
    )


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_ID",
    "GeometricPriorScorer",
    "JsonLogisticScorer",
    "OfflineGraspDecision",
    "OfflineScrewPlanner",
    "PLUGIN_API_VERSION",
    "SCREW_CLASS_ID",
    "ScrewShaftCandidateGenerator",
    "build_offline_screw_planner",
    "create",
    "create_simulation",
]
