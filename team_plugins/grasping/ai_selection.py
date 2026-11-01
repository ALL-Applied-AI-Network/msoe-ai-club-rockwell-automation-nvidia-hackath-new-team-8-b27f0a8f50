"""Fail-closed pretrained-model ranking of feasible grasp candidates.

The model is downstream of deterministic feasibility checks. It may rank
existing EvaluatedGrasp records, but it cannot create a pose, override a
rejection, or turn model confidence into grasp-success probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import hmac
import json
import math
import re
import secrets
import time
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from sorting_core.grasping import (
    EvaluatedGrasp,
    GraspPlanningError,
    HighestSuccessSelector,
)

REQUEST_SCHEMA_VERSION = "team8-feasible-grasp-ranking-request-v1"
RESPONSE_SCHEMA_VERSION = "team8-feasible-grasp-ranking-response-v1"
POLICY_VERSION = "team8-pretrained-ai-selection-policy-v2"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/+\-]{0,199}$")
_RESPONSE_KEYS = {
    "schema_version",
    "request_sha256",
    "model_version",
    "selected_candidate_id",
    "confidence",
    "ranking",
    "rationale",
}
_RANKING_KEYS = {"candidate_id", "score"}
_AI_SIMULATION_CAPABILITY = object()


class AIResponseValidationError(ValueError):
    """An untrusted provider response failed a validation gate."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _safe_id(value: Any, name: str) -> str:
    result = str(value).strip()
    if not _SAFE_ID.fullmatch(result):
        raise ValueError(f"{name} must be a safe 1-200 character identifier")
    return result


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit_interval(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _json_unit_interval(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number")
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used for request binding."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON data") from exc


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_object(text: str) -> dict[str, Any]:
    """Parse an object while rejecting duplicate keys and NaN/Infinity."""

    if not isinstance(text, str):
        raise ValueError("provider response must be JSON text")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("provider response is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("provider response must be a JSON object")
    return value


def _optional_nonnegative(value: Any, name: str) -> float | None:
    if value is None:
        return None
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _evaluation_payload(evaluation: EvaluatedGrasp) -> dict[str, Any]:
    if not isinstance(evaluation, EvaluatedGrasp):
        raise ValueError("all candidates must be EvaluatedGrasp records")
    if not evaluation.feasibility.accepted:
        raise ValueError("AI selection may receive only feasible candidates")
    candidate = evaluation.candidate
    opening = candidate.required_opening_m
    if opening is not None:
        opening = _finite(opening, "required_opening_m")
        if opening < 0.0:
            raise ValueError("required_opening_m must be non-negative")
    features = {
        str(name): _finite(value, f"feature.{name}")
        for name, value in sorted(evaluation.features.values.items())
    }
    metrics = {
        str(name): _finite(value, f"metric.{name}")
        for name, value in sorted(evaluation.feasibility.metrics.items())
    }
    score = evaluation.score
    return {
        "candidate_id": _safe_id(candidate.candidate_id, "candidate_id"),
        "generator_id": str(candidate.generator_id),
        "pick_tip_pose": {
            "xyz_m": [
                _finite(value, "pick_tip_pose.xyz_m")
                for value in candidate.pick_tip_pose.xyz_m
            ],
            "yaw_rad": _finite(
                candidate.pick_tip_pose.yaw_rad, "pick_tip_pose.yaw_rad"
            ),
        },
        "approach_tip_pose": {
            "xyz_m": [
                _finite(value, "approach_tip_pose.xyz_m")
                for value in candidate.approach_tip_pose.xyz_m
            ],
            "yaw_rad": _finite(
                candidate.approach_tip_pose.yaw_rad,
                "approach_tip_pose.yaw_rad",
            ),
        },
        "required_opening_m": opening,
        "feasibility": {
            "checker_id": str(evaluation.feasibility.checker_id),
            "metrics": metrics,
        },
        "features": {
            "schema_id": str(evaluation.features.schema_id),
            "values": features,
        },
        "baseline_score": {
            "success_probability": _unit_interval(
                score.success_probability, "success_probability"
            ),
            "expected_total_episode_time_s": _optional_nonnegative(
                score.expected_total_episode_time_s,
                "expected_total_episode_time_s",
            ),
            "expected_execution_time_s": _optional_nonnegative(
                score.expected_execution_time_s,
                "expected_execution_time_s",
            ),
            "expected_recovery_time_s": _optional_nonnegative(
                score.expected_recovery_time_s,
                "expected_recovery_time_s",
            ),
            "scorer_id": str(score.scorer_id),
            "model_version": str(score.model_version),
        },
    }


@dataclass(frozen=True)
class AISelectionRequest:
    """Canonical, scene-bound request sent to a pretrained model."""

    payload_json: str
    sha256: str
    scene_id: str
    target_track_id: str
    candidate_ids: tuple[str, ...]
    decision_nonce: str
    created_monotonic_s: float

    def as_dict(self) -> dict[str, Any]:
        return strict_json_object(self.payload_json)


def build_selection_request(
    evaluations: tuple[EvaluatedGrasp, ...],
    *,
    decision_nonce: str | None = None,
    request_context: Mapping[str, Any] | None = None,
) -> AISelectionRequest:
    """Build an order-invariant, per-decision request from feasible evaluations."""

    evaluations = tuple(evaluations)
    if not evaluations:
        raise GraspPlanningError("AI selector requires feasible candidates")
    scene_ids = {item.candidate.scene_id for item in evaluations}
    target_ids = {item.candidate.target_track_id for item in evaluations}
    if len(scene_ids) != 1:
        raise GraspPlanningError("AI candidates span multiple scenes")
    if len(target_ids) != 1:
        raise GraspPlanningError("AI candidates span multiple targets")
    by_id: dict[str, EvaluatedGrasp] = {}
    for item in evaluations:
        candidate_id = _safe_id(item.candidate.candidate_id, "candidate_id")
        if candidate_id in by_id:
            raise GraspPlanningError(f"duplicate AI candidate ID: {candidate_id}")
        by_id[candidate_id] = item
    if decision_nonce is None:
        decision_nonce = secrets.token_hex(16)
    decision_nonce = _safe_id(decision_nonce, "decision_nonce")
    if request_context is None:
        request_context = {}
    if not isinstance(request_context, Mapping):
        raise ValueError("request_context must be a mapping")
    normalized_context = strict_json_object(canonical_json(dict(request_context)))
    candidate_ids = tuple(sorted(by_id))
    payload = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "scene_id": next(iter(scene_ids)),
        "target_track_id": next(iter(target_ids)),
        "decision_nonce": decision_nonce,
        "request_context": normalized_context,
        "candidates": [
            _evaluation_payload(by_id[candidate_id]) for candidate_id in candidate_ids
        ],
    }
    payload_json = canonical_json(payload)
    return AISelectionRequest(
        payload_json=payload_json,
        sha256=hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
        scene_id=next(iter(scene_ids)),
        target_track_id=next(iter(target_ids)),
        candidate_ids=candidate_ids,
        decision_nonce=decision_nonce,
        created_monotonic_s=time.monotonic(),
    )


@runtime_checkable
class AISelectionProvider(Protocol):
    """External pretrained model service with bounded request time."""

    provider_id: str
    model_version: str

    def select(
        self, request_json: str, *, timeout_s: float
    ) -> str | Mapping[str, Any]: ...


@dataclass(frozen=True)
class AISelectionDecision:
    """Auditable result; model confidence is not success probability."""

    executed: EvaluatedGrasp
    fallback: EvaluatedGrasp
    mode: str
    status: str
    provider_id: str
    model_version: str
    prompt_version: str
    request_sha256: str
    response_sha256: str | None
    proposed_candidate_id: str | None
    executed_candidate_id: str
    confidence: float | None
    score_margin: float | None
    reason: str
    request_json: str
    response_json: str | None
    minimum_confidence: float
    minimum_score_margin: float
    timeout_s: float
    latency_s: float

    @property
    def used_fallback(self) -> bool:
        return self.status != "bound"

    def receipt(self, *, include_payloads: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": "team8-ai-selection-receipt-v1",
            "policy_version": POLICY_VERSION,
            "mode": self.mode,
            "status": self.status,
            "provider_id": self.provider_id,
            "model_version": self.model_version,
            "prompt_version": self.prompt_version,
            "request_schema_version": REQUEST_SCHEMA_VERSION,
            "response_schema_version": RESPONSE_SCHEMA_VERSION,
            "latency_s": self.latency_s,
            "request_sha256": self.request_sha256,
            "response_sha256": self.response_sha256,
            "proposed_candidate_id": self.proposed_candidate_id,
            "executed_candidate_id": self.executed_candidate_id,
            "fallback_candidate_id": self.fallback.candidate.candidate_id,
            "confidence": self.confidence,
            "score_margin": self.score_margin,
            "minimum_confidence": self.minimum_confidence,
            "minimum_score_margin": self.minimum_score_margin,
            "timeout_s": self.timeout_s,
            "reason": self.reason,
            "used_fallback": self.used_fallback,
        }
        if include_payloads:
            result["request"] = strict_json_object(self.request_json)
            result["response"] = (
                None
                if self.response_json is None
                else strict_json_object(self.response_json)
            )
        return result


@dataclass(frozen=True)
class AISelectionPolicy:
    """Validate a model ranking and bind it only in isolated simulation."""

    provider: AISelectionProvider
    mode: str = "shadow"
    minimum_confidence: float = 0.7
    minimum_score_margin: float = 0.1
    timeout_s: float = 20.0
    simulation_only: bool = True
    fallback_selector: Any = None
    request_context: Mapping[str, Any] | None = None
    _simulation_capability: object | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _safe_id(self.provider.provider_id, "provider.provider_id")
        _safe_id(self.provider.model_version, "provider.model_version")
        _safe_id(
            getattr(self.provider, "prompt_version", "unspecified"),
            "provider.prompt_version",
        )
        if self.mode not in {"shadow", "simulation_binding"}:
            raise ValueError("mode must be shadow or simulation_binding")
        if (
            self.mode == "simulation_binding"
            and self._simulation_capability is not _AI_SIMULATION_CAPABILITY
        ):
            raise ValueError(
                "simulation binding is available only through the guarded planner"
            )
        if self.mode == "simulation_binding" and self.simulation_only is not True:
            raise ValueError("simulation_binding requires simulation_only=True")
        object.__setattr__(
            self,
            "minimum_confidence",
            _unit_interval(self.minimum_confidence, "minimum_confidence"),
        )
        object.__setattr__(
            self,
            "minimum_score_margin",
            _unit_interval(self.minimum_score_margin, "minimum_score_margin"),
        )
        timeout = _finite(self.timeout_s, "timeout_s")
        if timeout <= 0.0 or timeout > 300.0:
            raise ValueError("timeout_s must be in (0, 300]")
        object.__setattr__(self, "timeout_s", timeout)
        context = {} if self.request_context is None else self.request_context
        if not isinstance(context, Mapping):
            raise ValueError("request_context must be a mapping")
        object.__setattr__(
            self,
            "request_context",
            strict_json_object(canonical_json(dict(context))),
        )
        if self.fallback_selector is None:
            object.__setattr__(self, "fallback_selector", HighestSuccessSelector())

    @property
    def selector_id(self) -> str:
        return (
            f"{POLICY_VERSION}:{self.mode}:"
            f"{self.provider.provider_id}:{self.provider.model_version}"
        )

    @property
    def selection_objective(self) -> str:
        return "pretrained_rank_feasible_candidates_with_fail_closed_fallback"

    def _fallback_decision(
        self,
        *,
        fallback: EvaluatedGrasp,
        request: AISelectionRequest,
        status: str,
        reason: str,
        response_json: str | None = None,
        proposed_candidate_id: str | None = None,
        confidence: float | None = None,
        score_margin: float | None = None,
    ) -> AISelectionDecision:
        response_sha256 = (
            None
            if response_json is None
            else hashlib.sha256(response_json.encode("utf-8")).hexdigest()
        )
        return AISelectionDecision(
            executed=fallback,
            fallback=fallback,
            mode=self.mode,
            status=status,
            provider_id=self.provider.provider_id,
            model_version=self.provider.model_version,
            prompt_version=getattr(self.provider, "prompt_version", "unspecified"),
            request_sha256=request.sha256,
            response_sha256=response_sha256,
            proposed_candidate_id=proposed_candidate_id,
            executed_candidate_id=fallback.candidate.candidate_id,
            confidence=confidence,
            score_margin=score_margin,
            reason=reason,
            request_json=request.payload_json,
            response_json=response_json,
            minimum_confidence=self.minimum_confidence,
            minimum_score_margin=self.minimum_score_margin,
            timeout_s=self.timeout_s,
            latency_s=max(0.0, time.monotonic() - request.created_monotonic_s),
        )

    def decide(self, evaluations: tuple[EvaluatedGrasp, ...]) -> AISelectionDecision:
        evaluations = tuple(evaluations)
        request = build_selection_request(
            evaluations,
            request_context=self.request_context,
        )
        fallback = self.fallback_selector.select(evaluations)
        by_id = {item.candidate.candidate_id: item for item in evaluations}
        try:
            raw_response = self.provider.select(
                request.payload_json, timeout_s=self.timeout_s
            )
        except Exception as exc:
            return self._fallback_decision(
                fallback=fallback,
                request=request,
                status="fallback_provider_error",
                reason=f"provider_error:{type(exc).__name__}",
            )
        response_json: str | None = None
        try:
            if isinstance(raw_response, str):
                response = strict_json_object(raw_response)
                response_json = canonical_json(response)
            elif isinstance(raw_response, Mapping):
                response_json = canonical_json(dict(raw_response))
                response = strict_json_object(response_json)
            else:
                raise AIResponseValidationError("response_type")
            if set(response) != _RESPONSE_KEYS:
                raise AIResponseValidationError("response_keys")
            if response["schema_version"] != RESPONSE_SCHEMA_VERSION:
                raise AIResponseValidationError("schema_version")
            if not isinstance(
                response["request_sha256"], str
            ) or not hmac.compare_digest(response["request_sha256"], request.sha256):
                raise AIResponseValidationError("request_sha256")
            if response["model_version"] != self.provider.model_version:
                raise AIResponseValidationError("model_version")
            rationale = response["rationale"]
            if not isinstance(rationale, str) or len(rationale) > 1000:
                raise AIResponseValidationError("rationale")
            confidence = _json_unit_interval(
                response["confidence"], "response.confidence"
            )
            ranking = response["ranking"]
            if not isinstance(ranking, list):
                raise AIResponseValidationError("ranking_type")
            ranked_ids: list[str] = []
            ranked_scores: list[float] = []
            for item in ranking:
                if not isinstance(item, dict) or set(item) != _RANKING_KEYS:
                    raise AIResponseValidationError("ranking_item")
                candidate_id = item["candidate_id"]
                if not isinstance(candidate_id, str):
                    raise AIResponseValidationError("ranking_candidate_id")
                ranked_ids.append(candidate_id)
                ranked_scores.append(
                    _json_unit_interval(item["score"], "ranking.score")
                )
            if len(ranked_ids) != len(set(ranked_ids)) or set(ranked_ids) != set(
                request.candidate_ids
            ):
                raise AIResponseValidationError("ranking_candidate_set")
            normalized_ranking = sorted(
                zip(ranked_ids, ranked_scores),
                key=lambda item: (-item[1], item[0]),
            )
            ranked_ids = [item[0] for item in normalized_ranking]
            ranked_scores = [item[1] for item in normalized_ranking]
            selected_id = response["selected_candidate_id"]
            if selected_id is None:
                return self._fallback_decision(
                    fallback=fallback,
                    request=request,
                    status="fallback_abstained",
                    reason="model_abstained",
                    response_json=response_json,
                    confidence=confidence,
                )
            if not isinstance(selected_id, str) or selected_id not in by_id:
                raise AIResponseValidationError("selected_candidate_id")
            margin = (
                1.0 if len(ranked_scores) == 1 else ranked_scores[0] - ranked_scores[1]
            )
            tied_top_ids = {
                candidate_id
                for candidate_id, score in zip(ranked_ids, ranked_scores)
                if score == ranked_scores[0]
            }

            if len(ranked_scores) > 1 and margin <= 0.0:
                if selected_id not in tied_top_ids:
                    raise AIResponseValidationError("selected_not_top_ranked")
                return self._fallback_decision(
                    fallback=fallback,
                    request=request,
                    status="fallback_tied",
                    reason="top_scores_tied",
                    response_json=response_json,
                    proposed_candidate_id=selected_id,
                    confidence=confidence,
                    score_margin=margin,
                )
            if not ranked_ids or selected_id != ranked_ids[0]:
                raise AIResponseValidationError("selected_not_top_ranked")
            if confidence < self.minimum_confidence:
                return self._fallback_decision(
                    fallback=fallback,
                    request=request,
                    status="fallback_low_confidence",
                    reason="confidence_below_gate",
                    response_json=response_json,
                    proposed_candidate_id=selected_id,
                    confidence=confidence,
                    score_margin=margin,
                )
            if margin < self.minimum_score_margin:
                return self._fallback_decision(
                    fallback=fallback,
                    request=request,
                    status="fallback_low_margin",
                    reason="score_margin_below_gate",
                    response_json=response_json,
                    proposed_candidate_id=selected_id,
                    confidence=confidence,
                    score_margin=margin,
                )
        except Exception as exc:
            code = (
                exc.code
                if isinstance(exc, AIResponseValidationError)
                else type(exc).__name__
            )
            return self._fallback_decision(
                fallback=fallback,
                request=request,
                status="fallback_invalid_response",
                reason=f"invalid_response:{code}",
                response_json=response_json,
            )
        if self.mode == "shadow":
            return self._fallback_decision(
                fallback=fallback,
                request=request,
                status="shadow_valid",
                reason="shadow_mode_never_binds",
                response_json=response_json,
                proposed_candidate_id=selected_id,
                confidence=confidence,
                score_margin=margin,
            )
        executed = by_id[selected_id]
        return AISelectionDecision(
            executed=executed,
            fallback=fallback,
            mode=self.mode,
            status="bound",
            provider_id=self.provider.provider_id,
            model_version=self.provider.model_version,
            prompt_version=getattr(self.provider, "prompt_version", "unspecified"),
            request_sha256=request.sha256,
            response_sha256=hashlib.sha256(response_json.encode("utf-8")).hexdigest(),
            proposed_candidate_id=selected_id,
            executed_candidate_id=selected_id,
            confidence=confidence,
            score_margin=margin,
            reason="validated_model_ranking",
            request_json=request.payload_json,
            response_json=response_json,
            minimum_confidence=self.minimum_confidence,
            minimum_score_margin=self.minimum_score_margin,
            timeout_s=self.timeout_s,
            latency_s=max(0.0, time.monotonic() - request.created_monotonic_s),
        )


def _create_ai_selection_policy(
    provider: AISelectionProvider,
    *,
    simulation_capability: object,
    mode: str,
    minimum_confidence: float,
    minimum_score_margin: float,
    timeout_s: float,
    request_context: Mapping[str, Any] | None = None,
) -> AISelectionPolicy:
    """Internal constructor for the simulation-only binding capability."""

    if simulation_capability is not _AI_SIMULATION_CAPABILITY:
        raise ValueError("AI binding requires the isolated simulation capability")
    return AISelectionPolicy(
        provider=provider,
        mode=mode,
        minimum_confidence=minimum_confidence,
        minimum_score_margin=minimum_score_margin,
        timeout_s=timeout_s,
        simulation_only=True,
        request_context=request_context,
        _simulation_capability=_AI_SIMULATION_CAPABILITY,
    )


class AIAssistedSelector:
    """GraspSelector adapter; a sink may receive each immutable decision."""

    def __init__(
        self,
        policy: AISelectionPolicy,
        *,
        decision_sink: Callable[[AISelectionDecision], None] | None = None,
    ) -> None:
        self.policy = policy
        self.decision_sink = decision_sink
        self.selector_id = policy.selector_id
        self.selection_objective = policy.selection_objective

    def select(self, evaluations: tuple[EvaluatedGrasp, ...]) -> EvaluatedGrasp:
        decision = self.policy.decide(evaluations)
        if self.decision_sink is not None:
            self.decision_sink(decision)
        return decision.executed


__all__ = [
    "AIAssistedSelector",
    "AIResponseValidationError",
    "AISelectionDecision",
    "AISelectionPolicy",
    "AISelectionProvider",
    "AISelectionRequest",
    "POLICY_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "RESPONSE_SCHEMA_VERSION",
    "build_selection_request",
    "canonical_json",
    "strict_json_object",
]
