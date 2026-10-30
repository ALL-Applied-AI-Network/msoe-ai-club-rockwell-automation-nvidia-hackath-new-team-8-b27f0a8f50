#!/usr/bin/env python3
"""Validate and summarize evaluation-result.v1 JSON files.

This intentionally uses only the Python standard library. The checked-in JSON
Schema is the complete interchange contract; this script repeats the important
cross-field checks that JSON Schema cannot express, such as candidate-set
partitioning and true-positive/dropout accounting.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
OUTCOME_STATUSES = (
    "complete_relocation_success",
    "failed_lift",
    "drop_or_slip",
    "wrong_destination",
    "cancelled",
    "unresolved",
    "not_executed",
)
OBSERVED_TERMINAL_STATUSES = {
    "complete_relocation_success",
    "failed_lift",
    "drop_or_slip",
    "wrong_destination",
}
TIMING_FIELDS = (
    "perception_s",
    "planning_s",
    "execution_s",
    "recovery_s",
    "total_episode_s",
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class EvaluationValidationError(ValueError):
    """Raised when a report violates the v1 contract."""


def _fail(path: str, message: str) -> None:
    raise EvaluationValidationError(f"{path}: {message}")


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    return value


def _array(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    return value


def _keys(
    value: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    path: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    extra = sorted(set(value) - allowed)
    if missing:
        _fail(path, f"missing required fields: {', '.join(missing)}")
    if extra:
        _fail(path, f"unknown fields: {', '.join(extra)}")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty string")
    return value


def _nullable_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(path, "must be a non-negative integer")
    return value


def _number(value: Any, path: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _fail(path, "must be a finite non-negative number")
    if maximum is not None and result > maximum:
        _fail(path, f"must be at most {maximum}")
    return result


def _nullable_number(
    value: Any, path: str, *, maximum: float | None = None
) -> float | None:
    if value is None:
        return None
    return _number(value, path, maximum=maximum)


def _nullable_bool(value: Any, path: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        _fail(path, "must be a boolean or null")
    return value


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "must be a boolean")
    return value


def _unique_strings(value: Any, path: str) -> list[str]:
    result = [_string(item, f"{path}[{index}]") for index, item in enumerate(_array(value, path))]
    if len(set(result)) != len(result):
        _fail(path, "must not contain duplicates")
    return result


def _validate_sha256(value: Any, path: str, *, nullable: bool) -> str | None:
    if value is None and nullable:
        return None
    result = _string(value, path)
    if not HEX64.fullmatch(result):
        _fail(path, "must be a lowercase 64-character SHA-256")
    return result


def _validate_created_at(value: Any, path: str) -> None:
    text = _string(value, path)
    if not text.endswith("Z"):
        _fail(path, "must be an explicit UTC date-time ending in Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        _fail(path, "must be a valid UTC date-time")
    if parsed.tzinfo != timezone.utc:
        _fail(path, "must use UTC")


def _validate_provenance(value: Any, path: str) -> None:
    record = _object(value, path)
    _keys(
        record,
        required=(
            "project_commit",
            "branch",
            "integration_profile",
            "object_catalog_version",
            "calibration_ids",
            "components",
            "source_run_ids",
        ),
        path=path,
    )
    commit = _string(record["project_commit"], f"{path}.project_commit")
    if not HEX40.fullmatch(commit):
        _fail(f"{path}.project_commit", "must be a lowercase 40-character Git SHA")
    _string(record["branch"], f"{path}.branch")
    _string(record["object_catalog_version"], f"{path}.object_catalog_version")
    _unique_strings(record["calibration_ids"], f"{path}.calibration_ids")
    source_run_ids = _unique_strings(
        record["source_run_ids"], f"{path}.source_run_ids"
    )
    if not source_run_ids:
        _fail(f"{path}.source_run_ids", "must contain at least one source run ID")

    profile = record["integration_profile"]
    if profile is not None:
        profile_record = _object(profile, f"{path}.integration_profile")
        _keys(
            profile_record,
            required=("profile_id", "profile_sha256"),
            path=f"{path}.integration_profile",
        )
        _string(profile_record["profile_id"], f"{path}.integration_profile.profile_id")
        _validate_sha256(
            profile_record["profile_sha256"],
            f"{path}.integration_profile.profile_sha256",
            nullable=False,
        )

    components = _array(record["components"], f"{path}.components")
    seen_components: set[str] = set()
    for index, component_value in enumerate(components):
        component_path = f"{path}.components[{index}]"
        component = _object(component_value, component_path)
        _keys(
            component,
            required=("component", "name", "version", "artifact_sha256"),
            path=component_path,
        )
        component_id = _string(component["component"], f"{component_path}.component")
        if component_id not in {"vision", "grasp", "task", "executor"}:
            _fail(f"{component_path}.component", "has an unsupported component type")
        if component_id in seen_components:
            _fail(f"{component_path}.component", "duplicates a component type")
        seen_components.add(component_id)
        _string(component["name"], f"{component_path}.name")
        _string(component["version"], f"{component_path}.version")
        _validate_sha256(
            component["artifact_sha256"],
            f"{component_path}.artifact_sha256",
            nullable=True,
        )


def _validate_vision(value: Any, path: str) -> None:
    if value is None:
        return
    vision = _object(value, path)
    _keys(
        vision,
        required=("frames_evaluated", "inference_latency_ms", "per_class"),
        path=path,
    )
    frames_evaluated = _integer(
        vision["frames_evaluated"], f"{path}.frames_evaluated"
    )
    latency_samples = _array(
        vision["inference_latency_ms"], f"{path}.inference_latency_ms"
    )
    if len(latency_samples) > frames_evaluated:
        _fail(
            f"{path}.inference_latency_ms",
            "sample count cannot exceed frames_evaluated",
        )
    for index, latency in enumerate(latency_samples):
        _number(latency, f"{path}.inference_latency_ms[{index}]")

    seen_classes: set[str] = set()
    for index, class_value in enumerate(_array(vision["per_class"], f"{path}.per_class")):
        class_path = f"{path}.per_class[{index}]"
        class_result = _object(class_value, class_path)
        _keys(
            class_result,
            required=(
                "class_id",
                "reference_instances",
                "true_positives",
                "false_positives",
                "dropouts",
                "pose_samples",
            ),
            path=class_path,
        )
        class_id = _string(class_result["class_id"], f"{class_path}.class_id")
        if class_id in seen_classes:
            _fail(f"{class_path}.class_id", "duplicates a class in one report")
        seen_classes.add(class_id)
        reference = _integer(class_result["reference_instances"], f"{class_path}.reference_instances")
        true_positives = _integer(class_result["true_positives"], f"{class_path}.true_positives")
        false_positives = _integer(
            class_result["false_positives"], f"{class_path}.false_positives"
        )
        dropouts = _integer(class_result["dropouts"], f"{class_path}.dropouts")
        if true_positives + dropouts != reference:
            _fail(class_path, "true_positives + dropouts must equal reference_instances")

        samples = _array(class_result["pose_samples"], f"{class_path}.pose_samples")
        if frames_evaluated == 0 and (
            reference or true_positives or false_positives or dropouts or samples
        ):
            _fail(
                class_path,
                "zero evaluated frames cannot have nonzero counts or pose samples",
            )
        if len(samples) > true_positives:
            _fail(f"{class_path}.pose_samples", "cannot exceed true_positives")
        for sample_index, sample_value in enumerate(samples):
            sample_path = f"{class_path}.pose_samples[{sample_index}]"
            sample = _object(sample_value, sample_path)
            _keys(
                sample,
                required=(
                    "position_error_mm",
                    "symmetry_aware_orientation_error_deg",
                    "symmetry_id",
                ),
                path=sample_path,
            )
            _number(sample["position_error_mm"], f"{sample_path}.position_error_mm")
            _nullable_number(
                sample["symmetry_aware_orientation_error_deg"],
                f"{sample_path}.symmetry_aware_orientation_error_deg",
                maximum=180.0,
            )
            _string(sample["symmetry_id"], f"{sample_path}.symmetry_id")


def _validate_grasp(value: Any, path: str) -> None:
    if value is None:
        return
    grasp = _object(value, path)
    _keys(
        grasp,
        required=(
            "generated_candidate_ids",
            "accepted_candidates",
            "rejected_candidates",
            "selected_candidate_id",
            "selected_candidate_outcome_observed",
            "selected_candidate_complete_grasp_success",
            "ranking_correct",
            "ranking_evidence_id",
        ),
        path=path,
    )
    generated = _unique_strings(grasp["generated_candidate_ids"], f"{path}.generated_candidate_ids")

    accepted_ids: list[str] = []
    for index, accepted_value in enumerate(_array(grasp["accepted_candidates"], f"{path}.accepted_candidates")):
        accepted_path = f"{path}.accepted_candidates[{index}]"
        accepted = _object(accepted_value, accepted_path)
        _keys(
            accepted,
            required=(
                "candidate_id",
                "predicted_success_probability",
                "scorer_id",
                "model_version",
            ),
            path=accepted_path,
        )
        accepted_ids.append(_string(accepted["candidate_id"], f"{accepted_path}.candidate_id"))
        _nullable_number(
            accepted["predicted_success_probability"],
            f"{accepted_path}.predicted_success_probability",
            maximum=1.0,
        )
        _nullable_string(accepted["scorer_id"], f"{accepted_path}.scorer_id")
        _nullable_string(accepted["model_version"], f"{accepted_path}.model_version")

    rejected_ids: list[str] = []
    for index, rejected_value in enumerate(_array(grasp["rejected_candidates"], f"{path}.rejected_candidates")):
        rejected_path = f"{path}.rejected_candidates[{index}]"
        rejected = _object(rejected_value, rejected_path)
        _keys(rejected, required=("candidate_id", "reasons"), path=rejected_path)
        rejected_ids.append(_string(rejected["candidate_id"], f"{rejected_path}.candidate_id"))
        reasons = _unique_strings(rejected["reasons"], f"{rejected_path}.reasons")
        if not reasons:
            _fail(f"{rejected_path}.reasons", "must contain at least one reason")

    partition = accepted_ids + rejected_ids
    if len(set(partition)) != len(partition):
        _fail(path, "accepted and rejected candidate IDs must be unique")
    if set(partition) != set(generated):
        _fail(path, "accepted and rejected candidates must exactly partition generated candidates")

    selected = _nullable_string(grasp["selected_candidate_id"], f"{path}.selected_candidate_id")
    if selected is not None and selected not in set(accepted_ids):
        _fail(f"{path}.selected_candidate_id", "must identify an accepted candidate")
    outcome_observed = _bool(
        grasp["selected_candidate_outcome_observed"],
        f"{path}.selected_candidate_outcome_observed",
    )
    if selected is None and outcome_observed:
        _fail(
            f"{path}.selected_candidate_outcome_observed",
            "cannot be true when no candidate was selected",
        )
    complete_grasp_success = _nullable_bool(
        grasp["selected_candidate_complete_grasp_success"],
        f"{path}.selected_candidate_complete_grasp_success",
    )
    if outcome_observed and complete_grasp_success is None:
        _fail(
            f"{path}.selected_candidate_complete_grasp_success",
            "must be boolean when the selected candidate outcome was observed",
        )
    if not outcome_observed and complete_grasp_success is not None:
        _fail(
            f"{path}.selected_candidate_complete_grasp_success",
            "must be null when the selected candidate outcome was not observed",
        )
    ranking_correct = _nullable_bool(
        grasp["ranking_correct"], f"{path}.ranking_correct"
    )
    ranking_evidence_id = _nullable_string(
        grasp["ranking_evidence_id"], f"{path}.ranking_evidence_id"
    )
    if ranking_correct is not None and selected is None:
        _fail(
            f"{path}.ranking_correct",
            "must be null when no candidate was selected",
        )
    if ranking_correct is not None and len(accepted_ids) < 2:
        _fail(
            f"{path}.ranking_correct",
            "requires at least two accepted candidates",
        )
    if ranking_correct is not None and ranking_evidence_id is None:
        _fail(
            f"{path}.ranking_evidence_id",
            "is required when ranking_correct is not null",
        )
    if ranking_correct is None and ranking_evidence_id is not None:
        _fail(
            f"{path}.ranking_evidence_id",
            "must be null when ranking_correct is null",
        )


def _validate_outcome(value: Any, path: str) -> None:
    outcome = _object(value, path)
    _keys(
        outcome,
        required=(
            "status",
            "contact_count",
            "operator_intervention_count",
            "retry_count",
            "verification",
        ),
        path=path,
    )
    status = _string(outcome["status"], f"{path}.status")
    if status not in OUTCOME_STATUSES:
        _fail(f"{path}.status", "has an unsupported outcome")
    _integer(outcome["contact_count"], f"{path}.contact_count")
    _integer(
        outcome["operator_intervention_count"],
        f"{path}.operator_intervention_count",
    )
    _integer(outcome["retry_count"], f"{path}.retry_count")
    verification = _unique_strings(outcome["verification"], f"{path}.verification")
    allowed = {"camera", "simulator_truth", "operator", "force_sensor"}
    invalid = sorted(set(verification) - allowed)
    if invalid:
        _fail(f"{path}.verification", f"unsupported methods: {', '.join(invalid)}")


def _validate_timing(value: Any, path: str) -> None:
    timing = _object(value, path)
    _keys(timing, required=(), optional=TIMING_FIELDS, path=path)
    for field, field_value in timing.items():
        _number(field_value, f"{path}.{field}")


def validate_evaluation_result(value: Any, *, source: str = "report") -> Mapping[str, Any]:
    """Validate one parsed evaluation result and return it unchanged."""

    report = _object(value, source)
    _keys(
        report,
        required=(
            "schema_version",
            "evaluation_id",
            "created_at_utc",
            "split",
            "backend",
            "provenance",
            "vision",
            "episodes",
        ),
        path=source,
    )
    if (
        isinstance(report["schema_version"], bool)
        or not isinstance(report["schema_version"], int)
        or report["schema_version"] != SCHEMA_VERSION
    ):
        _fail(f"{source}.schema_version", f"must equal {SCHEMA_VERSION}")
    _string(report["evaluation_id"], f"{source}.evaluation_id")
    _validate_created_at(report["created_at_utc"], f"{source}.created_at_utc")
    if report["split"] not in {"pilot", "train", "validation", "test"}:
        _fail(f"{source}.split", "has an unsupported split")
    if report["backend"] not in {"offline_replay", "isaac", "hardware"}:
        _fail(f"{source}.backend", "has an unsupported backend")
    _validate_provenance(report["provenance"], f"{source}.provenance")
    _validate_vision(report["vision"], f"{source}.vision")

    episodes = _array(report["episodes"], f"{source}.episodes")
    if not episodes:
        _fail(f"{source}.episodes", "must contain at least one episode")
    seen_episode_ids: set[str] = set()
    for index, episode_value in enumerate(episodes):
        episode_path = f"{source}.episodes[{index}]"
        episode = _object(episode_value, episode_path)
        _keys(
            episode,
            required=(
                "episode_id",
                "condition_id",
                "scene_id",
                "target_track_id",
                "target_class_id",
                "grasp_attempt",
                "outcome",
                "timing_s",
            ),
            path=episode_path,
        )
        episode_id = _string(episode["episode_id"], f"{episode_path}.episode_id")
        if episode_id in seen_episode_ids:
            _fail(f"{episode_path}.episode_id", "duplicates an episode ID")
        seen_episode_ids.add(episode_id)
        _string(episode["condition_id"], f"{episode_path}.condition_id")
        _string(episode["scene_id"], f"{episode_path}.scene_id")
        target_track_id = _nullable_string(
            episode["target_track_id"], f"{episode_path}.target_track_id"
        )
        target_class_id = _nullable_string(
            episode["target_class_id"], f"{episode_path}.target_class_id"
        )
        _validate_grasp(episode["grasp_attempt"], f"{episode_path}.grasp_attempt")
        _validate_outcome(episode["outcome"], f"{episode_path}.outcome")
        _validate_timing(episode["timing_s"], f"{episode_path}.timing_s")

        grasp = episode["grasp_attempt"]
        status = episode["outcome"]["status"]
        if grasp is not None and grasp["selected_candidate_outcome_observed"]:
            if grasp["selected_candidate_id"] is None:
                _fail(
                    f"{episode_path}.grasp_attempt.selected_candidate_outcome_observed",
                    "cannot be true when no candidate was selected",
                )
            if status in {"cancelled", "unresolved", "not_executed"}:
                _fail(
                    f"{episode_path}.grasp_attempt.selected_candidate_outcome_observed",
                    f"cannot be true for outcome status {status}",
                )

        if status in OBSERVED_TERMINAL_STATUSES:
            if target_track_id is None or target_class_id is None:
                _fail(
                    episode_path,
                    f"{status} requires non-null target IDs",
                )
            if grasp is None or grasp["selected_candidate_id"] is None:
                _fail(
                    episode_path,
                    f"{status} requires a selected accepted candidate",
                )
            if not grasp["selected_candidate_outcome_observed"]:
                _fail(
                    episode_path,
                    f"{status} requires selected_candidate_outcome_observed=true",
                )
            if not episode["outcome"]["verification"]:
                _fail(
                    f"{episode_path}.outcome.verification",
                    f"{status} requires verification evidence",
                )

        if status == "complete_relocation_success":
            if (
                grasp["selected_candidate_complete_grasp_success"] is not True
            ):
                _fail(
                    episode_path,
                    "complete_relocation_success requires an observed successful grasp",
                )
        if status in {"failed_lift", "drop_or_slip"}:
            if grasp["selected_candidate_complete_grasp_success"] is not False:
                _fail(
                    episode_path,
                    f"{status} requires selected_candidate_complete_grasp_success=false",
                )
    return report


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_evaluation_result(path: Path) -> Mapping[str, Any]:
    """Load JSON while rejecting duplicate keys and NaN/Infinity constants."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    EvaluationValidationError(
                        f"{path}: non-standard JSON constant {constant}"
                    )
                ),
            )
    except OSError as exc:
        raise EvaluationValidationError(f"{path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationValidationError(f"{path}: invalid JSON: {exc}") from exc
    return validate_evaluation_result(value, source=str(path))


def _metric(values: Sequence[float]) -> dict[str, int | float | None]:
    if not values:
        return {"sample_count": 0, "mean": None, "p95": None, "max": None}
    ordered = sorted(values)
    nearest_rank_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "sample_count": len(ordered),
        "mean": math.fsum(ordered) / len(ordered),
        "p95": ordered[nearest_rank_index],
        "max": ordered[-1],
    }


def _calibration_bins(samples: Sequence[tuple[float, int]]) -> list[dict[str, Any]]:
    bins: list[list[tuple[float, int]]] = [[] for _ in range(5)]
    for probability, observed in samples:
        bins[min(int(probability * 5), 4)].append((probability, observed))
    result: list[dict[str, Any]] = []
    for index, entries in enumerate(bins):
        low = index / 5
        high = (index + 1) / 5
        result.append(
            {
                "range": f"[{low:.1f}, {high:.1f}{']' if index == 4 else ')'}",
                "sample_count": len(entries),
                "mean_predicted_probability": (
                    math.fsum(item[0] for item in entries) / len(entries)
                    if entries
                    else None
                ),
                "observed_success_rate": (
                    math.fsum(item[1] for item in entries) / len(entries)
                    if entries
                    else None
                ),
            }
        )
    return result


def _configuration_signature(report: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the fields that must match for an ordinary aggregate."""

    provenance = report["provenance"]
    profile = provenance["integration_profile"]
    profile_signature = (
        None
        if profile is None
        else (profile["profile_id"], profile["profile_sha256"])
    )
    component_signature = tuple(
        sorted(
            (
                component["component"],
                component["name"],
                component["version"],
                component["artifact_sha256"],
            )
            for component in provenance["components"]
        )
    )
    return (
        provenance["project_commit"],
        report["split"],
        report["backend"],
        profile_signature,
        provenance["object_catalog_version"],
        tuple(sorted(provenance["calibration_ids"])),
        component_signature,
    )


def summarize_evaluation_results(
    reports: Sequence[Mapping[str, Any]],
    *,
    allow_mixed: bool = False,
) -> dict[str, Any]:
    """Aggregate validated reports without filling absent measurements.

    By default, reports must share the commit, split, backend, integration
    profile, catalog, calibration set, and component versions. ``allow_mixed``
    is an explicit escape hatch for descriptive summaries and emits a warning.
    """

    if not reports:
        raise EvaluationValidationError("at least one evaluation report is required")
    for index, report in enumerate(reports):
        validate_evaluation_result(report, source=f"reports[{index}]")

    configuration_signatures = {
        _configuration_signature(report) for report in reports
    }
    mixed_configurations = len(configuration_signatures) > 1
    mixed_warning = (
        "MIXED EVALUATION CONFIGURATIONS: this aggregate combines different "
        "commits, splits, backends, profiles, catalogs, calibrations, and/or "
        "components. Treat its rates as descriptive only, not as a matched "
        "policy comparison."
    )
    if mixed_configurations and not allow_mixed:
        raise EvaluationValidationError(
            "reports use heterogeneous evaluation configurations; summarize "
            "matched configurations separately, or explicitly pass "
            "allow_mixed=True (--allow-mixed in the CLI) for a descriptive "
            "aggregate"
        )

    outcomes: Counter[str] = Counter()
    contact_events = 0
    contact_episodes = 0
    intervention_events = 0
    intervention_episodes = 0
    retries = 0
    executed_episodes = 0
    timing_values: dict[str, list[float]] = {field: [] for field in TIMING_FIELDS}

    vision_frames_by_class: Counter[str] = Counter()
    vision_counts: dict[str, Counter[str]] = defaultdict(Counter)
    position_errors: dict[str, list[float]] = defaultdict(list)
    orientation_errors: dict[str, list[float]] = defaultdict(list)
    vision_latencies: list[float] = []

    grasp_attempts = 0
    generated_candidates = 0
    accepted_candidates = 0
    rejected_candidates = 0
    rejection_reasons: Counter[str] = Counter()
    prediction_samples: list[tuple[float, int]] = []
    ranking_samples: list[bool] = []

    evaluation_ids: list[str] = []
    seen_evaluation_ids: set[str] = set()
    commits: set[str] = set()
    branches: set[str] = set()
    profiles: set[tuple[str, str | None]] = set()
    backends: set[str] = set()
    splits: set[str] = set()
    object_catalog_versions: set[str] = set()
    calibration_ids: set[str] = set()
    components: set[tuple[str, str, str, str | None]] = set()
    source_run_ids: set[str] = set()
    episode_ids: set[str] = set()
    condition_ids: set[str] = set()

    for report in reports:
        if report["evaluation_id"] in seen_evaluation_ids:
            raise EvaluationValidationError(
                f"duplicate evaluation_id: {report['evaluation_id']}"
            )
        evaluation_ids.append(report["evaluation_id"])
        seen_evaluation_ids.add(report["evaluation_id"])
        provenance = report["provenance"]
        commits.add(provenance["project_commit"])
        branches.add(provenance["branch"])
        backends.add(report["backend"])
        splits.add(report["split"])
        profile = provenance["integration_profile"]
        profiles.add(
            (profile["profile_id"], profile["profile_sha256"])
            if profile is not None
            else ("baseline-unset", None)
        )
        object_catalog_versions.add(provenance["object_catalog_version"])
        calibration_ids.update(provenance["calibration_ids"])
        overlapping_run_ids = source_run_ids.intersection(
            provenance["source_run_ids"]
        )
        if overlapping_run_ids:
            raise EvaluationValidationError(
                "overlapping source_run_ids across reports: "
                + ", ".join(sorted(overlapping_run_ids))
            )
        source_run_ids.update(provenance["source_run_ids"])
        for component in provenance["components"]:
            components.add(
                (
                    component["component"],
                    component["name"],
                    component["version"],
                    component["artifact_sha256"],
                )
            )

        vision = report["vision"]
        if vision is not None:
            vision_latencies.extend(float(value) for value in vision["inference_latency_ms"])
            for class_result in vision["per_class"]:
                class_id = class_result["class_id"]
                vision_frames_by_class[class_id] += vision["frames_evaluated"]
                counts = vision_counts[class_id]
                for field in (
                    "reference_instances",
                    "true_positives",
                    "false_positives",
                    "dropouts",
                ):
                    counts[field] += class_result[field]
                for sample in class_result["pose_samples"]:
                    position_errors[class_id].append(float(sample["position_error_mm"]))
                    orientation = sample["symmetry_aware_orientation_error_deg"]
                    if orientation is not None:
                        orientation_errors[class_id].append(float(orientation))

        for episode in report["episodes"]:
            if episode["episode_id"] in episode_ids:
                raise EvaluationValidationError(
                    "duplicate episode_id across reports: "
                    + episode["episode_id"]
                )
            episode_ids.add(episode["episode_id"])
            condition_ids.add(episode["condition_id"])
            outcome = episode["outcome"]
            status = outcome["status"]
            outcomes[status] += 1
            if status != "not_executed":
                executed_episodes += 1
            contacts = outcome["contact_count"]
            contact_events += contacts
            contact_episodes += int(contacts > 0)
            interventions = outcome["operator_intervention_count"]
            intervention_events += interventions
            intervention_episodes += int(interventions > 0)
            retries += outcome["retry_count"]

            for field, value in episode["timing_s"].items():
                timing_values[field].append(float(value))

            grasp = episode["grasp_attempt"]
            if grasp is None:
                continue
            grasp_attempts += 1
            generated_candidates += len(grasp["generated_candidate_ids"])
            accepted_candidates += len(grasp["accepted_candidates"])
            rejected_candidates += len(grasp["rejected_candidates"])
            for rejected in grasp["rejected_candidates"]:
                rejection_reasons.update(rejected["reasons"])
            if grasp["ranking_correct"] is not None:
                ranking_samples.append(grasp["ranking_correct"])

            selected_id = grasp["selected_candidate_id"]
            selected = next(
                (
                    candidate
                    for candidate in grasp["accepted_candidates"]
                    if candidate["candidate_id"] == selected_id
                ),
                None,
            )
            if (
                grasp["selected_candidate_outcome_observed"]
                and selected is not None
                and selected["predicted_success_probability"] is not None
            ):
                prediction_samples.append(
                    (
                        float(selected["predicted_success_probability"]),
                        int(grasp["selected_candidate_complete_grasp_success"]),
                    )
                )

    per_class: dict[str, Any] = {}
    for class_id in sorted(vision_counts):
        counts = vision_counts[class_id]
        reference = counts["reference_instances"]
        frames = vision_frames_by_class[class_id]
        per_class[class_id] = {
            "frames_evaluated": frames,
            "reference_instances": reference,
            "true_positives": counts["true_positives"],
            "false_positives": counts["false_positives"],
            "dropouts": counts["dropouts"],
            "recall": counts["true_positives"] / reference if reference else None,
            "dropout_rate": counts["dropouts"] / reference if reference else None,
            "false_positives_per_frame": counts["false_positives"] / frames if frames else None,
            "position_error_mm": _metric(position_errors[class_id]),
            "symmetry_aware_orientation_error_deg": _metric(
                orientation_errors[class_id]
            ),
        }

    probability_values = [sample[0] for sample in prediction_samples]
    observed_values = [sample[1] for sample in prediction_samples]
    brier_values = [
        (probability - observed) ** 2
        for probability, observed in prediction_samples
    ]
    complete_successes = outcomes["complete_relocation_success"]
    return {
        "schema_version": 1,
        "warnings": [mixed_warning] if mixed_configurations else [],
        "configuration_count": len(configuration_signatures),
        "report_count": len(reports),
        "evaluation_ids": evaluation_ids,
        "provenance": {
            "project_commits": sorted(commits),
            "branches": sorted(branches),
            "splits": sorted(splits),
            "integration_profiles": [
                {"profile_id": profile_id, "profile_sha256": profile_sha256}
                for profile_id, profile_sha256 in sorted(
                    profiles, key=lambda item: (item[0], item[1] or "")
                )
            ],
            "backends": sorted(backends),
            "object_catalog_versions": sorted(object_catalog_versions),
            "calibration_ids": sorted(calibration_ids),
            "components": [
                {
                    "component": component,
                    "name": name,
                    "version": version,
                    "artifact_sha256": artifact_sha256,
                }
                for component, name, version, artifact_sha256 in sorted(
                    components,
                    key=lambda item: (item[0], item[1], item[2], item[3] or ""),
                )
            ],
            "source_run_ids": sorted(source_run_ids),
        },
        "episodes": {
            "total": sum(outcomes.values()),
            "unique_conditions": len(condition_ids),
            "condition_ids": sorted(condition_ids),
            "executed": executed_episodes,
            "outcomes": {status: outcomes[status] for status in OUTCOME_STATUSES},
            "complete_relocation_successes": complete_successes,
            "complete_relocation_success_rate": (
                complete_successes / executed_episodes if executed_episodes else None
            ),
            "contact_episodes": contact_episodes,
            "contact_events": contact_events,
            "operator_intervention_episodes": intervention_episodes,
            "operator_intervention_events": intervention_events,
            "retries": retries,
        },
        "vision": {
            "inference_latency_ms": _metric(vision_latencies),
            "per_class": per_class,
        },
        "grasp": {
            "attempts": grasp_attempts,
            "generated_candidates": generated_candidates,
            "accepted_candidates": accepted_candidates,
            "rejected_candidates": rejected_candidates,
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "selected_probability": {
                "sample_count": len(prediction_samples),
                "mean_predicted_probability": (
                    math.fsum(probability_values) / len(probability_values)
                    if probability_values
                    else None
                ),
                "observed_success_rate": (
                    math.fsum(observed_values) / len(observed_values)
                    if observed_values
                    else None
                ),
                "brier_score": (
                    math.fsum(brier_values) / len(brier_values)
                    if brier_values
                    else None
                ),
                "calibration_bins": _calibration_bins(prediction_samples),
            },
            "ranking": {
                "sample_count": len(ranking_samples),
                "correct": sum(ranking_samples),
                "accuracy": (
                    sum(ranking_samples) / len(ranking_samples)
                    if ranking_samples
                    else None
                ),
            },
        },
        "timing_s": {
            field: _metric(timing_values[field]) for field in TIMING_FIELDS
        },
    }


def _format_value(value: float | int | None, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def format_text_summary(summary: Mapping[str, Any]) -> str:
    """Render a compact, judge-readable summary."""

    episodes = summary["episodes"]
    grasp = summary["grasp"]
    lines: list[str] = []
    for warning in summary["warnings"]:
        lines.extend((f"WARNING: {warning}", ""))
    lines.extend([
        "Evaluation summary",
        f"Reports: {summary['report_count']}",
        (
            f"Episodes: {episodes['total']} total, {episodes['executed']} executed, "
            f"{episodes['complete_relocation_successes']} complete relocations "
            f"({_format_value(episodes['complete_relocation_success_rate'])})"
        ),
        "Outcomes: "
        + ", ".join(
            f"{status}={count}" for status, count in episodes["outcomes"].items()
        ),
        (
            f"Contacts: {episodes['contact_events']} events in "
            f"{episodes['contact_episodes']} episodes; interventions: "
            f"{episodes['operator_intervention_events']} events"
        ),
        (
            f"Grasp candidates: {grasp['generated_candidates']} generated, "
            f"{grasp['accepted_candidates']} accepted, "
            f"{grasp['rejected_candidates']} rejected"
        ),
    ])
    selected = grasp["selected_probability"]
    lines.append(
        "Selected-probability evidence: "
        f"n={selected['sample_count']}, Brier={_format_value(selected['brier_score'])}, "
        f"observed success={_format_value(selected['observed_success_rate'])}"
    )
    ranking = grasp["ranking"]
    lines.append(
        f"Ranking evidence: n={ranking['sample_count']}, "
        f"accuracy={_format_value(ranking['accuracy'])}"
    )

    per_class = summary["vision"]["per_class"]
    if per_class:
        lines.append("Vision by class:")
        for class_id, metrics in per_class.items():
            position = metrics["position_error_mm"]
            orientation = metrics["symmetry_aware_orientation_error_deg"]
            lines.append(
                f"  {class_id}: recall={_format_value(metrics['recall'])}, "
                f"FP/frame={_format_value(metrics['false_positives_per_frame'])}, "
                f"position mean={_format_value(position['mean'])} mm (n={position['sample_count']}), "
                f"orientation mean={_format_value(orientation['mean'])} deg "
                f"(n={orientation['sample_count']})"
            )
    else:
        lines.append("Vision by class: no vision measurements supplied")

    lines.append("Timing (measured samples only):")
    for field, metric in summary["timing_s"].items():
        lines.append(
            f"  {field}: mean={_format_value(metric['mean'])}, "
            f"p95={_format_value(metric['p95'])}, n={metric['sample_count']}"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and summarize one or more evaluation-result.v1 JSON files. "
            "Missing metrics remain n/a rather than becoming zero."
        )
    )
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="also write the aggregate summary as JSON",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print JSON instead of the default readable text",
    )
    parser.add_argument(
        "--allow-mixed",
        action="store_true",
        help=(
            "explicitly combine heterogeneous configurations for a descriptive "
            "summary; output includes a prominent warning"
        ),
    )
    args = parser.parse_args(argv)

    if args.json_output is not None:
        resolved_output = args.json_output.resolve()
        if any(path.resolve() == resolved_output for path in args.reports):
            parser.error("--json-output must not overwrite an input report")

    try:
        reports = [load_evaluation_result(path) for path in args.reports]
        summary = summarize_evaluation_results(
            reports, allow_mixed=args.allow_mixed
        )
    except EvaluationValidationError as exc:
        parser.error(str(exc))

    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded, encoding="utf-8")
    if args.json:
        print(encoded, end="")
    else:
        print(format_text_summary(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
