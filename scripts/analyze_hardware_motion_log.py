#!/usr/bin/env python3
"""Summarize JSON events embedded in one or more FANUC hardware log files.

The analyzer is offline and read-only. It reports indicators such as zero queue
depth or repeated short motion chunks; those indicators are not proof of a
controller pause because semantic checkpoints intentionally drain the queue.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


SCHEMA_VERSION = 2
SHORT_MOTION_MAX_GEOMETRIC_ROWS = 3
REPEATED_SHORT_MOTION_CALLS_PER_OCCURRENCE = 2
ZERO_QUEUE_MOTION_CALLS_PER_OCCURRENCE = 3
LONG_DISPATCH_MIN_ACTION_ROWS = 8
LATENCY_BUDGET_APPROACH_FRACTION = 0.8
VISIBILITY_RETRY_EVENTS = (
    "pre_descent_visibility_retry",
    "post_place_visibility_retry",
)
TIMESTAMP_PREFIX_PATTERN = re.compile(
    r"^\s*(?P<raw>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"
    r"(?:[.,]\d{1,6})?(?:Z|[+-]\d{2}:?\d{2})?)(?=\s|$)"
)


@dataclass
class _PhaseOccurrence:
    source_index: int
    source_path: Path
    occurrence_index: int
    phase_occurrence_index: int
    phase: str
    start_line: int
    end_line: int
    last_motion_call: int | None = None
    motion_calls: int = 0
    short_motion_calls: int = 0
    zero_queue_motion_calls: int = 0
    timestamped_phase_action_lines: int = 0
    timestamp_prefix_start: dict[str, Any] | None = None
    timestamp_prefix_end: dict[str, Any] | None = None

    @property
    def occurrence_id(self) -> str:
        return (
            f"source-{self.source_index:03d}:"
            f"occurrence-{self.occurrence_index:04d}"
        )

    def begins_new_occurrence(
        self, phase: str, motion_call: int | None
    ) -> bool:
        if phase != self.phase:
            return True
        return (
            motion_call is not None
            and self.last_motion_call is not None
            and motion_call <= self.last_motion_call
        )

    def observe(
        self,
        *,
        line_number: int,
        timestamp_prefix: dict[str, str] | None,
        kind: str,
        geometric_rows: int | None,
        queue_depth: int | None,
        motion_call: int | None,
    ) -> None:
        self.end_line = line_number
        if timestamp_prefix is not None:
            timestamp_evidence: dict[str, Any] = {
                "line": line_number,
                **timestamp_prefix,
            }
            if self.timestamp_prefix_start is None:
                self.timestamp_prefix_start = timestamp_evidence
            self.timestamp_prefix_end = timestamp_evidence
            self.timestamped_phase_action_lines += 1
        if kind == "motion":
            self.motion_calls += 1
            if (
                geometric_rows is not None
                and geometric_rows <= SHORT_MOTION_MAX_GEOMETRIC_ROWS
            ):
                self.short_motion_calls += 1
            if queue_depth == 0:
                self.zero_queue_motion_calls += 1
        if motion_call is not None:
            self.last_motion_call = motion_call

    def evidence(self) -> dict[str, Any]:
        return {
            "occurrence_id": self.occurrence_id,
            "source_index": self.source_index,
            "source": str(self.source_path),
            "phase": self.phase,
            "phase_occurrence_index": self.phase_occurrence_index,
            "line_start": self.start_line,
            "line_end": self.end_line,
            "motion_calls": self.motion_calls,
            "short_motion_calls": self.short_motion_calls,
            "zero_queue_motion_calls": self.zero_queue_motion_calls,
            "timestamped_phase_action_lines": (
                self.timestamped_phase_action_lines
            ),
            "timestamp_prefix_start": self.timestamp_prefix_start,
            "timestamp_prefix_end": self.timestamp_prefix_end,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, Any]:
    stat_before = path.stat()
    digest = _sha256(path)
    stat_after = path.stat()
    before_identity = (stat_before.st_size, stat_before.st_mtime_ns)
    after_identity = (stat_after.st_size, stat_after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError(f"source changed while it was being hashed: {path}")
    return {
        "sha256": digest,
        "size_bytes": stat_after.st_size,
        "modified_time_ns": stat_after.st_mtime_ns,
    }


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nonnegative_int(value: Any) -> int | None:
    number = _finite_number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _positive_number(value: Any) -> float | None:
    number = _finite_number(value)
    if number is None or number <= 0:
        return None
    return number


def _nonnegative_number(value: Any) -> float | None:
    number = _finite_number(value)
    if number is None or number < 0:
        return None
    return number


def _boolean(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _event_from_line(line: str) -> tuple[dict[str, Any] | None, bool]:
    """Return the first JSON event object and whether malformed JSON was seen."""

    decoder = json.JSONDecoder()
    malformed_candidate = False
    search_from = 0
    while True:
        start = line.find("{", search_from)
        if start < 0:
            return None, malformed_candidate
        try:
            value, _ = decoder.raw_decode(line[start:])
        except json.JSONDecodeError:
            malformed_candidate = True
            search_from = start + 1
            continue
        if isinstance(value, dict) and isinstance(value.get("event"), str):
            return value, malformed_candidate
        search_from = start + 1


def _timestamp_prefix(line: str) -> dict[str, str] | None:
    match = TIMESTAMP_PREFIX_PATTERN.match(line)
    if match is None:
        return None
    raw = match.group("raw")
    normalized = raw.replace(" ", "T", 1).replace(",", ".")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    timezone_without_colon = re.search(r"([+-]\d{2})(\d{2})$", normalized)
    if timezone_without_colon is not None:
        normalized = (
            normalized[: timezone_without_colon.start()]
            + timezone_without_colon.group(1)
            + ":"
            + timezone_without_colon.group(2)
        )
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return {"raw": raw, "iso8601": parsed.isoformat()}


def _depth_summary(depths: Iterable[int]) -> dict[str, Any]:
    values = list(depths)
    histogram = Counter(values)
    return {
        "observations": len(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "mean": round(sum(values) / len(values), 3) if values else None,
        "histogram": {str(depth): count for depth, count in sorted(histogram.items())},
    }


def _number_summary(values: Iterable[float]) -> dict[str, Any]:
    samples = sorted(float(value) for value in values)
    if not samples:
        return {
            "observations": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
            "p95_nearest_rank": None,
        }
    middle = len(samples) // 2
    if len(samples) % 2:
        median = samples[middle]
    else:
        median = (samples[middle - 1] + samples[middle]) / 2.0
    p95_index = max(0, math.ceil(0.95 * len(samples)) - 1)
    return {
        "observations": len(samples),
        "minimum": round(samples[0], 6),
        "maximum": round(samples[-1], 6),
        "mean": round(sum(samples) / len(samples), 6),
        "median": round(median, 6),
        "p95_nearest_rank": round(samples[p95_index], 6),
    }


def _boolean_summary(values: Iterable[bool], total_events: int) -> dict[str, int]:
    samples = list(values)
    true_count = sum(samples)
    return {
        "true": true_count,
        "false": len(samples) - true_count,
        "missing_or_invalid": total_events - len(samples),
    }


def _dispatch_evidence(
    *,
    source_index: int,
    source_path: Path,
    line_number: int,
    event: dict[str, Any],
) -> dict[str, Any]:
    """Return bounded, directly observed fields for one dispatch indicator."""

    keys = (
        "action_rows",
        "dispatch_pipeline_s",
        "dispatch_pipeline_ewma_s",
        "action_interval_s",
        "prefetch_lead_rows",
        "preview_queue_depth",
        "full_observation_queue_depth",
        "queue_preview_used",
        "replacement_used",
    )
    return {
        "source_index": source_index,
        "source": str(source_path),
        "line": line_number,
        "observed": {key: event.get(key) for key in keys},
    }


def _counter_summary(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def analyze_logs(paths: Iterable[Path]) -> dict[str, Any]:
    sources = tuple(Path(path).expanduser().resolve(strict=True) for path in paths)
    if not sources:
        raise ValueError("at least one log path is required")
    if len(set(sources)) != len(sources):
        raise ValueError("log paths must be unique")
    fingerprints_before = {path: _fingerprint(path) for path in sources}

    event_counts: Counter[str] = Counter()
    lines_total = 0
    json_event_lines = 0
    malformed_json_candidate_lines = 0
    phase_rows: dict[str, Counter[str]] = defaultdict(Counter)
    phase_profiles: dict[str, Counter[str]] = defaultdict(Counter)
    phase_queue_depths: dict[str, list[int]] = defaultdict(list)
    action_kind_rows: dict[str, Counter[str]] = defaultdict(Counter)
    profile_rows: dict[str, Counter[str]] = defaultdict(Counter)
    phase_motion_call_max: dict[str, int] = {}
    blocked_reasons: Counter[str] = Counter()
    retry_counts: Counter[str] = Counter()
    queue_depths: list[int] = []
    motion_queue_depths: list[int] = []
    phase_actions = 0
    motion_actions = 0
    dwell_actions = 0
    unknown_kind_actions = 0
    terminal_padding_rows = 0
    short_motion_calls = 0
    telemetry_coverage: Counter[str] = Counter()
    occurrences: list[_PhaseOccurrence] = []
    dispatch_events = 0
    dispatch_coverage: Counter[str] = Counter()
    dispatch_action_rows: list[int] = []
    dispatch_pipeline_s: list[float] = []
    dispatch_pipeline_ewma_s: list[float] = []
    dispatch_action_interval_s: list[float] = []
    dispatch_lead_rows: list[int] = []
    dispatch_preview_queue_depths: list[int] = []
    dispatch_full_queue_depths: list[int] = []
    dispatch_preview_used: list[bool] = []
    dispatch_replacement_used: list[bool] = []
    zero_lead_long_dispatches: list[dict[str, Any]] = []
    latency_budget_evidence: list[dict[str, Any]] = []
    latency_budget_unassessable = 0

    for source_index, path in enumerate(sources, start=1):
        active_occurrence: _PhaseOccurrence | None = None
        source_occurrence_index = 0
        phase_occurrence_counts: Counter[str] = Counter()
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, start=1):
                lines_total += 1
                event, malformed = _event_from_line(line)
                malformed_json_candidate_lines += int(malformed)
                if event is None:
                    continue
                json_event_lines += 1
                event_name = str(event["event"])
                event_counts[event_name] += 1
                if event_name in VISIBILITY_RETRY_EVENTS:
                    retry_counts[event_name] += 1
                if event_name == "policy_blocked":
                    reason = str(event.get("reason", "<missing reason>"))
                    blocked_reasons[reason] += 1
                if event_name == "client_action_dispatch":
                    dispatch_events += 1
                    action_rows = _nonnegative_int(event.get("action_rows"))
                    pipeline_s = _nonnegative_number(
                        event.get("dispatch_pipeline_s")
                    )
                    pipeline_ewma_s = _nonnegative_number(
                        event.get("dispatch_pipeline_ewma_s")
                    )
                    action_interval_s = _positive_number(
                        event.get("action_interval_s")
                    )
                    lead_rows = _nonnegative_int(
                        event.get("prefetch_lead_rows")
                    )
                    preview_queue_depth = _nonnegative_int(
                        event.get("preview_queue_depth")
                    )
                    full_queue_depth = _nonnegative_int(
                        event.get("full_observation_queue_depth")
                    )
                    preview_used = _boolean(event.get("queue_preview_used"))
                    replacement_used = _boolean(event.get("replacement_used"))
                    parsed_fields = (
                        ("action_rows", action_rows, dispatch_action_rows),
                        ("dispatch_pipeline_s", pipeline_s, dispatch_pipeline_s),
                        (
                            "dispatch_pipeline_ewma_s",
                            pipeline_ewma_s,
                            dispatch_pipeline_ewma_s,
                        ),
                        (
                            "action_interval_s",
                            action_interval_s,
                            dispatch_action_interval_s,
                        ),
                        ("prefetch_lead_rows", lead_rows, dispatch_lead_rows),
                        (
                            "preview_queue_depth",
                            preview_queue_depth,
                            dispatch_preview_queue_depths,
                        ),
                        (
                            "full_observation_queue_depth",
                            full_queue_depth,
                            dispatch_full_queue_depths,
                        ),
                    )
                    for field, value, values in parsed_fields:
                        if value is not None:
                            dispatch_coverage[field] += 1
                            values.append(value)
                    if preview_used is not None:
                        dispatch_coverage["queue_preview_used"] += 1
                        dispatch_preview_used.append(preview_used)
                    if replacement_used is not None:
                        dispatch_coverage["replacement_used"] += 1
                        dispatch_replacement_used.append(replacement_used)

                    evidence = _dispatch_evidence(
                        source_index=source_index,
                        source_path=path,
                        line_number=line_number,
                        event=event,
                    )
                    if (
                        preview_used is True
                        and action_rows is not None
                        and action_rows >= LONG_DISPATCH_MIN_ACTION_ROWS
                        and lead_rows == 0
                    ):
                        zero_lead_long_dispatches.append(evidence)

                    if preview_used is True:
                        if (
                            pipeline_s is None
                            or action_interval_s is None
                            or preview_queue_depth is None
                        ):
                            latency_budget_unassessable += 1
                        else:
                            available_s = preview_queue_depth * action_interval_s
                            if available_s == 0:
                                fraction = math.inf if pipeline_s > 0 else 0.0
                            else:
                                fraction = pipeline_s / available_s
                            if fraction >= LATENCY_BUDGET_APPROACH_FRACTION:
                                latency_budget_evidence.append(
                                    {
                                        **evidence,
                                        "available_preview_queue_time_s": round(
                                            available_s, 6
                                        ),
                                        "pipeline_fraction_of_available_time": (
                                            None
                                            if not math.isfinite(fraction)
                                            else round(fraction, 6)
                                        ),
                                        "classification": (
                                            "exceeded"
                                            if fraction >= 1.0
                                            else "approached"
                                        ),
                                    }
                                )
                if event_name != "phase_action":
                    if event_name == "phase_complete" and active_occurrence:
                        occurrences.append(active_occurrence)
                        active_occurrence = None
                    continue

                phase_actions += 1
                phase = str(event.get("phase", "<missing phase>"))
                motion_call = _nonnegative_int(event.get("motion_call"))
                if active_occurrence is not None and (
                    active_occurrence.begins_new_occurrence(phase, motion_call)
                ):
                    occurrences.append(active_occurrence)
                    active_occurrence = None
                if active_occurrence is None:
                    source_occurrence_index += 1
                    phase_occurrence_counts[phase] += 1
                    active_occurrence = _PhaseOccurrence(
                        source_index=source_index,
                        source_path=path,
                        occurrence_index=source_occurrence_index,
                        phase_occurrence_index=phase_occurrence_counts[phase],
                        phase=phase,
                        start_line=line_number,
                        end_line=line_number,
                    )
                raw_kind = event.get("action_kind")
                kind = raw_kind if isinstance(raw_kind, str) else "<missing/invalid>"
                raw_profile = event.get("motion_profile")
                profile = (
                    raw_profile
                    if isinstance(raw_profile, str) and raw_profile
                    else "<missing/invalid>"
                )
                action_rows = _nonnegative_int(event.get("action_rows"))
                if action_rows is not None:
                    telemetry_coverage["action_rows"] += 1
                    phase_rows[phase]["action_rows"] += action_rows
                    action_kind_rows[kind]["action_rows"] += action_rows
                    profile_rows[profile]["action_rows"] += action_rows
                phase_rows[phase]["calls"] += 1
                action_kind_rows[kind]["calls"] += 1
                profile_rows[profile]["calls"] += 1

                if kind == "motion":
                    motion_actions += 1
                    phase_rows[phase]["motion_calls"] += 1
                elif kind == "dwell":
                    dwell_actions += 1
                    phase_rows[phase]["dwell_calls"] += 1
                else:
                    unknown_kind_actions += 1
                    phase_rows[phase]["unknown_kind_calls"] += 1
                if isinstance(raw_kind, str):
                    telemetry_coverage["action_kind"] += 1

                if isinstance(raw_profile, str) and raw_profile:
                    telemetry_coverage["motion_profile"] += 1
                    phase_profiles[phase][profile] += 1

                padding = _nonnegative_int(event.get("terminal_padding_rows"))
                if padding is not None:
                    telemetry_coverage["terminal_padding_rows"] += 1
                    terminal_padding_rows += padding
                    phase_rows[phase]["terminal_padding_rows"] += padding
                    action_kind_rows[kind]["terminal_padding_rows"] += padding
                    profile_rows[profile]["terminal_padding_rows"] += padding

                geometric_rows = _nonnegative_int(event.get("geometric_rows_sent"))
                if geometric_rows is not None:
                    telemetry_coverage["geometric_rows_sent"] += 1
                    phase_rows[phase]["geometric_rows_sent"] += geometric_rows
                    action_kind_rows[kind]["geometric_rows_sent"] += geometric_rows
                    profile_rows[profile]["geometric_rows_sent"] += geometric_rows
                    if (
                        kind == "motion"
                        and geometric_rows <= SHORT_MOTION_MAX_GEOMETRIC_ROWS
                    ):
                        short_motion_calls += 1
                        phase_rows[phase]["short_motion_calls"] += 1

                queue_depth = _nonnegative_int(event.get("queued_at_observation"))
                if queue_depth is not None:
                    telemetry_coverage["queued_at_observation"] += 1
                    queue_depths.append(queue_depth)
                    phase_queue_depths[phase].append(queue_depth)
                    if kind == "motion":
                        motion_queue_depths.append(queue_depth)
                        if queue_depth == 0:
                            phase_rows[phase]["zero_queue_motion_calls"] += 1

                if motion_call is not None:
                    telemetry_coverage["motion_call"] += 1
                    phase_motion_call_max[phase] = max(
                        phase_motion_call_max.get(phase, 0), motion_call
                    )
                active_occurrence.observe(
                    line_number=line_number,
                    timestamp_prefix=_timestamp_prefix(line),
                    kind=kind,
                    geometric_rows=geometric_rows,
                    queue_depth=queue_depth,
                    motion_call=motion_call,
                )
        if active_occurrence is not None:
            occurrences.append(active_occurrence)

    try:
        fingerprints_after = {path: _fingerprint(path) for path in sources}
    except OSError as error:
        raise RuntimeError(
            "a source became unavailable while it was being analyzed"
        ) from error
    changed_sources = [
        path
        for path in sources
        if fingerprints_before[path] != fingerprints_after[path]
    ]
    if changed_sources:
        joined = ", ".join(str(path) for path in changed_sources)
        raise RuntimeError(f"source changed while it was being analyzed: {joined}")

    zero_queue_motion_calls = sum(depth == 0 for depth in motion_queue_depths)
    queue_starvation_occurrences = [
        occurrence.evidence()
        for occurrence in occurrences
        if occurrence.zero_queue_motion_calls
        >= ZERO_QUEUE_MOTION_CALLS_PER_OCCURRENCE
    ]
    repeated_micro_correction_occurrences = [
        occurrence.evidence()
        for occurrence in occurrences
        if occurrence.short_motion_calls
        >= REPEATED_SHORT_MOTION_CALLS_PER_OCCURRENCE
    ]
    queue_starvation_phases = sorted(
        {evidence["phase"] for evidence in queue_starvation_occurrences}
    )
    repeated_micro_correction_phases = sorted(
        {
            evidence["phase"]
            for evidence in repeated_micro_correction_occurrences
        }
    )
    phases: dict[str, Any] = {}
    for phase in sorted(phase_rows):
        row = dict(sorted(phase_rows[phase].items()))
        row["motion_profiles"] = dict(sorted(phase_profiles[phase].items()))
        row["queue_depths"] = _depth_summary(phase_queue_depths[phase])
        if phase in phase_motion_call_max:
            row["maximum_motion_call_index"] = phase_motion_call_max[phase]
        phases[phase] = row

    latency_budget_classifications = Counter(
        evidence["classification"] for evidence in latency_budget_evidence
    )
    preview_true_events = sum(dispatch_preview_used)
    latency_budget_evaluated = (
        preview_true_events - latency_budget_unassessable
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "sources": [
            {
                "path": str(path),
                "sha256_before": fingerprints_before[path]["sha256"],
                "sha256_after": fingerprints_after[path]["sha256"],
                "size_bytes": fingerprints_before[path]["size_bytes"],
                "modified_time_ns": fingerprints_before[path][
                    "modified_time_ns"
                ],
                "stable_during_analysis": True,
            }
            for path in sources
        ],
        "input": {
            "lines_total": lines_total,
            "json_event_lines": json_event_lines,
            "malformed_json_candidate_lines": malformed_json_candidate_lines,
        },
        "event_counts": dict(sorted(event_counts.items())),
        "phase_actions": {
            "calls": phase_actions,
            "motion_calls": motion_actions,
            "dwell_calls": dwell_actions,
            "unknown_kind_calls": unknown_kind_actions,
            "terminal_padding_rows": terminal_padding_rows,
            "short_motion_calls": short_motion_calls,
            "motion_calls_observed_with_zero_queue": zero_queue_motion_calls,
            "queue_depths": _depth_summary(queue_depths),
            "by_phase": phases,
            "by_action_kind": {
                key: _counter_summary(value)
                for key, value in sorted(action_kind_rows.items())
            },
            "by_motion_profile": {
                key: _counter_summary(value)
                for key, value in sorted(profile_rows.items())
            },
        },
        "client_dispatches": {
            "events": dispatch_events,
            "action_rows": _depth_summary(dispatch_action_rows),
            "dispatch_pipeline_s": _number_summary(dispatch_pipeline_s),
            "dispatch_pipeline_ewma_s": _number_summary(
                dispatch_pipeline_ewma_s
            ),
            "action_interval_s": _number_summary(dispatch_action_interval_s),
            "prefetch_lead_rows": _depth_summary(dispatch_lead_rows),
            "preview_queue_depth": _depth_summary(
                dispatch_preview_queue_depths
            ),
            "full_observation_queue_depth": _depth_summary(
                dispatch_full_queue_depths
            ),
            "queue_preview_used": _boolean_summary(
                dispatch_preview_used, dispatch_events
            ),
            "replacement_used": _boolean_summary(
                dispatch_replacement_used, dispatch_events
            ),
            "telemetry_coverage": {
                key: {
                    "events": dispatch_coverage.get(key, 0),
                    "of": dispatch_events,
                }
                for key in (
                    "action_rows",
                    "dispatch_pipeline_s",
                    "dispatch_pipeline_ewma_s",
                    "action_interval_s",
                    "prefetch_lead_rows",
                    "preview_queue_depth",
                    "full_observation_queue_depth",
                    "queue_preview_used",
                    "replacement_used",
                )
            },
            "indicators": {
                "zero_lead_long_dispatches": len(
                    zero_lead_long_dispatches
                ),
                "zero_lead_long_dispatch_minimum_action_rows": (
                    LONG_DISPATCH_MIN_ACTION_ROWS
                ),
                "zero_lead_long_dispatch_evidence": (
                    zero_lead_long_dispatches
                ),
                "latency_budget": {
                    "preview_events": preview_true_events,
                    "evaluated_events": latency_budget_evaluated,
                    "unassessable_events": latency_budget_unassessable,
                    "approached_events": latency_budget_classifications.get(
                        "approached", 0
                    ),
                    "exceeded_events": latency_budget_classifications.get(
                        "exceeded", 0
                    ),
                    "threshold_fraction": LATENCY_BUDGET_APPROACH_FRACTION,
                    "evidence": latency_budget_evidence,
                },
                "interpretation": (
                    "Dispatch indicators are diagnostic leads, not proof of a "
                    "controller pause. A zero-lead long dispatch is reported "
                    "only after a queue preview was used. Latency-budget rows "
                    "compare the complete measured dispatch pipeline with the "
                    "queue time remaining at the preview response; because the "
                    "pipeline measurement also includes the preview request, "
                    "this is a conservative pressure indicator. Camera and "
                    "semantic checkpoints may intentionally drain the queue. "
                    "Logs without action_interval_s remain unassessable rather "
                    "than assuming 0.1 seconds per row."
                ),
            },
        },
        "visibility_retries": dict(sorted(retry_counts.items())),
        "blocked_reasons": [
            {"reason": reason, "count": count}
            for reason, count in sorted(blocked_reasons.items())
        ],
        "telemetry_coverage": {
            key: {"events": telemetry_coverage.get(key, 0), "of": phase_actions}
            for key in (
                "action_rows",
                "action_kind",
                "motion_profile",
                "geometric_rows_sent",
                "terminal_padding_rows",
                "queued_at_observation",
                "motion_call",
            )
        },
        "indicators": {
            "possible_queue_starvation": bool(queue_starvation_phases),
            "queue_starvation_evidence_phases": queue_starvation_phases,
            "queue_starvation_evidence_occurrences": (
                queue_starvation_occurrences
            ),
            "repeated_micro_corrections": bool(
                repeated_micro_correction_phases
            ),
            "repeated_micro_correction_evidence_phases": (
                repeated_micro_correction_phases
            ),
            "repeated_micro_correction_evidence_occurrences": (
                repeated_micro_correction_occurrences
            ),
            "interpretation": (
                "Indicators are diagnostic leads, not proof. Queue depth can "
                "intentionally reach zero at camera, grasp, release, or semantic "
                "phase boundaries. Flags require repeated evidence within one "
                "segmented phase occurrence; correlate them with video and "
                "timestamps."
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="mixed text/JSON-lines log")
    parser.add_argument(
        "--output",
        type=Path,
        help="optional new JSON report path; stdout is always printed",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = tuple(path.expanduser().resolve(strict=True) for path in args.logs)
    output = args.output.expanduser().resolve() if args.output is not None else None
    if output is not None:
        if output in paths:
            raise ValueError("--output must not overwrite an input log")
        if not output.parent.is_dir():
            raise ValueError("--output parent directory must already exist")
    report = analyze_logs(paths)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if output is not None:
        try:
            with output.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
        except FileExistsError as error:
            raise ValueError("--output must be a new path") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
