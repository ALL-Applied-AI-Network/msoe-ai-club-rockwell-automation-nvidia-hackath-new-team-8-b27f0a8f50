"""Receipt-writing, no-actuation planner for pretrained AI grasp selection.

This module deliberately exposes planning only. It imports no robot client,
executor, hardware plugin, or physical-motion API.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import threading
from typing import Any

from team_plugins.grasping.ai_selection import (
    AISelectionDecision,
    canonical_json,
)
from team_plugins.grasping.multigrasp_pipeline import (
    _AI_SIMULATION_CAPABILITY,
    _create_ai_offline_screw_planner,
    _create_ai_simulation_pipeline,
)


def _fsync_directory(path: Path) -> None:
    # Windows does not expose directory handles through os.open. File contents
    # are still fsynced above; directory-entry fsync is a POSIX durability step.
    if os.name == "nt":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class DurableAIReceiptSink:
    """Append hash-chained selection receipts and fsync before returning."""

    def __init__(self, path: str | Path, *, factory_context: Any) -> None:
        raw_path = Path(path).expanduser()
        if raw_path.name in {"", ".", ".."}:
            raise ValueError("receipt path must name a file")
        parent_candidate = (
            raw_path.parent if raw_path.is_absolute() else Path.cwd() / raw_path.parent
        )
        parent_candidate.mkdir(parents=True, exist_ok=True)
        parent = parent_candidate.resolve(strict=True)
        self.path = parent / raw_path.name
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(parent)
        self.profile = {
            "profile_id": str(factory_context.profile_id),
            "profile_path": str(factory_context.profile_path),
            "profile_sha256": str(factory_context.profile_sha256),
        }
        self._lock = threading.Lock()
        self._sequence = 0
        self._previous_sha256: str | None = None
        self._poisoned = False

    def __call__(self, decision: AISelectionDecision) -> None:
        if not isinstance(decision, AISelectionDecision):
            raise TypeError("receipt sink requires AISelectionDecision")
        with self._lock:
            if self._poisoned:
                raise RuntimeError("receipt sink is poisoned after a write failure")
            record = decision.receipt(include_payloads=True)
            record.update(
                {
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "sequence": self._sequence,
                    "previous_record_sha256": self._previous_sha256,
                    "profile": self.profile,
                }
            )
            record_sha256 = hashlib.sha256(
                canonical_json(record).encode("utf-8")
            ).hexdigest()
            record["record_sha256"] = record_sha256
            line = (canonical_json(record) + "\n").encode("utf-8")
            flags = os.O_RDWR | os.O_APPEND
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(self.path, flags)
            original_size = os.fstat(descriptor).st_size
            try:
                view = memoryview(line)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("receipt write made no progress")
                    view = view[written:]
                os.fsync(descriptor)
            except BaseException:
                self._poisoned = True
                try:
                    os.ftruncate(descriptor, original_size)
                    os.fsync(descriptor)
                except OSError:
                    pass
                raise
            finally:
                os.close(descriptor)
            self._previous_sha256 = record_sha256
            self._sequence += 1


class AISelectedSimulationPlanner:
    """Planning-only facade; no method in this class can actuate a robot."""

    def __init__(self, pipeline, receipt_sink: DurableAIReceiptSink) -> None:
        self.pipeline = pipeline
        self.receipt_sink = receipt_sink

    @property
    def receipt_path(self) -> Path:
        return self.receipt_sink.path

    def decide_for_target(self, scene, target, context):
        return self.pipeline.decide_for_target(scene, target, context)


def build_ai_selected_simulation_planner(
    factory_context,
    provider,
    *,
    receipt_path: str | Path,
    mode: str = "shadow",
    minimum_confidence: float = 0.7,
    minimum_score_margin: float = 0.1,
    timeout_s: float = 20.0,
) -> AISelectedSimulationPlanner:
    """Build the only public AI-binding path, with mandatory durable receipts."""

    receipt_sink = DurableAIReceiptSink(receipt_path, factory_context=factory_context)
    pipeline = _create_ai_simulation_pipeline(
        factory_context,
        provider,
        simulation_capability=_AI_SIMULATION_CAPABILITY,
        mode=mode,
        minimum_confidence=minimum_confidence,
        minimum_score_margin=minimum_score_margin,
        timeout_s=timeout_s,
        decision_sink=receipt_sink,
    )
    return AISelectedSimulationPlanner(pipeline, receipt_sink)


class AISelectedOfflineScrewPlanner:
    """Offline screw facade; its decisions are never executable robot plans."""

    def __init__(self, planner, receipt_sink: DurableAIReceiptSink) -> None:
        self.planner = planner
        self.receipt_sink = receipt_sink

    @property
    def receipt_path(self) -> Path:
        return self.receipt_sink.path

    def decide(
        self,
        scene,
        target_track_id,
        *,
        pick_tip_z_m,
        approach_clearance_m,
        release_tip_pose,
    ):
        return self.planner.decide(
            scene,
            target_track_id,
            pick_tip_z_m=pick_tip_z_m,
            approach_clearance_m=approach_clearance_m,
            release_tip_pose=release_tip_pose,
        )


def build_ai_selected_offline_screw_planner(
    factory_context,
    provider,
    *,
    receipt_path: str | Path,
    mode: str = "shadow",
    minimum_confidence: float = 0.7,
    minimum_score_margin: float = 0.1,
    timeout_s: float = 20.0,
) -> AISelectedOfflineScrewPlanner:
    """Build a receipt-writing AI ranker for non-executable screw candidates."""

    receipt_sink = DurableAIReceiptSink(receipt_path, factory_context=factory_context)
    planner = _create_ai_offline_screw_planner(
        factory_context,
        provider,
        simulation_capability=_AI_SIMULATION_CAPABILITY,
        mode=mode,
        minimum_confidence=minimum_confidence,
        minimum_score_margin=minimum_score_margin,
        timeout_s=timeout_s,
        decision_sink=receipt_sink,
    )
    return AISelectedOfflineScrewPlanner(planner, receipt_sink)


__all__ = [
    "AISelectedOfflineScrewPlanner",
    "AISelectedSimulationPlanner",
    "DurableAIReceiptSink",
    "build_ai_selected_offline_screw_planner",
    "build_ai_selected_simulation_planner",
]
