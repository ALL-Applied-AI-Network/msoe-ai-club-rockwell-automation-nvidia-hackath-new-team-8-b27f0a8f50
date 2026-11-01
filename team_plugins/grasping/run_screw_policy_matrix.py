"""Run every isolated screw pilot case in a separate Isaac process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping

from team_plugins.grasping.isolated_sim_runner import (
    RUNNER_ID,
    _screw_retention_evidence,
    _screw_trainer_label,
)
from team_plugins.grasping.multigrasp_pipeline import SCREW_GENERATOR_ID
from team_plugins.grasping.prepare_screw_policy_matrix import (
    ROOT,
    TARGET_ITEM_ID,
    _source_state,
)


RUN_MATRIX_ID = "team8-isolated-screw-matrix-runner-v1"
EXPERIMENT_ID = "team8-isolated-screw-policy-pilot-v1"
RESOLVED_STATUSES = {"resolved_success", "resolved_failure"}
EXPECTED_CANDIDATES = {
    "screw-shaft-012mm-0": 12,
    "screw-shaft-016mm-1": 16,
    "screw-shaft-020mm-2": 20,
}
CASE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
RESERVED_CASE_IDS = {"logs", "_archived_incomplete"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _required_string(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_case_id(value: Any) -> str:
    case_id = _required_string(value, "case_id")
    if (
        CASE_ID_PATTERN.fullmatch(case_id) is None
        or case_id in {".", ".."}
        or case_id in RESERVED_CASE_IDS
    ):
        raise ValueError(f"unsafe case_id {case_id!r}")
    return case_id


def _safe_direct_child(root: Path, name: str) -> Path:
    root = root.expanduser().resolve()
    candidate = root / name
    resolved = candidate.resolve()
    if candidate.parent != root or resolved != candidate:
        raise ValueError(f"path escapes or aliases its root: {candidate}")
    return candidate


def _case_run_dir(runs_root: Path, case_id: Any) -> Path:
    return _safe_direct_child(runs_root, _validate_case_id(case_id))


def _case_log_path(logs_root: Path, case_id: Any) -> Path:
    return _safe_direct_child(
        logs_root,
        f"{_validate_case_id(case_id)}.log",
    )


def _validate_manifest(
    manifest: Any,
) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected experiment_id")
    commit = _required_string(manifest.get("source_commit"), "source_commit")
    if len(commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in commit.lower()
    ):
        raise ValueError("source_commit must be a hexadecimal Git object ID")
    commit = commit.lower()
    if manifest.get("relevant_source_tree_clean") is not True:
        raise ValueError("manifest was not prepared from a clean source tree")
    if manifest.get("relevant_source_dirty_entries") != []:
        raise ValueError("manifest dirty-source entries must be empty")
    if manifest.get("target_item_id") != TARGET_ITEM_ID:
        raise ValueError("manifest target item is not the isolated screw target")

    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest cases must be a non-empty list")
    if manifest.get("attempt_count") != len(cases) or len(cases) != 12:
        raise ValueError("pilot manifest must contain exactly 12 attempts")

    case_ids: set[str] = set()
    group_candidates: set[tuple[str, str]] = set()
    group_assignments: dict[str, tuple[str, str]] = {}
    family_assignments: dict[str, tuple[str, str]] = {}
    candidates_by_group: dict[str, set[str]] = {}
    split_by_family: dict[str, str] = {}
    validated_cases: list[dict[str, Any]] = []
    for raw_case in cases:
        if not isinstance(raw_case, dict):
            raise ValueError("every manifest case must be an object")
        case_id = _validate_case_id(raw_case.get("case_id"))
        family = _required_string(
            raw_case.get("layout_family_id"),
            "layout_family_id",
        )
        group = _required_string(raw_case.get("group_id"), "group_id")
        candidate = _required_string(
            raw_case.get("candidate_id"),
            "candidate_id",
        )
        split = _required_string(raw_case.get("split"), "split")
        if split not in {"train", "test"}:
            raise ValueError("case split must be train or test")
        if candidate not in EXPECTED_CANDIDATES:
            raise ValueError(f"unexpected pilot candidate {candidate!r}")
        if raw_case.get("pinch_from_head_mm") != EXPECTED_CANDIDATES[candidate]:
            raise ValueError("candidate ID and pinch offset disagree")
        if case_id in case_ids:
            raise ValueError("case_id values must be unique")
        case_ids.add(case_id)
        pair = (group, candidate)
        if pair in group_candidates:
            raise ValueError("candidate IDs must be unique within each group")
        group_candidates.add(pair)

        assignment = (family, split)
        if group in group_assignments and group_assignments[group] != assignment:
            raise ValueError("group crosses layout family or train/test split")
        group_assignments[group] = assignment
        family_assignment = (group, split)
        if (
            family in family_assignments
            and family_assignments[family] != family_assignment
        ):
            raise ValueError("layout family crosses group or train/test split")
        family_assignments[family] = family_assignment
        split_by_family[family] = split
        candidates_by_group.setdefault(group, set()).add(candidate)

        scenario_value = _required_string(
            raw_case.get("scenario_path"),
            "scenario_path",
        )
        scenario_path = Path(scenario_value).expanduser()
        if not scenario_path.is_absolute():
            raise ValueError("scenario_path must be absolute")
        scenario_path = scenario_path.resolve()
        scenario_sha = _required_string(
            raw_case.get("scenario_sha256"),
            "scenario_sha256",
        ).lower()
        if len(scenario_sha) != 64 or any(
            character not in "0123456789abcdef" for character in scenario_sha
        ):
            raise ValueError("scenario_sha256 must be hexadecimal")
        if not scenario_path.is_file() or _sha256(scenario_path) != scenario_sha:
            raise ValueError(f"scenario file or hash changed for {case_id}")
        validated_cases.append(raw_case)

    if manifest.get("layout_count") != len(group_assignments) or len(
        group_assignments
    ) != 4:
        raise ValueError("pilot manifest must contain exactly four layouts")
    expected_candidates = set(EXPECTED_CANDIDATES)
    if any(
        candidate_ids != expected_candidates
        for candidate_ids in candidates_by_group.values()
    ):
        raise ValueError("every layout must contain the complete candidate set")
    if list(split_by_family.values()).count("train") != 3 or list(
        split_by_family.values()
    ).count("test") != 1:
        raise ValueError("pilot split must contain three train and one test layout")
    if manifest.get("candidate_offsets_mm") != [12, 16, 20]:
        raise ValueError("unexpected pilot candidate offsets")
    if manifest.get("split_by_layout_family") != split_by_family:
        raise ValueError("split_by_layout_family does not match cases")
    if manifest.get("split_manifest_sha256") != _canonical_sha256(
        split_by_family
    ):
        raise ValueError("split manifest hash does not match cases")
    return commit, validated_cases


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _case_command(
    *,
    isaac_python: Path,
    case: Mapping[str, Any],
    run_dir: Path,
) -> list[str]:
    return [
        isaac_python.as_posix(),
        "-m",
        "team_plugins.grasping.isolated_sim_runner",
        "--scenario",
        str(case["scenario_path"]),
        "--run",
        str(run_dir),
        "--target-item",
        TARGET_ITEM_ID,
        "--layout-family-id",
        str(case["layout_family_id"]),
        "--group-id",
        str(case["group_id"]),
        "--candidate-id",
        str(case["candidate_id"]),
        "--split",
        str(case["split"]),
        "--physics-only",
    ]


def _existing_status(run_dir: Path) -> str | None:
    metadata_path = run_dir / "attempt_metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = _read_json(metadata_path)
    status = metadata.get("outcome_status")
    if status not in {"resolved_success", "resolved_failure", "unresolved"}:
        raise ValueError(f"invalid existing attempt status in {metadata_path}")
    return status


def _validate_prior_resolved_row(
    prior: Mapping[str, Any] | None,
    case_id: str,
) -> None:
    if prior is None:
        return
    if prior.get("return_code") != 0:
        raise ValueError(
            f"prior matrix row for {case_id} did not record a zero exit"
        )


def _validated_resolved_attempt(
    run_dir: Path,
    case: Mapping[str, Any],
    source_commit: str,
) -> str:
    if not run_dir.is_dir():
        raise ValueError("case output directory is missing")
    metadata_path = run_dir / "attempt_metadata.json"
    report_path = run_dir / "execution_report.json"
    if not metadata_path.is_file() or not report_path.is_file():
        raise ValueError("case output is incomplete")
    metadata = _read_json(metadata_path)
    report = _read_json(report_path)
    if not isinstance(metadata, dict) or not isinstance(report, dict):
        raise ValueError("case artifacts must be JSON objects")

    expected_identity = {
        "layout_family_id": case["layout_family_id"],
        "group_id": case["group_id"],
        "candidate_id": case["candidate_id"],
        "split": case["split"],
    }
    if metadata.get("identity") != expected_identity:
        raise ValueError("attempt identity does not match the manifest")
    if metadata.get("target_item_id") != TARGET_ITEM_ID:
        raise ValueError("attempt target does not match the manifest")
    if metadata.get("source_commit") != source_commit:
        raise ValueError("attempt source commit does not match the manifest")
    if metadata.get("relevant_source_tree_clean") is not True:
        raise ValueError("attempt used dirty relevant sources")
    if metadata.get("relevant_source_dirty_entries") != []:
        raise ValueError("attempt dirty-source entries are not empty")
    if metadata.get("runner_id") != RUNNER_ID:
        raise ValueError("attempt metadata uses an unexpected runner")
    for field in (
        "execution_exception",
        "augmentation_exception",
        "exception",
    ):
        if field not in metadata or metadata[field] is not None:
            raise ValueError(f"attempt metadata records or omits {field}")
    execution_trace = metadata.get("execution_trace")
    if not isinstance(execution_trace, dict):
        raise ValueError("attempt execution trace is missing")
    expected_scenario_sha = case["scenario_sha256"]
    if metadata.get("scenario_sha256") != expected_scenario_sha:
        raise ValueError("attempt scenario hash does not match the manifest")

    execution = report.get("isolated_candidate_execution")
    if not isinstance(execution, dict):
        raise ValueError("isolated execution provenance is missing")
    for name, expected in (
        ("layout_family_id", case["layout_family_id"]),
        ("group_id", case["group_id"]),
        ("candidate_id", case["candidate_id"]),
        ("split", case["split"]),
        ("target_item_id", TARGET_ITEM_ID),
        ("source_commit", source_commit),
        ("scenario_sha256", expected_scenario_sha),
    ):
        if execution.get(name) != expected:
            raise ValueError(f"execution field {name} does not match the manifest")
    if execution.get("relevant_source_tree_clean") is not True:
        raise ValueError("execution used dirty relevant sources")
    if execution.get("relevant_source_dirty_entries") != []:
        raise ValueError("execution dirty-source entries are not empty")
    expected_execution_fields = {
        "runner_id": RUNNER_ID,
        "candidate_generator_id": SCREW_GENERATOR_ID,
        "candidate_scene_id": case["group_id"],
        "decision_source": (
            "configured candidate transformed onto settled simulator target truth"
        ),
        "candidate_generation_source": "configured scenario target-only truth",
        "candidate_generation_capture_time_s": 0.0,
        "runtime_transform_source": "settled simulator target truth",
        "simulator_obstacle_truth_used_for_candidate_generation": False,
        "other_configured_objects_remained_as_physics_obstacles": True,
        "fresh_process_required_per_candidate": True,
        "label_scope": "target_only",
        "non_target_obstacle_final_state_evaluated": False,
        "collision_safe_label_eligible": False,
    }
    for name, expected in expected_execution_fields.items():
        if execution.get(name) != expected:
            raise ValueError(f"execution provenance field {name} is invalid")
    trace_bindings = {
        "executed_pick_tip_xyz_m": "pick_tip_xyz_m",
        "executed_pick_tip_yaw_rad": "pick_tip_yaw_rad",
        "commanded_spread_axis": "spread_axis",
        "configured_to_settled_position_disagreement_m": (
            "position_disagreement_m"
        ),
        "configured_to_settled_yaw_disagreement_rad": "yaw_disagreement_rad",
    }
    for report_name, trace_name in trace_bindings.items():
        if execution.get(report_name) != execution_trace.get(trace_name):
            raise ValueError(
                f"execution field {report_name} disagrees with command trace"
            )

    metadata_status = metadata.get("outcome_status")
    execution_status = execution.get("outcome_status")
    if metadata_status not in RESOLVED_STATUSES:
        raise ValueError("attempt metadata is not resolved")
    if execution_status != metadata_status:
        raise ValueError("metadata and execution outcome statuses disagree")
    retention_evidence = _screw_retention_evidence(
        report,
        TARGET_ITEM_ID,
        execution_trace,
    )
    if execution.get("screw_retention_evidence") != retention_evidence:
        raise ValueError("stored retention evidence does not match report truth")
    derived_label = _screw_trainer_label(
        report,
        TARGET_ITEM_ID,
        retention_evidence,
    )
    expected_label = metadata_status == "resolved_success"
    if derived_label is not expected_label:
        raise ValueError("resolved status and target-only label disagree")
    if execution.get("target_only_complete_grasp_label") is not derived_label:
        raise ValueError("stored target-only label does not match report truth")
    if execution.get("target_only_label_eligible") is not True:
        raise ValueError("target-only label is not eligible")
    return metadata_status


def _archive_attempt(
    *,
    runs_root: Path,
    run_dir: Path,
    log_path: Path,
    case_id: str,
    reason: str,
) -> Path:
    archive_root = _safe_direct_child(
        runs_root,
        "_archived_incomplete",
    )
    archive_root.mkdir(exist_ok=True)
    while True:
        archive_dir = _safe_direct_child(
            archive_root,
            f"{case_id}--{time.time_ns()}",
        )
        if not archive_dir.exists():
            break
    archive_dir.mkdir()
    if run_dir.exists():
        run_dir.replace(_safe_direct_child(archive_dir, "run_dir"))
    if log_path.exists():
        log_path.replace(_safe_direct_child(archive_dir, "runner.log"))
    _write_json(
        archive_dir / "archive_record.json",
        {
            "case_id": case_id,
            "reason": reason,
            "archived_time_unix_s": time.time(),
            "preserved_without_deletion": True,
        },
    )
    return archive_dir


def run_matrix(
    manifest_path: Path,
    runs_root: Path,
    *,
    isaac_python: Path,
    twin_runtime: Path,
    reference_assets: Path,
    resume: bool = False,
) -> Path:
    manifest_path = manifest_path.expanduser().resolve()
    runs_root = runs_root.expanduser().resolve()
    isaac_python = isaac_python.expanduser().resolve()
    twin_runtime = twin_runtime.expanduser().resolve()
    reference_assets = reference_assets.expanduser().resolve()
    if runs_root == ROOT or ROOT in runs_root.parents:
        raise ValueError("runs root must remain outside Git")
    for path, name in (
        (manifest_path, "manifest"),
        (isaac_python, "Isaac Python"),
        (twin_runtime, "twin runtime"),
        (reference_assets, "reference assets"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    manifest = _read_json(manifest_path)
    manifest_commit, cases = _validate_manifest(manifest)
    commit, source_clean, dirty_entries = _source_state(require_clean=True)
    if not source_clean or dirty_entries:
        raise RuntimeError("relevant source paths must be clean")
    if manifest_commit != commit:
        raise ValueError(
            "manifest commit does not match the checked-out source commit"
        )

    if runs_root.exists():
        if not resume:
            raise FileExistsError(f"runs root already exists: {runs_root}")
    else:
        runs_root.mkdir(parents=True)
    logs_root = _safe_direct_child(runs_root, "logs")
    logs_root.mkdir(exist_ok=True)
    report_path = _safe_direct_child(runs_root, "matrix_run_report.json")
    manifest_sha = _sha256(manifest_path)
    report: dict[str, Any]
    if report_path.is_file():
        report = _read_json(report_path)
        if not isinstance(report, dict):
            raise ValueError("existing matrix report must be a JSON object")
        if report.get("runner_id") != RUN_MATRIX_ID:
            raise ValueError("existing matrix report uses a different runner")
        if report.get("source_commit") != commit:
            raise ValueError("existing matrix report uses a different commit")
        if report.get("manifest_sha256") != manifest_sha:
            raise ValueError("existing matrix report uses a different manifest")
    else:
        report = {
            "schema_version": 1,
            "runner_id": RUN_MATRIX_ID,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "source_commit": commit,
            "started_time_unix_s": time.time(),
            "completed_time_unix_s": None,
            "attempts": [],
        }

    prior_by_case = {
        row["case_id"]: row
        for row in report.get("attempts", [])
        if isinstance(row, dict) and "case_id" in row
    }
    environment = os.environ.copy()
    current_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(ROOT), str(twin_runtime), current_pythonpath)
        if value
    )
    environment["ROCKWELL_TWIN_RUNTIME"] = str(twin_runtime)
    environment["ROCKWELL_REFERENCE_ASSETS"] = str(reference_assets)
    environment["PYTHONUNBUFFERED"] = "1"

    attempts = []
    for case in cases:
        case_id = _validate_case_id(case["case_id"])
        run_dir = _case_run_dir(runs_root, case_id)
        log_path = _case_log_path(logs_root, case_id)
        archived_previous: Path | None = None
        if run_dir.exists() or log_path.exists():
            try:
                existing = _validated_resolved_attempt(
                    run_dir,
                    case,
                    commit,
                )
                _validate_prior_resolved_row(
                    prior_by_case.get(case_id),
                    case_id,
                )
            except Exception as exc:
                if not resume:
                    raise RuntimeError(
                        f"existing case output is not reusable: {run_dir}: {exc}"
                    ) from exc
                archived_previous = _archive_attempt(
                    runs_root=runs_root,
                    run_dir=run_dir,
                    log_path=log_path,
                    case_id=case_id,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            else:
                if not resume:
                    raise FileExistsError(
                        f"case output already exists without --resume: {run_dir}"
                    )
                prior = prior_by_case.get(case_id)
                row = {
                    "case_id": case_id,
                    "layout_family_id": case["layout_family_id"],
                    "group_id": case["group_id"],
                    "candidate_id": case["candidate_id"],
                    "split": case["split"],
                    "status": existing,
                    "return_code": 0,
                    "prior_recorded_return_code": (
                        prior.get("return_code")
                        if isinstance(prior, dict)
                        else None
                    ),
                    "resumed_existing": True,
                    "run_dir": str(run_dir),
                    "log_path": str(log_path),
                }
                if log_path.is_file():
                    row["log_sha256"] = _sha256(log_path)
                attempts.append(row)
                report["attempts"] = attempts
                _write_json(report_path, report)
                continue

        command = _case_command(
            isaac_python=isaac_python,
            case=case,
            run_dir=run_dir,
        )
        started = time.time()
        with log_path.open("wb") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        artifact_validation_error = None
        try:
            if completed.returncode != 0:
                raise ValueError(
                    f"isolated runner exited with code {completed.returncode}"
                )
            status = _validated_resolved_attempt(run_dir, case, commit)
        except Exception as exc:
            status = "unresolved"
            artifact_validation_error = f"{type(exc).__name__}: {exc}"
        row = {
            "case_id": case_id,
            "layout_family_id": case["layout_family_id"],
            "group_id": case["group_id"],
            "candidate_id": case["candidate_id"],
            "split": case["split"],
            "status": status,
            "return_code": completed.returncode,
            "started_time_unix_s": started,
            "elapsed_s": time.time() - started,
            "run_dir": str(run_dir),
            "log_path": str(log_path),
            "log_sha256": _sha256(log_path),
            "resumed_existing": False,
        }
        if archived_previous is not None:
            row["archived_previous_attempt"] = str(archived_previous)
        if artifact_validation_error is not None:
            row["artifact_validation_error"] = artifact_validation_error
        attempts.append(row)
        report["attempts"] = attempts
        _write_json(report_path, report)

    report["attempts"] = attempts
    report["completed_time_unix_s"] = time.time()
    report["status_counts"] = {
        status: sum(row["status"] == status for row in attempts)
        for status in ("resolved_success", "resolved_failure", "unresolved")
    }
    report["all_attempts_resolved"] = (
        report["status_counts"]["unresolved"] == 0
        and len(attempts) == len(cases)
        and all(row.get("return_code") == 0 for row in attempts)
    )
    _write_json(report_path, report)
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("runs_root", type=Path)
    parser.add_argument(
        "--isaac-python",
        type=Path,
        default=Path("/isaac-sim/python.sh"),
    )
    parser.add_argument(
        "--twin-runtime",
        type=Path,
        default=(
            Path(os.environ["ROCKWELL_TWIN_RUNTIME"])
            if "ROCKWELL_TWIN_RUNTIME" in os.environ
            else None
        ),
        required="ROCKWELL_TWIN_RUNTIME" not in os.environ,
    )
    parser.add_argument(
        "--reference-assets",
        type=Path,
        default=(
            Path(os.environ["ROCKWELL_REFERENCE_ASSETS"])
            if "ROCKWELL_REFERENCE_ASSETS" in os.environ
            else None
        ),
        required="ROCKWELL_REFERENCE_ASSETS" not in os.environ,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    path = run_matrix(
        args.manifest,
        args.runs_root,
        isaac_python=args.isaac_python,
        twin_runtime=args.twin_runtime,
        reference_assets=args.reference_assets,
        resume=args.resume,
    )
    print(path)
    report = _read_json(path)
    return 0 if report["all_attempts_resolved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
