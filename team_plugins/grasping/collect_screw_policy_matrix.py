"""Collect isolated screw-policy reports into strict pilot rows and sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from demos.scripted_cube_screw_bucket_sort.scenario import load_scenario_config
from sorting_core import DetectedObject, Pose3D, SceneObservation, TipPoseBase

from team_plugins.grasping.isolated_sim_runner import (
    RUNNER_ID,
    _screw_retention_evidence,
    _screw_trainer_label,
)
from team_plugins.grasping.multigrasp_pipeline import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_ID,
    SCREW_CLASS_ID,
    SCREW_GENERATOR_ID,
    build_offline_screw_planner,
)
from team_plugins.grasping.prepare_screw_policy_matrix import (
    ROOT,
    TARGET_ITEM_ID,
    _configured_observation,
    _source_state,
)
from team_plugins.grasping.run_screw_policy_matrix import (
    RESOLVED_STATUSES,
    RUN_MATRIX_ID,
    _case_run_dir,
    _validate_manifest as _validate_matrix_manifest,
)


COLLECTOR_ID = "team8-isolated-screw-collector-v1"


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_strict_pairs,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nonempty_string(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _hex_sha256(value: Any, name: str) -> str:
    result = _nonempty_string(value, name).lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must be a hexadecimal SHA-256")
    return result


def _git_oid(value: Any, name: str) -> str:
    result = _nonempty_string(value, name).lower()
    if len(result) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must be a hexadecimal Git object ID")
    return result


def _rotate_vector(
    quaternion_xyzw: tuple[float, float, float, float],
    vector: tuple[float, float, float],
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


def _settled_observation(
    config: Any,
    group_id: str,
    source_truth: Mapping[str, Any],
) -> SceneObservation:
    configured = _configured_observation(config, group_id)
    origin = tuple(float(value) for value in source_truth["origin_world_m"])
    raw_wxyz = tuple(
        float(value) for value in source_truth["orientation_wxyz"]
    )
    if len(origin) != 3 or len(raw_wxyz) != 4:
        raise ValueError("settled target truth has invalid pose dimensions")
    quaternion_xyzw = (
        raw_wxyz[1],
        raw_wxyz[2],
        raw_wxyz[3],
        raw_wxyz[0],
    )
    axis = _rotate_vector(quaternion_xyzw, (0.0, 0.0, -1.0))
    recorded_axis = tuple(
        float(value) for value in source_truth["axis_world"]
    )
    if len(recorded_axis) != 3:
        raise ValueError("settled target axis must contain three values")
    axis_error = math.sqrt(
        sum((first - second) ** 2 for first, second in zip(axis, recorded_axis))
    )
    if axis_error > 1e-6:
        raise ValueError("settled target quaternion and axis disagree")

    replacement = DetectedObject(
        track_id=TARGET_ITEM_ID,
        class_id=SCREW_CLASS_ID,
        sort_label="red",
        capture_time_s=0.0,
        pose=Pose3D(
            position_base_m=origin,
            orientation_xyzw=quaternion_xyzw,
        ),
        dimensions_m=(0.012, 0.01199627, 0.030),
        confidence=1.0,
        pose_quality=1.0,
        visibility=1.0,
        source_camera_ids=("settled-isaac-truth",),
    )
    objects = tuple(
        replacement if item.track_id == TARGET_ITEM_ID else item
        for item in configured.objects
    )
    if sum(item.track_id == TARGET_ITEM_ID for item in objects) != 1:
        raise ValueError("configured scene must contain the target exactly once")
    return SceneObservation(
        scene_id=group_id,
        capture_time_s=0.0,
        objects=objects,
        provider_name="settled-target-plus-configured-obstacles",
        provider_version="1",
        calibration_ids=("simulation-only",),
    )


def _settled_candidate_evidence(
    config: Any,
    group_id: str,
    candidate_id: str,
    source_truth: Mapping[str, Any],
) -> dict[str, Any]:
    observation = _settled_observation(config, group_id, source_truth)
    cube_config = config.cube_config
    destination = cube_config.task.bin_centres_base_m["red"]
    planner = build_offline_screw_planner(cube_config)
    try:
        decision = planner.decide(
            observation,
            TARGET_ITEM_ID,
            pick_tip_z_m=cube_config.pick_tip_base_z_m,
            approach_clearance_m=cube_config.task.approach_clearance_m,
            release_tip_pose=TipPoseBase(
                (
                    destination[0],
                    destination[1],
                    cube_config.bin_release_tip_base_z_m,
                ),
                0.0,
            ),
        )
    except Exception as exc:
        return {
            "accepted": False,
            "rejection_reasons": [
                f"settled-scene planning failed: {type(exc).__name__}: {exc}"
            ],
            "metrics": None,
            "features": None,
        }

    for evaluation in decision.evaluations:
        if evaluation.candidate.candidate_id == candidate_id:
            if evaluation.features.schema_id != FEATURE_SCHEMA_ID:
                raise ValueError("settled features use an unexpected schema")
            if tuple(evaluation.features.values) != FEATURE_NAMES:
                raise ValueError("settled feature order is not canonical")
            return {
                "accepted": True,
                "rejection_reasons": [],
                "metrics": dict(evaluation.feasibility.metrics),
                "features": {
                    name: evaluation.features.values[name]
                    for name in FEATURE_NAMES
                },
            }
    for rejection in decision.rejections:
        if rejection.candidate.candidate_id == candidate_id:
            return {
                "accepted": False,
                "rejection_reasons": list(
                    rejection.feasibility.rejection_reasons
                ),
                "metrics": dict(rejection.feasibility.metrics),
                "features": None,
            }
    raise ValueError(f"settled planner omitted candidate {candidate_id!r}")


def _validate_manifest(manifest: Any) -> tuple[str, list[dict[str, Any]]]:
    return _validate_matrix_manifest(manifest)


def _validate_matrix_report(
    runs_root: Path,
    *,
    manifest_path: Path,
    source_commit: str,
    cases: list[dict[str, Any]],
) -> tuple[dict[str, str], Path, bool]:
    report_path = runs_root / "matrix_run_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"matrix run report is missing: {report_path}")
    report = _read_json(report_path)
    if not isinstance(report, dict):
        raise ValueError("matrix run report must be a JSON object")
    if report.get("runner_id") != RUN_MATRIX_ID:
        raise ValueError("matrix run report uses an unexpected runner")
    if report.get("source_commit") != source_commit:
        raise ValueError("matrix run report source commit does not match")
    if report.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("matrix run report manifest hash does not match")
    attempts = report.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != len(cases):
        raise ValueError("matrix run report attempt set is incomplete")
    by_case: dict[str, dict[str, Any]] = {}
    for row in attempts:
        if not isinstance(row, dict):
            raise ValueError("matrix run report attempts must be objects")
        case_id = _nonempty_string(row.get("case_id"), "matrix case_id")
        if case_id in by_case:
            raise ValueError("matrix run report contains duplicate case IDs")
        by_case[case_id] = row
    statuses: dict[str, str] = {}
    for case in cases:
        case_id = case["case_id"]
        row = by_case.get(case_id)
        if row is None:
            raise ValueError(f"matrix run report omitted {case_id}")
        for name in (
            "layout_family_id",
            "group_id",
            "candidate_id",
            "split",
        ):
            if row.get(name) != case[name]:
                raise ValueError(
                    f"matrix run report field {name} disagrees for {case_id}"
                )
        status = row.get("status")
        if status not in RESOLVED_STATUSES | {"unresolved"}:
            raise ValueError(f"matrix run report status is invalid for {case_id}")
        if status in RESOLVED_STATUSES and row.get("return_code") != 0:
            raise ValueError(
                f"resolved matrix attempt has nonzero return code for {case_id}"
            )
        statuses[case_id] = status
    extra = sorted(set(by_case) - set(statuses))
    if extra:
        raise ValueError(f"matrix run report has unknown case IDs: {extra}")
    all_resolved = all(
        status in RESOLVED_STATUSES for status in statuses.values()
    )
    if report.get("all_attempts_resolved") is not all_resolved:
        raise ValueError("matrix all_attempts_resolved flag is inconsistent")
    return statuses, report_path, all_resolved

def _attempt(
    case: Mapping[str, Any],
    runs_root: Path,
    source_commit: str,
    matrix_status: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    case_id = _nonempty_string(case.get("case_id"), "case_id")
    group_id = _nonempty_string(case.get("group_id"), "group_id")
    candidate_id = _nonempty_string(case.get("candidate_id"), "candidate_id")
    split = _nonempty_string(case.get("split"), "split")
    run_dir = _case_run_dir(runs_root, case_id)
    report_path = run_dir / "execution_report.json"
    metadata_path = run_dir / "attempt_metadata.json"
    sidecar: dict[str, Any] = {
        "case_id": case_id,
        "layout_family_id": case["layout_family_id"],
        "group_id": group_id,
        "candidate_id": candidate_id,
        "split": split,
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "attempt_metadata_path": str(metadata_path),
        "matrix_status": matrix_status,
        "status": "unresolved",
        "label_scope": "target_only",
        "target_only_complete_grasp_label": None,
        "collision_safe_label_eligible": False,
        "settled_preflight": None,
        "exclusion_reasons": [],
    }
    if matrix_status not in RESOLVED_STATUSES:
        sidecar["exclusion_reasons"].append("matrix attempt is unresolved")
        return sidecar, None
    if not report_path.is_file() or not metadata_path.is_file():
        raise ValueError(
            f"{case_id}: resolved matrix attempt is missing run artifacts"
        )

    report = _read_json(report_path)
    metadata = _read_json(metadata_path)
    if not isinstance(report, dict) or not isinstance(metadata, dict):
        raise ValueError(f"{case_id}: run artifacts must be JSON objects")
    identity = metadata.get("identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{case_id}: attempt identity is missing")
    expected_identity = {
        "layout_family_id": case["layout_family_id"],
        "group_id": group_id,
        "candidate_id": candidate_id,
        "split": split,
    }
    if identity != expected_identity:
        raise ValueError(f"{case_id}: attempt identity does not match manifest")
    if metadata.get("target_item_id") != TARGET_ITEM_ID:
        raise ValueError(f"{case_id}: attempt target does not match")
    if metadata.get("source_commit") != source_commit:
        raise ValueError(f"{case_id}: source commit does not match manifest")
    if metadata.get("relevant_source_tree_clean") is not True:
        raise ValueError(f"{case_id}: attempt used dirty relevant sources")
    if metadata.get("relevant_source_dirty_entries") != []:
        raise ValueError(f"{case_id}: dirty source entries are not empty")
    if metadata.get("runner_id") != RUNNER_ID:
        raise ValueError(f"{case_id}: metadata uses an unexpected runner")
    for field in (
        "execution_exception",
        "augmentation_exception",
        "exception",
    ):
        if field not in metadata or metadata[field] is not None:
            raise ValueError(
                f"{case_id}: metadata records or omits {field}"
            )
    execution_trace = metadata.get("execution_trace")
    if not isinstance(execution_trace, dict):
        raise ValueError(f"{case_id}: execution trace is missing")
    metadata_status = metadata.get("outcome_status")
    if metadata_status not in RESOLVED_STATUSES:
        raise ValueError(f"{case_id}: attempt metadata is unresolved")
    if metadata_status != matrix_status:
        raise ValueError(f"{case_id}: matrix and metadata statuses disagree")

    scenario_path = Path(
        _nonempty_string(case.get("scenario_path"), "scenario_path")
    )
    expected_scenario_sha = _hex_sha256(
        case.get("scenario_sha256"), "scenario_sha256"
    )
    if not scenario_path.is_file() or _sha256(scenario_path) != expected_scenario_sha:
        raise ValueError(f"{case_id}: scenario file or hash changed")
    if metadata.get("scenario_sha256") != expected_scenario_sha:
        raise ValueError(f"{case_id}: attempt scenario hash does not match")

    execution = report.get("isolated_candidate_execution")
    if not isinstance(execution, dict):
        raise ValueError(f"{case_id}: isolated execution provenance is missing")
    for name, expected_value in (
        ("layout_family_id", case["layout_family_id"]),
        ("group_id", group_id),
        ("candidate_id", candidate_id),
        ("split", split),
        ("target_item_id", TARGET_ITEM_ID),
        ("source_commit", source_commit),
        ("scenario_sha256", expected_scenario_sha),
    ):
        if execution.get(name) != expected_value:
            raise ValueError(f"{case_id}: report field {name} does not match")
    if (
        execution.get("simulation_only") is not True
        or execution.get("camera_perception") is not False
        or execution.get("hardware_integration") is not False
        or execution.get("simulator_truth_affects_runtime_control") is not True
        or execution.get("label_scope") != "target_only"
        or execution.get("collision_safe_label_eligible") is not False
        or execution.get("non_target_obstacle_final_state_evaluated") is not False
    ):
        raise ValueError(f"{case_id}: execution scope flags are invalid")
    if execution.get("relevant_source_tree_clean") is not True:
        raise ValueError(f"{case_id}: report used dirty relevant sources")
    if execution.get("relevant_source_dirty_entries") != []:
        raise ValueError(f"{case_id}: report dirty entries are not empty")
    expected_execution_fields = {
        "runner_id": RUNNER_ID,
        "candidate_generator_id": SCREW_GENERATOR_ID,
        "candidate_scene_id": group_id,
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
        "collision_safe_label_eligible": False,
        "non_target_obstacle_final_state_evaluated": False,
    }
    for name, expected in expected_execution_fields.items():
        if execution.get(name) != expected:
            raise ValueError(
                f"{case_id}: execution provenance field {name} is invalid"
            )
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
                f"{case_id}: {report_name} disagrees with command trace"
            )
    execution_status = execution.get("outcome_status")
    if execution_status != metadata_status:
        raise ValueError(f"{case_id}: metadata and report statuses disagree")

    events = report.get("events")
    if (
        not isinstance(events, list)
        or len(events) != 1
        or not isinstance(events[0], dict)
        or events[0].get("item_id") != TARGET_ITEM_ID
    ):
        raise ValueError(f"{case_id}: expected exactly one target event")
    if (
        report.get("expected_items") != 1
        or report.get("processed_items") != 1
        or report.get("expected_item_ids") != [TARGET_ITEM_ID]
    ):
        raise ValueError(f"{case_id}: target-only completion denominator is invalid")
    event = events[0]
    revalidation = event.get("pre_descent_revalidation")
    if not isinstance(revalidation, dict) or revalidation.get("passed") is not True:
        raise ValueError(f"{case_id}: resolved label lacks target revalidation")

    stored_retention_evidence = execution.get("screw_retention_evidence")
    execution_trace = metadata.get("execution_trace")
    if not isinstance(execution_trace, dict):
        raise ValueError(f"{case_id}: execution trace is missing")
    retention_evidence = _screw_retention_evidence(
        report,
        TARGET_ITEM_ID,
        execution_trace,
    )
    if stored_retention_evidence != retention_evidence:
        raise ValueError(
            f"{case_id}: stored retention evidence does not match report truth"
        )
    derived_label = _screw_trainer_label(
        report,
        TARGET_ITEM_ID,
        retention_evidence,
    )
    stored_label = execution.get("target_only_complete_grasp_label")
    if stored_label is not derived_label:
        raise ValueError(f"{case_id}: stored label does not match report evidence")
    expected_label = metadata_status == "resolved_success"
    if stored_label is not expected_label:
        raise ValueError(f"{case_id}: resolved status and target label disagree")
    if execution.get("target_only_label_eligible") is not True:
        raise ValueError(f"{case_id}: target-only label is not eligible")
    sidecar["target_only_complete_grasp_label"] = derived_label
    sidecar["screw_retention_evidence"] = retention_evidence
    sidecar["status"] = metadata_status

    config = load_scenario_config(scenario_path)
    source_truth = event.get("source_truth")
    if not isinstance(source_truth, dict):
        raise ValueError(f"{case_id}: settled source truth is missing")
    evidence = _settled_candidate_evidence(
        config,
        group_id,
        candidate_id,
        source_truth,
    )
    sidecar["settled_preflight"] = evidence
    if not evidence["accepted"]:
        sidecar["exclusion_reasons"].append(
            "candidate rejected after settled-target feature recomputation"
        )
        return sidecar, None

    row = {
        "episode_id": case_id,
        "group_id": group_id,
        "candidate_id": candidate_id,
        "split": split,
        "features": evidence["features"],
        "success": derived_label,
    }
    sidecar["status"] = (
        "eligible_success" if derived_label else "eligible_failure"
    )
    return sidecar, row

def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def collect(manifest_path: Path, runs_root: Path, output: Path) -> Path:
    manifest_path = manifest_path.expanduser().resolve()
    runs_root = runs_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("collection output must remain outside Git")
    if output.exists():
        raise FileExistsError(f"collection output already exists: {output}")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"experiment manifest does not exist: {manifest_path}"
        )
    if not runs_root.is_dir():
        raise FileNotFoundError(f"runs root does not exist: {runs_root}")

    manifest = _read_json(manifest_path)
    source_commit, cases = _validate_manifest(manifest)
    current_commit, source_clean, dirty_entries = _source_state(
        require_clean=True
    )
    if not source_clean or dirty_entries:
        raise RuntimeError("collector relevant source paths must be clean")
    if current_commit != source_commit:
        raise ValueError(
            "manifest commit does not match collector source commit"
        )
    matrix_statuses, matrix_report_path, all_matrix_attempts_resolved = (
        _validate_matrix_report(
            runs_root,
            manifest_path=manifest_path,
            source_commit=source_commit,
            cases=cases,
        )
    )

    sidecars: list[dict[str, Any]] = []
    provisional_rows: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        sidecar, row = _attempt(
            case,
            runs_root,
            source_commit,
            matrix_statuses[case["case_id"]],
        )
        sidecars.append(sidecar)
        if row is not None:
            provisional_rows.setdefault(row["group_id"], []).append(row)

    expected_by_group: dict[str, set[str]] = {}
    for case in cases:
        expected_by_group.setdefault(case["group_id"], set()).add(
            case["candidate_id"]
        )
    eligible_rows: list[dict[str, Any]] = []
    incomplete_groups: dict[str, dict[str, Any]] = {}
    for group_id, expected_ids in expected_by_group.items():
        rows = provisional_rows.get(group_id, [])
        observed_ids = {row["candidate_id"] for row in rows}
        if observed_ids != expected_ids:
            incomplete_groups[group_id] = {
                "expected_candidate_ids": sorted(expected_ids),
                "eligible_candidate_ids": sorted(observed_ids),
                "missing_candidate_ids": sorted(expected_ids - observed_ids),
            }
            continue
        eligible_rows.extend(rows)
    eligible_rows.sort(key=lambda row: (row["group_id"], row["candidate_id"]))
    sidecars.sort(key=lambda row: (row["group_id"], row["candidate_id"]))

    labels_by_split = {
        split: [
            row["success"] for row in eligible_rows if row["split"] == split
        ]
        for split in ("train", "test")
    }
    offsets_by_candidate = {
        case["candidate_id"]: int(case["pinch_from_head_mm"])
        for case in cases
    }
    axial_patterns: dict[str, list[dict[str, Any]]] = {}
    nonmonotone_groups: list[str] = []
    for group_id in sorted(expected_by_group):
        rows = [
            row for row in eligible_rows if row["group_id"] == group_id
        ]
        if len(rows) != len(expected_by_group[group_id]):
            continue
        ordered = sorted(
            rows,
            key=lambda row: offsets_by_candidate[row["candidate_id"]],
        )
        labels = [bool(row["success"]) for row in ordered]
        axial_patterns[group_id] = [
            {
                "pinch_from_head_mm": offsets_by_candidate[row["candidate_id"]],
                "success": bool(row["success"]),
            }
            for row in ordered
        ]
        if labels in ([False, True, False], [True, False, True]):
            nonmonotone_groups.append(group_id)

    train_has_both_classes = set(labels_by_split["train"]) == {False, True}
    test_nonempty = bool(labels_by_split["test"])
    structurally_valid = bool(
        all_matrix_attempts_resolved
        and not incomplete_groups
        and train_has_both_classes
        and test_nonempty
    )

    output.mkdir(parents=True)
    dataset_path = output / "dataset.jsonl"
    provenance_path = output / "attempt_provenance.jsonl"
    _write_jsonl(dataset_path, eligible_rows)
    _write_jsonl(provenance_path, sidecars)
    report = {
        "schema_version": 1,
        "collector_id": COLLECTOR_ID,
        "experiment_manifest_path": str(manifest_path),
        "experiment_manifest_sha256": _sha256(manifest_path),
        "matrix_run_report_path": str(matrix_report_path),
        "matrix_run_report_sha256": _sha256(matrix_report_path),
        "source_commit": source_commit,
        "collector_source_commit": current_commit,
        "collector_relevant_source_tree_clean": source_clean,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_order": list(FEATURE_NAMES),
        "pilot_only": True,
        "label_scope": "target_only",
        "collision_safe_label_eligible": False,
        "non_target_obstacle_final_state_evaluated": False,
        "feature_observation_scope": (
            "settled target truth plus configured obstacle poses"
        ),
        "feature_observation_limit": (
            "neighbor settling is not measured; rows are diagnostic pilot data"
        ),
        "camera_perception": False,
        "simulation_only": True,
        "cases": len(cases),
        "all_matrix_attempts_resolved": all_matrix_attempts_resolved,
        "resolved_labels": sum(
            row["target_only_complete_grasp_label"] is not None
            for row in sidecars
        ),
        "eligible_attempts_before_group_filter": sum(
            len(rows) for rows in provisional_rows.values()
        ),
        "dataset_rows": len(eligible_rows),
        "eligible_groups": len(
            {row["group_id"] for row in eligible_rows}
        ),
        "incomplete_groups": incomplete_groups,
        "train_successes": sum(labels_by_split["train"]),
        "train_failures": len(labels_by_split["train"])
        - sum(labels_by_split["train"]),
        "test_successes": sum(labels_by_split["test"]),
        "test_failures": len(labels_by_split["test"])
        - sum(labels_by_split["test"]),
        "raw_axial_offset_patterns": axial_patterns,
        "v1_raw_axial_offset_nonmonotone_groups": nonmonotone_groups,
        "v1_raw_axial_offset_monotone_compatible": not nonmonotone_groups,
        "training_input_structurally_valid": structurally_valid,
        "trainer_ready": False,
        "trainer_ready_reason": (
            "this 12-attempt target-only pilot is never a promotion-ready "
            "training dataset"
        ),
        "trainer_ready_requirements": {
            "all_matrix_attempts_resolved": all_matrix_attempts_resolved,
            "all_groups_complete": not incomplete_groups,
            "train_contains_both_classes": train_has_both_classes,
            "test_nonempty": test_nonempty,
            "all_obstacle_poses_settled": False,
            "collision_safe_labels": False,
            "pilot_promotion_allowed": False,
        },
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "attempt_provenance_path": str(provenance_path),
        "attempt_provenance_sha256": _sha256(provenance_path),
    }
    report_path = output / "collection_report.json"
    _write_json(report_path, report)
    return report_path

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(collect(args.manifest, args.runs_root, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
