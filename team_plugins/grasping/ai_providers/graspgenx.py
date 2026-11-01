"""Map GraspGenX 6-DoF proposals onto feasible catalog candidates.

GraspGenX output is advisory: proposals are transformed into robot_base and
matched to existing candidates. No model pose is ever returned for execution.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from team_plugins.grasping.ai_selection import (
    RESPONSE_SCHEMA_VERSION,
    canonical_json,
    strict_json_object,
)

PROPOSAL_SCHEMA_VERSION = "team8-graspgenx-proposals-v1"
MATCHER_VERSION = "team8-graspgenx-catalog-matcher-v1"
_ENVELOPE_KEYS = {
    "schema_version",
    "request_sha256",
    "scene_id",
    "model_version",
    "input_frame_id",
    "calibration_id",
    "fingertip_depth_m",
    "transform_robot_base_from_input",
    "proposals",
}
_PROPOSAL_KEYS = {
    "proposal_id",
    "transform_input_from_gripper",
    "confidence",
    "branch_tag",
}


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _unit_interval(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number")
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _se3(value: Any, name: str) -> tuple[tuple[float, ...], ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 4
    ):
        raise ValueError(f"{name} must be a 4x4 matrix")
    rows = []
    for row in value:
        if (
            isinstance(row, (str, bytes))
            or not isinstance(row, Sequence)
            or len(row) != 4
        ):
            raise ValueError(f"{name} must be a 4x4 matrix")
        rows.append(tuple(_finite(item, name) for item in row))
    matrix = tuple(rows)
    for actual, expected in zip(matrix[3], (0.0, 0.0, 0.0, 1.0)):
        if abs(actual - expected) > 1e-6:
            raise ValueError(f"{name} must have a homogeneous final row")
    rotation = tuple(
        tuple(matrix[row][column] for row in range(3)) for column in range(3)
    )
    for index, axis in enumerate(rotation):
        norm = math.sqrt(sum(item * item for item in axis))
        if abs(norm - 1.0) > 1e-4:
            raise ValueError(f"{name} rotation axis {index} is not unit")
    for first in range(3):
        for second in range(first + 1, 3):
            dot = sum(rotation[first][i] * rotation[second][i] for i in range(3))
            if abs(dot) > 1e-4:
                raise ValueError(f"{name} rotation is not orthogonal")
    r = matrix
    determinant = (
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    )
    if abs(determinant - 1.0) > 1e-4:
        raise ValueError(f"{name} rotation must be right-handed")
    return matrix


def _compose(
    outer: tuple[tuple[float, ...], ...],
    inner: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    result = []
    for row in range(4):
        result.append(
            tuple(
                sum(outer[row][k] * inner[k][column] for k in range(4))
                for column in range(4)
            )
        )
    return tuple(result)


def _parallel_angle_distance(first: float, second: float) -> float:
    def wrapped(value: float) -> float:
        return (value + math.pi) % (2.0 * math.pi) - math.pi

    return min(
        abs(wrapped(first - second)),
        abs(wrapped(first - second + math.pi)),
        abs(wrapped(first - second - math.pi)),
    )


@runtime_checkable
class GraspGenXProposalSource(Protocol):
    """Isolated inference service already bound to depth, mask, and calibration."""

    def infer(
        self, request_json: str, *, timeout_s: float
    ) -> str | Mapping[str, Any]: ...


class GraspGenXCatalogProvider:
    """Convert pretrained proposals into a ranking of feasible candidates."""

    provider_id = "graspgenx-catalog-provider-v1"
    prompt_version = MATCHER_VERSION

    def __init__(
        self,
        source: GraspGenXProposalSource,
        *,
        model_version: str,
        input_frame_id: str,
        calibration_id: str,
        fingertip_depth_m: float,
        maximum_position_error_m: float = 0.005,
        maximum_yaw_error_rad: float = math.radians(10.0),
        maximum_approach_tilt_rad: float = math.radians(15.0),
        expected_approach_axis_robot_base: Sequence[float] = (0.0, 0.0, -1.0),
    ) -> None:
        self.source = source
        self.model_version = _nonempty(model_version, "model_version")
        self.input_frame_id = _nonempty(input_frame_id, "input_frame_id")
        self.calibration_id = _nonempty(calibration_id, "calibration_id")
        self.fingertip_depth_m = _finite(fingertip_depth_m, "fingertip_depth_m")
        if self.fingertip_depth_m <= 0.0:
            raise ValueError("fingertip_depth_m must be positive")
        self.maximum_position_error_m = _finite(
            maximum_position_error_m, "maximum_position_error_m"
        )
        self.maximum_yaw_error_rad = _finite(
            maximum_yaw_error_rad, "maximum_yaw_error_rad"
        )
        self.maximum_approach_tilt_rad = _finite(
            maximum_approach_tilt_rad, "maximum_approach_tilt_rad"
        )
        if self.maximum_position_error_m <= 0.0:
            raise ValueError("maximum_position_error_m must be positive")
        if not 0.0 < self.maximum_yaw_error_rad <= math.pi / 2.0:
            raise ValueError("maximum_yaw_error_rad must be in (0, pi/2]")
        if not 0.0 < self.maximum_approach_tilt_rad <= math.pi / 2.0:
            raise ValueError("maximum_approach_tilt_rad must be in (0, pi/2]")
        if (
            isinstance(expected_approach_axis_robot_base, (str, bytes))
            or len(expected_approach_axis_robot_base) != 3
        ):
            raise ValueError("expected approach axis must have three values")
        axis = tuple(
            _finite(item, "expected_approach_axis_robot_base")
            for item in expected_approach_axis_robot_base
        )
        norm = math.sqrt(sum(item * item for item in axis))
        if norm <= 0.0:
            raise ValueError("expected approach axis must be nonzero")
        self.expected_approach_axis = tuple(item / norm for item in axis)

    def _load_envelope(
        self, raw: str | Mapping[str, Any]
    ) -> tuple[dict[str, Any], str]:
        if isinstance(raw, str):
            envelope = strict_json_object(raw)
            envelope_json = canonical_json(envelope)
        elif isinstance(raw, Mapping):
            envelope_json = canonical_json(dict(raw))
            envelope = strict_json_object(envelope_json)
        else:
            raise ValueError("GraspGenX source must return JSON data")
        if set(envelope) != _ENVELOPE_KEYS:
            raise ValueError("GraspGenX envelope has unexpected fields")
        return envelope, envelope_json

    def select(self, request_json: str, *, timeout_s: float) -> dict[str, Any]:
        request_payload = strict_json_object(request_json)
        scene_id = _nonempty(request_payload.get("scene_id"), "scene_id")
        candidates = request_payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("request must contain feasible candidates")
        request_sha256 = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
        raw = self.source.infer(request_json, timeout_s=timeout_s)
        envelope, _ = self._load_envelope(raw)
        if envelope["schema_version"] != PROPOSAL_SCHEMA_VERSION:
            raise ValueError("GraspGenX proposal schema is unsupported")
        if envelope["request_sha256"] != request_sha256:
            raise ValueError("GraspGenX proposals are stale")
        if envelope["scene_id"] != scene_id:
            raise ValueError("GraspGenX proposals belong to another scene")
        if envelope["model_version"] != self.model_version:
            raise ValueError("GraspGenX model version changed")
        if envelope["input_frame_id"] != self.input_frame_id:
            raise ValueError("GraspGenX input frame changed")
        if envelope["calibration_id"] != self.calibration_id:
            raise ValueError("GraspGenX calibration changed")
        reported_depth = _finite(envelope["fingertip_depth_m"], "fingertip_depth_m")
        if abs(reported_depth - self.fingertip_depth_m) > 1e-6:
            raise ValueError("GraspGenX fingertip depth changed")
        base_from_input = _se3(
            envelope["transform_robot_base_from_input"],
            "transform_robot_base_from_input",
        )
        raw_proposals = envelope["proposals"]
        if not isinstance(raw_proposals, list):
            raise ValueError("GraspGenX proposals must be an array")

        candidate_info = []
        seen_candidate_ids = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("candidate must be an object")
            candidate_id = _nonempty(candidate.get("candidate_id"), "candidate_id")
            if candidate_id in seen_candidate_ids:
                raise ValueError("candidate IDs must be unique")
            seen_candidate_ids.add(candidate_id)
            pick_pose = candidate.get("pick_tip_pose")
            if not isinstance(pick_pose, dict):
                raise ValueError("candidate pick_tip_pose must be an object")
            xyz = pick_pose.get("xyz_m")
            if (
                isinstance(xyz, (str, bytes))
                or not isinstance(xyz, Sequence)
                or len(xyz) != 3
            ):
                raise ValueError("candidate pick xyz must have three values")
            candidate_info.append(
                {
                    "candidate_id": candidate_id,
                    "xyz_m": tuple(_finite(item, "candidate pick xyz") for item in xyz),
                    "yaw_rad": _finite(pick_pose.get("yaw_rad"), "candidate yaw"),
                }
            )

        best_scores = {item["candidate_id"]: 0.0 for item in candidate_info}
        proposal_ids = set()
        matched_proposals = 0
        ambiguous_proposals = 0
        cosine_limit = math.cos(self.maximum_approach_tilt_rad)
        for proposal in raw_proposals:
            if not isinstance(proposal, dict) or set(proposal) != _PROPOSAL_KEYS:
                raise ValueError("GraspGenX proposal has unexpected fields")
            proposal_id = _nonempty(proposal["proposal_id"], "proposal_id")
            if proposal_id in proposal_ids:
                raise ValueError("GraspGenX proposal IDs must be unique")
            proposal_ids.add(proposal_id)
            branch_tag = proposal["branch_tag"]
            if branch_tag not in {None, "diff", "obb"}:
                raise ValueError("GraspGenX branch_tag is unsupported")
            confidence = _unit_interval(proposal["confidence"], "proposal.confidence")
            input_from_gripper = _se3(
                proposal["transform_input_from_gripper"],
                "transform_input_from_gripper",
            )
            base_from_gripper = _compose(base_from_input, input_from_gripper)
            approach_axis = tuple(base_from_gripper[row][2] for row in range(3))
            approach_cosine = sum(
                first * second
                for first, second in zip(approach_axis, self.expected_approach_axis)
            )
            if approach_cosine < cosine_limit:
                continue
            closing_axis = tuple(base_from_gripper[row][0] for row in range(3))
            closing_planar_norm = math.hypot(closing_axis[0], closing_axis[1])
            if closing_planar_norm <= 1e-9:
                continue
            proposal_yaw = math.atan2(-closing_axis[0], closing_axis[1])
            gripper_base = tuple(base_from_gripper[row][3] for row in range(3))
            proposal_tip = tuple(
                gripper_base[index] + self.fingertip_depth_m * approach_axis[index]
                for index in range(3)
            )
            matches = []
            for candidate in candidate_info:
                position_error = math.dist(proposal_tip, candidate["xyz_m"])
                yaw_error = _parallel_angle_distance(proposal_yaw, candidate["yaw_rad"])
                if (
                    position_error <= self.maximum_position_error_m
                    and yaw_error <= self.maximum_yaw_error_rad
                ):
                    cost = (position_error / self.maximum_position_error_m) ** 2 + (
                        yaw_error / self.maximum_yaw_error_rad
                    ) ** 2
                    matches.append((cost, candidate["candidate_id"]))
            if not matches:
                continue
            matches.sort()
            if len(matches) > 1 and abs(matches[0][0] - matches[1][0]) <= 1e-12:
                ambiguous_proposals += 1
                continue
            candidate_id = matches[0][1]
            best_scores[candidate_id] = max(best_scores[candidate_id], confidence)
            matched_proposals += 1

        ranking = sorted(
            (
                {"candidate_id": candidate_id, "score": score}
                for candidate_id, score in best_scores.items()
            ),
            key=lambda item: (-item["score"], item["candidate_id"]),
        )
        top_score = ranking[0]["score"]
        top_ids = [
            item["candidate_id"]
            for item in ranking
            if abs(item["score"] - top_score) <= 1e-12
        ]
        selected_id = top_ids[0] if top_score > 0.0 and len(top_ids) == 1 else None
        return {
            "schema_version": RESPONSE_SCHEMA_VERSION,
            "request_sha256": request_sha256,
            "model_version": self.model_version,
            "selected_candidate_id": selected_id,
            "confidence": top_score,
            "ranking": ranking,
            "rationale": (
                f"matched {matched_proposals} of {len(raw_proposals)} "
                f"GraspGenX proposals; {ambiguous_proposals} ambiguous"
            ),
        }


__all__ = [
    "GraspGenXCatalogProvider",
    "GraspGenXProposalSource",
    "MATCHER_VERSION",
    "PROPOSAL_SCHEMA_VERSION",
]
