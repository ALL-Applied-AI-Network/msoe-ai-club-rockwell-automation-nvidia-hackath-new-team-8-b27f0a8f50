"""Host-safe audit of paired real/sim table-camera framing landmarks.

The audit intentionally consumes manually reviewed 2D landmarks and source
hashes.  It diagnoses whether a discrepancy is too large to attribute to the
configured training-camera jitter, but never estimates or applies a camera or
table correction from two points.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_PATH = REPO_ROOT / "config" / "table_framing_landmarks.json"


class TableFramingError(ValueError):
    """Raised when framing evidence is malformed or source identity fails."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TableFramingError(f"{name} must be an object")
    return value


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TableFramingError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise TableFramingError(f"{name} must be finite")
    return result


def _point(value: Any, name: str, resolution: tuple[int, int]) -> tuple[float, float]:
    try:
        x, y = (_finite(item, name) for item in value)
    except (TypeError, ValueError) as exc:
        raise TableFramingError(f"{name} must contain x and y") from exc
    width, height = resolution
    if not (0.0 <= x < width and 0.0 <= y < height):
        raise TableFramingError(f"{name} lies outside {width}x{height}")
    return x, y


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_table_framing_evidence(path: str | Path | None = None) -> dict[str, Any]:
    """Load and strictly validate the landmark evidence JSON."""

    source = Path(path).resolve() if path is not None else DEFAULT_EVIDENCE_PATH
    try:
        raw = _mapping(json.loads(source.read_text(encoding="utf-8")), "evidence")
    except (OSError, json.JSONDecodeError) as exc:
        raise TableFramingError(f"cannot load framing evidence {source}: {exc}") from exc
    if raw.get("schema_version") != 1:
        raise TableFramingError("unsupported table-framing evidence schema")
    try:
        width, height = (int(value) for value in raw.get("image_resolution_px", ()))
    except (TypeError, ValueError) as exc:
        raise TableFramingError("image_resolution_px must contain two integers") from exc
    if width < 2 or height < 2:
        raise TableFramingError("image_resolution_px must be at least 2x2")
    resolution = (width, height)

    sources = _mapping(raw.get("sources"), "sources")
    if not sources:
        raise TableFramingError("sources must not be empty")
    for source_id, source_value in sources.items():
        entry = _mapping(source_value, f"source {source_id}")
        path_hint = str(entry.get("workspace_relative_path", "")).strip()
        digest = str(entry.get("sha256", "")).strip().lower()
        byte_size = entry.get("byte_size")
        if not path_hint or Path(path_hint).is_absolute() or ".." in Path(path_hint).parts:
            raise TableFramingError(
                f"source {source_id} workspace_relative_path must stay in workspace"
            )
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise TableFramingError(f"source {source_id} has invalid SHA-256")
        if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size < 1:
            raise TableFramingError(f"source {source_id} has invalid byte_size")

    frames = _mapping(raw.get("frames"), "frames")
    if not frames:
        raise TableFramingError("frames must not be empty")
    normalized_frames: dict[str, Any] = {}
    for frame_id, frame_value in frames.items():
        frame = _mapping(frame_value, f"frame {frame_id}")
        source_id = str(frame.get("source", "")).strip()
        if source_id not in sources:
            raise TableFramingError(f"frame {frame_id} references unknown source")
        uncertainty = _finite(
            frame.get("annotation_uncertainty_px"),
            f"frame {frame_id} annotation_uncertainty_px",
        )
        if uncertainty < 0.0:
            raise TableFramingError("annotation uncertainty must be non-negative")
        landmarks = _mapping(frame.get("landmarks_px"), f"frame {frame_id} landmarks")
        if not landmarks:
            raise TableFramingError(f"frame {frame_id} has no landmarks")
        normalized_frames[str(frame_id)] = {
            "source": source_id,
            "annotation_method": str(frame.get("annotation_method", "")).strip(),
            "annotation_uncertainty_px": uncertainty,
            "landmarks_px": {
                str(name): _point(value, f"{frame_id}.{name}", resolution)
                for name, value in landmarks.items()
            },
        }

    comparisons = raw.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise TableFramingError("comparisons must be a non-empty list")
    for index, comparison_value in enumerate(comparisons):
        comparison = _mapping(comparison_value, f"comparison {index}")
        reference = str(comparison.get("reference_frame", ""))
        candidate = str(comparison.get("candidate_frame", ""))
        if reference not in normalized_frames or candidate not in normalized_frames:
            raise TableFramingError(f"comparison {index} references unknown frame")
        names = comparison.get("landmarks")
        if not isinstance(names, list) or not names:
            raise TableFramingError(f"comparison {index} landmarks must be non-empty")
        for name in names:
            if (
                str(name) not in normalized_frames[reference]["landmarks_px"]
                or str(name) not in normalized_frames[candidate]["landmarks_px"]
            ):
                raise TableFramingError(
                    f"comparison {index} landmark {name!r} is not shared"
                )
        threshold = _finite(
            comparison.get("maximum_explainable_residual_px"),
            f"comparison {index} maximum_explainable_residual_px",
        )
        if threshold <= 0.0:
            raise TableFramingError("comparison threshold must be positive")

    result = dict(raw)
    result["frames"] = normalized_frames
    result["source_path"] = str(source)
    result["source_sha256"] = _sha256_file(source)
    return result


def verify_evidence_sources(
    evidence: Mapping[str, Any], workspace_root: str | Path
) -> dict[str, Any]:
    """Verify external evidence files without decoding or modifying them."""

    root = Path(workspace_root).resolve()
    rows: dict[str, Any] = {}
    for source_id, source_value in _mapping(evidence.get("sources"), "sources").items():
        source = _mapping(source_value, f"source {source_id}")
        path = (root / str(source["workspace_relative_path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise TableFramingError(f"source {source_id} escapes workspace") from exc
        if not path.is_file():
            rows[str(source_id)] = {
                "status": "missing",
                "path": str(path),
                "expected_sha256": source["sha256"],
                "expected_byte_size": source["byte_size"],
            }
            continue
        actual_size = path.stat().st_size
        actual_sha256 = _sha256_file(path)
        status = (
            "verified"
            if actual_size == source["byte_size"] and actual_sha256 == source["sha256"]
            else "identity_mismatch"
        )
        rows[str(source_id)] = {
            "status": status,
            "path": str(path),
            "expected_sha256": source["sha256"],
            "actual_sha256": actual_sha256,
            "expected_byte_size": source["byte_size"],
            "actual_byte_size": actual_size,
        }
    return {
        "workspace_root": str(root),
        "all_sources_verified": bool(rows) and all(
            row["status"] == "verified" for row in rows.values()
        ),
        "sources": rows,
    }


def _segment_metrics(
    first_reference: Sequence[float],
    second_reference: Sequence[float],
    first_candidate: Sequence[float],
    second_candidate: Sequence[float],
) -> dict[str, Any]:
    ref_dx = second_reference[0] - first_reference[0]
    ref_dy = second_reference[1] - first_reference[1]
    cand_dx = second_candidate[0] - first_candidate[0]
    cand_dy = second_candidate[1] - first_candidate[1]
    ref_length = math.hypot(ref_dx, ref_dy)
    cand_length = math.hypot(cand_dx, cand_dy)
    ref_midpoint = (
        (first_reference[0] + second_reference[0]) / 2.0,
        (first_reference[1] + second_reference[1]) / 2.0,
    )
    cand_midpoint = (
        (first_candidate[0] + second_candidate[0]) / 2.0,
        (first_candidate[1] + second_candidate[1]) / 2.0,
    )
    angle_delta = math.degrees(
        math.atan2(cand_dy, cand_dx) - math.atan2(ref_dy, ref_dx)
    )
    angle_delta = (angle_delta + 180.0) % 360.0 - 180.0
    return {
        "reference_length_px": ref_length,
        "candidate_length_px": cand_length,
        "length_error_px": cand_length - ref_length,
        "length_error_fraction": (
            (cand_length - ref_length) / ref_length if ref_length > 0.0 else None
        ),
        "reference_midpoint_px": list(ref_midpoint),
        "candidate_midpoint_px": list(cand_midpoint),
        "midpoint_residual_xy_px": [
            cand_midpoint[0] - ref_midpoint[0],
            cand_midpoint[1] - ref_midpoint[1],
        ],
        "midpoint_residual_norm_px": math.dist(ref_midpoint, cand_midpoint),
        "angle_error_deg": angle_delta,
    }


def audit_table_framing(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Compute landmark residuals and fail closed on unexplained framing error."""

    frames = _mapping(evidence.get("frames"), "frames")
    comparison_rows = []
    any_mismatch = False
    for comparison_value in evidence["comparisons"]:
        comparison = _mapping(comparison_value, "comparison")
        reference_id = str(comparison["reference_frame"])
        candidate_id = str(comparison["candidate_frame"])
        reference = _mapping(frames[reference_id], f"frame {reference_id}")
        candidate = _mapping(frames[candidate_id], f"frame {candidate_id}")
        ref_landmarks = _mapping(reference["landmarks_px"], "reference landmarks")
        cand_landmarks = _mapping(candidate["landmarks_px"], "candidate landmarks")
        threshold = float(comparison["maximum_explainable_residual_px"])
        residuals = {}
        norms = []
        for name_value in comparison["landmarks"]:
            name = str(name_value)
            ref = ref_landmarks[name]
            cand = cand_landmarks[name]
            dx, dy = cand[0] - ref[0], cand[1] - ref[1]
            norm = math.hypot(dx, dy)
            norms.append(norm)
            residuals[name] = {
                "reference_px": list(ref),
                "candidate_px": list(cand),
                "residual_xy_px": [dx, dy],
                "residual_norm_px": norm,
                "within_budget": norm <= threshold,
            }
        segment_rows = []
        for segment in comparison.get("segments", []):
            if not isinstance(segment, list) or len(segment) != 2:
                raise TableFramingError("comparison segment must contain two landmarks")
            first, second = (str(value) for value in segment)
            if first not in residuals or second not in residuals:
                raise TableFramingError("segment landmarks must be audited landmarks")
            segment_rows.append(
                {
                    "landmarks": [first, second],
                    **_segment_metrics(
                        ref_landmarks[first],
                        ref_landmarks[second],
                        cand_landmarks[first],
                        cand_landmarks[second],
                    ),
                }
            )
        maximum = max(norms)
        status = (
            "within_budget"
            if maximum <= threshold
            else "mismatch_requires_investigation"
        )
        any_mismatch |= status != "within_budget"
        comparison_rows.append(
            {
                "name": comparison["name"],
                "reference_frame": reference_id,
                "candidate_frame": candidate_id,
                "maximum_explainable_residual_px": threshold,
                "status": status,
                "expected_status": comparison.get("expected_status"),
                "status_matches_expectation": status
                == comparison.get("expected_status"),
                "landmarks": residuals,
                "residual_rms_px": math.sqrt(
                    sum(value * value for value in norms) / len(norms)
                ),
                "maximum_residual_px": maximum,
                "segments": segment_rows,
            }
        )

    return {
        "schema_version": 1,
        "evidence_path": evidence.get("source_path"),
        "evidence_sha256": evidence.get("source_sha256"),
        "image_resolution_px": evidence["image_resolution_px"],
        "camera_jitter_diagnostic": evidence["camera_jitter_diagnostic"],
        "comparisons": comparison_rows,
        "overall_status": (
            "mismatch_requires_investigation" if any_mismatch else "within_budget"
        ),
        "automatic_correction_permitted": False,
        "candidate_causes_to_test": list(evidence["candidate_causes_to_test"]),
        "limitations": list(evidence["limitations"]),
    }


__all__ = [
    "DEFAULT_EVIDENCE_PATH",
    "TableFramingError",
    "audit_table_framing",
    "load_table_framing_evidence",
    "verify_evidence_sources",
]
