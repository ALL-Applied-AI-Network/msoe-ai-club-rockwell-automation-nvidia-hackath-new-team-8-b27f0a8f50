"""Train a dependency-free logistic complete-grasp success model from JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from team_plugins.grasping.multigrasp_pipeline import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_ID,
)


ROOT = Path(__file__).resolve().parents[2]
TRAINER_VERSION = "team8-json-logistic-trainer-v1"


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _sigmoid(logit: float) -> float:
    if logit >= 0.0:
        value = math.exp(-logit)
        return 1.0 / (1.0 + value)
    value = math.exp(logit)
    return value / (1.0 + value)


def _rows(raw: bytes) -> list[dict[str, Any]]:
    result = []
    for line_number, raw_line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        row = json.loads(
            raw_line,
            parse_constant=_reject_constant,
            object_pairs_hook=_strict_pairs,
        )
        if not isinstance(row, dict) or set(row) != {
            "episode_id",
            "group_id",
            "candidate_id",
            "split",
            "features",
            "success",
        }:
            raise ValueError(f"line {line_number}: invalid row keys")
        for identifier in ("episode_id", "group_id", "candidate_id"):
            if (
                type(row[identifier]) is not str
                or not row[identifier].strip()
            ):
                raise ValueError(
                    f"line {line_number}: invalid {identifier}"
                )
            row[identifier] = row[identifier].strip()
        if row["split"] not in {"train", "test"}:
            raise ValueError(f"line {line_number}: split must be train or test")
        if type(row["success"]) is not bool:
            raise ValueError(f"line {line_number}: success must be boolean")
        features = row["features"]
        if not isinstance(features, dict) or tuple(features) != FEATURE_NAMES:
            raise ValueError(
                f"line {line_number}: features must use {FEATURE_SCHEMA_ID}"
            )
        checked = {}
        for name in FEATURE_NAMES:
            value = float(features[name])
            if not math.isfinite(value):
                raise ValueError(
                    f"line {line_number}: feature {name} must be finite"
                )
            checked[name] = value
        row["features"] = checked
        result.append(row)
    if not result:
        raise ValueError("dataset is empty")
    ids = [row["episode_id"] for row in result]
    if len(set(ids)) != len(ids):
        raise ValueError("episode_id values must be unique")
    candidate_keys = [
        (row["group_id"], row["candidate_id"]) for row in result
    ]
    if len(set(candidate_keys)) != len(candidate_keys):
        raise ValueError(
            "candidate_id values must be unique within each group_id"
        )
    group_splits: dict[str, str] = {}
    for row in result:
        prior = group_splits.setdefault(row["group_id"], row["split"])
        if prior != row["split"]:
            raise ValueError(
                f"group_id {row['group_id']!r} crosses train and test splits"
            )
    return result


def _matrix(rows: Iterable[dict[str, Any]]):
    selected = list(rows)
    return (
        [
            [row["features"][name] for name in FEATURE_NAMES]
            for row in selected
        ],
        [1.0 if row["success"] else 0.0 for row in selected],
    )


def _standardize(matrix):
    count = len(matrix)
    means = [
        sum(row[index] for row in matrix) / count
        for index in range(len(FEATURE_NAMES))
    ]
    scales = []
    for index, mean in enumerate(means):
        variance = sum(
            (row[index] - mean) ** 2 for row in matrix
        ) / count
        scales.append(max(math.sqrt(variance), 1e-9))
    values = [
        [
            (row[index] - means[index]) / scales[index]
            for index in range(len(FEATURE_NAMES))
        ]
        for row in matrix
    ]
    return values, means, scales


def _apply_standardization(matrix, means, scales):
    return [
        [
            (row[index] - means[index]) / scales[index]
            for index in range(len(FEATURE_NAMES))
        ]
        for row in matrix
    ]


def _fit(matrix, labels, *, iterations: int, learning_rate: float, l2: float):
    if not matrix or len(set(labels)) != 2:
        raise ValueError("training split must contain both success and failure")
    weights = [0.0] * len(FEATURE_NAMES)
    prevalence = min(max(sum(labels) / len(labels), 1e-4), 1.0 - 1e-4)
    intercept = math.log(prevalence / (1.0 - prevalence))
    for step in range(iterations):
        probabilities = [
            _sigmoid(
                intercept
                + sum(weight * value for weight, value in zip(weights, row))
            )
            for row in matrix
        ]
        scale = 1.0 / len(matrix)
        intercept_gradient = sum(
            probability - label
            for probability, label in zip(probabilities, labels)
        ) * scale
        gradients = [
            sum(
                (probability - label) * row[index]
                for probability, label, row in zip(
                    probabilities, labels, matrix
                )
            )
            * scale
            + l2 * weights[index]
            for index in range(len(weights))
        ]
        rate = learning_rate / math.sqrt(1.0 + step / 200.0)
        intercept -= rate * intercept_gradient
        weights = [
            weight - rate * gradient
            for weight, gradient in zip(weights, gradients)
        ]
    return weights, intercept


def _evaluate(rows, matrix, labels, weights, intercept):
    probabilities = [
        _sigmoid(
            intercept
            + sum(weight * value for weight, value in zip(weights, row))
        )
        for row in matrix
    ]
    clipped = [min(max(value, 1e-9), 1.0 - 1e-9) for value in probabilities]
    grouped: dict[str, list[tuple[float, float, str]]] = {}
    for row, probability, label in zip(rows, probabilities, labels):
        grouped.setdefault(row["group_id"], []).append(
            (probability, label, row["candidate_id"])
        )
    selected = [
        sorted(entries, key=lambda item: (-item[0], item[2]))[0]
        for entries in grouped.values()
    ]
    mixed = [
        entries
        for entries in grouped.values()
        if {entry[1] for entry in entries} == {0.0, 1.0}
    ]
    ranking_accuracy = (
        None
        if not mixed
        else sum(
            sorted(entries, key=lambda item: (-item[0], item[2]))[0][1] == 1.0
            for entries in mixed
        )
        / len(mixed)
    )
    return {
        "episodes": len(labels),
        "successes": int(sum(labels)),
        "brier_score": sum(
            (probability - label) ** 2
            for probability, label in zip(probabilities, labels)
        )
        / len(labels),
        "log_loss": -sum(
            label * math.log(probability)
            + (1.0 - label) * math.log(1.0 - probability)
            for probability, label in zip(clipped, labels)
        )
        / len(labels),
        "accuracy_at_0_5": sum(
            (probability >= 0.5) == bool(label)
            for probability, label in zip(probabilities, labels)
        )
        / len(labels),
        "selection": {
            "groups": len(grouped),
            "groups_with_mixed_outcomes": len(mixed),
            "top_candidate_success_rate": (
                sum(item[1] for item in selected) / len(selected)
            ),
            "top_candidate_ranking_accuracy": ranking_accuracy,
        },
    }





def _atomic_json_write(path: Path, payload: dict[str, Any], *, sort_keys: bool) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=sort_keys,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def train(
    dataset_path: Path,
    model_path: Path,
    metrics_path: Path,
    *,
    iterations: int = 2000,
    learning_rate: float = 0.08,
    l2: float = 0.01,
) -> dict[str, Any]:
    dataset_path = dataset_path.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    metrics_path = metrics_path.expanduser().resolve()
    if len({dataset_path, model_path, metrics_path}) != 3:
        raise ValueError("dataset, model, and metrics paths must be distinct")
    for output in (model_path, metrics_path):
        if output == ROOT or ROOT in output.parents:
            raise ValueError("model and metrics outputs must remain outside Git")
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    if iterations <= 0 or learning_rate <= 0.0 or l2 < 0.0:
        raise ValueError("training hyperparameters are invalid")

    raw = dataset_path.read_bytes()
    rows = _rows(raw)
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if not train_rows or not test_rows:
        raise ValueError("dataset must contain non-empty train and test splits")

    train_matrix, train_labels = _matrix(train_rows)
    test_matrix, test_labels = _matrix(test_rows)
    standardized_train, means, scales = _standardize(train_matrix)
    standardized_test = _apply_standardization(test_matrix, means, scales)
    weights, intercept = _fit(
        standardized_train,
        train_labels,
        iterations=iterations,
        learning_rate=learning_rate,
        l2=l2,
    )
    data_sha256 = hashlib.sha256(raw).hexdigest()
    hyperparameters = {
        "iterations": iterations,
        "learning_rate": learning_rate,
        "l2": l2,
    }
    identity = {
        "trainer_version": TRAINER_VERSION,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "training_data_sha256": data_sha256,
        "hyperparameters": hyperparameters,
    }
    training_run_sha256 = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    model = {
        "schema_version": 1,
        "model_type": "logistic_regression",
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_order": list(FEATURE_NAMES),
        "means": means,
        "scales": scales,
        "coefficients": weights,
        "intercept": intercept,
        "model_version": f"team8-logistic-{training_run_sha256[:16]}",
        "training_data_sha256": data_sha256,
    }
    metrics = {
        "schema_version": 1,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "model_version": model["model_version"],
        "trainer_version": TRAINER_VERSION,
        "training_data_sha256": data_sha256,
        "training_run_sha256": training_run_sha256,
        "train": _evaluate(
            train_rows, standardized_train, train_labels, weights, intercept
        ),
        "test": _evaluate(
            test_rows, standardized_test, test_labels, weights, intercept
        ),
        "hyperparameters": hyperparameters,
    }
    _atomic_json_write(model_path, model, sort_keys=False)
    _atomic_json_write(metrics_path, metrics, sort_keys=True)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.01)
    args = parser.parse_args()
    metrics = train(
        args.dataset,
        args.model,
        args.metrics,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    print(json.dumps(metrics["test"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
