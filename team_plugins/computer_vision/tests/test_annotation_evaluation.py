from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_evaluation import validate_evaluation_result
from team_plugins.computer_vision.evaluate_annotations import (
    AnnotationValidationError,
    evaluate_manifest,
    load_manifest,
    main,
    validate_manifest,
)


def pose_truth(*, class_id: str, x_m: float):
    return {
        "position_base_m": [x_m, 0.0, 0.01],
        "position_uncertainty_mm": 0.25,
        "cube_yaw_rad": 0.0 if class_id == "cube" else None,
        "screw_head_to_tip_unit_base": None if class_id == "cube" else [1.0, 0.0, 0.0],
        "symmetry_id": (
            "upright_square_yaw_modulo_90_deg"
            if class_id == "cube"
            else "catalog_axial_symmetry_unresolved"
        ),
        "measurement_method": "operator-taught registered fixture",
    }


def reference(
    *,
    instance_id: str,
    class_id: str,
    sort_label: str | None,
    left: float,
    x_m: float,
    axis_observability: str = "unobservable",
):
    polygon = [[left, 10.0], [left + 20.0, 10.0], [left + 20.0, 30.0], [left, 30.0]]
    is_screw = class_id == "screw_iso10642_m6x30"
    return {
        "instance_id": instance_id,
        "class_id": class_id,
        "sort_label": sort_label,
        "polygon_px": polygon,
        "bbox_xywh_px": [left, 10.0, 20.0, 20.0],
        "center_uv": [left + 10.0, 20.0],
        "head_uv": [left + 3.0, 20.0] if is_screw and axis_observability != "unobservable" else None,
        "tip_uv": [left + 17.0, 20.0] if is_screw and axis_observability != "unobservable" else None,
        "axis_observability": axis_observability,
        "occlusion": {"category": "none", "fraction": 0.0, "causes": []},
        "truncated": False,
        "touching_group_id": None,
        "ignore_for_scoring": False,
        "pose_truth": pose_truth(
            class_id="screw" if is_screw else "cube",
            x_m=x_m,
        ),
    }


def prediction(
    *,
    prediction_id: str,
    track_id: str,
    class_id: str,
    sort_label: str | None,
    center_uv: list[float],
    x_m: float,
    quaternion: list[float],
    axis_observability: str = "unobservable",
    head_uv: list[float] | None = None,
    tip_uv: list[float] | None = None,
):
    return {
        "prediction_id": prediction_id,
        "track_id": track_id,
        "class_id": class_id,
        "sort_label": sort_label,
        "center_uv": center_uv,
        "confidence": 0.9,
        "pose_quality": 0.8,
        "head_uv": head_uv,
        "tip_uv": tip_uv,
        "axis_observability": axis_observability,
        "pose": {
            "position_base_m": [x_m, 0.0, 0.01],
            "orientation_xyzw": quaternion,
        },
    }


def frame(*, frame_id: str, frame_index: int, references, predictions, regions=None):
    return {
        "frame_id": frame_id,
        "layout_id": "layout-static-1",
        "frame_index": frame_index,
        "capture_time_s": 100.0 + frame_index / 10.0,
        "image_sha256": f"{frame_index + 1:064x}",
        "inference_latency_ms": 12.5 + frame_index,
        "references": references,
        "predictions": predictions,
        "regions": [] if regions is None else regions,
    }


def valid_manifest():
    root = math.sqrt(0.5)
    cube_reference = reference(
        instance_id="cube-1",
        class_id="cube",
        sort_label="red",
        left=5.0,
        x_m=0.400,
    )
    screw_reference = reference(
        instance_id="screw-1",
        class_id="screw_iso10642_m6x30",
        sort_label="red",
        left=40.0,
        x_m=0.450,
        axis_observability="undirected",
    )
    first = frame(
        frame_id="frame-0",
        frame_index=0,
        references=[cube_reference, screw_reference],
        predictions=[
            prediction(
                prediction_id="pred-cube-0",
                track_id="cube-track-a",
                class_id="cube",
                sort_label="red",
                center_uv=[15.0, 20.0],
                x_m=0.401,
                quaternion=[0.0, 0.0, root, root],
            ),
            prediction(
                prediction_id="pred-screw-0",
                track_id="screw-track-a",
                class_id="screw_iso10642_m6x30",
                sort_label="red",
                center_uv=[50.0, 20.0],
                x_m=0.452,
                quaternion=[0.0, root, 0.0, root],
                axis_observability="undirected",
                head_uv=[57.0, 20.0],
                tip_uv=[43.0, 20.0],
            ),
        ],
    )
    second_cube_reference = deepcopy(cube_reference)
    second = frame(
        frame_id="frame-1",
        frame_index=1,
        references=[second_cube_reference],
        predictions=[
            prediction(
                prediction_id="pred-cube-1",
                track_id="cube-track-b",
                class_id="cube",
                sort_label="red",
                center_uv=[15.0, 20.0],
                x_m=0.399,
                quaternion=[0.0, 0.0, 0.0, 1.0],
            )
        ],
    )
    return {
        "schema_version": 1,
        "annotation_id": "synthetic-annotation-v1",
        "created_at_utc": "2026-10-31T12:00:00Z",
        "split": "pilot",
        "backend": "offline_replay",
        "provenance": {
            "project_commit": "a" * 40,
            "branch": "vision/test",
            "integration_profile": {
                "profile_id": "vision-test-profile",
                "profile_sha256": "b" * 64,
            },
            "provider": {
                "name": "test-provider",
                "version": "1",
                "artifact_sha256": "c" * 64,
            },
            "object_catalog_version": "sha256:" + "d" * 64,
            "calibration_ids": ["test-table-v1"],
            "review": {
                "annotator_id": "annotator-a",
                "review_status": "reviewed",
                "reviewer_id": "reviewer-b",
            },
        },
        "source": {
            "dataset_id": "synthetic-fixture",
            "source_run_id": "synthetic-run-1",
            "capture_sha256": "e" * 64,
            "camera_id": "table",
            "image_width_px": 100,
            "image_height_px": 80,
            "timestamp_source": "frame_index_over_fps",
        },
        "frames": [first, second],
    }


class AnnotationEvaluationTests(unittest.TestCase):
    def test_cube_and_screw_symmetry_pose_and_tracking_metrics(self):
        evaluation, diagnostics = evaluate_manifest(valid_manifest())
        validate_evaluation_result(evaluation, source="generated")
        by_class = {entry["class_id"]: entry for entry in evaluation["vision"]["per_class"]}
        self.assertEqual(by_class["cube"]["reference_instances"], 2)
        self.assertEqual(by_class["cube"]["true_positives"], 2)
        self.assertEqual(by_class["cube"]["dropouts"], 0)
        self.assertEqual(by_class["cube"]["false_positives"], 0)
        self.assertEqual(len(by_class["cube"]["pose_samples"]), 2)
        # A 90-degree cube yaw and the identity are symmetry-equivalent.
        self.assertAlmostEqual(
            by_class["cube"]["pose_samples"][0]["symmetry_aware_orientation_error_deg"],
            0.0,
            places=6,
        )
        screw = by_class["screw_iso10642_m6x30"]
        self.assertEqual(screw["true_positives"], 1)
        # The predicted axis is reversed, but the reference explicitly marks
        # head/tip direction as unobservable/undirected.
        self.assertAlmostEqual(
            screw["pose_samples"][0]["symmetry_aware_orientation_error_deg"],
            0.0,
            places=6,
        )
        self.assertEqual(diagnostics["id_switches_total"], 1)
        self.assertEqual(diagnostics["pose_truth_uncertainty_mm"]["samples"], 3)
        self.assertEqual(diagnostics["pose_truth_uncertainty_mm"]["maximum"], 0.25)
        self.assertIn("layout-static-1:cube-1", diagnostics["position_residual_jitter"])
        self.assertAlmostEqual(
            diagnostics["image_axis_error_deg"]["undirected"][0]["error_deg"],
            0.0,
            places=6,
        )
        self.assertNotIn("total_episode_s", evaluation["episodes"][0]["timing_s"])
        self.assertEqual(
            diagnostics["review_provenance"]["review_status"], "reviewed"
        )
        self.assertEqual(
            evaluation["created_at_utc"],
            diagnostics["evaluation_provenance"]["evaluated_at_utc"],
        )

    def test_matching_maximizes_cardinality_before_distance(self):
        manifest = valid_manifest()
        first_reference = reference(
            instance_id="a-reference",
            class_id="cube",
            sort_label="red",
            left=0.0,
            x_m=0.40,
        )
        second_reference = reference(
            instance_id="b-reference",
            class_id="cube",
            sort_label="red",
            left=10.0,
            x_m=0.42,
        )
        flexible = prediction(
            prediction_id="a-flexible",
            track_id="flexible",
            class_id="cube",
            sort_label="red",
            center_uv=[15.0, 20.0],
            x_m=0.42,
            quaternion=[0.0, 0.0, 0.0, 1.0],
        )
        constrained = prediction(
            prediction_id="z-constrained",
            track_id="constrained",
            class_id="cube",
            sort_label="red",
            center_uv=[5.0, 20.0],
            x_m=0.40,
            quaternion=[0.0, 0.0, 0.0, 1.0],
        )
        manifest["frames"] = [
            frame(
                frame_id="adversarial",
                frame_index=0,
                references=[first_reference, second_reference],
                predictions=[flexible, constrained],
            )
        ]
        evaluation, diagnostics = evaluate_manifest(manifest)
        cube = evaluation["vision"]["per_class"][0]
        self.assertEqual(cube["true_positives"], 2)
        self.assertEqual(cube["dropouts"], 0)
        self.assertEqual(cube["false_positives"], 0)
        self.assertEqual(diagnostics["matches"], 2)

    def test_matching_minimizes_total_distance_after_cardinality(self):
        manifest = valid_manifest()
        left_reference = reference(
            instance_id="left-reference",
            class_id="cube",
            sort_label="red",
            left=0.0,
            x_m=0.40,
        )
        right_reference = reference(
            instance_id="right-reference",
            class_id="cube",
            sort_label="red",
            left=40.0,
            x_m=0.50,
        )
        for item in (left_reference, right_reference):
            item["polygon_px"] = [[0.0, 5.0], [60.0, 5.0], [60.0, 35.0], [0.0, 35.0]]
            item["bbox_xywh_px"] = [0.0, 5.0, 60.0, 30.0]
        left_prediction = prediction(
            prediction_id="left-prediction",
            track_id="left-track",
            class_id="cube",
            sort_label="red",
            center_uv=[12.0, 20.0],
            x_m=0.40,
            quaternion=[0.0, 0.0, 0.0, 1.0],
        )
        right_prediction = prediction(
            prediction_id="right-prediction",
            track_id="right-track",
            class_id="cube",
            sort_label="red",
            center_uv=[48.0, 20.0],
            x_m=0.50,
            quaternion=[0.0, 0.0, 0.0, 1.0],
        )
        manifest["frames"] = [
            frame(
                frame_id="minimum-cost",
                frame_index=0,
                references=[left_reference, right_reference],
                predictions=[right_prediction, left_prediction],
            )
        ]
        evaluation, _ = evaluate_manifest(manifest)
        errors = evaluation["vision"]["per_class"][0]["pose_samples"]
        self.assertEqual([sample["position_error_mm"] for sample in errors], [0.0, 0.0])

    def test_unknown_and_unsupported_classes_are_abstentions_not_false_positives(self):
        manifest = valid_manifest()
        manifest["frames"] = [deepcopy(manifest["frames"][0])]
        manifest["frames"][0]["references"] = []
        unknown = deepcopy(manifest["frames"][0]["predictions"][0])
        unknown.update(
            {
                "prediction_id": "unknown-1",
                "track_id": "unknown-track",
                "class_id": "unknown_obstacle",
                "sort_label": None,
            }
        )
        unsupported = deepcopy(unknown)
        unsupported.update(
            {
                "prediction_id": "unsupported-1",
                "track_id": "unsupported-track",
                "class_id": "washer_unmodeled",
            }
        )
        manifest["frames"][0]["predictions"] = [unknown, unsupported]
        evaluation, diagnostics = evaluate_manifest(manifest)
        self.assertEqual(evaluation["vision"]["per_class"], [])
        self.assertEqual(diagnostics["abstentions_total"], 2)
        self.assertEqual(diagnostics["abstentions_by_class"]["unknown_obstacle"], 1)
        self.assertEqual(diagnostics["abstentions_by_class"]["washer_unmodeled"], 1)
        self.assertFalse(diagnostics["semantic_candidate_recognition"]["available"])

    def test_semantic_candidate_recognition_keeps_withheld_target_a_primary_dropout(self):
        manifest = valid_manifest()
        source_frame = deepcopy(manifest["frames"][0])
        screw_reference = source_frame["references"][1]
        withheld = deepcopy(source_frame["predictions"][1])
        withheld.update(
            {
                "prediction_id": "withheld-screw",
                "track_id": "withheld-screw-track",
                "class_id": "unknown_obstacle",
                "sort_label": None,
                "pose_quality": 0.0,
                "head_uv": None,
                "tip_uv": None,
                "axis_observability": "unobservable",
                "pose": None,
                "semantic_candidate_class_id": "screw_iso10642_m6x30",
                "provider_output_eligibility": "withheld_non_executable",
                "eligibility_reasons": ["footprint_outside_pickup_region"],
            }
        )
        source_frame["references"] = [screw_reference]
        source_frame["predictions"] = [withheld]
        manifest["frames"] = [source_frame]

        evaluation, diagnostics = evaluate_manifest(manifest)
        screw = evaluation["vision"]["per_class"][0]
        self.assertEqual(screw["true_positives"], 0)
        self.assertEqual(screw["dropouts"], 1)
        self.assertEqual(diagnostics["abstentions_total"], 1)
        semantic = diagnostics["semantic_candidate_recognition"]
        self.assertTrue(semantic["available"])
        self.assertEqual(
            semantic["by_class"]["screw_iso10642_m6x30"],
            {
                "reference_instances": 1,
                "matched_semantic_total": 1,
                "matched_published_supported": 0,
                "matched_withheld_non_executable": 1,
                "dropouts": 0,
                "unmatched_candidate_predictions": 0,
                "unmatched_published_supported": 0,
                "unmatched_withheld_non_executable": 0,
                "semantic_candidate_recall": 1.0,
            },
        )
        self.assertEqual(
            semantic["matched_withheld_reasons"],
            {"footprint_outside_pickup_region": 1},
        )
        self.assertIn("remains non-executable", semantic["evidence_limit"])

    def test_explicit_published_eligibility_does_not_change_primary_metrics(self):
        legacy_manifest = valid_manifest()
        explicit_manifest = deepcopy(legacy_manifest)
        for frame_value in explicit_manifest["frames"]:
            for prediction_value in frame_value["predictions"]:
                prediction_value.update(
                    {
                        "semantic_candidate_class_id": prediction_value["class_id"],
                        "provider_output_eligibility": "published_supported",
                        "eligibility_reasons": [],
                    }
                )

        legacy_evaluation, legacy_diagnostics = evaluate_manifest(legacy_manifest)
        explicit_evaluation, explicit_diagnostics = evaluate_manifest(explicit_manifest)

        self.assertEqual(
            explicit_evaluation["vision"],
            legacy_evaluation["vision"],
        )
        self.assertEqual(
            explicit_diagnostics["abstentions_total"],
            legacy_diagnostics["abstentions_total"],
        )
        self.assertEqual(
            explicit_diagnostics["abstentions_by_class"],
            legacy_diagnostics["abstentions_by_class"],
        )
        self.assertTrue(
            explicit_diagnostics["semantic_candidate_recognition"]["available"]
        )
        self.assertFalse(
            legacy_diagnostics["semantic_candidate_recognition"]["available"]
        )

    def test_provider_eligibility_metadata_is_strict_and_all_or_none(self):
        cases = []
        partial = valid_manifest()
        partial["frames"][0]["predictions"][0][
            "semantic_candidate_class_id"
        ] = "cube"
        cases.append(("partial group", partial))

        inconsistent = valid_manifest()
        target = inconsistent["frames"][0]["predictions"][0]
        target.update(
            {
                "semantic_candidate_class_id": "cube",
                "provider_output_eligibility": "withheld_non_executable",
                "eligibility_reasons": ["outside"],
            }
        )
        cases.append(("withheld supported class", inconsistent))

        mixed = valid_manifest()
        target = mixed["frames"][0]["predictions"][0]
        target.update(
            {
                "semantic_candidate_class_id": "cube",
                "provider_output_eligibility": "published_supported",
                "eligibility_reasons": [],
            }
        )
        cases.append(("mixed legacy and extended predictions", mixed))

        supported_marked_non_candidate = valid_manifest()
        for frame in supported_marked_non_candidate["frames"]:
            for prediction in frame["predictions"]:
                prediction.update(
                    {
                        "semantic_candidate_class_id": None,
                        "provider_output_eligibility": "not_supported_candidate",
                        "eligibility_reasons": [],
                    }
                )
        cases.append(
            ("published supported class marked as non-candidate", supported_marked_non_candidate)
        )

        comma_reason = valid_manifest()
        comma_reason["frames"] = [comma_reason["frames"][0]]
        target = comma_reason["frames"][0]["predictions"][0]
        target.update(
            {
                "class_id": "unknown_obstacle",
                "sort_label": None,
                "pose": None,
                "semantic_candidate_class_id": "cube",
                "provider_output_eligibility": "withheld_non_executable",
                "eligibility_reasons": ["outside,blurred"],
            }
        )
        for other in comma_reason["frames"][0]["predictions"][1:]:
            other.update(
                {
                    "semantic_candidate_class_id": other["class_id"],
                    "provider_output_eligibility": "published_supported",
                    "eligibility_reasons": [],
                }
            )
        cases.append(("non-atomic reason", comma_reason))

        for name, manifest in cases:
            with self.subTest(name=name), self.assertRaises(AnnotationValidationError):
                validate_manifest(manifest)

    def test_scored_reference_wins_over_overlapping_ignore_region(self):
        manifest = valid_manifest()
        manifest["frames"] = [deepcopy(manifest["frames"][0])]
        manifest["frames"][0]["references"] = [manifest["frames"][0]["references"][0]]
        manifest["frames"][0]["predictions"] = [manifest["frames"][0]["predictions"][0]]
        manifest["frames"][0]["regions"] = [
            {
                "region_id": "broad-ignore",
                "kind": "ignore",
                "polygon_px": [[0.0, 0.0], [30.0, 0.0], [30.0, 40.0], [0.0, 40.0]],
            }
        ]
        evaluation, diagnostics = evaluate_manifest(manifest)
        cube = evaluation["vision"]["per_class"][0]
        self.assertEqual(cube["true_positives"], 1)
        self.assertEqual(diagnostics["ignored_predictions"], 0)

    def test_diagnostics_stratify_confidence_and_reference_difficulty(self):
        manifest = valid_manifest()
        manifest["frames"] = [deepcopy(manifest["frames"][0])]
        target = manifest["frames"][0]["references"][0]
        target["occlusion"] = {
            "category": "partial",
            "fraction": 0.25,
            "causes": ["cable"],
        }
        target["truncated"] = True
        target["touching_group_id"] = "touching-a"
        manifest["frames"][0]["references"] = [target]
        predicted = manifest["frames"][0]["predictions"][0]
        predicted["confidence"] = 0.4
        manifest["frames"][0]["predictions"] = [predicted]
        _, diagnostics = evaluate_manifest(manifest)
        self.assertEqual(
            diagnostics["confidence_strata"]["[0.0,0.5)"]["matched_supported"],
            1,
        )
        self.assertEqual(
            diagnostics["reference_strata"]["occlusion"]["partial"]["true_positives"],
            1,
        )
        self.assertEqual(
            diagnostics["reference_strata"]["truncation"]["truncated"]["reference_instances"],
            1,
        )
        self.assertEqual(
            diagnostics["reference_strata"]["touching"]["touching"]["true_positives"],
            1,
        )

    def test_wrong_colour_is_dropout_and_false_positive(self):
        manifest = valid_manifest()
        manifest["frames"] = [deepcopy(manifest["frames"][0])]
        manifest["frames"][0]["references"] = [manifest["frames"][0]["references"][0]]
        wrong = deepcopy(manifest["frames"][0]["predictions"][0])
        wrong["sort_label"] = "blue"
        manifest["frames"][0]["predictions"] = [wrong]
        evaluation, _ = evaluate_manifest(manifest)
        cube = evaluation["vision"]["per_class"][0]
        self.assertEqual(cube["reference_instances"], 1)
        self.assertEqual(cube["true_positives"], 0)
        self.assertEqual(cube["dropouts"], 1)
        self.assertEqual(cube["false_positives"], 1)

    def test_cube_orientation_metric_does_not_hide_tilt(self):
        manifest = valid_manifest()
        manifest["frames"] = [deepcopy(manifest["frames"][0])]
        manifest["frames"][0]["references"] = [manifest["frames"][0]["references"][0]]
        tilted = deepcopy(manifest["frames"][0]["predictions"][0])
        half_angle = math.radians(10.0)
        tilted["pose"]["orientation_xyzw"] = [
            math.sin(half_angle),
            0.0,
            0.0,
            math.cos(half_angle),
        ]
        manifest["frames"][0]["predictions"] = [tilted]
        evaluation, _ = evaluate_manifest(manifest)
        error = evaluation["vision"]["per_class"][0]["pose_samples"][0][
            "symmetry_aware_orientation_error_deg"
        ]
        self.assertAlmostEqual(error, 20.0, places=6)

    def test_directed_image_axis_preserves_head_tip_polarity(self):
        manifest = valid_manifest()
        screw = reference(
            instance_id="directed-screw",
            class_id="screw_iso10642_m6x30",
            sort_label="red",
            left=40.0,
            x_m=0.45,
            axis_observability="directed",
        )
        predicted = prediction(
            prediction_id="directed-prediction",
            track_id="directed-track",
            class_id="screw_iso10642_m6x30",
            sort_label="red",
            center_uv=[50.0, 20.0],
            x_m=0.45,
            quaternion=[0.0, 0.0, 0.0, 1.0],
            axis_observability="directed",
            head_uv=[57.0, 20.0],
            tip_uv=[43.0, 20.0],
        )
        manifest["frames"] = [
            frame(
                frame_id="directed-axis",
                frame_index=0,
                references=[screw],
                predictions=[predicted],
            )
        ]
        _, diagnostics = evaluate_manifest(manifest)
        self.assertAlmostEqual(
            diagnostics["image_axis_error_deg"]["directed"][0]["error_deg"],
            180.0,
            places=6,
        )

    def test_prediction_in_unknown_region_is_not_counted_as_false_positive(self):
        manifest = valid_manifest()
        manifest["frames"] = [deepcopy(manifest["frames"][0])]
        manifest["frames"][0]["references"] = []
        manifest["frames"][0]["predictions"] = [manifest["frames"][0]["predictions"][0]]
        manifest["frames"][0]["regions"] = [
            {
                "region_id": "unknown-1",
                "kind": "unknown",
                "polygon_px": [[0.0, 0.0], [30.0, 0.0], [30.0, 40.0], [0.0, 40.0]],
            }
        ]
        evaluation, diagnostics = evaluate_manifest(manifest)
        self.assertEqual(evaluation["vision"]["per_class"], [])
        self.assertEqual(diagnostics["ignored_predictions"], 1)

    def test_strict_validation_rejects_unknown_fields_and_invalid_semantics(self):
        cases = []
        unknown = valid_manifest()
        unknown["invented"] = True
        cases.append((unknown, "unknown fields"))
        cube_head = valid_manifest()
        cube_head["frames"][0]["references"][0]["head_uv"] = [10.0, 20.0]
        cases.append((cube_head, "cube head/tip"))
        bad_quaternion = valid_manifest()
        bad_quaternion["frames"][0]["predictions"][0]["pose"]["orientation_xyzw"] = [0.0, 0.0, 0.0, 2.0]
        cases.append((bad_quaternion, "unit quaternion"))
        duplicate_frame = valid_manifest()
        duplicate_frame["frames"][1]["frame_id"] = duplicate_frame["frames"][0]["frame_id"]
        cases.append((duplicate_frame, "duplicates"))
        unreviewed_test = valid_manifest()
        unreviewed_test["split"] = "test"
        unreviewed_test["provenance"]["review"] = {
            "annotator_id": "annotator-a",
            "review_status": "unreviewed",
            "reviewer_id": None,
        }
        cases.append((unreviewed_test, "must be reviewed or adjudicated"))
        fake_unreviewed_reviewer = valid_manifest()
        fake_unreviewed_reviewer["provenance"]["review"] = {
            "annotator_id": "annotator-a",
            "review_status": "unreviewed",
            "reviewer_id": "not-actually-a-review",
        }
        cases.append((fake_unreviewed_reviewer, "must be null"))
        missing_calibration = valid_manifest()
        missing_calibration["provenance"]["calibration_ids"] = []
        cases.append((missing_calibration, "calibration provenance"))
        wrong_symmetry = valid_manifest()
        wrong_symmetry["frames"][0]["references"][0]["pose_truth"]["symmetry_id"] = "catalog_axial_symmetry_unresolved"
        cases.append((wrong_symmetry, "must be 'upright_square_yaw_modulo_90_deg'"))
        whitespace = valid_manifest()
        whitespace["annotation_id"] = " annotation-with-whitespace "
        cases.append((whitespace, "leading or trailing whitespace"))
        uppercase_sha = valid_manifest()
        uppercase_sha["source"]["capture_sha256"] = "A" * 64
        cases.append((uppercase_sha, "lowercase 64-character"))
        backwards_across_null = valid_manifest()
        middle = deepcopy(backwards_across_null["frames"][1])
        middle.update(
            {
                "frame_id": "frame-middle-null",
                "frame_index": 1,
                "capture_time_s": None,
                "image_sha256": "3" * 64,
            }
        )
        final = deepcopy(backwards_across_null["frames"][1])
        final.update(
            {
                "frame_id": "frame-final-backward",
                "frame_index": 2,
                "capture_time_s": 99.0,
                "image_sha256": "4" * 64,
            }
        )
        backwards_across_null["source"]["timestamp_source"] = "unknown"
        backwards_across_null["frames"] = [
            backwards_across_null["frames"][0],
            middle,
            final,
        ]
        cases.append((backwards_across_null, "must not run backwards"))
        for manifest, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(AnnotationValidationError, message):
                    validate_manifest(manifest)

    def test_loader_rejects_duplicate_keys_and_cli_outputs_validate(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaisesRegex(AnnotationValidationError, "duplicate JSON key"):
                load_manifest(duplicate)

            manifest_path = root / "manifest.json"
            evaluation_path = root / "evaluation.json"
            diagnostics_path = root / "diagnostics.json"
            manifest_path.write_text(
                json.dumps(valid_manifest(), sort_keys=True),
                encoding="utf-8",
            )
            self.assertEqual(
                main(
                    [
                        str(manifest_path),
                        "--evaluation-out",
                        str(evaluation_path),
                        "--diagnostics-out",
                        str(diagnostics_path),
                    ]
                ),
                0,
            )
            validate_evaluation_result(
                json.loads(evaluation_path.read_text(encoding="utf-8")),
                source=str(evaluation_path),
            )
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            self.assertEqual(diagnostics["id_switches_total"], 1)


if __name__ == "__main__":
    unittest.main()
