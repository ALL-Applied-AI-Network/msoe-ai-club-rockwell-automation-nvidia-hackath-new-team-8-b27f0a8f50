import hashlib
import json
import math
import unittest

from sorting_core import TipPoseBase
from sorting_core.grasping import (
    EvaluatedGrasp,
    FeasibilityResult,
    GraspCandidate,
    GraspFeatures,
    GraspPlanningError,
    GraspScore,
)

from team_plugins.grasping.ai_selection import (
    AIAssistedSelector,
    AISelectionPolicy,
    RESPONSE_SCHEMA_VERSION,
    _AI_SIMULATION_CAPABILITY,
    _create_ai_selection_policy,
    build_selection_request,
    strict_json_object,
)


def _evaluation(
    candidate_id,
    success_probability,
    *,
    scene_id="scene-1",
    target_track_id="target-1",
    feature_value=1.0,
    yaw_rad=0.0,
):
    candidate = GraspCandidate(
        candidate_id=candidate_id,
        scene_id=scene_id,
        target_track_id=target_track_id,
        pick_tip_pose=TipPoseBase((0.4, 0.0, 0.1), yaw_rad),
        approach_tip_pose=TipPoseBase((0.4, 0.0, 0.2), yaw_rad),
        required_opening_m=0.012,
        generator_id="fixture-generator-v1",
    )
    return EvaluatedGrasp(
        candidate=candidate,
        feasibility=FeasibilityResult(
            True,
            checker_id="fixture-checker-v1",
            metrics={"clearance_m": 0.01},
        ),
        features=GraspFeatures(
            "fixture-features-v1",
            {"value": feature_value, "visibility": 0.9},
        ),
        score=GraspScore(
            success_probability,
            scorer_id="fixture-scorer-v1",
            model_version="fixture-score-v1",
        ),
    )


class _Provider:
    provider_id = "fixture-pretrained-provider"
    model_version = "fixture-model@abc123"

    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def select(self, request_json, *, timeout_s):
        self.calls.append((request_json, timeout_s))
        if isinstance(self.responder, BaseException):
            raise self.responder
        if callable(self.responder):
            return self.responder(request_json)
        return self.responder


def _valid_response(
    request_json,
    *,
    selected="candidate-b",
    confidence=0.9,
    scores=(0.9, 0.6),
):
    request_sha256 = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
    candidate_ids = [
        item["candidate_id"] for item in strict_json_object(request_json)["candidates"]
    ]
    score_by_id = {
        candidate_id: score
        for candidate_id, score in zip(("candidate-b", "candidate-a"), scores)
    }
    ranking = sorted(
        (
            {
                "candidate_id": candidate_id,
                "score": score_by_id[candidate_id],
            }
            for candidate_id in candidate_ids
        ),
        key=lambda item: (-item["score"], item["candidate_id"]),
    )
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "request_sha256": request_sha256,
        "model_version": _Provider.model_version,
        "selected_candidate_id": selected,
        "confidence": confidence,
        "ranking": ranking,
        "rationale": "fixture ranking",
    }


class AISelectionTests(unittest.TestCase):
    def setUp(self):
        self.candidate_a = _evaluation("candidate-a", 0.8)
        self.candidate_b = _evaluation("candidate-b", 0.4)
        self.evaluations = (self.candidate_a, self.candidate_b)

    def _policy(self, responder, **kwargs):
        defaults = {
            "mode": "simulation_binding",
            "minimum_confidence": 0.7,
            "minimum_score_margin": 0.1,
            "timeout_s": 3.0,
        }
        defaults.update(kwargs)
        return _create_ai_selection_policy(
            _Provider(responder),
            simulation_capability=_AI_SIMULATION_CAPABILITY,
            **defaults,
        )

    def test_request_hash_is_order_invariant(self):
        first = build_selection_request(self.evaluations, decision_nonce="fixed-nonce")
        second = build_selection_request(
            tuple(reversed(self.evaluations)),
            decision_nonce="fixed-nonce",
        )
        self.assertEqual(first.payload_json, second.payload_json)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.candidate_ids, ("candidate-a", "candidate-b"))

    def test_request_hash_binds_scene_pose_and_features(self):
        baseline = build_selection_request(
            self.evaluations,
            decision_nonce="fixed-nonce",
            request_context={"profile_sha256": "a" * 64},
        ).sha256
        variants = (
            (
                _evaluation("candidate-a", 0.8, scene_id="scene-2"),
                _evaluation("candidate-b", 0.4, scene_id="scene-2"),
            ),
            (
                _evaluation("candidate-a", 0.8, yaw_rad=0.1),
                self.candidate_b,
            ),
            (
                _evaluation("candidate-a", 0.8, feature_value=2.0),
                self.candidate_b,
            ),
        )
        for evaluations in variants:
            with self.subTest(evaluations=evaluations):
                self.assertNotEqual(
                    build_selection_request(
                        evaluations,
                        decision_nonce="fixed-nonce",
                        request_context={"profile_sha256": "a" * 64},
                    ).sha256,
                    baseline,
                )

    def test_request_rejects_mixed_or_duplicate_identity(self):
        mixed_scene = (
            self.candidate_a,
            _evaluation("candidate-b", 0.4, scene_id="scene-2"),
        )
        mixed_target = (
            self.candidate_a,
            _evaluation("candidate-b", 0.4, target_track_id="target-2"),
        )
        duplicate = (self.candidate_a, self.candidate_a)
        for evaluations, message in (
            ((), "requires feasible"),
            (mixed_scene, "multiple scenes"),
            (mixed_target, "multiple targets"),
            (duplicate, "duplicate AI candidate"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(GraspPlanningError, message):
                    build_selection_request(evaluations)

    def test_strict_json_rejects_unsafe_encodings(self):
        for value in (
            '{"key": 1, "key": 2}',
            '{"key": NaN}',
            '{"key": Infinity}',
            'markdown-prefix {"key": 1}',
            '{"key": 1} trailing',
            "[]",
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    strict_json_object(value)

    def test_valid_binding_selects_only_existing_feasible_candidate(self):
        policy = self._policy(_valid_response)
        decision = policy.decide(self.evaluations)
        self.assertEqual(decision.status, "bound")
        self.assertIs(decision.executed, self.candidate_b)
        self.assertEqual(decision.proposed_candidate_id, "candidate-b")
        self.assertFalse(decision.used_fallback)
        self.assertEqual(decision.confidence, 0.9)
        self.assertAlmostEqual(decision.score_margin, 0.3)
        receipt = decision.receipt(include_payloads=True)
        self.assertEqual(receipt["executed_candidate_id"], "candidate-b")
        self.assertEqual(receipt["fallback_candidate_id"], "candidate-a")
        self.assertEqual(
            receipt["request"]["schema_version"],
            "team8-feasible-grasp-ranking-request-v1",
        )

    def test_bound_same_candidate_is_not_mislabeled_as_fallback(self):
        def responder(request_json):
            return _valid_response(
                request_json,
                selected="candidate-a",
                scores=(0.6, 0.9),
            )

        decision = self._policy(responder).decide(self.evaluations)
        self.assertEqual(decision.status, "bound")
        self.assertIs(decision.executed, self.candidate_a)
        self.assertIs(decision.fallback, self.candidate_a)
        self.assertFalse(decision.used_fallback)
        self.assertFalse(decision.receipt()["used_fallback"])

    def test_shadow_mode_never_binds(self):
        policy = self._policy(_valid_response, mode="shadow")
        decision = policy.decide(self.evaluations)
        self.assertEqual(decision.status, "shadow_valid")
        self.assertIs(decision.executed, self.candidate_a)
        self.assertEqual(decision.proposed_candidate_id, "candidate-b")
        self.assertTrue(decision.used_fallback)

    def test_binding_is_refused_without_guarded_factory(self):
        with self.assertRaisesRegex(ValueError, "guarded planner"):
            AISelectionPolicy(
                _Provider(_valid_response),
                mode="simulation_binding",
                simulation_only=True,
            )

    def test_invalid_responses_fall_back_with_explicit_status(self):
        def stale_hash(response):
            response["request_sha256"] = "0" * 64

        def wrong_model(response):
            response["model_version"] = "other-model@revision"

        def wrong_schema(response):
            response["schema_version"] = "wrong-schema"

        def extra_field(response):
            response["unexpected"] = True

        def unknown_selection(response):
            response["selected_candidate_id"] = "not-feasible"

        def omitted_candidate(response):
            response["ranking"].pop()

        def duplicate_candidate(response):
            response["ranking"][1]["candidate_id"] = response["ranking"][0][
                "candidate_id"
            ]

        def selected_not_top(response):
            response["selected_candidate_id"] = "candidate-a"

        def nonfinite(response):
            response["ranking"][0]["score"] = math.nan

        def boolean_confidence(response):
            response["confidence"] = True

        def string_score(response):
            response["ranking"][0]["score"] = "0.9"

        modifiers = {
            "stale_hash": stale_hash,
            "wrong_model": wrong_model,
            "wrong_schema": wrong_schema,
            "extra_field": extra_field,
            "unknown_selection": unknown_selection,
            "omitted_candidate": omitted_candidate,
            "duplicate_candidate": duplicate_candidate,
            "selected_not_top": selected_not_top,
            "nonfinite": nonfinite,
            "boolean_confidence": boolean_confidence,
            "string_score": string_score,
        }
        for name, modifier in modifiers.items():

            def responder(request_json, modifier=modifier):
                response = _valid_response(request_json)
                modifier(response)
                return response

            with self.subTest(name=name):
                decision = self._policy(responder).decide(self.evaluations)
                self.assertEqual(decision.status, "fallback_invalid_response")
                self.assertIs(decision.executed, self.candidate_a)
                self.assertTrue(decision.reason.startswith("invalid_response:"))

    def test_ranking_is_canonicalized_by_scores(self):
        def responder(request_json):
            response = _valid_response(request_json)
            response["ranking"].reverse()
            return response

        decision = self._policy(responder).decide(self.evaluations)
        self.assertEqual(decision.status, "bound")
        self.assertIs(decision.executed, self.candidate_b)
        raw_response = strict_json_object(decision.response_json)
        self.assertEqual(raw_response["ranking"][0]["candidate_id"], "candidate-a")

    def test_duplicate_response_json_keys_fall_back(self):
        def responder(request_json):
            response = _valid_response(request_json)
            encoded = json.dumps(response)
            return encoded.replace(
                '"confidence": 0.9',
                '"confidence": 0.9, "confidence": 0.8',
            )

        decision = self._policy(responder).decide(self.evaluations)
        self.assertEqual(decision.status, "fallback_invalid_response")
        self.assertIs(decision.executed, self.candidate_a)

    def test_confidence_margin_and_tie_gates(self):
        cases = (
            ("bound", 0.7, (0.8, 0.7)),
            ("fallback_low_confidence", 0.699, (0.9, 0.6)),
            ("fallback_low_margin", 0.9, (0.8, 0.7001)),
            ("fallback_tied", 0.9, (0.8, 0.8)),
        )
        for expected, confidence, scores in cases:

            def responder(
                request_json,
                confidence=confidence,
                scores=scores,
            ):
                return _valid_response(
                    request_json,
                    confidence=confidence,
                    scores=scores,
                )

            with self.subTest(expected=expected):
                decision = self._policy(responder).decide(self.evaluations)
                self.assertEqual(decision.status, expected)

    def test_tied_top_does_not_hide_lower_ranked_selection(self):
        candidate_c = _evaluation("candidate-c", 0.2)
        evaluations = self.evaluations + (candidate_c,)

        def responder(request_json):
            return {
                "schema_version": RESPONSE_SCHEMA_VERSION,
                "request_sha256": hashlib.sha256(
                    request_json.encode("utf-8")
                ).hexdigest(),
                "model_version": _Provider.model_version,
                "selected_candidate_id": "candidate-c",
                "confidence": 0.8,
                "ranking": [
                    {"candidate_id": "candidate-a", "score": 0.8},
                    {"candidate_id": "candidate-b", "score": 0.8},
                    {"candidate_id": "candidate-c", "score": 0.1},
                ],
                "rationale": "malformed lower-ranked selection",
            }

        decision = self._policy(responder).decide(evaluations)
        self.assertEqual(decision.status, "fallback_invalid_response")
        self.assertEqual(
            decision.reason,
            "invalid_response:selected_not_top_ranked",
        )

    def test_abstention_and_provider_error_fall_back(self):
        def abstain(request_json):
            return _valid_response(request_json, selected=None)

        abstained = self._policy(abstain).decide(self.evaluations)
        failed = self._policy(RuntimeError("offline")).decide(self.evaluations)
        self.assertEqual(abstained.status, "fallback_abstained")
        self.assertEqual(failed.status, "fallback_provider_error")
        self.assertEqual(failed.reason, "provider_error:RuntimeError")
        self.assertIs(abstained.executed, self.candidate_a)
        self.assertIs(failed.executed, self.candidate_a)

    def test_system_exit_and_keyboard_interrupt_are_not_swallowed(self):
        for error in (KeyboardInterrupt(), SystemExit()):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)):
                    self._policy(error).decide(self.evaluations)

    def test_empty_input_never_calls_provider(self):
        provider = _Provider(_valid_response)
        policy = AISelectionPolicy(provider)
        with self.assertRaisesRegex(GraspPlanningError, "requires feasible"):
            policy.decide(())
        self.assertFalse(provider.calls)

    def test_selector_adapter_emits_immutable_decision(self):
        decisions = []
        selector = AIAssistedSelector(
            self._policy(_valid_response),
            decision_sink=decisions.append,
        )
        selected = selector.select(self.evaluations)
        self.assertIs(selected, self.candidate_b)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].status, "bound")


if __name__ == "__main__":
    unittest.main()
