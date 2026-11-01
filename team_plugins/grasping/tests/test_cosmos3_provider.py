import hashlib
import math
import os
import unittest
from unittest import mock

from team_plugins.grasping.ai_providers.cosmos3 import (
    Cosmos3OpenAIProvider,
    PROMPT_VERSION,
)
from team_plugins.grasping.ai_selection import (
    REQUEST_SCHEMA_VERSION,
    RESPONSE_SCHEMA_VERSION,
    canonical_json,
    strict_json_object,
)


class _HTTPResponse:
    def __init__(self, body, *, content_length=None):
        self.body = body
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, amount):
        return self.body[:amount]


def _request_json():
    return canonical_json(
        {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "scene_id": "scene-1",
            "target_track_id": "target-1",
            "candidates": [
                {"candidate_id": "candidate-a"},
                {"candidate_id": "candidate-b"},
            ],
        }
    )


def _model_response(request_json):
    request_sha256 = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
    return canonical_json(
        {
            "schema_version": RESPONSE_SCHEMA_VERSION,
            "request_sha256": request_sha256,
            "model_version": "nvidia/Cosmos3-Nano@abc123",
            "selected_candidate_id": "candidate-a",
            "confidence": 0.8,
            "ranking": [
                {"candidate_id": "candidate-a", "score": 0.8},
                {"candidate_id": "candidate-b", "score": 0.6},
            ],
            "rationale": "fixture",
        }
    )


def _envelope(content, *, finish_reason="stop", model="nvidia/Cosmos3-Nano"):
    return canonical_json(
        {
            "id": "fixture",
            "model": model,
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": content},
                }
            ],
        }
    ).encode("utf-8")


class Cosmos3ProviderTests(unittest.TestCase):
    def _provider(self, **kwargs):
        defaults = {
            "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
            "model": "nvidia/Cosmos3-Nano",
            "model_revision": "abc123",
        }
        defaults.update(kwargs)
        return Cosmos3OpenAIProvider(**defaults)

    def test_constructor_rejects_unsafe_configuration(self):
        cases = (
            {"endpoint": "ftp://localhost/model"},
            {"endpoint": "http://user:secret@localhost/v1"},
            {"endpoint": "http://localhost/v1?api_key=secret"},
            {"model": "bad model"},
            {"model_revision": "bad revision"},
            {"max_tokens": 1},
            {"max_response_bytes": 10},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self._provider(**overrides)

    @mock.patch("team_plugins.grasping.ai_providers.cosmos3._OPENER.open")
    def test_select_sends_constrained_schema_and_returns_json_text(self, urlopen):
        request_json = _request_json()
        model_response = _model_response(request_json)
        urlopen.return_value = _HTTPResponse(_envelope(model_response))
        provider = self._provider()
        result = provider.select(request_json, timeout_s=4.0)
        self.assertEqual(result, model_response)
        http_request = urlopen.call_args.args[0]
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 4.0)
        body = strict_json_object(http_request.data.decode("utf-8"))
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["model"], "nvidia/Cosmos3-Nano")
        response_format = body["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        schema = response_format["json_schema"]["schema"]
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertFalse(schema["additionalProperties"])
        selected_schema = schema["properties"]["selected_candidate_id"]
        self.assertEqual(
            selected_schema["anyOf"][0]["enum"],
            ["candidate-a", "candidate-b"],
        )
        prompt = body["messages"][1]["content"]
        self.assertIn(PROMPT_VERSION, prompt)
        self.assertIn(request_json, prompt)

    @mock.patch("team_plugins.grasping.ai_providers.cosmos3._OPENER.open")
    def test_api_key_is_read_at_call_time_and_redacted(self, urlopen):
        request_json = _request_json()
        model_response = _model_response(request_json)
        urlopen.return_value = _HTTPResponse(_envelope(model_response))
        provider = self._provider(api_key_env="COSMOS_FIXTURE_KEY")
        self.assertNotIn("COSMOS_FIXTURE_KEY", repr(provider))
        with mock.patch.dict(
            os.environ, {"COSMOS_FIXTURE_KEY": "super-secret"}, clear=False
        ):
            provider.select(request_json, timeout_s=4.0)
        http_request = urlopen.call_args.args[0]
        self.assertEqual(
            http_request.get_header("Authorization"),
            "Bearer super-secret",
        )
        self.assertNotIn("super-secret", repr(provider))

    def test_missing_api_key_fails_before_network(self):
        provider = self._provider(api_key_env="COSMOS_MISSING_KEY")
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "is unset"):
                provider.select(_request_json(), timeout_s=4.0)

    @mock.patch("team_plugins.grasping.ai_providers.cosmos3._OPENER.open")
    def test_response_size_is_bounded(self, urlopen):
        provider = self._provider(max_response_bytes=1024)
        urlopen.return_value = _HTTPResponse(b"{}", content_length=1025)
        with self.assertRaisesRegex(ValueError, "size limit"):
            provider.select(_request_json(), timeout_s=4.0)
        urlopen.return_value = _HTTPResponse(b"x" * 1025)
        with self.assertRaisesRegex(ValueError, "size limit"):
            provider.select(_request_json(), timeout_s=4.0)

    @mock.patch("team_plugins.grasping.ai_providers.cosmos3._OPENER.open")
    def test_invalid_transport_envelopes_fail_closed(self, urlopen):
        request_json = _request_json()
        cases = (
            b"\xff",
            b"{}",
            _envelope(_model_response(request_json), finish_reason="length"),
            _envelope(
                _model_response(request_json),
                model="unexpected-model",
            ),
            canonical_json(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": {"not": "text"}},
                        }
                    ]
                }
            ).encode("utf-8"),
        )
        for body in cases:
            with self.subTest(body=body):
                urlopen.return_value = _HTTPResponse(body)
                with self.assertRaises(ValueError):
                    self._provider().select(request_json, timeout_s=4.0)

    @mock.patch("team_plugins.grasping.ai_providers.cosmos3._OPENER.open")
    def test_total_deadline_includes_response_read(self, urlopen):
        request_json = _request_json()
        model_response = _model_response(request_json)
        urlopen.return_value = _HTTPResponse(_envelope(model_response))
        with mock.patch("team_plugins.grasping.ai_providers.cosmos3.time") as clock:
            clock.monotonic.side_effect = [10.0, 10.25, 11.01]
            with self.assertRaisesRegex(TimeoutError, "total deadline"):
                self._provider().select(request_json, timeout_s=1.0)

    def test_invalid_request_and_timeout_fail_before_network(self):
        provider = self._provider()
        invalid_requests = (
            "{}",
            canonical_json(
                {
                    "schema_version": "wrong",
                    "candidates": [{"candidate_id": "candidate-a"}],
                }
            ),
            canonical_json(
                {
                    "schema_version": REQUEST_SCHEMA_VERSION,
                    "candidates": [],
                }
            ),
            canonical_json(
                {
                    "schema_version": REQUEST_SCHEMA_VERSION,
                    "candidates": [
                        {"candidate_id": "same"},
                        {"candidate_id": "same"},
                    ],
                }
            ),
        )
        for request_json in invalid_requests:
            with self.subTest(request_json=request_json):
                with self.assertRaises(ValueError):
                    provider.select(request_json, timeout_s=4.0)
        for timeout in (0.0, math.nan, 301.0):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    provider.select(_request_json(), timeout_s=timeout)


if __name__ == "__main__":
    unittest.main()
