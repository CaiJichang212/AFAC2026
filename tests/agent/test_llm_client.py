"""Tests for agent/llm_client.py — mock-based unit tests + one real smoke test."""

from __future__ import annotations

import json
import os
import time

import pytest

from agent.llm_client import (
    LitellmApiCaller,
    LLMClient,
    LLMResponse,
    MockApiCaller,
    NonRetryableError,
)
from agent.schemas import UsageRecord
from agent.token_meter import TokenMeter


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _canned_response(content: str = "OK", *, model: str = "ark-code-latest") -> dict:
    """Build a minimal OpenAI-compatible canned response dict."""
    return {
        "choices": [{"message": {"content": content}}],
        "model": model,
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    }


def _simple_messages() -> list[dict]:
    return [{"role": "user", "content": "Hello"}]


# ---------------------------------------------------------------------------
# Mock-based unit tests (NO network)
# ---------------------------------------------------------------------------


class TestMockApiCaller:
    """Verify MockApiCaller itself behaves correctly."""

    def test_returns_canned_response(self) -> None:
        canned = _canned_response("hello")
        mock = MockApiCaller(responses=[canned])
        result = mock(model="test", messages=[])
        assert result["choices"][0]["message"]["content"] == "hello"

    def test_cycles_through_responses(self) -> None:
        mock = MockApiCaller(responses=[
            _canned_response("first"),
            _canned_response("second"),
        ])
        r1 = mock(model="x", messages=[])
        r2 = mock(model="x", messages=[])
        assert r1["choices"][0]["message"]["content"] == "first"
        assert r2["choices"][0]["message"]["content"] == "second"

    def test_sticks_to_last_response(self) -> None:
        mock = MockApiCaller(responses=[_canned_response("only")])
        mock(model="x", messages=[])
        r2 = mock(model="x", messages=[])
        r3 = mock(model="x", messages=[])
        # Beyond the list length, always returns the last one
        assert r2["choices"][0]["message"]["content"] == "only"
        assert r3["choices"][0]["message"]["content"] == "only"

    def test_records_calls(self) -> None:
        mock = MockApiCaller(responses=[_canned_response("a")])
        mock(model="m1", messages=[{"role": "user", "content": "hi"}], temperature=0.5)
        assert len(mock.calls) == 1
        assert mock.calls[0]["model"] == "m1"
        assert mock.calls[0]["kwargs"]["temperature"] == 0.5

    def test_raises_on_configured_call_number(self) -> None:
        mock = MockApiCaller(
            responses=[_canned_response("ok")],
            raise_on_call={2: RuntimeError("transient")},
        )
        mock(model="m", messages=[])  # call 1 → ok
        with pytest.raises(RuntimeError, match="transient"):
            mock(model="m", messages=[])  # call 2 → raise

    def test_default_fallback_when_no_responses(self) -> None:
        mock = MockApiCaller()
        result = mock(model="m", messages=[])
        content = result["choices"][0]["message"]["content"]
        assert json.loads(content) == {"_mock": True}


class TestLLMClientWithMock:
    """Unit tests for LLMClient.chat() using MockApiCaller (no network)."""

    def test_chat_returns_content_and_usage(self) -> None:
        mock = MockApiCaller(responses=[_canned_response("Hello, world!")])
        client = LLMClient(model="ark-code-latest", api_caller=mock)
        response = client.chat(messages=_simple_messages())
        assert response.content == "Hello, world!"
        assert response.model == "ark-code-latest"
        assert response.prompt_tokens == 50
        assert response.completion_tokens == 10
        assert response.total_tokens == 60
        assert response.latency_ms >= 0

    def test_chat_records_usage_in_token_meter(self) -> None:
        """Verify LLMResponse fields can be mapped to a UsageRecord."""
        mock = MockApiCaller(responses=[_canned_response("data")])
        client = LLMClient(model="test-model", api_caller=mock)
        response = client.chat(messages=_simple_messages())

        # Simulate what the pipeline does: record usage after each call
        meter = TokenMeter()
        meter.record(UsageRecord(
            qid="ins_a_001",
            stage="evidence",
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_ms=response.latency_ms,
            success=True,
        ))
        assert meter.record_count == 1
        summary = meter.summary()
        assert summary["total_prompt_tokens"] == 50
        assert summary["total_completion_tokens"] == 10
        assert summary["total_tokens"] == 60

    def test_retry_on_transient_error_then_succeeds(self) -> None:
        """First call raises, second call succeeds — retry must work."""
        mock = MockApiCaller(
            responses=[_canned_response("recovered")],
            raise_on_call={1: ConnectionError("transient")},
        )
        client = LLMClient(model="m", api_caller=mock, max_retries=2, base_delay=0.001)
        response = client.chat(messages=_simple_messages())
        assert response.content == "recovered"
        assert len(mock.calls) == 2, f"Expected 2 calls (1 failed + 1 success), got {len(mock.calls)}"

    def test_retry_exhausted_raises(self) -> None:
        """All calls fail → RuntimeError."""
        mock = MockApiCaller(
            raise_on_call={1: ConnectionError("fail1"), 2: ConnectionError("fail2"), 3: ConnectionError("fail3")},
        )
        client = LLMClient(model="m", api_caller=mock, max_retries=2, base_delay=0.001)
        with pytest.raises(RuntimeError, match="LLM call failed after 3 attempts"):
            client.chat(messages=_simple_messages())
        assert len(mock.calls) == 3

    def test_json_validation_object_type(self) -> None:
        """When json_schema requests an object, valid JSON objects pass validation."""
        mock = MockApiCaller(responses=[_canned_response('{"key": "value"}')])
        client = LLMClient(model="m", api_caller=mock)
        response = client.chat(messages=_simple_messages(), json_schema={"type": "object"})
        assert response.content == '{"key": "value"}'

    def test_json_validation_rejects_array_when_object_expected(self) -> None:
        """When json_schema expects object but response is array, NonRetryableError is raised."""
        mock = MockApiCaller(responses=[_canned_response("[1, 2, 3]")])
        client = LLMClient(model="m", api_caller=mock)
        with pytest.raises(NonRetryableError, match="Expected JSON object but got list"):
            client.chat(messages=_simple_messages(), json_schema={"type": "object"})

    def test_json_validation_rejects_invalid_json(self) -> None:
        """Invalid JSON in response raises NonRetryableError."""
        mock = MockApiCaller(responses=[_canned_response("not json")])
        client = LLMClient(model="m", api_caller=mock)
        with pytest.raises(NonRetryableError, match="LLM response is not valid JSON"):
            client.chat(messages=_simple_messages(), json_schema={"type": "object"})

    def test_json_validation_accepts_array_when_expected(self) -> None:
        """When json_schema expects array, valid JSON arrays pass validation."""
        mock = MockApiCaller(responses=[_canned_response("[1, 2, 3]")])
        client = LLMClient(model="m", api_caller=mock)
        response = client.chat(messages=_simple_messages(), json_schema={"type": "array"})
        assert response.content == "[1, 2, 3]"

    def test_latency_is_positive(self) -> None:
        """Latency must be non-negative and reasonable."""
        mock = MockApiCaller(responses=[_canned_response("ok")])
        client = LLMClient(model="m", api_caller=mock)
        t0 = time.monotonic()
        response = client.chat(messages=_simple_messages())
        wall = (time.monotonic() - t0) * 1000
        assert response.latency_ms >= 0
        # Latency should not exceed wall-clock by more than a small margin
        assert response.latency_ms <= wall + 50  # 50ms tolerance

    def test_from_config_with_force_mock(self) -> None:
        """from_config(force_mock=True) creates a skeleton client even when key is set."""
        from agent.config import AgentConfig
        config = AgentConfig()
        client = LLMClient.from_config(config, force_mock=True)
        assert client._api_caller is None
        assert client.model == "ark-code-latest"

    def test_from_config_without_key_falls_back(self, monkeypatch) -> None:
        """When env var is missing, from_config falls back to skeleton (no ApiCaller)."""
        from agent.config import AgentConfig
        # Ensure the env var is NOT set
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        config = AgentConfig()
        client = LLMClient.from_config(config)
        assert client._api_caller is None  # skeleton mode


# ---------------------------------------------------------------------------
# Real smoke test (guarded by ARK_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestRealArkEndpoint:
    """Smoke test hitting the real ARK coding plan endpoint.

    These tests are skipped when ARK_API_KEY is not in the environment.
    """

    def test_trivial_prompt_returns_ok(self) -> None:
        """Send a trivial prompt to ark-code-latest and verify a valid response.

        Note: the ARK coding model (ark-code-latest) requires temperature > 0;
        at temperature=0.0 all completion tokens are internal reasoning tokens
        and ``content`` is empty.
        """
        api_key = os.environ.get("ARK_API_KEY")
        if not api_key:
            pytest.skip("ARK_API_KEY not set")

        from agent.config import AgentConfig
        config = AgentConfig()
        assert config.inference_api_key is not None, "API key should be readable from env"

        caller = LitellmApiCaller(
            base_url=config.inference_base_url,
            api_key=config.inference_api_key,  # type: ignore[arg-type]
        )
        client = LLMClient(model=config.inference_model, api_caller=caller)

        response = client.chat(
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            temperature=0.6,  # ARK coding model requires t > 0 for visible content
            max_tokens=32,
        )

        # Content should be non-empty
        assert response.content, "Response content must not be empty"
        assert len(response.content.strip()) > 0

        # Token usage must be non-negative integers
        assert isinstance(response.prompt_tokens, int) and response.prompt_tokens >= 0
        assert isinstance(response.completion_tokens, int) and response.completion_tokens >= 0
        assert isinstance(response.total_tokens, int) and response.total_tokens >= 0
        assert response.total_tokens == response.prompt_tokens + response.completion_tokens

        # Latency must be positive
        assert response.latency_ms > 0

        # Model should match
        assert response.model == "ark-code-latest"
