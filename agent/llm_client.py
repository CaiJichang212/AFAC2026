"""Mockable LLM client skeleton for Qwen-compatible APIs.

Design principles:
- Uses dependency injection so tests can monkeypatch or inject a fake client.
- Wraps JSON schema validation around responses.
- Provides retry logic and error wrapping.
- Skeleton only: real network calls will be wired in later tasks.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol / interface for the underlying API caller
# ---------------------------------------------------------------------------


class ApiCaller(Protocol):
    """Protocol for the low-level API call so it can be mocked.

    Implementations should accept messages, a model name, and kwargs,
    and return a dict resembling a litellm / OpenAI-compatible response.
    """

    def __call__(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# Response wrapper
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Normalised response from an LLM call."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    raw: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


class LLMClient:
    """Skeleton LLM client that delegates to an injectable API caller.

    Usage::

        client = LLMClient(model="dashscope/qwen3.6-plus")
        response = client.chat(messages=[{"role": "user", "content": "Hello"}])
    """

    def __init__(
        self,
        model: str = "dashscope/qwen3.6-plus",
        *,
        api_caller: ApiCaller | None = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        self.model = model
        self._api_caller = api_caller
        self.max_retries = max_retries
        self.base_delay = base_delay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat completion request with optional JSON schema validation.

        Args:
            messages: OpenAI-format message list.
            json_schema: If provided, request structured output matching this schema.
            temperature: Sampling temperature.
            max_tokens: Maximum completion tokens.

        Returns:
            LLMResponse with content, token counts, and latency.
        """
        kwargs: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
            # Store schema for post-hoc validation
            kwargs["_json_schema"] = json_schema

        return self._call_with_retry(messages=messages, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_retry(self, *, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        """Retry loop with exponential backoff and error wrapping."""
        last_error: Exception | None = None
        json_schema = kwargs.pop("_json_schema", None)

        for attempt in range(self.max_retries + 1):
            try:
                return self._do_call(messages=messages, json_schema=json_schema, **kwargs)
            except Exception as exc:
                last_error = exc
                logger.warning("LLM call attempt %d/%d failed: %s", attempt + 1, self.max_retries + 1, exc)
                if attempt < self.max_retries:
                    delay = self.base_delay * (2**attempt)
                    time.sleep(delay)

        raise RuntimeError(f"LLM call failed after {self.max_retries + 1} attempts") from last_error

    def _do_call(
        self, *, messages: list[dict[str, Any]], json_schema: dict[str, Any] | None = None, **kwargs: Any
    ) -> LLMResponse:
        """Execute a single call (real or mocked)."""
        t0 = time.monotonic()

        if self._api_caller is None:
            # Skeleton mode: return a dummy response so callers can be tested.
            raw: dict[str, Any] = {
                "choices": [{"message": {"content": json.dumps({"_skeleton": True})}}],
                "model": self.model,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        else:
            raw = self._api_caller(model=self.model, messages=messages, **kwargs)

        latency_ms = (time.monotonic() - t0) * 1000.0

        # Extract content
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Extract usage
        usage = raw.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        # Validate JSON schema if requested
        if json_schema is not None and content:
            self._validate_json_response(content, json_schema)

        return LLMResponse(
            content=content,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            raw=raw,
        )

    @staticmethod
    def _validate_json_response(content: str, schema: dict[str, Any]) -> None:
        """Validate that the response content conforms to the expected JSON schema.

        Raises ValueError on parse or validation failure.
        """
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM response is not valid JSON: {exc}") from exc

        # Full JSON Schema validation requires the jsonschema library,
        # which is a transitive dependency of litellm. We use a lightweight
        # structural check for now; full validation can be added later.
        if schema.get("type") == "object" and not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object but got {type(parsed).__name__}")
        if schema.get("type") == "array" and not isinstance(parsed, list):
            raise ValueError(f"Expected JSON array but got {type(parsed).__name__}")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_client(config: "AgentConfig") -> LLMClient:  # noqa: F821
    """Create an LLMClient from an AgentConfig.

    Import is deferred to avoid circular imports.
    """
    return LLMClient(model=config.inference_model)
