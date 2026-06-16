"""Mockable LLM client for OpenAI-compatible APIs (litellm-backed).

Design principles:
- Uses dependency injection so tests can inject a MockApiCaller.
- Wraps JSON schema validation around responses.
- Provides retry logic and error wrapping.
- Real caller uses litellm to hit OpenAI-compatible endpoints (e.g. ARK).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NonRetryableError(Exception):
    """Raised for failures that should NOT be retried (e.g. JSON validation).

    The retry loop re-raises these immediately instead of backing off.
    """


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
# Real litellm-backed API caller
# ---------------------------------------------------------------------------


class LitellmApiCaller:
    """Calls an OpenAI-compatible endpoint via litellm.

    The *model* parameter is automatically prefixed with ``"openai/"`` so
    litellm routes to the correct provider.

    .. note::

        The ARK coding model (ark-code-latest) returns only internal
        reasoning tokens when ``temperature=0.0``, producing empty
        ``content``.  Use ``temperature > 0`` (e.g. 0.6) or switch to a
        non-coding model for final submission.
    """

    def __init__(self, *, base_url: str, api_key: str) -> None:
        self._base_url = base_url
        self._api_key = api_key

    def __call__(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        import litellm

        # litellm needs the provider prefix for a custom OpenAI-compatible base
        litellm_model = f"openai/{model}"

        response = litellm.completion(
            model=litellm_model,
            messages=messages,
            api_base=self._base_url,
            api_key=self._api_key,
            **kwargs,
        )

        # Build a plain predictable dict; avoid model_dump() / .dict() as
        # some litellm response types on custom endpoints lose nested content
        # fields during serialization.
        return _litellm_response_to_dict(response)


def _litellm_response_to_dict(response: Any) -> dict[str, Any]:
    """Convert a litellm ModelResponse to a predictable plain dict.

    We extract content and usage directly via attribute access (which is
    reliable), then build a dict that matches the OpenAI-compatible shape
    expected by the rest of the pipeline.
    """
    # -- choices ---------------------------------------------------------------
    choices: list[dict[str, Any]] = []
    for choice in getattr(response, "choices", []) or []:
        msg = getattr(choice, "message", None)
        msg_dict: dict[str, Any] = {}
        if msg is not None:
            msg_dict["content"] = getattr(msg, "content", "") or ""
            msg_dict["role"] = getattr(msg, "role", "assistant")
        choices.append(
            {
                "finish_reason": getattr(choice, "finish_reason", None),
                "index": getattr(choice, "index", 0),
                "message": msg_dict,
            }
        )

    # -- usage -----------------------------------------------------------------
    usage_raw = getattr(response, "usage", None)
    usage: dict[str, Any] = {}
    if usage_raw is not None:
        usage = {
            "prompt_tokens": getattr(usage_raw, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage_raw, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage_raw, "total_tokens", 0) or 0,
        }

    return {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "choices": choices,
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# Mock API caller (for deterministic offline tests)
# ---------------------------------------------------------------------------


class MockApiCaller:
    """Configurable mock that returns canned responses.

    Supports:
    - A list of canned responses (cycled through on successive calls).
    - A dict mapping call-number -> exception to raise (for testing retries).
    - Records every call for later inspection.

    Example::

        mock = MockApiCaller(responses=[
            canned_response_1,
            canned_response_2,
        ])
        # First call returns canned_response_1, second returns canned_response_2.
    """

    def __init__(
        self,
        responses: list[dict[str, Any]] | None = None,
        raise_on_call: dict[int, Exception] | None = None,
    ) -> None:
        self.responses: list[dict[str, Any]] = list(responses) if responses else []
        self.raise_on_call: dict[int, Exception] = dict(raise_on_call) if raise_on_call else {}
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        call_record = {"model": model, "messages": messages, "kwargs": kwargs}
        self.calls.append(call_record)
        call_num = len(self.calls)

        if call_num in self.raise_on_call:
            raise self.raise_on_call[call_num]

        if self.responses:
            idx = min(call_num - 1, len(self.responses) - 1)
            return self.responses[idx]

        # Default fallback response
        return {
            "choices": [{"message": {"content": '{"_mock": true}'}}],
            "model": model,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


class LLMClient:
    """LLM client that delegates to an injectable API caller.

    Usage::

        client = LLMClient(model="ark-code-latest")
        response = client.chat(messages=[{"role": "user", "content": "Hello"}])
    """

    def __init__(
        self,
        model: str = "ark-code-latest",
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
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any, *, force_mock: bool = False) -> "LLMClient":
        """Build a client from an AgentConfig instance.

        When *force_mock* is False and an API key is available, a real
        ``LitellmApiCaller`` is wired in.  Otherwise a warning is logged
        and the client falls back to the skeleton (no-op) mode.
        """
        api_key: str | None = getattr(config, "inference_api_key", None)
        if force_mock or not api_key:
            if not force_mock:
                key_env: str = getattr(config, "inference_api_key_env", "UNKNOWN")
                logger.warning(
                    "No API key found (env var %s); falling back to skeleton mock caller.",
                    key_env,
                )
            return cls(model=getattr(config, "inference_model", "ark-code-latest"))

        caller = LitellmApiCaller(
            base_url=getattr(config, "inference_base_url", ""),
            api_key=api_key,
        )
        return cls(model=getattr(config, "inference_model", "ark-code-latest"), api_caller=caller)

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
            except NonRetryableError:
                raise  # JSON validation failures etc. — do NOT retry
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

        Raises NonRetryableError on parse or validation failure (these are
        deterministic and should not trigger a retry).
        """
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise NonRetryableError(f"LLM response is not valid JSON: {exc}") from exc

        # Full JSON Schema validation requires the jsonschema library,
        # which is a transitive dependency of litellm. We use a lightweight
        # structural check for now; full validation can be added later.
        if schema.get("type") == "object" and not isinstance(parsed, dict):
            raise NonRetryableError(f"Expected JSON object but got {type(parsed).__name__}")
        if schema.get("type") == "array" and not isinstance(parsed, list):
            raise NonRetryableError(f"Expected JSON array but got {type(parsed).__name__}")
