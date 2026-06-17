from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from agent.schemas import UsageRecord


class JsonTransport(Protocol):
    def __call__(
        self,
        *,
        model: str,
        prompt: str,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        ...


class LLMClientError(RuntimeError):
    """Raised when the configured LLM transport fails."""


@dataclass
class LLMClientResponse:
    content: dict[str, Any]
    usage: UsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMClient:
    model: str
    transport: JsonTransport | None = None
    api_key_env: str = "DASHSCOPE_API_KEY"
    base_url_env: str = "DASHSCOPE_BASE_URL"
    time_fn: Callable[[], float] = time.time

    def generate_json(
        self,
        *,
        qid: str,
        stage: str,
        prompt: str,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMClientResponse:
        if self.transport is None:
            raise LLMClientError(
                "LLM transport is not configured. Inject a transport before formal inference."
            )

        start = self.time_fn()
        try:
            payload = self.transport(
                model=self.model,
                prompt=prompt,
                json_schema=json_schema,
                temperature=temperature,
            )
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise LLMClientError(str(exc)) from exc

        elapsed_ms = int((self.time_fn() - start) * 1000)
        usage_data = payload.get("usage", {})
        usage = UsageRecord(
            qid=qid,
            stage=stage,
            model=payload.get("model", self.model),
            prompt_tokens=int(usage_data.get("prompt_tokens", 0)),
            completion_tokens=int(usage_data.get("completion_tokens", 0)),
            total_tokens=int(usage_data.get("total_tokens", 0)),
            latency_ms=elapsed_ms,
            success=bool(payload.get("success", True)),
            error=payload.get("error"),
        )
        return LLMClientResponse(
            content=payload.get("content", {}),
            usage=usage,
            raw=payload,
        )

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env)

    @property
    def base_url(self) -> str | None:
        return os.getenv(self.base_url_env)
