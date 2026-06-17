from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping


ARK_CODING_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
DASHSCOPE_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass(frozen=True)
class OpenAICompatibleSettings:
    api_key: str
    base_url: str
    api_key_env: str
    base_url_env: str
    strip_model_prefix: str | None = None


@dataclass(frozen=True)
class OpenAICompatibleTransport:
    settings: OpenAICompatibleSettings
    timeout: float = 120.0

    def __call__(
        self,
        *,
        model: str,
        prompt: str,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        from openai import OpenAI

        request_model = _normalize_request_model(model, self.settings.strip_model_prefix)
        client = OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            timeout=self.timeout,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是保险问答裁判。只能输出一个 JSON 对象，不要输出 Markdown、解释或代码块。"
                ),
            },
            {
                "role": "user",
                "content": _append_schema_instruction(prompt, json_schema),
            },
        ]
        response = client.chat.completions.create(
            model=request_model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        usage = response.usage
        return {
            "model": model,
            "content": _load_json_object(content),
            "usage": {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            },
            "success": True,
        }


def resolve_openai_compatible_settings(
    model: str, env: Mapping[str, str] | None = None
) -> OpenAICompatibleSettings | None:
    values = os.environ if env is None else env
    if model == "ark-code-latest" or model.startswith("ark-"):
        api_key = values.get("ARK_API_KEY")
        if not api_key:
            return None
        return OpenAICompatibleSettings(
            api_key=api_key,
            base_url=values.get("ARK_BASE_URL", ARK_CODING_BASE_URL),
            api_key_env="ARK_API_KEY",
            base_url_env="ARK_BASE_URL",
        )

    if model.startswith("dashscope/") or model.startswith("qwen"):
        api_key = values.get("DASHSCOPE_API_KEY")
        if not api_key:
            return None
        return OpenAICompatibleSettings(
            api_key=api_key,
            base_url=values.get("DASHSCOPE_BASE_URL", DASHSCOPE_COMPATIBLE_BASE_URL),
            api_key_env="DASHSCOPE_API_KEY",
            base_url_env="DASHSCOPE_BASE_URL",
            strip_model_prefix="dashscope/",
        )

    return None


def create_openai_compatible_transport(
    model: str, env: Mapping[str, str] | None = None
) -> OpenAICompatibleTransport | None:
    settings = resolve_openai_compatible_settings(model, env)
    if settings is None:
        return None
    return OpenAICompatibleTransport(settings=settings)


def _normalize_request_model(model: str, strip_prefix: str | None) -> str:
    if strip_prefix and model.startswith(strip_prefix):
        return model.removeprefix(strip_prefix)
    return model


def _append_schema_instruction(prompt: str, json_schema: dict[str, Any] | None) -> str:
    if not json_schema:
        return prompt
    return (
        f"{prompt}\n\n"
        "输出 JSON 必须满足以下 JSON Schema：\n"
        f"{json.dumps(json_schema, ensure_ascii=False)}"
    )


def _load_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("LLM response must be a JSON object")
    return data
