from agent.llm_transport import resolve_openai_compatible_settings


def test_resolve_ark_coding_settings_from_environment() -> None:
    settings = resolve_openai_compatible_settings(
        "ark-code-latest",
        {
            "ARK_API_KEY": "test-key",
        },
    )

    assert settings is not None
    assert settings.api_key == "test-key"
    assert settings.base_url == "https://ark.cn-beijing.volces.com/api/coding/v3"
    assert settings.api_key_env == "ARK_API_KEY"


def test_resolve_settings_returns_none_when_api_key_is_missing() -> None:
    settings = resolve_openai_compatible_settings("ark-code-latest", {})

    assert settings is None
