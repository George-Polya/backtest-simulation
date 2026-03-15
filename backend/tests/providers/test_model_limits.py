"""Tests for model token-limit resolution helpers."""

from backend.providers.llm.model_limits import resolve_model_limits, resolve_openrouter_limits


def test_resolve_openrouter_limits_for_gpt_5_4() -> None:
    """GPT-5.4 should use the preset when config omits explicit limits."""
    context_tokens, output_tokens = resolve_openrouter_limits(
        model_id="openai/gpt-5.4",
        configured_max_tokens=None,
        configured_max_context_tokens=None,
    )

    assert context_tokens == 1_000_000
    assert output_tokens == 128_000


def test_resolve_openrouter_limits_for_gpt_5_3_codex() -> None:
    """GPT-5.3-Codex should use the preset when config omits explicit limits."""
    context_tokens, output_tokens = resolve_openrouter_limits(
        model_id="openai/gpt-5.3-codex",
        configured_max_tokens=None,
        configured_max_context_tokens=None,
    )

    assert context_tokens == 400_000
    assert output_tokens == 128_000


def test_resolve_openrouter_limits_without_config_uses_global_fallback() -> None:
    """Without configured limits, unknown models should still fall back to 8k."""
    context_tokens, output_tokens = resolve_openrouter_limits(
        model_id="openai/unknown-model",
        configured_max_tokens=None,
        configured_max_context_tokens=None,
    )

    assert context_tokens == 128_000
    assert output_tokens == 8_000


def test_resolve_openrouter_limits_prefers_explicit_override() -> None:
    """Explicit request config should override only the provided field."""
    context_tokens, output_tokens = resolve_openrouter_limits(
        model_id="openai/gpt-5.4",
        configured_max_tokens=4_096,
        configured_max_context_tokens=None,
    )

    assert context_tokens == 1_000_000
    assert output_tokens == 4_096


def test_resolve_model_limits_supports_provider_prefix_fallback() -> None:
    """Provider prefixes should be stripped when fallback lookup is enabled."""
    context_tokens, output_tokens = resolve_model_limits(
        model_id="openai/gpt-5.4",
        model_limits={"gpt-5.4": (1_000_000, 128_000)},
        configured_max_tokens=None,
        configured_max_context_tokens=None,
        allow_provider_prefix_fallback=True,
    )

    assert context_tokens == 1_000_000
    assert output_tokens == 128_000
