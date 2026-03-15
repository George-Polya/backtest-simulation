"""
Model token-limit presets and helpers.

These presets let the backend infer sane model-specific defaults when
`llm.max_tokens` / `llm.max_context_tokens` are omitted from config.
"""

from typing import Mapping

# Conservative global fallbacks for unknown models.
DEFAULT_MAX_CONTEXT_TOKENS = 128000
DEFAULT_MAX_OUTPUT_TOKENS = 8000

# Tuple shape: (max_context_tokens, max_output_tokens)
ModelLimits = tuple[int, int]

OPENROUTER_MODEL_LIMITS: Mapping[str, ModelLimits] = {
    "anthropic/claude-opus-4.6": (1_000_000, 128_000),
    "openai/gpt-5.4": (1_000_000, 128_000),
    "openai/gpt-5.3-codex": (400_000, 128_000),
    "openai/gpt-5.2-codex": (400_000, 128_000),
    "minimax/minimax-m2.5": (204_800, 131_072),
}

def _normalize_model_id(model_id: str) -> str:
    """Normalize model IDs for case-insensitive map lookup."""
    return model_id.strip().lower()


def _lookup_model_limits(
    model_id: str,
    model_limits: Mapping[str, ModelLimits],
    allow_provider_prefix_fallback: bool,
) -> ModelLimits | None:
    """Find model limits by full ID first, then optionally by stripped ID."""
    normalized_model_id = _normalize_model_id(model_id)
    candidate_ids = [normalized_model_id]

    if allow_provider_prefix_fallback and "/" in normalized_model_id:
        candidate_ids.append(normalized_model_id.split("/", 1)[1])

    for candidate in candidate_ids:
        limits = model_limits.get(candidate)
        if limits is not None:
            return limits

    return None


def resolve_model_limits(
    model_id: str,
    model_limits: Mapping[str, ModelLimits],
    configured_max_tokens: int | None,
    configured_max_context_tokens: int | None,
    *,
    fallback_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    fallback_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    allow_provider_prefix_fallback: bool = False,
) -> ModelLimits:
    """
    Resolve effective model limits.

    Priority:
    1. Explicit config values
    2. Model preset values
    3. Global fallback values
    """
    preset_limits = _lookup_model_limits(
        model_id=model_id,
        model_limits=model_limits,
        allow_provider_prefix_fallback=allow_provider_prefix_fallback,
    )

    default_context = (
        preset_limits[0] if preset_limits is not None else fallback_context_tokens
    )
    default_output = (
        preset_limits[1] if preset_limits is not None else fallback_output_tokens
    )

    resolved_context = (
        configured_max_context_tokens
        if configured_max_context_tokens is not None
        else default_context
    )
    resolved_output = (
        configured_max_tokens if configured_max_tokens is not None else default_output
    )

    return (resolved_context, resolved_output)


def resolve_openrouter_limits(
    model_id: str,
    configured_max_tokens: int | None,
    configured_max_context_tokens: int | None,
) -> ModelLimits:
    """Resolve limits for OpenRouter models."""
    return resolve_model_limits(
        model_id=model_id,
        model_limits=OPENROUTER_MODEL_LIMITS,
        configured_max_tokens=configured_max_tokens,
        configured_max_context_tokens=configured_max_context_tokens,
        allow_provider_prefix_fallback=True,
    )
