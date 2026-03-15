"""
Shared OpenRouter/LangChain client helpers.

Keeps the OpenRouter-specific ChatOpenAI wiring in one place so the direct
adapter and the agent backend use the same defaults for headers, model info,
token limits, and request parameters.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from backend.core.config import LLMConfig
from backend.providers.llm.base import GenerationConfig, ModelInfo

# OpenRouter API base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default model costs (per 1k tokens, in USD)
DEFAULT_MODEL_COSTS: dict[str, tuple[Decimal, Decimal]] = {
    "anthropic/claude-3.5-sonnet": (Decimal("0.003"), Decimal("0.015")),
    "anthropic/claude-sonnet-4": (Decimal("0.003"), Decimal("0.015")),
    "anthropic/claude-3-opus": (Decimal("0.015"), Decimal("0.075")),
    "anthropic/claude-3-haiku": (Decimal("0.00025"), Decimal("0.00125")),
    "openai/gpt-4o": (Decimal("0.005"), Decimal("0.015")),
    "openai/gpt-4o-mini": (Decimal("0.00015"), Decimal("0.0006")),
    "moonshotai/kimi-k2-thinking": (Decimal("0.0006"), Decimal("0.002")),
}


def resolve_generation_config(
    llm_config: LLMConfig,
    config: GenerationConfig | None,
    *,
    resolved_max_output_tokens: int,
) -> GenerationConfig:
    """Return the effective generation config for a request without mutating input."""
    if config is None:
        return GenerationConfig(
            temperature=llm_config.temperature,
            max_tokens=resolved_max_output_tokens,
            seed=llm_config.seed,
        )

    return GenerationConfig(
        temperature=(
            config.temperature
            if config.temperature is not None
            else llm_config.temperature
        ),
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        top_k=config.top_k,
        stop_sequences=list(config.stop_sequences),
        frequency_penalty=config.frequency_penalty,
        presence_penalty=config.presence_penalty,
        seed=config.seed,
        extra=dict(config.extra),
    )


def build_openrouter_default_headers(llm_config: LLMConfig) -> dict[str, str] | None:
    """Build optional OpenRouter headers from app metadata."""
    default_headers: dict[str, str] = {}
    if llm_config.site_url:
        default_headers["HTTP-Referer"] = llm_config.site_url
    if llm_config.site_name:
        default_headers["X-Title"] = llm_config.site_name
    return default_headers or None


def build_openrouter_extra_body(
    llm_config: LLMConfig,
    config: GenerationConfig,
    *,
    resolved_max_output_tokens: int,
) -> dict[str, Any] | None:
    """Build OpenRouter-specific request payload additions."""
    extra: dict[str, Any] = {}

    config_extra = dict(config.extra)
    web_search_override = config_extra.pop("web_search_enabled", None)
    if config_extra:
        extra.update(config_extra)

    if llm_config.reasoning_enabled:
        reasoning: dict[str, Any] = {"enabled": True}
        if llm_config.reasoning_effort is not None:
            reasoning["effort"] = llm_config.reasoning_effort.value
        else:
            reasoning_tokens = (
                llm_config.reasoning_max_tokens
                or config.max_tokens
                or resolved_max_output_tokens
            )
            reasoning["max_tokens"] = reasoning_tokens
        extra["reasoning"] = reasoning

    web_search_enabled = (
        web_search_override
        if web_search_override is not None
        else llm_config.web_search_enabled
    )
    if web_search_enabled:
        web_plugin: dict[str, Any] = {
            "id": "web",
            "max_results": llm_config.web_search_max_results,
        }
        if llm_config.web_search_prompt:
            web_plugin["search_prompt"] = llm_config.web_search_prompt

        existing_plugins = list(extra.get("plugins", []))
        existing_plugins.append(web_plugin)
        extra["plugins"] = existing_plugins

    return extra or None


def build_chat_openai_kwargs(
    api_key: str,
    llm_config: LLMConfig,
    config: GenerationConfig,
    *,
    resolved_max_output_tokens: int,
    include_extra_body: bool,
) -> dict[str, Any]:
    """Build kwargs shared by LangChain ChatOpenAI clients."""
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": OPENROUTER_BASE_URL,
        "model": llm_config.model_name,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens or resolved_max_output_tokens,
        "default_headers": build_openrouter_default_headers(llm_config),
        # OpenRouter is not the official OpenAI Responses API and some parameters
        # such as `seed` are rejected when langchain-openai auto-switches there.
        "use_responses_api": False,
    }

    if config.top_p != 1.0:
        kwargs["top_p"] = config.top_p
    if config.frequency_penalty != 0.0:
        kwargs["frequency_penalty"] = config.frequency_penalty
    if config.presence_penalty != 0.0:
        kwargs["presence_penalty"] = config.presence_penalty
    if config.seed is not None:
        kwargs["seed"] = config.seed
    if config.stop_sequences:
        kwargs["stop_sequences"] = list(config.stop_sequences)
    if include_extra_body:
        extra_body = build_openrouter_extra_body(
            llm_config,
            config,
            resolved_max_output_tokens=resolved_max_output_tokens,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body

    return kwargs


def build_model_info(
    llm_config: LLMConfig,
    *,
    resolved_max_context_tokens: int,
    resolved_max_output_tokens: int,
) -> ModelInfo:
    """Build provider-facing model metadata from shared config."""
    model_id = llm_config.model_name
    costs = DEFAULT_MODEL_COSTS.get(model_id, (Decimal("0"), Decimal("0")))

    return ModelInfo(
        model_id=model_id,
        provider="langchain",
        display_name=model_id.split("/")[-1] if "/" in model_id else model_id,
        max_context_tokens=resolved_max_context_tokens,
        max_output_tokens=resolved_max_output_tokens,
        cost_per_1k_input=costs[0],
        cost_per_1k_output=costs[1],
        supports_system_prompt=True,
        supports_streaming=True,
    )
