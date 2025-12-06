"""
PLP Global Configuration.

Manages global settings for the PLP library, including:
- Default LLM model and parameters
- Global system prompts
- API keys and base URLs

Provides context managers for scoped configuration overrides.
"""

import os
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Iterator, Optional

from langchain_openai import ChatOpenAI


@dataclass
class PLPConfig:
    """Global configuration for PLP."""

    model: str = "gpt-4.1-mini"
    temperature: float = 0.7
    base_url: Optional[str] = None
    system_prompt: Optional[str] = None
    api_key: Optional[str] = None  # Falls back to env var if None
    additional_kwargs: dict = field(default_factory=dict)


# Singleton global config instance
_config = PLPConfig()

# Cache for LLM instances to avoid re-initialization overhead
_llm_cache: dict[str, ChatOpenAI] = {}


def configure(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Set global PLP configuration.

    Args:
        model: OpenAI model name (default: gpt-4.1-mini)
        temperature: Sampling temperature (default: 0.7)
        base_url: Optional API base URL
        system_prompt: Default system prompt for all calls
        api_key: OpenAI API key (optional, uses env var if None)
        **kwargs: Additional arguments passed to ChatOpenAI
    """
    global _config
    if model is not None:
        _config.model = model
    if temperature is not None:
        _config.temperature = temperature
    if base_url is not None:
        _config.base_url = base_url
    if system_prompt is not None:
        _config.system_prompt = system_prompt
    if api_key is not None:
        _config.api_key = api_key

    _config.additional_kwargs.update(kwargs)

    # Clear cache when config changes significantly
    _llm_cache.clear()


def get_config() -> PLPConfig:
    """Get current global configuration."""
    return _config


def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> ChatOpenAI:
    """
    Get or create an LLM instance based on current config or overrides.

    Args:
        model: Override model name
        temperature: Override temperature
        base_url: Override base URL
        api_key: Override API key
        **kwargs: Additional overrides

    Returns:
        ChatOpenAI instance
    """
    # Resolve parameters against global config
    cfg = _config
    eff_model = model if model is not None else cfg.model
    eff_temp = temperature if temperature is not None else cfg.temperature
    eff_base_url = base_url if base_url is not None else cfg.base_url
    eff_api_key = api_key if api_key is not None else cfg.api_key

    # Merge kwargs
    eff_kwargs = cfg.additional_kwargs.copy()
    eff_kwargs.update(kwargs)

    # Create cache key
    # Note: we don't cache based on system_prompt as that's handled at call time
    cache_key = (
        eff_model,
        eff_temp,
        eff_base_url,
        eff_api_key,
        tuple(sorted(eff_kwargs.items())),
    )

    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    # Create new instance
    # Only pass api_key if it's explicitly set in config
    # Otherwise let LangChain/OpenAI SDK find it in env vars
    init_kwargs = {
        "model": eff_model,
        "temperature": eff_temp,
        "base_url": eff_base_url,
        **eff_kwargs,
    }

    if eff_api_key is not None:
        init_kwargs["api_key"] = eff_api_key

    llm = ChatOpenAI(**init_kwargs)

    _llm_cache[cache_key] = llm
    return llm


@contextmanager
def model(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    system: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> Iterator[PLPConfig]:
    """
    Context manager for scoped LLM configuration.

    Temporarily overrides global settings within the `with` block.

    Args:
        model: Override model name
        temperature: Override temperature
        system: Override system prompt
        base_url: Override base URL
        **kwargs: Additional overrides

    Example:
        >>> with plp.model("gpt-4o", temperature=0.2):
        ...     plp.sample("Generate code")
    """
    global _config
    original_config = PLPConfig(
        model=_config.model,
        temperature=_config.temperature,
        base_url=_config.base_url,
        system_prompt=_config.system_prompt,
        api_key=_config.api_key,
        additional_kwargs=_config.additional_kwargs.copy(),
    )

    # Apply overrides
    configure(
        model=model,
        temperature=temperature,
        base_url=base_url,
        system_prompt=system,
        **kwargs,
    )

    try:
        yield _config
    finally:
        # Restore original config
        _config = original_config
        # Note: we don't clear cache here as we might want to reuse instances
        # created during the context later, and restoring config doesn't invalidate
        # the validity of cached instances, just which one is "default"
