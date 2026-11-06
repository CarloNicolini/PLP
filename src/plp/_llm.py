"""Utility helpers for interacting with LLM-like objects."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from tenacity import (
    AsyncRetrying,
    Retrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMLike(Protocol):
    """Protocol for synchronous LLM-like objects used inside PLP (``invoke``)."""

    model: str | None
    temperature: float | None

    def invoke(self, messages: list[Any]) -> Any:
        ...


@runtime_checkable
class AsyncLLMLike(Protocol):
    """
    Protocol for async LLM-like objects (:meth:`ainvoke`), e.g. :class:`plp.client.PLPChatClient`.

    Used by :func:`plp.primitives.asample` and :func:`plp.runtime.arun_prompt`.
    Implementations may also provide :meth:`invoke` for synchronous code paths.
    """

    model: str | None
    temperature: float | None

    async def ainvoke(self, messages: list[Any]) -> Any:
        ...


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------


@dataclass
class RetryConfig:
    """Tunable retry policy for LLM calls.

    Controls how ``invoke_llm`` / ``ainvoke_llm`` guard against transient
    provider errors (rate limits, connection blips, 5xx). Disable via
    ``enabled = False`` in tests that do not want retry delays.
    """

    max_attempts: int = 6
    max_wait_s: float = 30.0
    multiplier: float = 1.0
    enabled: bool = True


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


_retry_config = RetryConfig(
    max_attempts=_env_int("PLP_LLM_MAX_ATTEMPTS", 6),
    max_wait_s=_env_float("PLP_LLM_MAX_WAIT_S", 30.0),
)


def get_retry_config() -> RetryConfig:
    """Return the current module-level retry policy."""
    return _retry_config


def set_retry_config(config: RetryConfig) -> None:
    """Replace the module-level retry policy (e.g. from tests)."""
    global _retry_config
    _retry_config = config


def _retryable_exceptions() -> tuple[type[BaseException], ...]:
    """Exception types worth retrying.

    Resolved lazily so that this module loads even if ``openai`` is not
    installed or is replaced by a custom client.
    """
    exc_types: list[type[BaseException]] = [TimeoutError, ConnectionError]
    try:
        import openai

        for attr in (
            "RateLimitError",
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "APIStatusError",
        ):
            cls = getattr(openai, attr, None)
            if isinstance(cls, type) and issubclass(cls, BaseException):
                exc_types.append(cls)
    except ImportError:  # pragma: no cover - openai is a declared dep
        pass
    return tuple(exc_types)


def llm_name(llm: Any) -> str:
    """Build a human-readable representation of the LLM."""
    model = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    temperature = getattr(llm, "temperature", None)
    parts: list[str] = []

    if model:
        parts.append(str(model))
    else:
        parts.append(llm.__class__.__name__)

    if temperature is not None:
        parts.append(f"{temperature}")

    return "@".join(parts)


def _build_messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def invoke_llm_with_attempts(
    llm: LLMLike | Any, prompt: str, system_prompt: str | None = None
) -> tuple[str, int]:
    """
    Send a prompt to the LLM and report how many attempts were made.

    Wraps ``llm.invoke`` with a tenacity retry loop against transient
    provider errors. Returns ``(content, attempts)`` where ``attempts`` is
    the number of calls actually made (>= 1). ``invoke_llm`` is the
    string-only convenience form.
    """
    messages = _build_messages(prompt, system_prompt)
    config = get_retry_config()

    if not config.enabled or config.max_attempts <= 1:
        response = llm.invoke(messages)
        return getattr(response, "content", str(response)), 1

    retrying = Retrying(
        retry=retry_if_exception_type(_retryable_exceptions()),
        stop=stop_after_attempt(config.max_attempts),
        wait=wait_random_exponential(
            multiplier=config.multiplier, max=config.max_wait_s
        ),
        reraise=True,
    )

    attempts = 0
    last_response: Any = None

    try:
        for attempt in retrying:
            with attempt:
                attempts += 1
                last_response = llm.invoke(messages)
    except RetryError as err:  # pragma: no cover - reraise=True avoids this
        raise err.last_attempt.exception()  # type: ignore[misc]

    return getattr(last_response, "content", str(last_response)), attempts


async def ainvoke_llm_with_attempts(
    llm: AsyncLLMLike | Any, prompt: str, system_prompt: str | None = None
) -> tuple[str, int]:
    """
    Async counterpart of :func:`invoke_llm_with_attempts`.

    Wraps ``llm.ainvoke`` with a tenacity ``AsyncRetrying`` loop. Returns
    ``(content, attempts)``.
    """
    messages = _build_messages(prompt, system_prompt)
    config = get_retry_config()

    if not config.enabled or config.max_attempts <= 1:
        response = await llm.ainvoke(messages)
        return getattr(response, "content", str(response)), 1

    retrying = AsyncRetrying(
        retry=retry_if_exception_type(_retryable_exceptions()),
        stop=stop_after_attempt(config.max_attempts),
        wait=wait_random_exponential(
            multiplier=config.multiplier, max=config.max_wait_s
        ),
        reraise=True,
    )

    attempts = 0
    last_response: Any = None

    try:
        async for attempt in retrying:
            with attempt:
                attempts += 1
                last_response = await llm.ainvoke(messages)
    except RetryError as err:  # pragma: no cover - reraise=True avoids this
        raise err.last_attempt.exception()  # type: ignore[misc]

    return getattr(last_response, "content", str(last_response)), attempts


def invoke_llm(
    llm: LLMLike | Any, prompt: str, system_prompt: str | None = None
) -> str:
    """String-returning convenience wrapper around :func:`invoke_llm_with_attempts`."""
    output, _ = invoke_llm_with_attempts(llm, prompt, system_prompt=system_prompt)
    return output


async def ainvoke_llm(
    llm: AsyncLLMLike | Any, prompt: str, system_prompt: str | None = None
) -> str:
    """String-returning convenience wrapper around :func:`ainvoke_llm_with_attempts`."""
    output, _ = await ainvoke_llm_with_attempts(
        llm, prompt, system_prompt=system_prompt
    )
    return output
