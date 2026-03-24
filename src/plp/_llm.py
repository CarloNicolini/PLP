"""Utility helpers for interacting with LLM-like objects."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


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


def invoke_llm(llm: LLMLike | Any, prompt: str, system_prompt: str | None = None) -> str:
    """
    Send a prompt to the LLM using OpenAI-style chat messages (dicts).

    Compatible with :class:`plp.client.PLPChatClient` and any object exposing
    ``invoke(messages) -> response`` where ``response`` has ``content``.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = llm.invoke(messages)
    return getattr(response, "content", str(response))


async def ainvoke_llm(
    llm: AsyncLLMLike | Any, prompt: str, system_prompt: str | None = None
) -> str:
    """
    Async chat completion using OpenAI-style messages.

    Compatible with :class:`plp.client.PLPChatClient` and any object exposing
    ``async def ainvoke(messages) -> response`` where ``response`` has ``content``.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = await llm.ainvoke(messages)
    return getattr(response, "content", str(response))
