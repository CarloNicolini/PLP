"""Utility helpers for interacting with LLM-like objects."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import HumanMessage, SystemMessage


@runtime_checkable
class LLMLike(Protocol):
    """Protocol for LLM-like objects used inside PLP."""

    model: str | None
    temperature: float | None

    def invoke(self, messages: list[Any]) -> Any:
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
    Send a prompt to the LLM using ChatOpenAI-style messaging.

    This mirrors the previous helper behavior for invoking an LLM.
    """
    messages: list[Any] = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))

    messages.append(HumanMessage(content=prompt))

    response = llm.invoke(messages)
    return getattr(response, "content", str(response))
