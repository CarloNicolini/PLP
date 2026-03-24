"""
Runtime helpers shared across PLP primitives.

The goal of this module is to keep prompt execution and trace logging
uniform across the primitive layer.
"""

from __future__ import annotations

from typing import Any

from ._llm import LLMLike, invoke_llm, llm_name, ainvoke_llm
from .handlers import check_handlers
from .state import PLPState, TraceEvent


def ensure_llm(llm: LLMLike) -> None:
    """Best-effort guard that the object looks like a synchronous LLM client."""
    if not hasattr(llm, "invoke"):
        raise TypeError(f"{llm!r} does not expose an 'invoke' method")


def ensure_async_llm(llm: Any) -> None:
    """Best-effort guard that the object supports async chat completion."""
    if not callable(getattr(llm, "ainvoke", None)):
        raise TypeError(f"{llm!r} does not expose a callable 'ainvoke' method")


def run_prompt(
    llm: LLMLike,
    state: PLPState,
    *,
    name: str,
    prompt: str,
    system_prompt: str | None = None,
    meta: dict[str, Any] | None = None,
) -> tuple[PLPState, str]:
    """
    Execute a prompt against an LLM and append the interaction to the trace.

    This function is the single entry-point used by all sampling helpers.
    It automatically checks active effect handlers, so replay/block/do
    semantics stay consistent no matter which API the user chooses.
    """
    ensure_llm(llm)
    cached_output, handler_used = check_handlers(name)
    output = (
        cached_output
        if cached_output is not None
        else invoke_llm(llm, prompt, system_prompt=system_prompt)
    )

    new_state = state.copy()
    event_meta: dict[str, Any] = {"llm": llm_name(llm)}
    if handler_used != "none":
        event_meta["handler"] = handler_used
    if meta:
        event_meta.update(meta)

    new_state.trace.append(
        TraceEvent(
            primitive=name,
            prompt=prompt,
            output=output,
            meta=event_meta,
        )
    )
    return new_state, output


async def arun_prompt(
    llm: Any,
    state: PLPState,
    *,
    name: str,
    prompt: str,
    system_prompt: str | None = None,
    meta: dict[str, Any] | None = None,
) -> tuple[PLPState, str]:
    """
    Async counterpart of :func:`run_prompt`: same handler semantics, ``ainvoke`` for the LLM.

    Replay / do / condition / block handlers are evaluated synchronously before awaiting
    the model, matching :func:`run_prompt`.
    """
    ensure_async_llm(llm)
    cached_output, handler_used = check_handlers(name)
    output = (
        cached_output
        if cached_output is not None
        else await ainvoke_llm(llm, prompt, system_prompt=system_prompt)
    )

    new_state = state.copy()
    event_meta: dict[str, Any] = {"llm": llm_name(llm)}
    if handler_used != "none":
        event_meta["handler"] = handler_used
    if meta:
        event_meta.update(meta)

    new_state.trace.append(
        TraceEvent(
            primitive=name,
            prompt=prompt,
            output=output,
            meta=event_meta,
        )
    )
    return new_state, output

