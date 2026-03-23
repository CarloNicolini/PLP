"""
Stateful helpers for interacting with LLMs the PLP way.

The Session class keeps a PLPState, routes every prompt through the
shared runtime (so handlers work), and records trace events.
"""

from __future__ import annotations

from typing import Any

from ._config import get_config, get_llm
from ._llm import LLMLike
from .runtime import run_prompt, ensure_llm
from .state import PLPState, TraceEvent


class Session:
    """
    Stateful wrapper around repeated prompt interactions.

    A Session owns:
      - an LLM client (shared across calls)
      - a default system prompt
      - a PLPState that accumulates trace events
    """

    def __init__(
        self,
        llm: LLMLike | None = None,
        *,
        system: str | None = None,
        state: PLPState | None = None,
        model: str | None = None,
        temperature: float | None = None,
        **llm_kwargs: Any,
    ):
        self.llm = _resolve_llm(
            llm,
            model=model,
            temperature=temperature,
            **llm_kwargs,
        )
        cfg = get_config()
        self.system_prompt = system if system is not None else cfg.system_prompt
        self.state = state.copy() if state is not None else PLPState()

    def ask(
        self,
        prompt: str,
        *,
        name: str | None = None,
        system: str | None = None,
    ) -> str:
        """
        Send a prompt through the session and record the TraceEvent.

        Args:
            prompt: User content to send to the LLM.
            name: Optional site name for tracing (defaults to "session.ask").
            system: Optional system prompt override for this call.
        """
        site = name or "session.ask"
        sys_prompt = system if system is not None else self.system_prompt
        self.state, output = run_prompt(
            self.llm,
            self.state,
            name=site,
            prompt=prompt,
            system_prompt=sys_prompt,
        )
        return output

    def trace(self) -> list[TraceEvent]:
        """Return a shallow copy of the execution trace."""
        return list(self.state.trace)

    def fork(self, *, share_llm: bool = True) -> "Session":
        """Create a new Session with a copied state."""
        return Session(
            llm=self.llm if share_llm else None,
            system=self.system_prompt,
            state=self.state.copy(),
        )

    def reset(self) -> None:
        """Reset the accumulated state and trace."""
        self.state = PLPState()


def session(
    *,
    llm: LLMLike | None = None,
    system: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    **llm_kwargs: Any,
) -> Session:
    """
    Convenience factory for Session objects.

    Args:
        llm: Optional pre-configured LLM client.
        system: Default system prompt.
        model: Override model name (if llm not supplied).
        temperature: Override sampling temperature.
        llm_kwargs: Additional kwargs passed to ChatOpenAI via get_llm.
    """
    return Session(
        llm=llm,
        system=system,
        model=model,
        temperature=temperature,
        **llm_kwargs,
    )


def _resolve_llm(
    llm: LLMLike | None,
    *,
    model: str | None = None,
    temperature: float | None = None,
    **llm_kwargs: Any,
) -> LLMLike:
    """Resolve an LLM client from explicit instance or global config."""
    if llm is not None:
        ensure_llm(llm)
        return llm

    cfg = get_config()
    kwargs = cfg.additional_kwargs.copy()
    kwargs.update(llm_kwargs)
    return get_llm(
        model=model if model is not None else cfg.model,
        temperature=temperature if temperature is not None else cfg.temperature,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        **kwargs,
    )

