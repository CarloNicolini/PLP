"""
Engine base protocol.

Every concrete engine implements ``async def run(graph, *, K, T, init_state, **kwargs)``
and returns a list of ``PLPState`` particles. The engines differ only in
how they traverse the graph, score factors, and combine particles.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable

from plp.handlers import get_seed_context
from plp.state import PLPState, TraceEvent

from ..graph import ChainGraph, FactorSite, SampleSite


@runtime_checkable
class Engine(Protocol):
    """Inference engine contract."""

    name: str

    async def run(
        self,
        graph: ChainGraph,
        *,
        K: int = 1,
        T: int = 1,
        init_state: PLPState | None = None,
        **sample_kwargs: Any,
    ) -> list[PLPState]:
        ...


async def invoke_sample(
    site: SampleSite, state: PLPState, **kwargs: Any
) -> tuple[PLPState, str]:
    """Call a sample site's fn, awaiting if it returns a coroutine.

    @asample decorators return async functions; @sample decorators return
    sync ones. Engines should not care.
    """
    result = site.fn(state.copy(), **kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result


def apply_factor(factor_site: FactorSite, state: PLPState) -> PLPState:
    """Apply a factor to a state, reading its parents from ``state.vars``.

    @factor decorators are synchronous by construction in PLP.
    """
    args = [state.vars.get(parent, "") for parent in factor_site.parents]
    return factor_site.fn(state, *args)


def append_engine_trace(
    particles: list[PLPState],
    graph: ChainGraph,
    *,
    engine: str,
    primitive: str,
    prompt: str,
    k: int,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    """Append the same engine-summary ``TraceEvent`` to every particle."""
    seed_ctx = get_seed_context()
    for p in particles:
        meta: dict[str, Any] = {
            "engine": engine,
            "K": k,
            "seed": seed_ctx.seed if seed_ctx else None,
            "output_site": graph.output_site,
        }
        if extra_meta:
            meta.update(extra_meta)
        if graph.output_site is not None and graph.output_site in p.vars:
            meta["output_value"] = p.vars[graph.output_site]
        p.trace.append(
            TraceEvent(
                primitive=primitive,
                prompt=prompt,
                output="",
                meta=meta,
            )
        )
