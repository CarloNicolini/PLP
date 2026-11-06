"""Shared topological DAG sweep for particle engines (internal)."""

from __future__ import annotations

import asyncio
from typing import Any

from plp.state import PLPState

from ..graph import ChainGraph
from .base import apply_factor, invoke_sample


def acyclic_order_or_raise(graph: ChainGraph, *, message: str) -> list[str]:
    order = graph.topological_order()
    if order is None:
        raise ValueError(message)
    return order


async def propose_site_for_particles(
    graph: ChainGraph,
    site_name: str,
    particles: list[PLPState],
    **sample_kwargs: Any,
) -> list[PLPState]:
    site = graph.sites_by_name[site_name]
    tasks = [invoke_sample(site, p, **sample_kwargs) for p in particles]
    results = await asyncio.gather(*tasks)
    out: list[PLPState] = []
    for new_state, output in results:
        new_state.vars[site_name] = output
        out.append(new_state)
    return out


def apply_newly_ready_factors(
    graph: ChainGraph, particles: list[PLPState], applied: set[str],
) -> None:
    if not particles:
        return
    for factor in graph.factors:
        if factor.name in applied:
            continue
        if all(p in particles[0].vars for p in factor.parents):
            particles[:] = [apply_factor(factor, p) for p in particles]
            applied.add(factor.name)
