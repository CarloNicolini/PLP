"""
Forward engine for single pass inference over a Bayes net in topological order, no resampling.

Correct when the sample-parent graph is acyclic (Bayes net).
If factors are present, they are applied but weights are ignored.

The returned particles (PLPStates) are independent forward samples from pi_D.
"""

from __future__ import annotations

from typing import Any

from plp.state import PLPState

from ..graph import ChainGraph
from .base import append_engine_trace, Engine
from ._dag_sweep import (
    acyclic_order_or_raise,
    apply_newly_ready_factors,
    propose_site_for_particles,
)


class Forward(Engine):
    """Forward pass over a Bayes net.

    Draws K independent particles by iterating over the topological
    order of sample sites. Factors are applied (so log_weight accumulates)
    but no resampling or reweighting is performed — the caller decides
    what to do with the weights.

    Requires graph.topological_order() to be non-None. Raises
    ValueError otherwise.
    """

    name = "Forward"

    async def run(
        self,
        graph: ChainGraph,
        *,
        K: int = 1,
        T: int = 1,  # unused; kept for signature compatibility
        init_state: PLPState | None = None,
        **sample_kwargs: Any,
    ) -> list[PLPState]:
        order = acyclic_order_or_raise(
            graph,
            message=(
                "Forward engine requires an acyclic sample-parent graph. "
                "Use LoopyParticleBP for Case B (cyclic parents)."
            ),
        )

        base = init_state or PLPState()
        particles = [base.copy() for _ in range(K)]
        applied: set[str] = set()

        for site_name in order:
            particles = await propose_site_for_particles(
                graph, site_name, particles, **sample_kwargs,
            )
            apply_newly_ready_factors(graph, particles, applied)

        append_engine_trace(
            particles,
            graph,
            engine="Forward",
            primitive="engine:forward",
            prompt=f"(Forward K={K})",
            k=K,
        )

        return particles
