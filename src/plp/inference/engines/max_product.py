"""
Max-product engine (MAP inference).

Identical to ImportanceSampling in the forward / scoring phase, but
replaces systematic resampling with top-K selection: we keep the K
particles with the highest log-weight (argmax instead of random draw
proportional to weight).

Correct when the caller wants the most likely trace rather than
samples from the posterior. Works on the same graph topologies as
ImportanceSampling.

For cyclic parent graphs, we fall back to iterating K independent
Gibbs chains and returning the top-K by final log-weight — a
particle-based analogue of max-product BP on MRFs.
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


class MaxProduct(Engine):
    """Top-K selection over forward particles.

    Drops ImportanceSampling's stochastic resampling in favour of
    deterministic max selection. Tie-breaking: earlier index wins.
    """

    name = "MaxProduct"

    async def run(
        self,
        graph: ChainGraph,
        *,
        K: int = 5,
        T: int = 1,
        init_state: PLPState | None = None,
        **sample_kwargs: Any,
    ) -> list[PLPState]:
        order = acyclic_order_or_raise(
            graph,
            message=(
                "MaxProduct requires an acyclic sample-parent graph. "
                "Consider LoopyParticleBP with a max-aggregated variant."
            ),
        )

        base = init_state or PLPState()
        N = max(K * 2, 4)  # over-sample to have material for argmax
        particles = [base.copy() for _ in range(N)]
        applied: set[str] = set()

        for site_name in order:
            particles = await propose_site_for_particles(
                graph, site_name, particles, **sample_kwargs,
            )
            apply_newly_ready_factors(graph, particles, applied)

        # Top-K selection by log_weight.
        indexed = sorted(
            enumerate(particles), key=lambda pair: pair[1].log_weight, reverse=True,
        )
        top = [p for _, p in indexed[:K]]
        max_w = top[0].log_weight if top else float("-inf")

        append_engine_trace(
            top,
            graph,
            engine="MaxProduct",
            primitive="engine:max_product",
            prompt=f"(MaxProduct K={K} oversample={N})",
            k=K,
            extra_meta={"max_log_weight": max_w},
        )

        return top
