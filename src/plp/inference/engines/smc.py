"""
Sequential Monte Carlo engine.

For long directed chains with per-site factors. Each site is proposed
for K particles, factors on the already-instantiated sites are applied,
then we resample. This concentrates compute on promising prefixes and
avoids the exponential particle collapse of plain importance sampling
on long chains.

Correct when:
  - Sample-parent graph is a DAG (topological order exists).
  - Factors are typically unary or short-range (pairwise between adjacent
    sites). For dense / long-range factor structures, LoopyParticleBP
    is usually a better fit.

Reference: Murphy ch. 23.5 on particle filtering for state-space models
and Del Moral's Sequential Monte Carlo Samplers (2006).
"""

from __future__ import annotations

from typing import Any

from plp.numerics import logmeanexp
from plp.state import PLPState

from ..graph import ChainGraph
from ..utils import systematic_resample

from .base import append_engine_trace, Engine
from ._dag_sweep import (
    acyclic_order_or_raise,
    apply_newly_ready_factors,
    propose_site_for_particles,
)


class SMC(Engine):
    """Per-site sequential Monte Carlo.

    For each site in topological order:
      1. Propose K samples (one per particle) via its @asample function.
      2. Apply every factor whose parents are now all instantiated and
         have not been applied yet — contributing to log_weight.
      3. Resample the K particles systematically.

    The final particles represent the posterior. Use with caution on
    graphs whose factor cliques skip over many sites: the effective
    sample size collapses when late factors disagree with early
    resampling decisions.
    """

    name = "SMC"

    def __init__(self, resample_each_site: bool = True) -> None:
        self.resample_each_site = resample_each_site

    async def run(
        self,
        graph: ChainGraph,
        *,
        K: int = 5,
        T: int = 1,  # unused (SMC is inherently single-pass)
        init_state: PLPState | None = None,
        **sample_kwargs: Any,
    ) -> list[PLPState]:
        order = acyclic_order_or_raise(
            graph,
            message=(
                "SMC requires an acyclic sample-parent graph. "
                "Use LoopyParticleBP for Case B (cyclic parents)."
            ),
        )

        base = init_state or PLPState()
        particles = [base.copy() for _ in range(K)]
        applied: set[str] = set()
        per_site_log_z: list[float] = []

        for site_name in order:
            particles = await propose_site_for_particles(
                graph, site_name, particles, **sample_kwargs,
            )
            apply_newly_ready_factors(graph, particles, applied)

            # Record per-site log-evidence before resampling.
            log_z = logmeanexp([p.log_weight for p in particles])
            per_site_log_z.append(log_z)

            if self.resample_each_site:
                particles = systematic_resample(particles, K)

        append_engine_trace(
            particles,
            graph,
            engine="SMC",
            primitive="engine:smc",
            prompt=f"(SMC K={K})",
            k=K,
            extra_meta={"per_site_log_evidence": per_site_log_z},
        )

        return particles
