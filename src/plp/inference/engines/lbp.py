"""
Loopy particle belief propagation.

Case B (cyclic parents) and loopy Case C (cyclic moralised graph).

This is the particle-based analogue of loopy sum-product BP (Murphy ch.
22). We maintain K particles — each a full assignment to all sample
sites — and iterate:

  1. SCORE:     apply every factor, accumulating log-weight on each particle.
  2. RESAMPLE:  systematic resample so the population concentrates on
                high-scoring joints.
  3. REPROPOSE: re-sample every site using @asample, with state.vars set
                to the particle's current assignment. Conditionally
                independent sites are re-proposed concurrently.

Across rounds this is Gibbs sampling on an implicit MRF defined by the
conditional kernels K_i(y_i | y_{pa(i)}). The log-evidence diagnostic
logsumexp(w) - log K is recorded in the trace.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from plp.numerics import logmeanexp
from plp.state import PLPState

from ..graph import ChainGraph

from ..utils import (
    parallel_groups_for as _parallel_groups,
    systematic_resample as _utils_systematic_resample,
)
from .base import append_engine_trace, apply_factor, invoke_sample, Engine


def systematic_resample(
    particles: list[PLPState],
    K: int,
    *,
    rng: random.Random | None = None,
) -> list[PLPState]:
    """Systematic resampling (low-variance), scoped to ``lbp.resample``."""
    return _utils_systematic_resample(particles, K, rng=rng, scope="lbp.resample")


async def repropose_by_groups(
    graph: ChainGraph,
    particles: list[PLPState],
    groups: list[list[str]],
    sample_kwargs: dict[str, Any],
) -> list[PLPState]:
    """Concurrent proposals per parallel group, in order of ``groups``."""
    K = len(particles)
    for group in groups:
        tasks = []
        for name in group:
            site = graph.sites_by_name[name]
            for p in particles:
                tasks.append(invoke_sample(site, p, **sample_kwargs))
        results = await asyncio.gather(*tasks)
        idx = 0
        for name in group:
            for i in range(K):
                new_state, output = results[idx]
                particles[i] = new_state
                particles[i].vars[name] = output
                idx += 1
    return particles


class LoopyParticleBP(Engine):
    """Particle-based loopy BP: T rounds of score / resample / re-propose."""

    name = "LoopyParticleBP"

    async def run(
        self,
        graph: ChainGraph,
        *,
        K: int = 5,
        T: int = 3,
        init_state: PLPState | None = None,
        **sample_kwargs: Any,
    ) -> list[PLPState]:
        base = init_state or PLPState()
        groups = _parallel_groups(graph)

        particles = [base.copy() for _ in range(K)]
        particles = await repropose_by_groups(graph, particles, groups, sample_kwargs)
        log_evidence: list[float] = []

        for t in range(T):
            particles = self._score_all_factors(graph, particles)
            log_z = logmeanexp([p.log_weight for p in particles])
            log_evidence.append(log_z)
            particles = systematic_resample(particles, K)
            if t < T - 1:
                particles = await repropose_by_groups(
                    graph, particles, groups, sample_kwargs,
                )

        particles = self._score_all_factors(graph, particles)

        append_engine_trace(
            particles,
            graph,
            engine="LoopyParticleBP",
            primitive="engine:lbp",
            prompt=f"(LoopyParticleBP T={T} K={K})",
            k=K,
            extra_meta={"T": T, "log_evidence": log_evidence},
        )

        return particles

    # -- internals --

    @staticmethod
    def _score_all_factors(
        graph: ChainGraph, particles: list[PLPState]
    ) -> list[PLPState]:
        out = []
        for p in particles:
            state = p
            for factor in graph.factors:
                state = apply_factor(factor, state)
            out.append(state)
        return out
