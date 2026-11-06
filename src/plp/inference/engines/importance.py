"""
Importance sampling engine.

Case A + tree-shaped factor cliques. Draws K independent forward
particles, applies all factors (accumulating log_weight), then performs
one systematic resample to concentrate particles on high-scoring joints.

Correct when:
  - Sample-parent graph is acyclic.
  - Factor cliques do not induce cycles in the moralised graph (tree-like).

For richer structures, use SMC (per-site weighting) or LoopyParticleBP.
"""

from __future__ import annotations

import random
from typing import Any

from plp.numerics import logmeanexp
from plp.state import PLPState

from ..graph import ChainGraph
from ..utils import systematic_resample as _utils_systematic_resample
from .base import append_engine_trace, Engine
from ._dag_sweep import (
    acyclic_order_or_raise,
    apply_newly_ready_factors,
    propose_site_for_particles,
)


def _systematic_resample(
    particles: list[PLPState],
    K: int,
    *,
    rng: "random.Random | None" = None,
) -> list[PLPState]:
    """Systematic resampling (scope ``is.resample``)."""
    import random  # noqa: F401 — forward ref for type hint

    return _utils_systematic_resample(particles, K, rng=rng, scope="is.resample")


class ImportanceSampling(Engine):
    """K-particle importance sampling.

    1. Forward-sample K particles in topological order of sample sites.
    2. Apply every factor once its parents are instantiated; the factor's
       log-weight accumulates on ``state.log_weight``.
    3. One systematic resample at the end so the returned particles
       represent the posterior.

    The per-particle log-weights are preserved on the resampled particles
    by zeroing them out (they have been absorbed into the resampling).
    Callers wanting the un-resampled posterior marginals can pass
    ``resample=False``.
    """

    name = "ImportanceSampling"

    def __init__(self, resample: bool = True) -> None:
        self.resample = resample

    async def run(
        self,
        graph: ChainGraph,
        *,
        K: int = 5,
        T: int = 1,  # unused
        init_state: PLPState | None = None,
        **sample_kwargs: Any,
    ) -> list[PLPState]:
        order = acyclic_order_or_raise(
            graph,
            message=(
                "ImportanceSampling requires an acyclic sample-parent graph. "
                "Use LoopyParticleBP"
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

        log_z = logmeanexp([p.log_weight for p in particles])
        append_engine_trace(
            particles,
            graph,
            engine="ImportanceSampling",
            primitive="engine:importance",
            prompt=f"(ImportanceSampling K={K})",
            k=K,
            extra_meta={"log_evidence": log_z},
        )

        if self.resample:
            particles = _systematic_resample(particles, K)

        return particles
