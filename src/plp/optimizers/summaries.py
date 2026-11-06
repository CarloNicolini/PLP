"""Summary and posterior helpers for amortized PLP guides."""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field

from plp.state import PLPState


@dataclass
class InferenceSummary:
    """Post-inference digest fed back to an amortized guide.

    ``evidence_proxy_trajectory`` is a diagnostic signal such as
    ``logmeanexp_k(w_k^{(t)})`` from a particle run. It is useful for guide
    adaptation, but should not be interpreted as a literal ELBO unless the
    objective profile says so.
    """

    evidence_proxy_trajectory: list[float]
    factor_contributions: dict[str, float] = field(default_factory=dict)
    best_output: str = ""
    site_previews: dict[str, str] = field(default_factory=dict)
    objective_signal_name: str = "evidence_weighted_utility"
    evidence_proxy_name: str = "particle_logmeanexp"

    @property
    def J_trajectory(self) -> list[float]:
        """Backward-compatible alias for old examples and reports."""
        return self.evidence_proxy_trajectory


@dataclass(frozen=True)
class SummaryPosterior:
    """Weighted posterior estimate over extracted summary values."""

    support: dict[Hashable, float]
    normalizer: float
    num_particles: int
    extractor_name: str


def estimate_summary_posterior(
    particles: list[PLPState],
    extractor: Callable[[PLPState], Hashable],
    *,
    extractor_name: str = "summary_extractor",
) -> SummaryPosterior:
    """Estimate a posterior over summary variables extracted from traces.

    The estimate is a softmax over particle log-weights grouped by summary.
    This provides tractable marginal-like objects over semantic summaries even
    when full-string marginals are intractable.
    """
    if not particles:
        return SummaryPosterior(
            support={},
            normalizer=0.0,
            num_particles=0,
            extractor_name=extractor_name,
        )

    max_w = max(p.log_weight for p in particles)
    grouped: dict[Hashable, float] = {}
    for particle in particles:
        summary_value = extractor(particle)
        grouped[summary_value] = grouped.get(summary_value, 0.0) + math.exp(
            particle.log_weight - max_w
        )

    normalizer = sum(grouped.values())
    support = (
        {k: 0.0 for k in grouped}
        if normalizer == 0.0
        else {k: v / normalizer for k, v in grouped.items()}
    )

    return SummaryPosterior(
        support=support,
        normalizer=normalizer,
        num_particles=len(particles),
        extractor_name=extractor_name,
    )


# Backward-compatible name used by the earlier controller implementation.
RunSummary = InferenceSummary
