"""Compatibility imports for the old controller module path.

New code should import generic guide abstractions from ``plp.optimizers`` (or
the more specific modules below). Soft-NAND circuit schemas now live in the
application layer, not in the core optimizer library.
"""

from __future__ import annotations

from .base import AmortizedGuide, AmortizedInferer, ProposalFamily, SpecFamily
from .objectives import DensityRegime, ObjectiveProfile, SurrogateObjective
from .specs import DeterministicFactor, FactorSpec, JudgeFactor, ProgramSpec, SampleSite
from .summaries import (
    InferenceSummary,
    RunSummary,
    SummaryPosterior,
    estimate_summary_posterior,
)

__all__ = [
    "AmortizedGuide",
    "AmortizedInferer",
    "DensityRegime",
    "DeterministicFactor",
    "FactorSpec",
    "InferenceSummary",
    "JudgeFactor",
    "ObjectiveProfile",
    "ProgramSpec",
    "ProposalFamily",
    "RunSummary",
    "SampleSite",
    "SpecFamily",
    "SummaryPosterior",
    "SurrogateObjective",
    "estimate_summary_posterior",
]
