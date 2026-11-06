"""Objective profiles for amortized PLP guides."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DensityRegime(str, Enum):
    """Whether a guide/proposal family exposes tractable density terms."""

    DENSITY_EVALUABLE = "density_evaluable"
    SEMI_IMPLICIT = "semi_implicit"
    IMPLICIT = "implicit"


class SurrogateObjective(str, Enum):
    """Primary objective families for approximate inference in PLP."""

    ELBO = "elbo"
    IWAE = "iwae"
    BBVI = "bbvi"
    WAKE_SLEEP = "wake_sleep"
    EVIDENCE_WEIGHTED_UTILITY = "evidence_weighted_utility"


@dataclass(frozen=True)
class ObjectiveProfile:
    """Inference profile for a schema/spec family.

    This separates an optimization target from proxy diagnostics. For example,
    a particle log-mean-exp trajectory can be a useful evidence proxy without
    being a literal ELBO.
    """

    density_regime: DensityRegime
    candidate_objectives: tuple[SurrogateObjective, ...]
    primary_signal: str
    diagnostic_signals: tuple[str, ...] = ()
