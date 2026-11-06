"""
Stable scalar helpers backed by SciPy.

This module is the single import site for ``scipy.special`` in PLP.
Do not add circuit- or NAND-specific semantics here — keep those in
callers (e.g. :mod:`examples.soft_nand_circuit`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from scipy.special import expit, logsumexp

# Logistic sigmoid; same callable as :func:`scipy.special.expit`.
sigmoid = expit


def logmeanexp(a: Sequence[float]) -> float:
    """``log(mean_i exp(a[i]))`` in a numerically stable way.

    Implemented as ``logsumexp(log_weights) - log(n)`` with ``n = len(log_weights)``.
    An empty sequence returns ``-inf``, matching :func:`scipy.special.logsumexp` on
    no inputs.
    """
    n = len(a)
    if n == 0:
        return float("-inf")
    return float(logsumexp(a) - math.log(n))


def soft_nand(a: float, b: float) -> float:
    """Soft NAND factor: phi(a, b) = 1 - sigma(a) * sigma(b)."""
    return 1.0 - sigmoid(a) * sigmoid(b)


def log_soft_nand(a: float, b: float) -> float:
    """Log of the soft NAND factor, clamped for numerical safety."""
    return math.log(max(soft_nand(a, b), 1e-15))
