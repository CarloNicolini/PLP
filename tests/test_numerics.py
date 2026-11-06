"""Tests for :mod:`plp.numerics`."""

from __future__ import annotations

import math

import pytest

from plp.numerics import logmeanexp, logsumexp


def test_log_mean_exp_matches_manual() -> None:
    w = [1.0, 0.5, 0.25]
    assert logmeanexp(w) == pytest.approx(logsumexp(w) - math.log(len(w)))


def test_logmeanexp_empty() -> None:
    assert logmeanexp([]) == float("-inf")
