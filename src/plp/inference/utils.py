"""
Shared inference utilities.

Single source of truth for the numerical and graph-colouring helpers
that every engine and the legacy :mod:`plp.factor_graph` shim need.
Re-exports :func:`~plp.numerics.logsumexp` for legacy callers and tests.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from plp.handlers import rng_for
from plp.numerics import logsumexp
from plp.state import PLPState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .graph import ChainGraph


def systematic_resample(
    particles: list[PLPState],
    K: int,
    *,
    rng: random.Random | None = None,
    scope: str = "inference.resample",
) -> list[PLPState]:
    """Systematic resampling (low-variance). Resets log_weight to 0.

    Falls back to :func:`plp.handlers.rng_for` (scope ``scope``) when no
    ``rng`` is supplied so that ``with plp.seed(k): ...`` is honoured.
    """
    if not particles:
        return []

    if rng is None:
        rng = rng_for(scope)

    log_weights = [p.log_weight for p in particles]
    m = max(log_weights)
    if m == float("-inf"):
        indices = [i % len(particles) for i in range(K)]
    else:
        w = [math.exp(lw - m) for lw in log_weights]
        total = sum(w)
        normalised = [wi / total for wi in w]
        positions = [(rng.random() + i) / K for i in range(K)]
        indices = []
        cumulative = 0.0
        j = 0
        for pos in positions:
            while cumulative + normalised[j] < pos and j < len(normalised) - 1:
                cumulative += normalised[j]
                j += 1
            indices.append(j)

    out: list[PLPState] = []
    for idx in indices:
        new = particles[idx].copy()
        new.log_weight = 0.0
        out.append(new)
    return out


def build_adjacency(
    variable_names: list[str],
    factor_vars: list[list[str]],
) -> dict[str, set[str]]:
    """Pairwise adjacency induced by factor cliques.

    Each factor couples every pair of variables in its parent clique;
    this is the minimal adjacency used by the legacy ``FactorGraph``
    colouring (no moralisation of directed edges).
    """
    adj: dict[str, set[str]] = {v: set() for v in variable_names}
    for connected in factor_vars:
        for v in connected:
            for u in connected:
                if u != v:
                    adj[v].add(u)
    return adj


def colour_groups(
    variable_names: list[str],
    adj: dict[str, set[str]],
) -> list[list[str]]:
    """Greedy graph colouring.

    Nodes sharing a colour have no edges between them in ``adj`` and so
    can be re-sampled in parallel without interference.
    """
    colours: dict[str, int] = {}
    for v in variable_names:
        taken = {colours[u] for u in adj[v] if u in colours}
        c = 0
        while c in taken:
            c += 1
        colours[v] = c
    groups: dict[int, list[str]] = {}
    for v, c in colours.items():
        groups.setdefault(c, []).append(v)
    return list(groups.values())


def parallel_groups_for(graph: "ChainGraph") -> list[list[str]]:
    """Colour the moralised graph of a :class:`ChainGraph`.

    This is the correct version for engines like LoopyParticleBP: sites
    in the same group are conditionally independent given the rest,
    accounting for sample-parent edges, moralisation of co-parents, and
    factor cliques.
    """
    adj = graph.moralised_adjacency()
    return colour_groups(graph.site_names, adj)
