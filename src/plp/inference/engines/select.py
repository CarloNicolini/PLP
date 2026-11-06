"""
Topology-driven engine selection.

``choose_engine(graph)`` returns the appropriate Engine class by
inspecting the sample-parent graph and the moralised factor graph.

Rules (first match wins):
  1. If prefer_map is set → MaxProduct.
  2. If the sample-parent graph has a cycle → LoopyParticleBP
     (Case B: pi_D is an MRF, Gibbs iteration required).
  3. If there are no factors and parents are acyclic → Forward
     (Case A: pure Bayes net, single pass suffices).
  4. If the moralised graph (sample edges + factor cliques) has a cycle
     → LoopyParticleBP (Case C loopy: factors couple sites cyclically).
  5. If the sample-parent graph is a long chain with per-site factors
     → SMC (Case A + per-site weights).
  6. Otherwise → ImportanceSampling.

Users who disagree with the heuristic pass ``engine=<EngineClass>``
explicitly or call the engine directly.
"""

from __future__ import annotations

from .base import Engine
from .forward import Forward
from .importance import ImportanceSampling
from .lbp import LoopyParticleBP
from .max_product import MaxProduct
from .smc import SMC
from ..graph import ChainGraph


def _is_long_chain(graph: ChainGraph) -> bool:
    """True iff the sample-parent graph is a single directed path of at least 3 sites."""
    order = graph.topological_order()
    if order is None or len(order) < 3:
        return False
    adj = graph.sample_parent_adjacency()
    # Each internal site has in-degree and out-degree of 1; ends have exactly one.
    for i, name in enumerate(order):
        successors = adj[name]
        n_out = len(successors)
        n_in = sum(1 for other in graph.site_names if name in adj[other])
        if i == 0 and (n_in != 0 or n_out != 1):
            return False
        if 0 < i < len(order) - 1 and (n_in != 1 or n_out != 1):
            return False
        if i == len(order) - 1 and (n_in != 1 or n_out != 0):
            return False
    return True


def _factors_are_per_site(graph: ChainGraph) -> bool:
    """True iff every factor is unary (attached to a single site)."""
    return all(len(f.parents) == 1 for f in graph.factors)


def choose_engine(graph: ChainGraph, *, prefer_map: bool = False) -> type[Engine]:
    """Pick the right engine for the graph's topology."""
    if prefer_map:
        return MaxProduct

    if graph.has_cycle_in_parents():
        # Case B: pi is an MRF; Forward / IS / SMC all assume acyclicity.
        return LoopyParticleBP

    if not graph.factors:
        # Case A: pure Bayes net.
        return Forward

    if graph.has_cycle_in_moralised_graph():
        # Case C loopy: factor cliques couple the DAG cyclically.
        return LoopyParticleBP

    if _is_long_chain(graph) and _factors_are_per_site(graph):
        # Case A + per-site weights along a chain.
        return SMC

    return ImportanceSampling
