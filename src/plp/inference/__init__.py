"""
PLP inference package.

A PLP program induces a chain graph (Lauritzen & Wermuth, 1989):

    p(tau) ∝ pi(tau) · Phi(tau)

where pi is the proposal (induced by @sample sites with declared parents)
and Phi is the potential (induced by @factor cliques).

Depending on the declared structure the joint can be:

  — a directed Bayes net if parents form a DAG and there are no
  factors. Exact samples come from a single forward pass.

  - an MRF if parents form a cycle (sites read each other across
  rounds). Samples come from Gibbs iteration.

  — a chain graph in general: some parts are directed, others are
  coupled undirectedly by factor cliques or parent cycles.

This package provides:

- ``ChainGraph``: the pure data structure encoding (i) the directed
  sample-parent graph, and (ii) the undirected factor cliques.
- Engines in ``plp.inference.engines``: Forward, ImportanceSampling, SMC,
  LoopyParticleBP, MaxProduct. Each consumes a ChainGraph and returns
  ``list[PLPState]`` particles.
- ``choose_engine``: a topology-driven selector that picks the right
  engine from the graph's class.
"""

from .engines import (
    SMC,
    Engine,
    Forward,
    ImportanceSampling,
    LoopyParticleBP,
    MaxProduct,
    systematic_resample,
)
from .graph import ChainGraph, FactorSite, SampleSite
from .engines.select import choose_engine

__all__ = [
    "ChainGraph",
    "Engine",
    "FactorSite",
    "Forward",
    "ImportanceSampling",
    "LoopyParticleBP",
    "MaxProduct",
    "SMC",
    "SampleSite",
    "choose_engine",
    "systematic_resample",
]
