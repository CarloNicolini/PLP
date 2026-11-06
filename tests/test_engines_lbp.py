"""LoopyParticleBP — resampling helpers + live cyclic graph."""

from __future__ import annotations

import asyncio

import pytest

from plp import PLPChatClient, PLPState, asample, factor
from plp.inference.engines.lbp import (
    LoopyParticleBP,
    _parallel_groups,
    systematic_resample,
)
from plp.inference.graph import ChainGraph, FactorSite, SampleSite


def test_systematic_resample_high_weight_dominated() -> None:
    low = PLPState(vars={"x": "low"}, log_weight=-10.0)
    high = PLPState(vars={"x": "high"}, log_weight=10.0)
    out = systematic_resample([low, high], 10)
    assert sum(1 for s in out if s.vars["x"] == "high") >= 8


def test_parallel_groups_separate_factor_clique() -> None:
    g = ChainGraph(
        sites=[
            SampleSite(name="a", fn=lambda s: "a"),
            SampleSite(name="b", fn=lambda s: "b"),
        ],
        factors=[FactorSite(
            name="phi", fn=lambda s, a, b: 0.0, parents=("a", "b"),
        )],
    )
    groups = _parallel_groups(g)
    group_of = {n: i for i, grp in enumerate(groups) for n in grp}
    assert group_of["a"] != group_of["b"]


def test_lbp_cyclic_two_site_debate(llm: PLPChatClient) -> None:
    """Case B: mutual parents; LBP runs Gibbs-style outer iterations."""

    @asample(llm, name="pro")
    def pro(state: PLPState, topic: str) -> str:
        opp = state.vars.get("con", "")
        extra = f"\nOpponent said: {opp}" if opp else ""
        return (
            f"Topic: {topic}\nArgue FOR in two short sentences. Start with PRO:{extra}"
        )

    @asample(llm, name="con")
    def con(state: PLPState, topic: str) -> str:
        opp = state.vars.get("pro", "")
        extra = f"\nOpponent said: {opp}" if opp else ""
        return (
            f"Topic: {topic}\nArgue AGAINST in two short sentences. Start with CON:{extra}"
        )

    graph = ChainGraph(
        sites=[
            SampleSite(name="pro", fn=pro, parents=("con",)),
            SampleSite(name="con", fn=con, parents=("pro",)),
        ],
    )
    assert graph.topological_order() is None

    particles = asyncio.run(
        LoopyParticleBP().run(graph, K=2, T=5, topic="cats vs dogs as pets"),
    )
    assert len(particles) == 2
    for p in particles:
        assert p.vars["pro"].strip()
        assert p.vars["con"].strip()
