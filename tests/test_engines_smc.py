"""SMC engine — live LLM smoke."""

from __future__ import annotations

import asyncio

from plp import PLPChatClient, PLPState, asample, factor
from plp.inference.engines.smc import SMC
from plp.inference.graph import ChainGraph, FactorSite, SampleSite


def test_smc_chain_and_resampling(llm: PLPChatClient) -> None:
    @asample(llm, name="a")
    def a(state: PLPState) -> str:
        return "One letter only: A"

    @asample(llm, name="b", parents=("a",))
    def b(state: PLPState) -> str:
        return f"Prior was: {state.vars['a']}\nRespond with one letter only: B"

    @factor(name="phi")
    def phi(state: PLPState, b: str) -> float:
        return 0.2 if len(b.strip()) > 0 else -0.2

    graph = ChainGraph(
        sites=[
            SampleSite(name="a", fn=a),
            SampleSite(name="b", fn=b, parents=("a",)),
        ],
        factors=[FactorSite(name="phi", fn=phi, parents=("b",))],
    )
    particles = asyncio.run(SMC().run(graph, K=3))
    assert len(particles) == 3
    for p in particles:
        assert p.vars["a"].strip()
        assert p.vars["b"].strip()
