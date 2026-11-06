"""Forward engine — live LLM checks."""

from __future__ import annotations

import asyncio

import pytest

from plp import PLPChatClient, PLPState, asample, factor
from plp.inference.engines.forward import Forward
from plp.inference.graph import ChainGraph, FactorSite, SampleSite


def test_forward_rejects_directed_cycle(llm: PLPChatClient) -> None:
    """Forward requires a DAG; cycles are rejected before any sampling."""

    @asample(llm, name="x", parents=("y",))
    def x(state: PLPState) -> str:
        return "x"

    @asample(llm, name="y", parents=("x",))
    def y(state: PLPState) -> str:
        return "y"

    graph = ChainGraph(
        sites=[
            SampleSite(name="x", fn=x, parents=("y",)),
            SampleSite(name="y", fn=y, parents=("x",)),
        ],
    )
    with pytest.raises(ValueError, match="acyclic"):
        asyncio.run(Forward().run(graph, K=1))


def test_forward_chain_and_unary_factor(llm: PLPChatClient) -> None:
    """Two-site chain with a unary factor; parent context is visible to the child."""
    question = "What is 2 + 2? Put only one digit on the last line."

    @asample(llm, name="work")
    def work(state: PLPState, question: str = "") -> str:
        return f"Solve briefly.\n{question}"

    @asample(llm, name="final", parents=("work",))
    def final(state: PLPState, question: str = "") -> str:
        prior = state.vars.get("work", "")
        return f"{question}\nWork:\n{prior}\nLast line: single digit only."

    @factor(name="digit_bonus")
    def digit_bonus(state: PLPState, final: str) -> float:
        return 0.25 if any(c.isdigit() for c in final) else -0.25

    graph = ChainGraph(
        sites=[
            SampleSite(name="work", fn=work),
            SampleSite(name="final", fn=final, parents=("work",)),
        ],
        factors=[FactorSite(name="digit_bonus", fn=digit_bonus, parents=("final",))],
        output_site="final",
    )
    particles = asyncio.run(Forward().run(graph, K=2, question=question))
    assert len(particles) == 2
    for p in particles:
        assert p.vars["work"].strip()
        assert p.vars["final"].strip()
        assert "4" in p.vars["final"].replace(" ", "")
