"""ImportanceSampling / MaxProduct — logsumexp numerics + live smoke."""

from __future__ import annotations

import asyncio
import math

import pytest

from plp import PLPChatClient, PLPState, asample, factor
from plp.numerics import logsumexp
from plp.inference.engines.importance import ImportanceSampling
from plp.inference.engines.max_product import MaxProduct
from plp.inference.graph import ChainGraph, FactorSite, SampleSite


def test_logsumexp_basic() -> None:
    assert logsumexp([]) == float("-inf")
    assert logsumexp([3.0]) == pytest.approx(3.0)
    assert logsumexp([1.0, 1.0]) == pytest.approx(1.0 + math.log(2))


def test_importance_sampling_smoke(llm: PLPChatClient) -> None:
    @asample(llm, name="x")
    def x(state: PLPState) -> str:
        return "Reply with one word: hello"

    @factor(name="phi")
    def phi(state: PLPState, x: str) -> float:
        return 0.1 * min(len(x.strip()), 80)

    graph = ChainGraph(
        sites=[SampleSite(name="x", fn=x)],
        factors=[FactorSite(name="phi", fn=phi, parents=("x",))],
        output_site="x",
    )
    particles = asyncio.run(ImportanceSampling().run(graph, K=3))
    assert len(particles) == 3
    ev = [e for e in particles[0].trace if e.primitive == "engine:importance"]
    assert len(ev) == 1
    assert "log_evidence" in ev[0].meta


def test_max_product_orders_by_weight(llm: PLPChatClient) -> None:
    @asample(llm, name="x")
    def x(state: PLPState) -> str:
        return "Reply with exactly: OK"

    @factor(name="phi")
    def phi(state: PLPState, x: str) -> float:
        return 1.5 if "ok" in x.lower() else -1.0

    graph = ChainGraph(
        sites=[SampleSite(name="x", fn=x)],
        factors=[FactorSite(name="phi", fn=phi, parents=("x",))],
        output_site="x",
    )
    particles = asyncio.run(MaxProduct().run(graph, K=3))
    assert len(particles) == 3
    ws = [p.log_weight for p in particles]
    assert ws == sorted(ws, reverse=True)
