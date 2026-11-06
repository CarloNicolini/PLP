"""``seed`` / ``rng_for`` plumbing + LBP trace meta (live LLM)."""

from __future__ import annotations

import asyncio

import plp
from plp import PLPChatClient, PLPState, asample, factor, seed
from plp.inference.utils import systematic_resample
from plp.handlers import get_seed_context, rng_for
from plp.inference.engines.importance import _systematic_resample
from plp.inference.engines.lbp import LoopyParticleBP
from plp.inference.graph import ChainGraph


def test_rng_for_and_seed_context() -> None:
    a = rng_for()
    b = rng_for()
    assert a is not b

    with seed(7) as ctx:
        r1 = rng_for("x")
        r2 = rng_for("x")
        r3 = rng_for("y")
        assert r1 is r2
        assert r1 is not r3
        assert get_seed_context() is ctx


def test_seed_block_deterministic() -> None:
    def draw() -> list[float]:
        with seed(123):
            r = rng_for("draw")
            return [r.random() for _ in range(5)]

    assert draw() == draw()


def test_systematic_resample_deterministic_under_seed() -> None:
    particles = [PLPState(vars={"x": str(i)}, log_weight=float(i)) for i in range(5)]
    with seed(0):
        a = [p.vars["x"] for p in systematic_resample([p.copy() for p in particles], 5)]
    with seed(0):
        b = [p.vars["x"] for p in systematic_resample([p.copy() for p in particles], 5)]
    assert a == b


def test_importance_resample_respects_seed() -> None:
    particles = [PLPState(vars={"i": str(i)}, log_weight=float(i)) for i in range(6)]
    with seed(99):
        a = [p.vars["i"] for p in _systematic_resample([p.copy() for p in particles], 6)]
    with seed(99):
        b = [p.vars["i"] for p in _systematic_resample([p.copy() for p in particles], 6)]
    assert a == b


def test_lbp_trace_records_seed(llm: PLPChatClient) -> None:
    @asample(llm, name="x")
    def propose(state: PLPState) -> str:
        return "Say hi in one word."

    @factor(name="score", parents=["x"])
    def score(state: PLPState, x: str) -> float:
        return 0.0

    graph = ChainGraph.from_decorated(propose, score)
    with seed(17):
        particles = asyncio.run(LoopyParticleBP().run(graph, K=2, T=2))
    for p in particles:
        events = [e for e in p.trace if e.primitive == "engine:lbp"]
        assert events and events[-1].meta["seed"] == 17


def test_plp_seed_export() -> None:
    assert plp.seed is seed
