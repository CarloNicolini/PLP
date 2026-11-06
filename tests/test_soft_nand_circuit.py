"""Tests for `examples.soft_nand_circuit`: compile + minimal live run."""

from __future__ import annotations

import asyncio
import math

from plp import PLPChatClient
from plp.inference.graph import ChainGraph
from soft_nand_circuit import (
    CircuitSpec,
    DeterministicFactor,
    SampleSite,
    _MINIMAL_SPEC,
    build_graph,
    run_spec,
)


def test_build_graph_compiles_spec(llm: PLPChatClient) -> None:
    spec = CircuitSpec(
        sites=[
            SampleSite(
                name="answer",
                prompt_template="{question}\nReply briefly.",
                parents=[],
            )
        ],
        factors=[
            DeterministicFactor(
                name="non_empty",
                parents=["answer"],
                code=(
                    "def phi(state, *args, **kwargs):\n"
                    "    return 0.5 if state.vars['answer'].strip() else -0.5"
                ),
            )
        ],
        output_site="answer",
    )
    graph = build_graph(spec, worker=llm)
    assert isinstance(graph, ChainGraph)
    assert graph.output_site == "answer"
    assert {s.name for s in graph.sites} == {"answer"}
    assert [f.name for f in graph.factors] == ["non_empty"]


def test_run_minimal_spec_returns_J_trajectory(llm: PLPChatClient) -> None:
    particles, J = asyncio.run(
        run_spec(_MINIMAL_SPEC, "Reply with the digit 4.", worker=llm, K=1, T=1)
    )
    assert len(particles) == 1
    assert particles[0].vars.get("direct", "").strip()
    assert J and math.isfinite(J[-1])
