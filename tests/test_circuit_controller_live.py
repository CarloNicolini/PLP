"""
Live-LLM checks for :class:`plp.optimizers.controller.CircuitGuide`.

The guide emits a typed :class:`CircuitSpec` via structured output; the
example's compile path turns it into a :class:`ChainGraph` and runs LBP.
Uses the session-scoped ``llm`` fixture from :mod:`tests.conftest` (skips
when ``OPENAI_API_KEY`` is unset). Live-LLM tests are not unit tests; they
verify that the guide emits a valid spec and that the spec runs end-to-end
through LBP. They are not exhaustive tests of the guide's ability to
synthesize good specs; that is the province of the :mod:`examples` module.
"""

from __future__ import annotations

import asyncio
import math

from plp import PLPChatClient
from soft_nand_circuit import (
    CircuitGuide,
    CircuitSpec,
    NandLeaf,
    NandNode,
    NandTreeFactor,
    DeterministicFactor,
    SampleSite,
    build_graph,
    run_spec,
)


def test_controller_proposes_valid_spec_and_runs(llm: PLPChatClient) -> None:
    """`propose` returns a CircuitSpec; the spec runs end-to-end through LBP."""
    question = "What is 2 + 2? Reply with a single digit on the last line."
    controller = CircuitGuide(client=llm)

    spec = controller.propose([], question)
    assert isinstance(spec, CircuitSpec)
    assert spec.sites, "spec must declare at least one sample site"
    assert spec.output_site in {s.name for s in spec.sites}

    # Judge is shared with the worker — any judge factor the controller emits
    # will hit the same endpoint. That is fine for a smoke test.
    graph = build_graph(spec, worker=llm, judge=llm)
    assert graph.output_site == spec.output_site

    particles, J = asyncio.run(run_spec(spec, question, worker=llm, judge=llm, K=1, T=1))
    assert particles, "expected at least one particle"
    assert J, "expected a non-empty J trajectory from LBP"
    assert math.isfinite(J[-1])


def test_hand_built_nand_tree_runs_end_to_end(llm: PLPChatClient) -> None:
    """A 2-leaf NAND tree factor compiles and scores a live particle."""
    spec = CircuitSpec(
        sites=[
            SampleSite(
                name="draft",
                prompt_template="{question}\nReply concisely.",
                parents=[],
            ),
            SampleSite(
                name="final",
                prompt_template=(
                    "Draft:\n{draft}\n\nQuestion: {question}\n"
                    "Last line: single digit answer only."
                ),
                parents=["draft"],
            ),
        ],
        factors=[
            NandTreeFactor(
                name="both_non_empty",
                parents=["draft", "final"],
                tree=NandNode(
                    children=[
                        NandLeaf(
                            source=DeterministicFactor(
                                name="draft_ok",
                                parents=["draft"],
                                code=(
                                    "def phi(state, *args, **kwargs):\n"
                                    "    return 1.0 if state.vars['draft'].strip() else -1.0"
                                ),
                            )
                        ),
                        NandLeaf(
                            source=DeterministicFactor(
                                name="final_ok",
                                parents=["final"],
                                code=(
                                    "def phi(state, *args, **kwargs):\n"
                                    "    return 1.0 if state.vars['final'].strip() else -1.0"
                                ),
                            )
                        ),
                    ]
                ),
            )
        ],
        output_site="final",
    )

    particles, J = asyncio.run(run_spec(spec, "What is 1 + 1?", worker=llm, judge=llm, K=1, T=1))
    assert len(particles) == 1
    assert "final" in particles[0].vars
    assert J and math.isfinite(J[-1])
