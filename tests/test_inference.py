"""High-level ``plp.guarantee`` wrapping a live ``sample``."""

from __future__ import annotations

import plp
from plp import PLPChatClient, PLPState, sample


def test_guarantee_majority_wraps_live_sample(llm: PLPChatClient) -> None:
    @plp.guarantee(llm=llm, eps=0.2, method="majority", max_calls=6, assumed_p=0.85)
    def run(state: PLPState) -> tuple[PLPState, str]:
        @sample(llm, name="inner")
        def inner(state: PLPState) -> str:
            return "Output exactly the word: PASS"

        return inner(state)

    state, out = run(PLPState())
    assert "pass" in out.lower()
    assert any(e.primitive.startswith("guarantee(") for e in state.trace)
