"""Split guarantee: eps accounting, selectors, one live @guarantee run."""

from __future__ import annotations

import warnings

import pytest

from plp import PLPChatClient, PLPState, guarantee, sample, seed
from plp.guarantee import (
    MajoritySelector,
    SPRTSelector,
    achieved_eps_from_k,
    repeat_select,
    required_k_for_eps,
)
from plp.handlers import rng_for


def test_required_k_and_achieved_eps_are_inverses() -> None:
    for p in (0.7, 0.9):
        for eps in (1e-3, 1e-6):
            k = required_k_for_eps(p, eps)
            assert achieved_eps_from_k(p, k) <= eps + 1e-12


def test_majority_selector() -> None:
    sel = MajoritySelector()
    idx, info = sel.select(["a", "b", "a", "c", "a"], rng=None)
    assert info["winning_key"] == "a"
    assert ["a", "b", "a", "c", "a"][idx] == "a"


def test_sprt_early_stop() -> None:
    def judge(a: str, b: str) -> str:
        if a == "good" and b != "good":
            return "a"
        if b == "good" and a != "good":
            return "b"
        return "tie"

    sel = SPRTSelector(judge_fn=judge, k_target=2)
    with seed(0):
        idx, info = sel.select(["bad", "good", "bad", "good", "bad"], rng=rng_for("t"))
    assert info["stopped_early"] is True
    assert ["bad", "good", "bad", "good", "bad"][idx] == "good"


def test_guarantee_live_majority(llm: PLPChatClient) -> None:
    @guarantee(
        llm,
        eps=0.15,
        assumed_p=0.85,
        max_calls=6,
        group_by=lambda s: s.strip().lower(),
    )
    def ask(state: PLPState, q: str) -> tuple[PLPState, str]:
        @sample(llm, name="answer")
        def inner(state: PLPState, q: str) -> str:
            return f"{q}\nOutput EXACTLY the word: yes"

        return inner(state, q)

    state, out = ask(PLPState(), "yes/no?")
    assert "yes" in out.strip().lower()
    ev = [e for e in state.trace if e.primitive.startswith("guarantee(")]
    assert ev and ev[0].meta.get("selector") == "majority"


def test_guarantee_warns_when_eps_unreachable(llm: PLPChatClient) -> None:
    @guarantee(llm, eps=1e-30, assumed_p=0.7, max_calls=3)
    def ask(state: PLPState) -> tuple[PLPState, str]:
        @sample(llm, name="answer")
        def inner(state: PLPState) -> str:
            return "ping"

        return inner(state)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        state, _ = ask(PLPState())
    assert any("eps" in str(x.message) and "cap" in str(x.message) for x in w)
    ev = [e for e in state.trace if e.primitive.startswith("guarantee(")]
    assert ev[0].meta.get("clamped") is True


def test_repeat_select_majority(llm: PLPChatClient) -> None:
    @sample(llm, name="s")
    def inner(state: PLPState) -> str:
        return "Output exactly the digit 1 and nothing else."

    new_state, out, info = repeat_select(
        inner,
        PLPState(),
        args=(),
        kwargs={},
        n=5,
        selector=MajoritySelector(),
    )
    assert info["num_calls"] == 5
    assert len([e for e in new_state.trace if e.primitive == "s"]) == 5
    assert "1" in out
