from itertools import cycle

import plp


def test_guarantee_majority_vote_without_judge(llm):
    outputs = ["good", "bad", "good", "good"]
    call_count = {"value": 0}

    def simple_site(state: plp.PLPState) -> tuple[plp.PLPState, str]:
        choice = outputs[call_count["value"] % len(outputs)]
        call_count["value"] += 1
        new_state = state.copy()
        new_state.trace.append(
            plp.TraceEvent(
                primitive="simple_site",
                prompt="(synthetic)",
                output=choice,
                meta={},
            )
        )
        return new_state, choice

    @plp.guarantee(llm=llm, eps=1e-3, judge_fn=None, method="majority")
    def guarded(state: plp.PLPState) -> tuple[plp.PLPState, str]:
        return simple_site(state)

    state = plp.PLPState()
    new_state, answer = guarded(state)

    assert answer == "good"
    outputs = [event.output for event in new_state.trace if event.primitive == "simple_site"]
    assert outputs.count("good") >= 2


def test_guarantee_with_judge_prefers_target_candidate(llm):
    outputs = cycle(["beta", "alpha", "alpha", "beta", "alpha"])

    def simple_site(state: plp.PLPState) -> tuple[plp.PLPState, str]:
        choice = next(outputs)
        new_state = state.copy()
        new_state.trace.append(
            plp.TraceEvent(
                primitive="simple_site",
                prompt="(synthetic)",
                output=choice,
                meta={},
            )
        )
        return new_state, choice

    def judge(a: str, b: str) -> str:
        if a == "alpha" and b != "alpha":
            return "a"
        if b == "alpha" and a != "alpha":
            return "b"
        return "tie"

    @plp.guarantee(llm=llm, eps=1e-3, judge_fn=judge, method="sprt_discrete")
    def guarded(state: plp.PLPState) -> tuple[plp.PLPState, str]:
        return simple_site(state)

    _, answer = guarded(plp.PLPState())
    assert answer == "alpha"
