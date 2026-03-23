from collections import Counter

import plp


def test_sprt_discrete_prefers_candidate_with_consistent_wins():
    candidates = ["alpha", "beta", "gamma"]

    def judge(a: str, b: str) -> str:
        if a == "alpha" and b != "alpha":
            return "a"
        if b == "alpha" and a != "alpha":
            return "b"
        return "tie"

    winner = plp.InferEngine.sprt_discrete(
        candidates=candidates,
        judge_fn=judge,
        k=2,
        max_rounds=50,
    )

    assert winner == "alpha"


def test_vote_uses_callable_judge():
    candidates = ["cat", "dog"]

    def judge(a: str, b: str) -> str:
        if a == "cat" and b == "dog":
            return "a"
        if a == "dog" and b == "cat":
            return "b"
        return "tie"

    winner = plp.vote(candidates, judge=judge, k=2)
    assert winner == "cat"


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
    occurrences = Counter(event.output for event in new_state.trace if event.primitive == "simple_site")
    assert occurrences["good"] >= 2
