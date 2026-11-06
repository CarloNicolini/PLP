import os

import plp


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _label(text: str) -> str:
    return text.strip().upper().split()[0] if text.strip() else ""


def _pick_index(text: str, n: int) -> int:
    for token in text.replace(".", " ").split():
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < n:
                return idx
    return 0


def test_reliability_quantities_for_open_ended_search(llm):
    generator = plp.PLPChatClient(
        model=llm.model,
        temperature=0.6,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    task = (
        "In 2-3 sentences, explain when recursive decomposition helps an LLM "
        "scaffold and when it hurts."
    )
    branches = 2

    @plp.sample(generator, name="direct_answer")
    def direct_answer(state: plp.PLPState, task: str) -> str:
        return f"Answer directly and concisely.\n\nTask:\n{task}\n\nReturn only the answer."

    @plp.sample(generator, name="search_answer")
    def search_answer(state: plp.PLPState, task: str) -> str:
        return (
            "Answer the task with one distinct framing or angle.\n\n"
            f"Task:\n{task}\n\nReturn only the answer."
        )

    @plp.sample(llm, name="judge_answer")
    def judge_answer(state: plp.PLPState, task: str, answer: str) -> str:
        return f"""
You are grading an answer to a conceptual question about LLM scaffolds.

Task:
{task}

Candidate answer:
{answer}

PASS iff the answer is on-task, reasonably concise, and captures both sides:
- branching helps when it creates alternative solution paths that can be verified or selected
- branching hurts when children are all essential or composition is brittle

Return PASS or FAIL only.
""".strip()

    @plp.guarantee(llm=llm, eps=0.2, method="majority", max_calls=3)
    def robust_verdict(
        state: plp.PLPState, task: str, answer: str
    ) -> tuple[plp.PLPState, str]:
        return judge_answer(state, task, answer)

    @plp.sample(llm, name="judge_search_validity")
    def judge_search_validity(state: plp.PLPState, task: str) -> str:
        return f"""
Task:
{task}

Is it a valid search decomposition to generate {branches} alternative full answers
to this task and then select the best one?

Return VALID or INVALID only.
""".strip()

    @plp.guarantee(llm=llm, eps=0.2, method="majority", max_calls=3)
    def robust_validity(state: plp.PLPState, task: str) -> tuple[plp.PLPState, str]:
        return judge_search_validity(state, task)

    @plp.sample(llm, name="select_best_branch")
    def select_best_branch(
        state: plp.PLPState, task: str, answers: list[str]
    ) -> str:
        numbered = "\n\n".join(
            f"{i + 1}. {answer}" for i, answer in enumerate(answers)
        )
        return f"""
Task:
{task}

Candidate answers:
{numbered}

Return only the index of the best answer.
""".strip()

    @plp.observe(name="has_good_child")
    def has_good_child(state: plp.PLPState, child_passes: list[bool]) -> bool:
        return any(child_passes)

    def direct_trial(state: plp.PLPState, task: str) -> tuple[plp.PLPState, float]:
        state, answer = direct_answer(state, task)
        state, verdict = robust_verdict(state, task, answer)
        return state, 1.0 if _label(verdict) == "PASS" else 0.0

    def validity_trial(state: plp.PLPState, task: str) -> tuple[plp.PLPState, float]:
        state, verdict = robust_validity(state, task)
        return state, 1.0 if _label(verdict) == "VALID" else 0.0

    def search_trial(state: plp.PLPState, task: str) -> tuple[plp.PLPState, float]:
        state, answer = search_answer(state, task)
        state, verdict = robust_verdict(state, task, answer)
        return state, 1.0 if _label(verdict) == "PASS" else 0.0

    def selector_trial(
        state: plp.PLPState, task: str, branches: int
    ) -> tuple[plp.PLPState, dict[str, float]]:
        answers: list[str] = []
        child_passes: list[bool] = []

        # Prompts ignore state, so reusing the state here only accumulates trace.
        with plp.plate("children", branches) as children:
            for _ in children:
                state, answer = search_answer(state, task)
                state, verdict = robust_verdict(state, task, answer)
                answers.append(answer)
                child_passes.append(_label(verdict) == "PASS")

        state = has_good_child(state, child_passes)
        valid = 1.0 if state.is_valid() else 0.0

        state, picked = select_best_branch(state, task, answers)
        chosen = answers[_pick_index(picked, len(answers))]
        state, verdict = robust_verdict(state, task, chosen)
        return state, {
            "valid": valid,
            "selected_passed": 1.0 if _label(verdict) == "PASS" else 0.0,
        }

    def selector_rate(rows: list[dict[str, float]]) -> float:
        valid = [row["selected_passed"] for row in rows if row["valid"] > 0.0]
        return _mean(valid)

    a_state, a_hat = plp.plate("direct_trials", 2).reduce(direct_trial, _mean, task)
    v_state, v_hat = plp.plate("validity_trials", 2).reduce(
        validity_trial, _mean, task
    )
    r_state, r_hat = plp.plate("search_trials", 2).reduce(search_trial, _mean, task)
    c_state, c_hat = plp.plate("search_batches", 2).reduce(
        selector_trial,
        selector_rate,
        task,
        branches,
    )

    d_hat = v_hat * c_hat * (1.0 - (1.0 - r_hat) ** branches)

    for value in [a_hat, v_hat, r_hat, c_hat, d_hat]:
        assert 0.0 <= value <= 1.0

    assert a_state.trace[-1].primitive == "plate_reduce(direct_trials)"
    assert v_state.trace[-1].primitive == "plate_reduce(validity_trials)"
    assert r_state.trace[-1].primitive == "plate_reduce(search_trials)"
    assert c_state.trace[-1].primitive == "plate_reduce(search_batches)"
    assert any(event.primitive == "search_answer" for event in c_state.trace)
    assert any(event.primitive == "select_best_branch" for event in c_state.trace)
    assert any(event.primitive == "has_good_child" for event in c_state.trace)
    assert any(
        event.primitive == "guarantee(robust_verdict)" for event in c_state.trace
    )
