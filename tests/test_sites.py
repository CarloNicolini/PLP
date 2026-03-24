import plp


def test_sample_logs_trace(llm):
    @plp.sample(llm, name="demo_site")
    def ask(state: plp.PLPState) -> str:
        return "Reply with the single word acknowledged."

    state = plp.PLPState()
    new_state, output = ask(state)

    assert "acknowledged" in output.lower()
    assert new_state.trace[-1].primitive == "demo_site"
    assert new_state.trace[-1].meta["llm"].startswith("openai/gpt-4.1-mini")


def test_observe_marks_invalid_state():
    @plp.observe(name="must_pass")
    def check(state: plp.PLPState, text: str) -> bool:
        return text == "pass"

    new_state = check(plp.PLPState(), "fail")
    assert not new_state.is_valid()
    assert new_state.trace[-1].meta["observe"] is True


def test_factor_updates_log_weight():
    @plp.factor(name="score")
    def score(state: plp.PLPState, value: float) -> float:
        return value

    new_state = score(plp.PLPState(), 1.5)
    assert new_state.log_weight == 1.5
    assert new_state.trace[-1].meta["factor"] is True
