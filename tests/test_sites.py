import plp


def test_text_sample_logs_trace(llm):
    state = plp.PLPState()
    new_state, output = plp.text_sample(
        state,
        llm,
        prompt="Reply with a short acknowledgement.",
        name="demo_site",
    )

    assert output.strip()
    assert new_state.trace[-1].primitive == "demo_site"


def test_text_sampler_wraps_prompt_builder(llm):
    @plp.text_sampler(llm, name="city_greet")
    def greet(state: plp.PLPState, city: str) -> str:
        """You are a concise greeter."""
        return f"Say hello to {city} in one short sentence."

    base_state = plp.PLPState()
    new_state, reply = greet(base_state, "Naples")

    assert reply.strip()
    assert new_state.trace[-1].primitive == "city_greet"


def test_infer_rejection_returns_last_attempt_on_failure(llm):
    def prompt_fn(topic: str) -> str:
        return f"Answer this question briefly: {topic}"

    def always_fail(state: plp.PLPState, answer: str, topic: str) -> bool:
        return False

    site = plp.make_text_site(
        name="failing_site",
        llm=llm,
        prompt_fn=prompt_fn,
        observe_fn=always_fail,
    )

    result = plp.infer_rejection(site, "What is PLP?", max_attempts=2)

    assert not result.state.is_valid()
    assert result.state.trace  # trace is populated even when all attempts fail


def test_text_transition_logs_edit(llm):
    state = plp.PLPState()
    instruction = "Improve the sentence for clarity."
    current = "The cat was sat on by the mat."

    new_state, revised = plp.text_transition(
        state,
        llm,
        current=current,
        instruction=instruction,
        name="transition_demo",
    )

    assert revised.strip()
    assert new_state.trace[-1].primitive == "transition_demo"


def test_self_consistency_sampler_majority(llm):
    @plp.self_consistency_sampler(llm, n_samples=2, name="yes_no")
    def decide(state: plp.PLPState) -> str:
        """Answer YES or NO only."""
        return "Respond with YES or NO for: Is the sky blue?"

    state = plp.PLPState()
    new_state, answer = decide(state)

    assert answer.strip()
    assert any(event.primitive.startswith("yes_no.sample_") for event in new_state.trace)
