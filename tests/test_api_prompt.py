import plp


def test_prompt_can_return_state(llm):
    output, state = plp.prompt(
        "Reply with a single emoji.",
        llm=llm,
        name="api.prompt.demo",
        return_state=True,
    )

    assert output.strip()
    assert state.trace[-1].primitive == "api.prompt.demo"


def test_sampler_respects_existing_state(llm):
    @plp.sampler(name="hello_city", llm=llm)
    def greet(city: str) -> str:
        return f"Say hi to {city} in three words."

    state = plp.PLPState()
    reply, new_state = greet(state, "Rome", return_state=True)

    assert reply.strip()
    assert new_state is not state
    assert new_state.trace[-1].primitive == "hello_city"
