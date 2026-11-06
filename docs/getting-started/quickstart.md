# No-Key Quickstart

This example uses a tiny scripted object with the same `invoke(messages)` method as a real chat client. It runs offline and shows the core PLP semantics.

```python
from types import SimpleNamespace

import plp


class ScriptedLLM:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.model_name = "scripted"

    def invoke(self, messages):
        return SimpleNamespace(content=self.outputs.pop(0))


def remember_answer(state: plp.PLPState, output: str) -> None:
    state.vars["answer"] = output


llm = ScriptedLLM(["Paris"])


@plp.sample(llm, name="capital", update_state=remember_answer)
def capital(state: plp.PLPState, country: str) -> str:
    return f"Name the capital of {country}. Reply with one word."


state = plp.PLPState()
state, answer = capital(state, "France")

print(answer)
print(state.vars)
print(state.trace[-1])
```

A `sample` site does three things:

1. builds a prompt from the current `PLPState`;
2. calls an object with an `invoke` method;
3. returns a copied state with a new `TraceEvent`.

The optional `update_state` callback is where you decide which generated strings become named variables for later sites.

## Add Evidence

A trace is useful because it can be scored.

```python
@plp.observe(name="is_one_word")
def is_one_word(state: plp.PLPState) -> bool:
    return len(state.vars["answer"].split()) == 1


@plp.factor(name="known_capital", parents=["capital"])
def known_capital(state: plp.PLPState) -> float:
    return 0.0 if state.vars["answer"] == "Paris" else -4.0


state = is_one_word(state)
state = known_capital(state)
print(state.log_weight)
```

`observe` is a hard constraint. If it fails, the trace receives `-inf` weight. `factor` is softer: it adds a finite log score.

This is the basic PLP pattern: generate strings, store the relevant parts of the state, score the trace, and let inference compare alternatives.
