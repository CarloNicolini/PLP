# Example: Offline Scoring And Graphs

Save this example as `offline_inference.py` and run it with
`uv run python offline_inference.py`. The scripted model supplies three
candidate answers, and the factor gives the two correct answers a higher weight.

```python
import asyncio
from types import SimpleNamespace

import plp


class ScriptedLLM:
    def __init__(self, answers):
        self.answers = iter(answers)

    def invoke(self, messages):
        return SimpleNamespace(content=next(self.answers))


def store_answer(state, answer):
    state.vars["answer"] = answer


@plp.sample(ScriptedLLM(["4", "5", "4"]), name="answer", update_state=store_answer)
def propose(state, question):
    return f"Answer with one number: {question}"


@plp.factor(name="arithmetic_check", parents=["answer"])
def arithmetic_check(state, answer):
    return 0.0 if answer == "4" else -2.0


async def main():
    graph = plp.ChainGraph.from_decorated(propose, arithmetic_check, output_site="answer")
    engine_type = plp.choose_engine(graph)
    particles = await engine_type(resample=False).run(graph, K=3, question="What is 2 + 2?")

    print(engine_type.__name__)
    for particle in particles:
        print(particle.vars["answer"], particle.log_weight)


asyncio.run(main())
```

`choose_engine` returns `ImportanceSampling`. The graph has one sample site
and one factor, so the engine generates candidate traces and weights them
using the factor scores.
With `resample=False`, the returned particles keep those scores in
`log_weight`. By default, the engine resamples and resets the log weights.
