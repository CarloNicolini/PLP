# Probabilistic Language Programming (PLP)

PLP is a small, semantics-first library for treating LLM calls as probabilistic program sites. The package is intentionally narrow: it focuses on traces, primitive operations, plate-style repetition, and a minimal reliability wrapper.

For the theoretical framing, see [`docs/plp_theoretical_foundations.md`](docs/plp_theoretical_foundations.md).

## Public API

Top-level `plp` exports only the core surface:

- State and trace: `PLPState`, `TraceEvent`
- LLM protocols and client: `LLMLike`, `AsyncLLMLike`, `PLPChatClient`, `ChatOpenAI`
- LLM call helpers: `invoke_llm`, `ainvoke_llm`, `normalize_chat_messages`
- Primitives: `sample`, `asample`, `observe`, `factor`, `plate`
- Plate runners: `plate(...).map(...)`, `plate(...).reduce(...)`, `plate(...).amap(...)`, `plate(...).areduce(...)`
- Reliability wrapper: `guarantee`

There is no high-level prompt/session API, agent abstraction, or TextGrad layer in the current package.

## Installation

```bash
pip install plp
```

Set `OPENAI_API_KEY` in the environment, then construct a client explicitly:

```python
import plp

llm = plp.ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
```

## Core Example

Define a stochastic site with `@sample`, then inspect the resulting trace.

```python
import plp

llm = plp.ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)


@plp.sample(llm, name="solver")
def solve(state: plp.PLPState, question: str) -> str:
    return f"Answer briefly: {question}"


state = plp.PLPState()
new_state, answer = solve(state, "What is probabilistic programming?")

print(answer)
print(new_state.trace[-1])
```

`TraceEvent` records the primitive name, prompt, output, and metadata for each step.

## Primitive Semantics

- `sample` / `asample`: draw a string from an LLM-conditioned distribution and append a trace event.
- `observe`: apply a hard Boolean constraint by setting `log_weight = -inf` on failure.
- `factor`: apply a soft log-weight update.
- `plate`: represent conditionally independent repetition.
- `guarantee`: repeat a stochastic subcomputation and select a representative answer under a fixed error budget.

## Plate and Concurrency

PLP supports both sequential and concurrent plate execution.

For simple repeated runs:

```python
states, outputs = plp.plate("candidates", 5).map(
    solve,
    "Why use plate?",
    init_state=plp.PLPState(),
)
```

For async workloads:

```python
import asyncio
import plp

llm = plp.ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)


@plp.asample(llm, name="async_solver")
def solve_async(state: plp.PLPState, question: str) -> str:
    return f"Answer briefly: {question}"

states, outputs = asyncio.run(
    plp.plate("candidates", 4).amap(
        solve_async,
        "Why use plate?",
        max_concurrency=4,
    )
)
```

`plate(...).amap(...)` and `plate(...).areduce(...)` use task-local context plus an `asyncio.Semaphore`, so concurrent async runs do not share plate state accidentally.

## Reliability Example

```python
import plp

llm = plp.ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)


@plp.sample(llm, name="step_solver")
def solve_step(state: plp.PLPState, task: str) -> str:
    return f"Compute the result for: {task}. Reply with just the answer."


def judge(a: str, b: str) -> str:
    return "a" if len(a) <= len(b) else "b"


@plp.guarantee(llm=llm, eps=1e-3, judge_fn=judge)
def robust_step(state: plp.PLPState, task: str) -> tuple[plp.PLPState, str]:
    return solve_step(state, task)
```

## Notes

- Handlers such as `replay`, `block`, `do`, and `condition` still exist in `plp.handlers`, but they are supporting machinery rather than the advertised top-level API.
- The library currently assumes OpenAI-compatible chat clients at the transport boundary.

## Development

Run tests with:

```bash
source .venv/bin/activate
pytest
```
