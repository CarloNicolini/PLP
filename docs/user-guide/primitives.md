# Primitives

PLP has a small primitive vocabulary. That is a design choice: the scientific semantics should fit in your head.

## `sample` And `asample`

`sample` decorates a function that turns a state into a prompt string. The wrapper calls an LLM-like object and returns `(new_state, output)`.

```python
@plp.sample(llm, name="draft", parents=["plan"])
def draft(state: plp.PLPState, topic: str) -> str:
    plan = state.vars.get("plan", "")
    return f"Using this plan: {plan}
Draft a short answer about {topic}."
```

The `parents` argument declares graph structure. If a site reads `state.vars["plan"]`, declare `parents=["plan"]` so `ChainGraph` can infer the dependency.

`asample` is the async version. It expects an object with `ainvoke(messages)` and composes with async plate helpers.

## `observe`

`observe` is a hard constraint.

```python
@plp.observe(name="has_number")
def has_number(state: plp.PLPState) -> bool:
    return any(ch.isdigit() for ch in state.vars["answer"])
```

If the predicate returns `False`, `log_weight` becomes `-inf`. Use this only for constraints you really want to kill.

## `factor`

`factor` is a soft log-weight update.

```python
@plp.factor(name="brevity", scale=0.1, parents=["answer"])
def brevity(state: plp.PLPState) -> float:
    return -len(state.vars["answer"].split())
```

Positive values favor a trace. Negative values penalize it. A factor is often a verifier, a rubric, a symbolic checker, or a domain score.

## `parents` And Scientific Honesty

Parents are not just metadata. They encode a claim about conditional dependence. If a site reads another site's output but you do not declare the parent, engine selection may be too optimistic.
