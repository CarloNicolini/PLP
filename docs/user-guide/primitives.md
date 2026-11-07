# Primitives

PLP has a small primitive vocabulary. That is a design choice: the scientific semantics should fit in your head.

## `sample` And `asample`

`sample` decorates a function that turns a state into a prompt string. The wrapper calls an LLM-like object and returns `(new_state, output)`.

```python
def store_draft(state, draft_text):
    state.vars["draft"] = draft_text


@plp.sample(llm, name="draft", parents=["plan"], update_state=store_draft)
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
    return any(ch.isdigit() for ch in state.vars["draft"])
```

If the predicate returns `False`, `log_weight` becomes `-inf`. Use this only for constraints you really want to kill.

## `factor`

`factor` is a soft log-weight update.

```python
@plp.factor(name="brevity", scale=0.1, parents=["draft"])
def brevity(state: plp.PLPState, draft_text: str) -> float:
    return -len(draft_text.split())
```

Positive values favor a trace. Negative values penalize it. A factor is often a verifier, a rubric, a symbolic checker, or a domain score.

During graph inference, PLP looks up each parent by name in `state.vars`
and passes the values after `state`, in the order given by `parents`.
Store each parent output under its site name, as `store_draft` does above.
For a direct call, use `brevity(state, state.vars["draft"])`.

## `parents` And Scientific Honesty

Parents are not just metadata. They encode a claim about conditional dependence. If a site reads another site's output but you do not declare the parent, engine selection may be too optimistic.
