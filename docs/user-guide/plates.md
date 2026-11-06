# Plates And Repetition

A plate represents repeated conditionally independent work. In LLM systems this often appears as self-consistency, candidate generation, committee sampling, or repeated critique.

```python
states, answers = plp.plate("candidates", 5).map(
    solve,
    "Why use probabilistic programming for LLMs?",
    init_state=plp.PLPState(),
)
```

The result is a list of states and a list of outputs. Each run has its own trace.

## Async Plates

For hosted LLM calls, concurrency matters:

```python
states, answers = await plp.plate("candidates", 8).amap(
    solve_async,
    "Explain effective sample size.",
    max_concurrency=4,
)
```

Async plate helpers use task-local plate context and an `asyncio.Semaphore`, so independent runs do not share plate-local state accidentally.

## What A Plate Does Not Prove

A plate gives repeated samples from the same procedure. If the samples are correlated by prompt wording, provider behavior, shared misconceptions, or a weak judge, increasing the plate size may plateau. This is why the theory pages discuss `K_eff`, the effective number of independent samples.
