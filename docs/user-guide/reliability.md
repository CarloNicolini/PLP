# Reliability With `guarantee`

`guarantee` repeats a stochastic subcomputation and selects a representative answer under a fixed error budget. It is the PLP version of a reliability wrapper around an uncertain step.

A simple selector might prefer shorter answers, exact matches, or judge-comparison winners. More advanced selectors can use sequential tests.

```python
@plp.guarantee(llm=llm, eps=1e-3, judge_fn=judge)
def robust_step(state: plp.PLPState, task: str) -> tuple[plp.PLPState, str]:
    return solve_step(state, task)
```

## What Reliability Can And Cannot Do

Repetition can reduce variance when errors are independent enough and the selector is aligned with the target. It cannot fix a support problem: if the model never proposes the correct answer, selection will not recover it. It also cannot fix a judge that consistently rewards the wrong behavior.

Use `guarantee` as an engineering tool, not as a mathematical certificate that the answer is true.
