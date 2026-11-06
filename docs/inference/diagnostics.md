# Diagnostics And Failure Modes

PLP makes failures easier to name. It does not make them disappear.

## Effective Sample Size

When a few particles dominate the weights, the effective sample size is low. This usually means proposal-target mismatch: the LLM is generating traces that the factors mostly reject.

Responses:

- improve prompts so useful traces are proposed more often;
- move evidence earlier with SMC when the graph is a chain;
- soften overly brittle factors;
- increase particles only after support and scoring are plausible.

## Correlation And `K_eff`

Self-consistency can produce many strings that share the same misconception. The nominal sample count `K` is not the same as the effective independent count `K_eff`.

Correlation can come from shared prompts, shared model weights, common retrieval context, or a judge that rewards the same flawed pattern.

## Four Failure Families

| Failure | Meaning | Typical symptom |
| --- | --- | --- |
| Support | Correct traces are rarely or never proposed. | All particles are plausible-looking but wrong. |
| Selection | The factor or judge rewards the wrong trace. | The best-weighted output is worse than a lower-weight output. |
| Dependence | Samples are more correlated than assumed. | More samples stop improving results. |
| Composition | The declared graph misses a dependency. | Engine choice or confidence is too optimistic. |

A good PLP workflow records enough trace data to debug all four.
