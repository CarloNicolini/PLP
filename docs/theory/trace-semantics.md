# Trace Semantics

Fix a deployment `D`: model weights, decoding parameters, tools, retry policy, and provider behavior. Running a PLP program produces a trace `tau`.

The proposal distribution `pi_D(tau | x)` is induced by the LLM calls and any other stochastic tools. Factors define a potential `Phi(tau, x)`. PLP's unnormalized target is:

```{math}
p_D(\tau \mid x) \propto \pi_D(\tau \mid x) \Phi(\tau, x)
```

This equation is the center of the library. It says that a workflow is not merely a sequence of prompts. It is a proposal mechanism plus a set of preferences or constraints.

## Same-Proposal Reweighting

PLP does not require pointwise LLM probabilities. Many engines use same-proposal reweighting: draw traces from the deployment, score them with factors, and compare weighted particles.

This is why factors matter. They are the bridge between raw generated traces and the target distribution you actually care about.
