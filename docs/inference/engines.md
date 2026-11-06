# Inference Engines

All engines consume a `ChainGraph` and return particles, represented as `PLPState` objects.

## `Forward`

Use when the graph is a directed acyclic sample graph with no factors. It performs a forward pass and records traces.

## `ImportanceSampling`

Use when proposals come from the model but factors define the target. It samples from `pi_D` and reweights by `Phi`.

## `SMC`

Use for long chains with local factors. It alternates proposal, scoring, and resampling so particles do not collapse only at the end.

## `LoopyParticleBP`

Use for cyclic parent structure or loopy factor coupling. Messages are represented by particles rather than tables because PLP sites are strings and LLM kernels are implicit.

## `MaxProduct`

Use when you want a high-scoring trace rather than a weighted sample approximation. This is useful for best-of-K style selection and MAP-like workflows.

## Choosing Particle Counts

More particles help only when the proposal has support over useful traces and the weights are not completely degenerate. Always inspect diagnostics when available: top weights, distinct outputs, repeated failure modes, and effective sample size.
