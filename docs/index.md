# Probabilistic Language Programming

PLP is a small, semantics-first Python library for treating language-model calls as probabilistic program sites. It gives LLM workflows a trace, a weight, a graph, and an inference story.

The point is not to hide the uncertainty of language models. The point is to expose it clearly enough that you can reason about it, score it, repeat it, and diagnose it.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Start from zero
:link: getting-started/quickstart
:link-type: doc
Install PLP, run a no-key example, inspect a trace, and understand what changed in the state.
:::

:::{grid-item-card} Learn the model
:link: theory/trace-semantics
:link-type: doc
Read PLP as probabilistic programming over strings: traces, kernels, factors, and the unnormalized target.
:::

:::{grid-item-card} Run inference
:link: inference/index
:link-type: doc
Move from decorated sample sites to chain graphs, engine selection, particles, and diagnostics.
:::

:::{grid-item-card} Use the API
:link: api/index
:link-type: doc
Browse generated references for the public package, inference engines, and optimizer helpers.
:::

::::

## Why PLP Exists

Modern LLM systems often grow as a scaffold zoo: chain-of-thought prompts, self-consistency, verifier loops, multi-agent debates, repair chains, retrieval tools, and hand-tuned controller logic. These systems can work, but the object being approximated is often implicit.

PLP asks a simple question: what probabilistic program did we just build?

A PLP run produces a trace `tau`. LLM calls induce proposal kernels. `factor` and `observe` add preferences and constraints. Inference engines turn repeated traces into weighted particles. The scientific target is:

```{math}
p_D(\tau \mid x) \propto \pi_D(\tau \mid x) \Phi(\tau, x)
```

where `pi_D` is the distribution induced by the fixed deployment and `Phi` is the product of checks, judges, and domain-specific potentials.

```{toctree}
:hidden:
:caption: Documentation

getting-started/index
user-guide/index
inference/index
examples/index
theory/index
api/index
```
