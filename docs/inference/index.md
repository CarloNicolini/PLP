# Inference Overview

```{toctree}
:hidden:
:caption: Inference

chain-graphs
engine-selection
engines
diagnostics
```

PLP programs generate traces. Inference compares traces.

A decorated PLP program can be summarized as a `ChainGraph`: sample sites form directed dependencies, and factors form undirected cliques. The graph topology determines whether a simple forward pass is enough or whether weighted particles, SMC, max-product search, or loopy particle belief propagation are needed.

The practical workflow is:

1. decorate sample and factor sites;
2. declare `parents` honestly;
3. build a `ChainGraph`;
4. call `choose_engine` or choose an engine explicitly;
5. inspect particles and weights;
6. diagnose failure modes before trusting the final answer.
