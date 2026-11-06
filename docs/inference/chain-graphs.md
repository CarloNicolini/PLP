# Chain Graphs

A `ChainGraph` is the static representation of a PLP program:

- sample sites are directed nodes;
- sample `parents` are directed dependencies;
- factor `parents` define undirected cliques;
- `output_site` names the site you care about most.

```python
import plp


graph = plp.ChainGraph.from_decorated(
    propose,
    critique,
    score_pair,
    output_site="critique",
)
```

The graph is not an execution trace. It is a structural claim about how traces will be generated and scored.

## Why Structure Matters

A pure acyclic sample graph can be sampled in one pass. Add factors, and traces need weights. Add long chains with local evidence, and SMC becomes attractive. Add cycles in parents or in the moralized factor graph, and loopy particle BP becomes the appropriate approximation.

Wrong structure can lead to wrong confidence. If two agents read each other in a debate loop, declare the cycle.
