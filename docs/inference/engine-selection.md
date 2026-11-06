# Engine Selection

`choose_engine(graph)` chooses an engine from topology.

| Graph structure | Engine |
| --- | --- |
| `prefer_map=True` | `MaxProduct` |
| cycle in sample parents | `LoopyParticleBP` |
| acyclic sample graph, no factors | `Forward` |
| cycle in moralized graph | `LoopyParticleBP` |
| long chain with unary per-site factors | `SMC` |
| other acyclic graph with factors | `ImportanceSampling` |

This heuristic is intentionally conservative. You can always instantiate an engine class directly when you know more than the topology reveals.

## Reading The Decision

Engine selection is a diagnostic. If you expected `Forward` but got `LoopyParticleBP`, the graph has a cycle or factor coupling that deserves attention. If you expected `SMC` but got `ImportanceSampling`, your graph may not look like a single long chain with local evidence.
