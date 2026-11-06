# Example: Offline Scoring And Graphs

The runnable file `examples/offline_chain_graph.py` shows generation, scoring, graph construction, engine selection, and particle execution without API keys.

The key pieces are:

```python
graph = plp.ChainGraph.from_decorated(plan, solution, uses_math, output_site="solution")
engine_type = plp.choose_engine(graph)
particles = await engine_type().run(graph, K=3, problem="Find a robust method.")
```

The graph has an acyclic sample dependency plus a factor, so `choose_engine` selects an engine that can reweight traces.

Run it with:

```bash
source .venv/bin/activate
python examples/offline_chain_graph.py
```

Because the outputs are scripted, any surprising result is a PLP semantics issue rather than provider nondeterminism.
