# Examples

```{toctree}
:hidden:
:caption: Examples

scripted-llm
offline-inference
complex-workflow
live-gallery
```

The examples are split into two groups.

Offline examples use scripted clients and compile without credentials. Start with these when learning the semantics.

Live examples use hosted or local language models. They are useful for experiments, but they depend on API keys, router configuration, model downloads, and provider behavior.

## Offline Scripts

| Script | Role |
| --- | --- |
| `examples/scripted_quickstart.py` | Minimal state, trace, `observe`, and `factor` run with no API key. |
| `examples/offline_chain_graph.py` | Builds a `ChainGraph`, selects an engine, and runs weighted particles with scripted outputs. |

## Existing Repository Examples

| Script | Role | Requirements |
| --- | --- | --- |
| `examples/factor_graph_bp.py` | Factor graph and belief-propagation style demo. | `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`. |
| `examples/graphical_models.py` | Graphical model patterns for PLP workflows. | `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`. |
| `examples/multimodel_debate.py` | Multi-model debate via OpenAI-compatible routers. | Router-capable `OPENAI_BASE_URL` and key. |
| `examples/demo_library_puzzle.py` | Soft-NAND/library puzzle controller demo. | Hosted LLM credentials. |
| `examples/intmult/intmult_rrt.py` | Local Hugging Face integer multiplication experiment. | `torch`, model download, CPU/GPU resources. |

The `examples/old/` directory is archival and should not be treated as the primary learning path.
