# Live Example Gallery

Live scripts should be run explicitly from the repository root after configuring credentials.

```bash
source .venv/bin/activate
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...  # optional
python examples/graphical_models.py
```

## Recommended Order

1. `examples/graphical_models.py` for the broad modeling vocabulary.
2. `examples/factor_graph_bp.py` for factor-graph intuition.
3. `examples/multimodel_debate.py` for router-backed multi-model systems.
4. `examples/demo_library_puzzle.py` for the Soft-NAND case study.
5. `examples/intmult/intmult_rrt.py` for local Hugging Face experiments.

If a script fails before any PLP logic runs, check credentials and provider compatibility first. If it fails after traces are generated, inspect `PLPState.trace` and `log_weight` before changing prompts.
