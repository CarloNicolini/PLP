# PLP tests

- `conftest.py` — session `llm` fixture: loads project `.env`, builds `PLPChatClient` with `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`. Tests that need a model request this fixture and **skip** if the key is missing.
- Most integration tests call the real API (small prompts, low temperature). Retry behaviour is covered by patching `invoke` / `ainvoke` on that same client so failures are synthetic but completions still come from the provider when the wrapper succeeds.

Run (from repo root, with venv activated):

```bash
pytest tests/
```

The default configuration excludes `test_reliability_example.py`, a live experiment.
Currently `test_circuit_controller_live.py` and `test_soft_nand_circuit.py` import
example modules absent from this checkout (`soft_nand_circuit`). Until those examples are restored, run the remaining suite with:

```bash
uv run pytest tests/ --ignore=tests/test_circuit_controller_live.py --ignore=tests/test_soft_nand_circuit.py
```

Packaging and documentation checks are described in the root README. They do
not make hosted model calls.
