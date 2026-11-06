# PLP tests

- `conftest.py` — session `llm` fixture: loads project `.env`, builds `PLPChatClient` with `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`. Tests that need a model request this fixture and **skip** if the key is missing.
- Most integration tests call the real API (small prompts, low temperature). Retry behaviour is covered by patching `invoke` / `ainvoke` on that same client so failures are synthetic but completions still come from the provider when the wrapper succeeds.

Run (from repo root, with venv activated):

```bash
pytest tests/
```
