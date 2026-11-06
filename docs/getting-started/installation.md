# Installation

PLP targets Python 3.12 or newer.

From a checkout of the repository:

```bash
python -m venv .venv
source .venv/bin/activate
uv sync --all-groups
```

For package-only development:

```bash
source .venv/bin/activate
uv sync
```

For documentation work:

```bash
source .venv/bin/activate
uv sync --group docs
sphinx-build -b html docs docs/_build/html
```

To treat warnings as failures, which is recommended before publishing:

```bash
source .venv/bin/activate
sphinx-build -W -b html docs docs/_build/html
```

## Environment Variables

Most documentation examples do not require network access. Live LLM examples use these variables:

| Variable | Required | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | For hosted LLM calls | API key for OpenAI or an OpenAI-compatible provider. |
| `OPENAI_BASE_URL` | Optional | Alternate OpenAI-compatible endpoint, often used for routers. |
| `PLP_LLM_MAX_ATTEMPTS` | Optional | Retry attempts for `invoke_llm` and `ainvoke_llm`. |
| `PLP_LLM_MAX_WAIT_S` | Optional | Maximum retry wait in seconds. |

A typical `.env` file for hosted examples looks like:

```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

Do not commit real credentials.
