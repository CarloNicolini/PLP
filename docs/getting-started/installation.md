# Installation

PLP targets Python 3.12 or newer.

Install from the repository (the package name alone may refer to an unrelated index release):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "git+https://github.com/CarloNicolini/PLP.git"
```

On Windows, activate with `.venv\Scripts\activate`. The current dependencies
include PyTorch and Transformers, even for hosted clients and scripted examples.

## Development checkout

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone https://github.com/CarloNicolini/PLP.git
cd PLP
uv sync --all-groups
uv run pytest tests/ \
  --ignore=tests/test_circuit_controller_live.py \
  --ignore=tests/test_soft_nand_circuit.py
```

`uv sync` creates `.venv` and installs the project in editable mode. For only the
package and default development group, use `uv sync`.

## Build the documentation

From the checkout root:

```bash
make docs-strict
```

This installs the docs dependency group and runs Sphinx with warnings treated as
errors. Use `make docs-serve` for a live preview at `http://127.0.0.1:8000`.
The equivalent build command is:

```bash
uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
```

## Build distributions

```bash
uv run python -m build
uv run twine check dist/*
```

The wheel contains `plp` from `src/plp`; the source distribution also includes
this documentation and the tests. The version comes from `plp.__version__`.

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
