# Probabilistic Language Programming (PLP)

PLP is a Python library for treating language-model calls as probabilistic program
sites. Decorated functions produce traces; hard constraints and soft factors score
those traces; graph-based inference compares repeated runs.

Requires **Python 3.12 or newer**. PLP is experimental: weighted particles and
reliability selection depend on the model, scoring functions, and sampling assumptions.
They do not certify that generated answers are true.

## Installation

Install this repository in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "git+https://github.com/CarloNicolini/PLP.git"
```

On Windows, activate with `.venv\Scripts\activate`. The current dependency set
includes PyTorch and Transformers, so allow space and time for those downloads.

## Offline quickstart

This complete example requires no credentials or model downloads:

```python
from types import SimpleNamespace
import plp


class ScriptedLLM:
    def invoke(self, messages):
        return SimpleNamespace(content="Paris")


@plp.sample(ScriptedLLM(), name="capital")
def capital(state: plp.PLPState, country: str) -> str:
    return f"Name the capital of {country}."


state, answer = capital(plp.PLPState(), "France")
print(answer)                    # Paris
print(state.trace[-1].primitive) # capital
print(state.log_weight)          # 0.0
```

Continue with the [quickstart](docs/getting-started/quickstart.md) to store answers
and add evidence with `observe` and `factor`.

## Core concepts and API

- `PLPState` and `TraceEvent` hold state, log weights, and execution history.
- `sample` / `asample` call a model and append a trace event.
- `observe` rejects a trace on a failed Boolean constraint; `factor` adds a log score.
- `plate` runs repeated computations, including async execution with bounded concurrency.
- `ChainGraph` and `choose_engine` expose graph structure and inference selection.
- `guarantee` repeats a computation and selects an answer under its statistical assumptions.
- `replay`, `block`, `do`, `condition`, and `seed` control execution.

See the [API reference sources](docs/api/index.md) and
[inference guide](docs/inference/index.md) for details and available engines.

## Hosted models

Set `OPENAI_API_KEY`, then replace `ScriptedLLM()` with an explicit client:

```python
llm = plp.ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
```

`OPENAI_BASE_URL` selects an OpenAI-compatible endpoint; choose a model identifier
supported by that provider. Live requests may incur charges. See
[installation and environment variables](docs/getting-started/installation.md) and
the [live LLM guide](docs/getting-started/live-llm.md).

## Documentation

Start with [installation](docs/getting-started/installation.md), the
[offline quickstart](docs/getting-started/quickstart.md), and
[examples](docs/examples/index.md). The [scientific context](docs/theory/index.md)
explains trace semantics, assumptions, and failure modes. The longer
[theoretical foundations](docs/plp_theoretical_foundations.md) remain a research reference.

## Development and validation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/CarloNicolini/PLP.git
cd PLP
uv sync --all-groups
uv run pytest tests/ \
  --ignore=tests/test_circuit_controller_live.py \
  --ignore=tests/test_soft_nand_circuit.py
make docs-strict
uv run python -m build
uv run twine check dist/*
```

Tests using the live `llm` fixture skip without `OPENAI_API_KEY`; see
[tests/README.md](tests/README.md) for details and the excluded reliability example.
Use `make docs-serve` for a local documentation preview. Documentation changes
should pass `make docs-strict`; keep examples runnable offline unless explicitly
labeled as live, and update the API and installation guides when behavior changes.
