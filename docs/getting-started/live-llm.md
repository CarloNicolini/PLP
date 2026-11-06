# First Live LLM Run

After the no-key quickstart, replace the scripted model with a real OpenAI-compatible client.

```python
import os

import plp

llm = plp.ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.3,
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL"),
)
```

Then use the same primitive shape:

```python
@plp.sample(llm, name="solver")
def solve(state: plp.PLPState, question: str) -> str:
    return f"Answer briefly: {question}"


state, answer = solve(plp.PLPState(), "What is probabilistic programming?")
```

## Provider Notes

`ChatOpenAI` is an alias for `PLPChatClient`, a thin wrapper around the OpenAI Python client. It works with OpenAI and with providers that implement an OpenAI-compatible chat-completions API.

If you use a router with model names such as `meta-llama/...`, `qwen/...`, or `mistralai/...`, set `OPENAI_BASE_URL` to that router's endpoint.

## Why Live Runs Are Optional In The Docs

The documentation builds without API keys. Live model calls are valuable for experiments, but they make builds flaky and expensive. The teaching examples use scripted clients so the semantics are reproducible. Hosted examples are presented as scripts you can run explicitly when credentials are available.
