# LLM Clients

A PLP sample site only needs a small protocol:

- sync sites need `invoke(messages)`;
- async sites need `ainvoke(messages)`;
- returned objects should expose `content` or be convertible to text.

`PLPChatClient` implements this protocol for OpenAI-compatible chat completions. `ChatOpenAI` is an alias.

```python
llm = plp.ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.2,
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL"),
)
```

`LocalHuggingFaceClient` implements the same interface for a local `transformers` backend. It is useful when you want no hosted API, but it can require large model downloads and appropriate CPU/GPU resources.

## Retries

`invoke_llm` and `ainvoke_llm` use retry settings controlled by:

- `PLP_LLM_MAX_ATTEMPTS`;
- `PLP_LLM_MAX_WAIT_S`.

Provider failures are part of the deployment. For scientific runs, record the provider, model, temperature, and retry policy next to your results.
