# Hosted Model Example

Follow the [first live LLM run](../getting-started/live-llm.md) to configure an
OpenAI-compatible client and run a sample site. Use the provider's model
identifier and set `OPENAI_BASE_URL` if the provider uses a custom endpoint.

If a request fails before returning a response, check the credentials, model
identifier, and endpoint. After a successful call, inspect the returned state's
`trace` to see the prompt and response.
