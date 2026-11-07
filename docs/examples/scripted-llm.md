# Example: A Scripted LLM

This tiny client is enough to run `sample` without a network. The [offline quickstart](../getting-started/quickstart.md) shows how to run a complete trace with a scripted client.

```python
from types import SimpleNamespace


class ScriptedLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.model_name = "scripted"

    def invoke(self, messages):
        if not self.outputs:
            raise RuntimeError("ScriptedLLM has no outputs left")
        return SimpleNamespace(content=self.outputs.pop(0))
```

Use this pattern in tests and documentation. It gives deterministic traces while preserving the same primitive interface as a real client.
