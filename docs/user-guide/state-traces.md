# State And Traces

`PLPState` is the runtime object passed through a PLP program. It has three fields:

- `vars`: named natural-language variables;
- `log_weight`: the accumulated log score of the trace;
- `trace`: a list of `TraceEvent` objects.

```python
import plp

state = plp.PLPState(vars={"question": "2 + 2"})
print(state.log_weight)
print(state.is_valid())
```

A `TraceEvent` records the primitive name, prompt, output, and metadata. The trace is deliberately plain: it should be easy to inspect after a surprising answer.

## Copy Semantics

PLP primitives return new state objects rather than mutating the input state in place. The state copy is shallow: `vars` and `trace` are copied containers, while the string values inside are reused.

This keeps examples easy to reason about while still making trace branching cheap.
