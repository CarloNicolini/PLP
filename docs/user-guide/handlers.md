# Handlers

Handlers are supporting tools for controlling execution. They are useful for tests, counterfactual runs, and debugging.

| Handler | Use |
| --- | --- |
| `replay` | Reuse previously recorded outputs for named sites. |
| `block` | Skip selected primitives while still recording that they were blocked. |
| `do` | Intervene on a named site with a forced value. |
| `condition` | Condition execution on named observations. |
| `seed` | Control randomness used by PLP internals. |

Handlers are powerful because they change execution semantics. In documentation and experiments, prefer making the intervention explicit in the surrounding prose.
