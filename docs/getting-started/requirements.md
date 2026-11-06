# What A Scientific PLP Site Should Give You

A documentation site for a scientific library has two jobs. It must make the package easy to use, and it must make the scientific object behind the package legible.

For PLP, that means the site should include the following.

## A Clear User Path

A new user should be able to move through the site in this order:

1. Install the package in a clean Python 3.12 environment.
2. Run a no-key example that uses a scripted LLM object.
3. Inspect `PLPState`, `TraceEvent`, and `log_weight`.
4. Add `observe` and `factor` to understand hard and soft evidence.
5. Build a `ChainGraph` from decorated functions.
6. Let `choose_engine` select an inference engine.
7. Run a more complex workflow and interpret particles, weights, and failure modes.
8. Only then configure live LLM providers.

The no-key path matters because a scientific reader should not need an API key to understand the semantics.

## Complete Scientific Context

The site should explain the ideas that make PLP different from ordinary prompt engineering:

- LLM calls are stochastic sites.
- A run is a trace, not just a final string.
- `factor` and `observe` define an unnormalized target over traces.
- Plates represent repeated conditionally independent runs.
- Chain graphs expose the structure that decides which inference engine is appropriate.
- Weighted particles are approximations, not guarantees of truth.

## Usability Requirements

The documentation should be concrete enough for users to copy commands, but precise enough for researchers to trust:

- exact installation commands;
- documented environment variables;
- examples that compile without hosted LLM credentials;
- live examples isolated behind `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`;
- API reference generated from source;
- troubleshooting guidance for low effective sample size, correlated samples, weak judges, and provider failures.

## Style

PLP benefits from a serious but engaging style. The prose should avoid hype. It should show why the abstraction is useful, then expose the edge cases. A good PLP page starts with a small program, shows the trace, then names the probabilistic idea the program just demonstrated.

The goal is for a user to leave with two forms of understanding: how to write PLP programs, and how to reason about the science behind those programs.
