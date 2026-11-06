# PLP Inference Zoo

The classical prompting strategies in the LLM literature — chain-of-thought,
self-consistency, best-of-K, self-refine, tree-of-thoughts, multi-agent
debate, amortised structured variational inference — all fall out of one
framework: a PLP chain graph plus a choice of inference engine. This
document gives the mapping.

## 1. The framework in one page

A PLP program defines

```
p_D(tau | x) ∝ pi_D(tau | x) · Phi(tau, x)
```

where `pi_D` is the forward execution law (induced by `@sample` sites
with declared `parents`) and `Phi` is the potential (induced by
`@factor` sites). Together these form a **chain graph** (Lauritzen and
Wermuth, 1989).

Three cases matter for inference:

| Case | Sample-parent graph | Factor graph | Joint class | Engine |
|------|---------------------|--------------|-------------|--------|
| A    | DAG                 | no factors   | Bayes net   | `Forward` |
| A+C  | DAG                 | tree cliques | chain graph, tree-like | `ImportanceSampling` |
| A+C  | DAG (long chain)    | per-site unary | chain graph, Markovian | `SMC` |
| A+C  | DAG                 | cyclic cliques | chain graph, loopy | `LoopyParticleBP` |
| B    | cyclic              | any          | MRF         | `LoopyParticleBP` |
| any  | MAP requested       | any          | any         | `MaxProduct` |

Rule of thumb:
  * Start with the topology you need.
  * Let `plp.inference.choose_engine` pick the algorithm.
  * Override the choice with `engine=<EngineClass>` if you know better.

## 2. Minimal template

```python
from plp import asample, factor, PLPChatClient, PLPState
from plp.inference import ChainGraph, choose_engine

llm = PLPChatClient(model="openai/gpt-4.1-mini")

@asample(llm, name="answer", parents=[])
def answer(state: PLPState, question: str) -> str:
    return f"Answer concisely: {question}"

graph = ChainGraph.from_decorated(answer, output_site="answer")
engine = choose_engine(graph)
particles = await engine().run(graph, K=1, question="What is 7 * 8?")
print(particles[0].vars["answer"])
```

`from_decorated` reads the `parents` metadata from each decorator.
Everything else is inferred by the engine selector.

## 3. Classical strategies as chain-graph parameterisations

### 3.1 Zero-shot / Chain-of-Thought (Case A)

A single sample site — or a short directed chain. No factors. The
selector picks `Forward`.

```python
@asample(llm, name="reason", parents=[])
def reason(state, question): return f"Think step by step: {question}"

@asample(llm, name="answer", parents=["reason"])
def answer(state, question):
    return f"Reasoning:\n{state.vars['reason']}\n\nFinal answer: {question}"

graph = ChainGraph.from_decorated(reason, answer, output_site="answer")
# choose_engine(graph) => Forward, one pass.
```

### 3.2 Self-Consistency (Wang et al., 2022)

K independent samples from one site with no factors. Majority-vote
outside the graph.

```python
graph = ChainGraph.from_decorated(answer_site, output_site="answer")
particles = await choose_engine(graph)().run(graph, K=20, question=q)
answers = [p.vars["answer"] for p in particles]
consensus = Counter(answers).most_common(1)[0][0]
```

### 3.3 Best-of-K with a judge (MAP)

One site plus a unary judge factor. Ask for MAP via
`prefer_map=True` to get the top-K by weight instead of resampled
posterior draws.

```python
@asample(worker, name="summary", parents=[]) ...
@factor(name="quality", parents=["summary"]) ...  # LLM-judge score

graph = ChainGraph.from_decorated(summary_site, quality, output_site="summary")
engine = choose_engine(graph, prefer_map=True)  # -> MaxProduct
particles = await engine().run(graph, K=8, text=text)
best = particles[0]  # MaxProduct returns top-K by weight, descending
```

### 3.4 Self-Refine / Reflexion (Case B)

A site that declares itself as a parent → cyclic sample-parent graph
→ MRF → `LoopyParticleBP`. Each BP round re-samples the site given its
previous draft.

```python
@asample(llm, name="draft", parents=["draft"])
def draft(state, task):
    prev = state.vars.get("draft", "")
    if prev:
        return f"Improve this draft:\n{prev}\n\nTask: {task}"
    return f"First draft of: {task}"

graph = ChainGraph.from_decorated(draft, output_site="draft")
# choose_engine(graph) => LoopyParticleBP (cycle detected in parents).
particles = await engine().run(graph, K=1, T=3, task="...")
```

### 3.5 Tree of Thoughts

A shallow tree of sample sites with a unary heuristic factor per
node. Long chain + per-site factors → `SMC`, which resamples after
each site (the particle-filter interpretation of beam pruning).

```python
@asample(llm, name="step1", parents=[])                     ...
@asample(llm, name="step2", parents=["step1"])              ...
@asample(llm, name="step3", parents=["step2"])              ...
@factor(name="f1", parents=["step1"])                        ...
@factor(name="f2", parents=["step2"])                        ...
@factor(name="f3", parents=["step3"])                        ...
# choose_engine => SMC.
```

### 3.6 Debate (multi-LLM with engagement factors)

Case C loopy: panelists read each other via `parents`; pairwise
engagement factors form a ring of dependencies; the moralised graph
is cyclic. `choose_engine` picks `LoopyParticleBP`.

See `examples/multimodel_debate.py` for a worked instance with five
different OpenRouter models and six judge-evaluated factors.

### 3.7 Soft-NAND chain-graph search (demo)

Controller-proposed topology (JSON) with mixed python/judge factors and
NAND edges lives in `examples/soft_nand_circuit.py`: compile with
`build_chain_graph`, run with `run_chain_spec` / `choose_engine` inside
that module. Engine selection follows the usual `choose_engine` rules

  * e.g. single site + unary factors → `ImportanceSampling`;
  * cyclic parents or loopy moralised graphs → `LoopyParticleBP`.

```python
# See examples/soft_nand_circuit.py — async beam_search(question, workers, controller, judge_llm, ...)
```

## 4. When to override the selector

The selector is a heuristic, not a law. Cases where explicit override
makes sense:

  * **You want MAP**: `engine=MaxProduct` or `prefer_map=True`.
  * **You trust a dense factor graph to converge without BP rounds**:
    pass `engine=ImportanceSampling` to skip the iterative loop.
  * **You are benchmarking**: hold the engine fixed across topologies
    so the only variable is graph structure.

## 5. The engines

Each engine in `plp.inference.engines` exposes:

```python
class Engine(Protocol):
    name: str
    async def run(
        self, graph: ChainGraph, *,
        K: int = 1, T: int = 1, init_state: PLPState | None = None,
        **sample_kwargs: Any,
    ) -> list[PLPState]: ...
```

  * `Forward`: K independent forward passes. Applies factors once
    their parents are instantiated. No resampling.
  * `ImportanceSampling`: same forward pass, then one systematic
    resample at the end.
  * `SMC`: forward + factor application + resample, per site.
  * `LoopyParticleBP`: T rounds of score / resample / re-propose.
    Uses graph colouring over the moralised graph to re-propose
    conditionally-independent sites concurrently.
  * `MaxProduct`: forward with over-sampling; top-K by log-weight at
    the end instead of stochastic resampling.

All engines record a trace event on each returned particle, including
the log-evidence diagnostic `logsumexp(w) - log K`.

## 6. Relation to classical BP (Murphy ch. 22)

Classical loopy sum-product BP maintains messages on the factor graph
and iterates:

```
mu_{x -> f}(x) = prod_{g != f} mu_{g -> x}(x)
mu_{f -> x}(x) = sum_{y_c \ x} psi(y_c) prod_{y != x} mu_{y -> f}(y)
```

For LLM-valued variables these messages are intractable, so we replace
them with particle populations. The correspondence:

| Classical BP   | PLP particle-based BP |
|----------------|----------------------|
| Variable node `x_i` | `@asample` site |
| Factor node `psi_c` | `@factor` site |
| Message `mu_{f -> x}` | Factor log-weight contribution on each particle |
| Belief `b(x_i)`      | Empirical distribution of resampled particles |
| One BP iteration     | SCORE → RESAMPLE → RE-PROPOSE |

This is what `LoopyParticleBP` does. For tree-shaped graphs a single
round is enough (BP on trees is exact). For loopy graphs, convergence
is not guaranteed, but the log-evidence `logsumexp(w) - log K`
monotonically tracks the population's concentration on high-scoring
joints and gives a cheap sanity check.

## 7. References

  * Lauritzen, S. and Wermuth, N. (1989). Graphical models for associations
    between variables, some of which are qualitative and some quantitative.
    *Annals of Statistics* 17(1): 31–57.
  * Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*.
    MIT Press. (Chapter 22 on belief propagation.)
  * Del Moral, P., Doucet, A., and Jasra, A. (2006). Sequential Monte
    Carlo samplers. *JRSS B* 68(3): 411–436.
  * Wang, X. et al. (2022). Self-consistency improves chain-of-thought
    reasoning in language models. *arXiv:2203.11171*.
  * Yao, S. et al. (2023). Tree of Thoughts: deliberate problem solving
    with large language models. *NeurIPS 36*.
  * Madaan, A. et al. (2023). Self-Refine: iterative refinement with
    self-feedback. *NeurIPS 36*.
