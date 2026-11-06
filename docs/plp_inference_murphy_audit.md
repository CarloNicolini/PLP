---
layout: post
title: "PLP Inference Audited Against Murphy"
description: "How the current implementation of PLP covers the Kevin Murphy's book"
date: 2026-04-21
published: true
---

Kevin Murphy's treatment of inference on pp. 435-471 of *Probabilistic Machine Learning: Advanced Topics* gives us a clean gold standard for what exact and approximate inference algorithms are supposed to mean. Those pages cover the forwards-backwards algorithm for hidden Markov models, sum-product and max-product message passing on trees, factor-graph belief propagation, loopy belief propagation and its caveats, variable elimination, and the junction-tree viewpoint. They do **not** present these methods as vague metaphors. They present them as specific operators with specific mathematical guarantees.

This post audits the current `src/plp/inference` implementation against that standard.

The conclusion is mixed, but encouraging. PLP has already made one very important move correctly: it separates a **proposal law** induced by prompt-generating sample sites from **factor potentials** induced by soft scoring functions. That is the right abstraction for probabilistic language programming. But once we move from semantics to algorithms, several current engine names and correctness claims are too strong. In particular, today's `LoopyParticleBP` is not belief propagation in Murphy's sense, and today's `MaxProduct` is not max-product message passing. They are useful particle heuristics, but they are not the textbook algorithms their names suggest.

If PLP wants to become the gold standard for mathematically correct prompting methods, the path is clear: keep the current chain-graph semantics, but distinguish sharply between

1. exact message-passing algorithms when the local distributions are explicit and tractable, and
2. particle or sampling heuristics when the local distributions are black-box LLM conditionals.

That distinction will make the library both more honest and more powerful.

## The textbook baseline

Murphy's pages 437-441 present the forwards and forwards-backwards recursions for a discrete HMM. The central equations are

$$
\alpha_{t|t-1}(j) = \sum_i A_{i,j}\alpha_{t-1}(i),
$$

$$
\alpha_t(j) = \frac{1}{Z_t}\lambda_t(j)\alpha_{t|t-1}(j),
\qquad
Z_t = \sum_j \lambda_t(j)\alpha_{t|t-1}(j),
$$

$$
\beta_{t-1}(i) = \sum_j A_{i,j}\lambda_t(j)\beta_t(j),
$$

and

$$
\gamma_t(j) \propto \alpha_t(j)\beta_t(j).
$$

The point is not just that these equations exist. The point is that they are **exact dynamic programs** on chain-structured models. Murphy is explicit that normalizing the forwards message at every step is the right implementation, both mathematically and numerically.

On pp. 448-453 the same idea is generalized from chains to trees and then to factor graphs. For a tree, the sum-product equations are

$$
\mathrm{bel}_s(z_s) \propto \psi_s(z_s)\prod_{t \in \mathrm{ch}_s} m_{t \to s}(z_s),
$$

$$
m_{s \to t}(z_t) = \sum_{z_s}\psi_{st}(z_s, z_t)\,\mathrm{bel}_s(z_s).
$$

For a factor graph, Murphy writes the message updates as

$$
m_{x \to f}(x) = \prod_{h \in \mathrm{nbr}(x)\setminus\{f\}} m_{h \to x}(x),
$$

$$
m_{f \to x}(x) = \sum_{x_{\setminus x}} f(x_{\mathrm{nbr}(f)})\prod_{x' \in \mathrm{nbr}(f)\setminus\{x\}} m_{x' \to f}(x'),
$$

$$
\mathrm{bel}(x) \propto \prod_{f \in \mathrm{nbr}(x)} m_{f \to x}(x).
$$

On pp. 455-457 he then makes the usual caution explicit: loopy BP can work well, but it may fail to converge, and damping or residual scheduling are engineering tools for managing that fixed-point iteration. Finally, pp. 462-468 give the exact variable-elimination and junction-tree story: exact inference is exponential in treewidth, elimination order matters, and message passing on trees is dynamic programming that avoids redundant recomputation.

That is the standard. So how does PLP compare?

## What PLP has done right

The core semantic move in PLP is good, and it is more important than it may look at first glance. The package models a prompt program as a directed proposal together with soft factors:

$$
p(\tau) \propto \pi_D(\tau)\,\Phi(\tau).
$$

Since factors in PLP contribute additive log-weights, the more explicit form is

$$
p(\tau) \propto \pi_D(\tau)\exp\left(\sum_f \ell_f(\tau_{\mathrm{pa}(f)})\right),
$$

where each factor returns a log-potential contribution. That matches the code exactly:

```python
raw_logf = fn(state, *args, **kwargs)
scaled_logf = raw_logf * scale

new_state = state.copy()
new_state.log_weight += scaled_logf
```

This is the right abstraction for probabilistic language programming because prompt sites naturally behave like conditional proposal kernels, while critique, preference, format, consistency, or task-success signals naturally behave like soft factors.

PLP also gets several structural decisions right.

First, it represents the model as a chain graph with directed sample-parent edges and undirected factor cliques. That is a principled way to unify directed prompt generation with undirected scoring couplings.

Second, the DAG-based engines enforce a topological order. That is the correct structural precondition for ancestral sampling:

```python
for site_name in order:
    particles = await propose_site_for_particles(
        graph, site_name, particles, **sample_kwargs,
    )
    apply_newly_ready_factors(graph, particles, applied)
```

Third, the implementation stores weights in log-space and uses stable normalization helpers such as `logmeanexp` and systematic resampling. Those are good numerical choices.

Fourth, `choose_engine` uses the moralized graph as a structural signal. That is a reasonable first heuristic. It is not the full exact-inference story, but it is much better than pretending the graph structure does not matter.

So the foundation is not the problem. The problem is that the current algorithms are sometimes named or described as though they were Murphy's algorithms, when they are really proposal-based particle approximations.

## The PLP target distribution

Before critiquing the engines, it helps to state clearly what they are trying to approximate.

If the `@sample` sites define a forward proposal

$$
\pi_D(\tau) = \prod_i q_i(\tau_i \mid \tau_{\mathrm{pa}(i)}),
$$

and the `@factor` sites contribute log-potentials \(\ell_f\), then the unnormalized target is

$$
\tilde{p}(\tau) = \pi_D(\tau)\exp\left(\sum_f \ell_f(\tau_f)\right).
$$

Under that interpretation, simple importance sampling from \(\pi_D\) with weight

$$
w(\tau) = \exp\left(\sum_f \ell_f(\tau_f)\right)
$$

is mathematically coherent. This is one of the strongest ideas in the current package: the proposal is the prompt-generation law, and the factor stack is the correction from "what the model says" to "what we want."

But this also means that exact message passing requires more than the current API exposes. Murphy's algorithms assume explicit local probability tables, potentials, or messages that can be multiplied, summed, and maximized over. For open-ended LLM text outputs, PLP often has only a black-box sampler. Once that is true, exact forwards-backwards or exact sum-product are no longer automatically available.

This is not a flaw. It is a fact. The mistake is when the library uses the names of exact message-passing algorithms for engines that do not implement those equations.

## Why exact inference is not the main benchmark for PLP

The most important correction to my original audit is this: for PLP, exact inference is usually **not** the right primary benchmark.

Murphy's exact algorithms are the correct ground truth for what words like "sum-product," "max-product," "forward-backward," and "junction tree" mean. But PLP lives in a radically harder regime. Its latent variables are often long strings, multi-step traces, or entire reasoning trajectories. The support of a single site may be astronomically large, and in many applications the notion of semantic correctness is itself noisy, contextual, or only partially observable. On open problems, there may not even be a crisp gold answer to condition on.

So the real lesson from Murphy is not that PLP should aspire to exact inference over arbitrary strings. It is that PLP should aspire to **mathematically explicit approximate inference**.

That means:

1. exact inference on tractable discrete submodels should be used as a **reference point** and calibration harness,
2. particle methods should be used when only black-box proposals are available,
3. variational methods should be used when we want a learnable approximation family and an optimization objective, and
4. every algorithm should be honest about whether it is exact, asymptotically consistent, heuristic, amortized, biased, or merely a search procedure.

This shift matters because it changes the standard by which PLP should be judged. The goal is not "recover Murphy's exact posterior over strings." The goal is "approximate the right target distribution with the clearest possible objective and the best possible computational tradeoff."

## A variational view of PLP

Murphy's variational framework on pp. 474-505 is almost tailor-made for the setting PLP actually inhabits. The central identity is

$$
\log p_\theta(x) = \mathbb{E}_{q_\psi(z)}[\log p_\theta(x,z) - \log q_\psi(z)] + D_{\mathrm{KL}}(q_\psi(z)\,\|\,p_\theta(z \mid x)).
$$

Equivalently,

$$
\mathrm{ELBO}(q_\psi)
=
\mathbb{E}_{q_\psi(z)}[\log p_\theta(x,z) - \log q_\psi(z)]
\le
\log p_\theta(x).
$$

The gap is exactly the reverse KL to the posterior. Murphy also rewrites the ELBO as

$$
\mathrm{ELBO}(q_\psi)
=
\mathbb{E}_{q_\psi(z)}[\log p_\theta(x,z)] + \mathbb{H}(q_\psi),
$$

or equivalently

$$
\mathrm{ELBO}(q_\psi)
=
\mathbb{E}_{q_\psi(z)}[\log p_\theta(x \mid z)]
-
D_{\mathrm{KL}}(q_\psi(z)\,\|\,p_\theta(z)).
$$

For PLP, the natural latent variable is not just a finite-dimensional vector \(z\). It is the full execution trace \(\tau\), possibly including intermediate drafts, debate states, plans, outputs, and even topology choices. If the PLP target is

$$
p(\tau \mid x) \propto \pi_D(\tau \mid x)\,\Phi(\tau, x),
$$

then with \(\log \Phi(\tau, x) = \sum_f \ell_f(\tau_f, x)\), the corresponding variational objective is

$$
\mathrm{ELBO}(q_\psi)
=
\mathbb{E}_{q_\psi(\tau \mid x)}
\left[
\log \pi_D(\tau \mid x)
+
\sum_f \ell_f(\tau_f, x)
-
\log q_\psi(\tau \mid x)
\right].
$$

This equation is more appropriate for PLP than exact tree message passing because it does not assume that the posterior is tractable, only that we can define a variational family \(q_\psi\) and optimize a lower bound or a surrogate to it.

In other words, the mathematically correct long-term story for PLP is probably not

$$
\text{prompt program} \longrightarrow \text{exact inference},
$$

but rather

$$
\text{prompt program} \longrightarrow \text{target over traces} \longrightarrow \text{variational / Monte Carlo approximation}.
$$

## Engine-by-engine audit

### `Forward`: right as ancestral sampling, wrong as posterior inference

The current `Forward` engine is a valid ancestral sampler on a DAG. It samples each site in topological order and applies factors after the relevant parents exist. As a sampler from the proposal law \(\pi_D\), it is correct.

But it is not Murphy's forwards algorithm. Murphy's forwards algorithm propagates a **belief state** \(\alpha_t(j) = p(z_t=j \mid y_{1:t})\) using matrix-vector recursion and normalization at each step. PLP's `Forward` propagates **particles**, not filtered marginals.

That difference matters. The code says:

```python
"""
Correct when the sample-parent graph is acyclic (Bayes net).
If factors are present, they are applied but weights are ignored.
"""
```

This is too loose. If there are no factors, then yes: the engine returns exact samples from the directed model. If factors are present, the engine is still a correct ancestral sampler from \(\pi_D\), but it is **not** an inference algorithm for the posterior target \(p(\tau) \propto \pi_D(\tau)\Phi(\tau)\). The factors are merely recorded; no resampling, normalization, or marginalization uses them.

So the right mathematical description is:

$$
\tau^{(k)} \sim \pi_D(\tau),
\qquad
\log w^{(k)} = \sum_f \ell_f(\tau_f^{(k)}),
$$

with no posterior correction step. That is useful, but it should be called what it is: forward proposal sampling, not textbook filtering.

### `ImportanceSampling`: coherent Monte Carlo, but not exact tree inference

The current `ImportanceSampling` engine is mathematically defensible, and in spirit it is the cleanest approximation in the package. It samples from the forward proposal and then weights by the factors:

$$
\bar{w}^{(k)} \propto \exp\left(\sum_f \ell_f(\tau_f^{(k)})\right).
$$

That is exactly what the code does before its optional final resampling step.

This is good probabilistic programming practice. But the docstring says it is "correct" for acyclic parents with tree-shaped factor cliques. Relative to Murphy, that is still too strong.

On a tree-structured discrete graphical model with explicit local probabilities, Murphy's answer is not importance sampling. It is sum-product, which gives exact marginals in linear time in the number of nodes and quadratic time in the local state size. The existence of a tree does **not** make importance sampling exact. It only makes exact dynamic programming available.

So `ImportanceSampling` is a good approximate engine for PLP's black-box proposal setting, but it should be described as

1. asymptotically consistent as \(K \to \infty\) under the PLP target above,
2. finite-sample and variance-sensitive, and
3. an approximation even on tree-structured graphs unless the proposal already equals the posterior.

That is not a small wording issue. It is the difference between a Monte Carlo approximation and an exact inference algorithm.

### `SMC`: a reasonable particle method, but outside the scope of these Murphy pages

The current `SMC` engine performs sequential propose-weight-resample updates across a topological order:

$$
w_t^{(k)} \propto \prod_{f \in \mathcal{F}_t} \exp\left(\ell_f(\tau_f^{(k)})\right),
$$

where \(\mathcal{F}_t\) denotes factors that become ready at step \(t\).

This is a sensible particle method for long directed chains with local factors. The implementation also acknowledges its own weakness: if important factors arrive late, early resampling can kill good trajectories too soon. That is exactly the right warning.

But Murphy pp. 435-471 do not present SMC as the reference algorithm here. On those pages, the exact chain-structured gold standard is forwards-backwards, not particle filtering. So the right comparison is not "SMC is wrong." The right comparison is "SMC is an approximate fallback for cases where exact message passing is unavailable or impractical."

For PLP, that can be the correct engineering choice, especially because LLM outputs are open-ended. But the library should say so explicitly: SMC is an approximation for black-box prompt kernels, not the canonical exact solution that Murphy develops for discrete chains.

### `LoopyParticleBP`: the biggest naming error in the module

This is the engine whose current name diverges most sharply from Murphy's theory.

Murphy's belief propagation, whether on trees or factor graphs, is a local message-passing algorithm. It updates messages \(m_{s \to t}\) or \(m_{f \to x}\) using sums, products, and possibly maxima over neighboring state spaces. The whole point of BP is that it passes **localized summaries** across edges.

PLP's `LoopyParticleBP` does something else:

```python
for t in range(T):
    particles = self._score_all_factors(graph, particles)
    log_z = logmeanexp([p.log_weight for p in particles])
    log_evidence.append(log_z)
    particles = systematic_resample(particles, K)
    if t < T - 1:
        particles = await repropose_by_groups(
            graph, particles, groups, sample_kwargs,
        )
```

This is not message passing. It is repeated full-particle scoring, resampling, and reproposal. No message objects exist. No variable-to-factor or factor-to-variable messages exist. No fixed-point equations of the form Murphy gives on p. 453 are implemented.

That does not mean the method is useless. It may be a practical heuristic for cyclic prompt systems. But mathematically it is much closer to a population Monte Carlo or rejuvenated particle heuristic than to belief propagation.

The docstring goes further and says:

> Across rounds this is Gibbs sampling on an implicit MRF defined by the conditional kernels \(K_i(y_i \mid y_{\mathrm{pa}(i)})\).

This is not justified by the code. A true Gibbs sampler draws each variable from its **full conditional under the target distribution**. Here, each site is redrawn from its prompt proposal, then the whole particle is rescored, then particles are resampled. There is no acceptance ratio and no proof that the update kernel leaves the desired target invariant. So the method should not be described as Gibbs unless that invariant-measure argument is actually established.

This is the clearest place where PLP should become more conservative and more precise. A better name would be something like `LoopyParticleReweighting`, `PopulationReproposal`, or `ParticleGraphHeuristic`.

### `MaxProduct`: useful best-of-\(K\), but not Murphy's max-product

Murphy's max-product algorithm replaces summation by maximization in a message-passing recursion and is closely related to Viterbi on chains. It computes max-marginals or MAP configurations through dynamic programming.

PLP's `MaxProduct` does this instead:

```python
indexed = sorted(
    enumerate(particles), key=lambda pair: pair[1].log_weight, reverse=True,
)
top = [p for _, p in indexed[:K]]
```

This is top-\(K\) selection among an oversampled set of proposal trajectories. In other words, if the engine draws \(N\) ancestral samples, then it returns

$$
\operatorname{TopK}\left\{\tau^{(1)}, \ldots, \tau^{(N)}\right\}
$$

ranked by total factor score. That is a perfectly reasonable best-of-\(K\) search heuristic. But it is not max-product message passing.

There is also a direct inconsistency between documentation and implementation. The file header claims a cyclic fallback based on Gibbs chains, but the implementation immediately enforces acyclicity and raises on cyclic parents. So even as documentation, the current file overpromises.

Again, the right fix is not to delete the engine. The right fix is to rename and reframe it, perhaps as `TopKProposalSearch` or `BeamMAP`.

## What Murphy's theory says PLP is currently missing

The biggest gap is not a bug in the current engines. It is the absence of the exact algorithms Murphy puts at center stage.

But after reading the variational chapter, the more precise statement is this: PLP is missing both

1. exact algorithms for the tractable subcases where they should exist, and
2. explicit variational objectives for the intractable string-valued cases where exactness is unrealistic.

### Missing exact chain inference

There is no engine in `src/plp/inference` that implements exact forwards-backwards smoothing or Viterbi-style dynamic programming on discrete chain models. That means the package currently lacks the exact analog of the most classical inference problem in the whole chapter.

For prompt programs with finite state spaces or enumerable local supports, PLP should have this. But this should be viewed primarily as a **calibration tier**: a way to validate semantics, compare approximations, and test claims on small structured models, not as the default inference story for open-ended language traces.

### Missing exact tree inference

There is no true sum-product or max-product implementation over tree-structured factor graphs. Today, a two-node tree with a pairwise factor is sent to `ImportanceSampling` by `choose_engine`. Murphy would send that model to exact sum-product immediately.

The current choice is understandable because LLM outputs are usually not enumerable. But the library should separate two cases:

1. black-box sample sites, where importance sampling may be the only practical option, and
2. explicit discrete sites, where exact sum-product should be available and preferred.

### Missing variable elimination and junction tree

Murphy's pp. 462-468 make an important point that PLP does not yet exploit: loops do not imply "use loopy BP." Some loopy graphs still admit efficient exact inference when the induced width is small enough. That is the whole point of variable elimination and the junction-tree algorithm.

Current engine selection is therefore too blunt:

```python
if graph.has_cycle_in_moralised_graph():
    return LoopyParticleBP
```

That is a fine heuristic for black-box prompt spaces. It is not a mathematically complete inference policy.

### Missing a genuine variational layer

Once we accept that PLP is mostly operating in intractable string spaces, the most important missing object is not another exact solver. It is an explicit variational family \(q_\psi\) over traces, together with a training or adaptation rule that can honestly be called VI.

This matters because Murphy's VI chapter gives a principled language for exactly the regime PLP is in:

1. the posterior is intractable,
2. exact normalization is unrealistic,
3. the approximate posterior should come from an optimization problem over distributions, and
4. the choice of approximation family matters as much as the inference algorithm.

Today PLP has pieces of this story, but not the full object.

## What PLP already has that looks variational

The strongest variational clue in the current codebase is `src/plp/optimizers/controller.py`, whose file-level docstring explicitly describes the controller as an "amortised variational proposal over chain-graph topologies."

The implementation says this very directly:

```python
"""
Circuit Controller — amortised variational proposal over chain-graph topologies.

In PLP notation, the controller is ``q_psi(topology | x, history)`` ...
"""
```

That is a promising direction. If a controller emits a topology \(c\) and perhaps other proposal choices conditioned on the input \(x\) and history \(h\), then one natural formalization is

$$
q_\psi(c,\tau \mid x,h)
=
q_\psi(c \mid x,h)\,q_\psi(\tau \mid c,x,h).
$$

This is already close in spirit to Murphy's amortized VI on pp. 478-479, where an inference network predicts variational parameters from the observed input instead of optimizing fresh local parameters from scratch each time.

The controller also keeps a scalar summary called `J_trajectory`, described as an "ELBO-style log-evidence" signal. Conceptually, this is heading in the right direction: the system wants a learned proposal mechanism that gets better over repeated tasks.

That signal is currently recorded as:

```python
J_trajectory: list[float]
...
"J_trajectory" is the per-round ELBO-style log-evidence recorded by
LoopyParticleBP (``logmeanexp_k(w_k^{(t)})``).
```

But mathematically the current story is still incomplete. `J_trajectory` is currently just a particle statistic based on `logmeanexp` of current weights. That is **not**, by itself, an ELBO. Murphy's ELBO requires an expectation of a joint term minus a log variational density term:

$$
\mathbb{E}_{q_\psi}[\log p - \log q_\psi].
$$

Without the \(-\log q_\psi\) term, and without a clear statement of what the variational family is, the controller is better described as an amortized **proposal search** or amortized **architecture search** mechanism than as full variational inference.

That is not a criticism of the idea. It is a request for one more mathematical step.

## What the correct VI comparison suggests for PLP

Murphy's VI chapter suggests a better benchmark for PLP than "implement exact inference over strings." The right benchmark is something like:

1. define the target distribution over traces clearly,
2. define the approximation family clearly,
3. define the optimization objective clearly,
4. characterize the bias introduced by the approximation family, and
5. use exact methods only on toy or structured subcases where they are genuinely available.

In that light, several current PLP design choices look better than they did under a pure exact-inference lens.

First, proposal-target separation is precisely what one wants before doing VI. One needs a tractable or at least sampleable proposal family before one can even talk about approximate posteriors.

Second, the controller is naturally interpreted as an **amortized inference network** over prompt-program structure.

Third, the particle engines can be reinterpreted as Monte Carlo approximations that may eventually support variational training, wake-sleep style updates, or importance-weighted objectives.

But the VI lens also exposes what is currently missing.

### No explicit \(\log q\) accounting

Murphy's ELBO depends on \(-\log q_\psi\). Current PLP sample sites return strings, traces, and weights, but they do not generally expose tractable log densities for the full proposal path. Without that, one cannot honestly compute the ELBO.

This is the single biggest mathematical blocker for a true VI story.

### No clearly defined variational family over traces

At the moment, PLP has sample sites, particle engines, and a controller. But it does not yet specify whether the approximate posterior family is

1. fully factorized over sites,
2. structured by the chain graph,
3. autoregressive over traces,
4. hierarchical over topology plus execution,
5. implicit rather than density-evaluable,
6. or some hybrid of the above.

Murphy emphasizes that the approximation family is central. For PLP, this is where a lot of the research novelty could live.

### No BBVI-style gradient estimator for discrete string latents

Murphy's black-box VI section is especially relevant here. He explicitly treats the case where we can evaluate \(\log p(x,z) - \log q_\psi(z)\) pointwise but cannot differentiate through the model, including discrete latent variables. The score-function estimator gives

$$
\nabla_\psi \mathrm{ELBO}
=
\mathbb{E}_{q_\psi(z)}
\left[
\hat{\mathcal{L}}(\psi, z)\,\nabla_\psi \log q_\psi(z)
\right],
$$

with control variates needed to reduce variance.

That is much closer to the PLP regime than standard reparameterized Gaussian VI. If PLP can obtain token-level or site-level log-probabilities from its proposal models, then BBVI becomes a plausible mathematical route for learning better prompt proposals over discrete traces.

### No structured variational family yet

Murphy's structured mean-field discussion is also directly relevant. He argues that fully factorized mean field is often too crude, and that one should preserve tractable blocks of dependence when possible. For time series this means sequence blocks; for PLP it could mean things like

1. a full reasoning chain as one block,
2. a debate panel as one coupled block,
3. a plan-code pair as one structured block,
4. topology choices as global variables and output traces as local variables.

That is likely a much better inductive bias than pretending all prompt sites are independent under the approximate posterior.

## A precise summary of what is right, wrong, and improvable

The cleanest way to summarize the audit is in a table.

| PLP component | Verdict | Why |
| --- | --- | --- |
| Chain-graph semantics | Right | The split between proposal law and factor potentials is mathematically sound and well suited to prompt programs. |
| Log-weight factors | Right | Additive log-factors correspond naturally to multiplicative potentials. |
| DAG enforcement for ancestral engines | Right | Topological ordering is the correct structural requirement. |
| Stable numerics and systematic resampling | Right | Good implementation discipline. |
| `Forward` naming as "forward inference" | Misleading | It is ancestral proposal sampling, not Murphy's filtered belief recursion. |
| `ImportanceSampling` correctness claims | Too strong | It is a valid approximation, not exact tree inference. |
| `LoopyParticleBP` name | Wrong | It does not implement BP messages or Murphy's factor-graph equations. |
| `LoopyParticleBP` "Gibbs" claim | Unjustified | The code does not sample exact full conditionals of the target. |
| `MaxProduct` name | Wrong | It is top-\(K\) sampled search, not max-product dynamic programming. |
| `choose_engine` cycle heuristic | Incomplete | It ignores exact VE/JTA possibilities on low-treewidth loopy graphs. |
| Amortized controller language | Promising but incomplete | It gestures at VI, but no explicit ELBO/KL over a defined \(q_\psi\) is currently optimized. |
| Test suite | Not enough for inference claims | Mostly structural and smoke tests; not enough exact-posterior validation. |

## The most important implementation snippets

To make the comparison concrete, it helps to place the implementation beside the theory.

Murphy's factor-graph BP on p. 453 is explicitly local:

```text
m_{x -> f}(x) = prod_{h in nbr(x) \ {f}} m_{h -> x}(x)
m_{f -> x}(x) = sum_{x' in nbr(f) \ {x}} f(x, x') prod m_{x' -> f}(x')
bel(x) propto prod_{f in nbr(x)} m_{f -> x}(x)
```

PLP's current "LBP" engine is global and particle-based:

```python
for p in particles:
    state = p
    for factor in graph.factors:
        state = apply_factor(factor, state)
    out.append(state)
```

Murphy's max-product is dynamic programming over local maxima. PLP's current `MaxProduct` is ranking over sampled trajectories:

```python
N = max(K * 2, 4)
particles = [base.copy() for _ in range(N)]
...
top = [p for _, p in indexed[:K]]
```

And Murphy's exact tree and junction-tree story says that cached local messages are what avoid repeated work. PLP's current exact-style caching is limited to "apply each factor once when ready" in DAG sweeps, which is useful but not a replacement for general message reuse:

```python
if all(p in particles[0].vars for p in factor.parents):
    particles[:] = [apply_factor(factor, p) for p in particles]
    applied.add(factor.name)
```

This is operationally clean, but it is not variable elimination, not forwards-backwards, and not junction-tree propagation.

## What PLP should do next to become the gold standard

The right roadmap is not to force everything into exact message passing. The right roadmap is to make the boundary between exact and approximate inference mathematically explicit.

### 1. Rename the current stochastic engines honestly

This is the fastest high-value improvement.

`Forward` should become something like `AncestralSampling` or `ForwardProposal`.

`ImportanceSampling` can keep its name, but its docstring should say "approximate posterior inference by self-normalized importance sampling."

`LoopyParticleBP` should be renamed to something that does not claim belief propagation, unless it is rewritten to actually pass messages.

`MaxProduct` should become `TopKProposalMAP`, `BestOfKMAP`, or `BeamMAP`.

Honest names are not cosmetic. They prevent users from making false theoretical assumptions.

### 2. Add an explicit variational layer over traces

For PLP's real operating regime, this is more important than adding more exact solvers.

The library should define a family \(q_\psi(\tau \mid x)\) or \(q_\psi(c,\tau \mid x)\) and say which of the following it supports:

1. density-evaluable autoregressive proposals,
2. implicit proposals that are sampleable but not density-evaluable,
3. amortized topology proposals,
4. structured block posteriors,
5. wake-sleep or importance-weighted surrogates when the plain ELBO is impractical.

Even if the first version is partial, having this object explicitly in the theory would dramatically improve the mathematical clarity of the project.

### 3. Treat exact discrete inference as a calibration and regression layer

To match Murphy properly, PLP should add exact engines for cases where local supports are tractable.

The minimal exact layer is:

1. `ForwardBackward` for discrete chains,
2. `Viterbi` for MAP on chains,
3. `TreeSumProduct` for tree-structured factor graphs,
4. `TreeMaxProduct` for tree MAP,
5. `VariableElimination` for exact marginals on small-treewidth graphs,
6. `JunctionTree` for reusable exact marginals when many queries are needed.

The key architectural requirement is that a sample site must optionally expose more than a sampler. It must expose either

1. an enumerable support together with local log-probabilities, or
2. a tractable local potential representation.

Without that, Murphy-style exact algorithms are impossible on open-ended LLM outputs. So these exact engines should be framed as a calibration tier for tractable subspaces, not as the default path for general prompt inference.

### 4. Introduce a two-tier API: black-box sites vs explicit sites

PLP currently treats all sample sites as generative black boxes. That is realistic for LLM calls, but it collapses together two very different use cases.

The library should support:

1. **black-box prompt sites**, for which only particle methods are available, and
2. **explicit probabilistic sites**, for which exact message passing becomes available.

This two-tier design would let PLP be mathematically exact where possible and pragmatically approximate where necessary.

### 5. If keeping a loopy particle engine, define its target and update law

If a particle rejuvenation method is going to remain in the core library, it should come with a precise target and update kernel.

At minimum, the documentation should specify whether each round is intended to approximate

$$
p(\tau) \propto \pi_D(\tau)\exp\left(\sum_f \ell_f(\tau_f)\right),
$$

and by what argument the update kernel preserves or approximates that target.

If no such argument exists, the method should be presented as a heuristic search-and-reweight scheme, not as Gibbs or BP.

### 6. Add BBVI- and amortization-aware objectives

Murphy's VI chapter points to three especially relevant directions for PLP:

1. **BBVI for discrete traces**, if site log-probabilities can be exposed,
2. **semi-amortized inference**, where a controller proposes a topology or trace family and local optimization refines it,
3. **importance-weighted bounds or wake-sleep style surrogates**, when direct ELBO optimization is too brittle.

This is where PLP could become genuinely novel: not by pretending strings are small discrete states, but by adapting the right approximate-inference machinery to prompt traces.

### 7. Add exact-vs-approximate tests on small closed-form models

This is the second fastest high-value improvement.

Right now the tests are mostly structural or smoke-based. To make inference claims credible, PLP should include small models with closed-form answers:

1. two-state HMMs with known filtering and smoothing marginals,
2. tiny tree factor graphs where exact sum-product can be computed by hand,
3. small loopy graphs where exact marginals are available by brute-force enumeration,
4. MAP tests where Viterbi or VE gives the known optimum.

Then each approximate engine can be benchmarked against exact reference values.

### 8. Move `choose_engine` from topology-only to capability-aware

Murphy's chapter is really about a triple:

1. graph structure,
2. algebra of local factors, and
3. query type.

PLP currently looks mostly at structure. A stronger selector would ask:

1. is the graph acyclic?
2. are local state spaces finite and enumerable?
3. is the moralized graph a tree?
4. what is the induced width?
5. is the query marginal, evidence, or MAP?

Only then should it pick between exact dynamic programming, exact elimination, or stochastic approximation.

## The deeper opportunity for PLP

The most exciting part of this audit is not the criticism. It is the opportunity.

Prompt engineering has lacked a semantics-first library that says, cleanly:

$$
\text{prompting method} = \text{proposal family} + \text{factor family} + \text{inference algorithm}.
$$

PLP is already very close to having that language. The proposal side is there. The factor side is there. The missing step is to align the algorithm names, approximation families, and guarantees with classical probabilistic inference.

If that alignment happens, PLP can become more than a useful experimentation library. It can become the place where prompting methods are described with the same mathematical discipline that we expect in graphical models:

1. exact when exact is possible,
2. approximate when approximation is necessary,
3. explicit about the target distribution,
4. explicit about invariance, bias, consistency, and convergence,
5. and benchmarked against textbook reference algorithms.

That is how probabilistic language programming stops being a metaphor and becomes a real inference discipline.

## Final verdict

PLP has done the hardest conceptual part correctly: it treats prompt generation as a probabilistic proposal process and prompt evaluation as factor-based reweighting on a chain graph. That is a strong and publishable abstraction.

What it has **not** done yet is implement Murphy's exact inference algorithms under those names. More importantly, it has not yet turned its promising "amortized variational proposal" language into an explicit variational inference framework with a defined \(q_\psi\), an ELBO or surrogate objective, and clear approximation guarantees. The current module mostly contains

1. ancestral proposal sampling,
2. importance sampling,
3. sequential Monte Carlo,
4. a loopy particle heuristic, and
5. a top-\(K\) MAP heuristic.

Those are valuable tools. But they should be presented as such.

For PLP, the gold standard should therefore not be "exact inference over arbitrary strings." It should be "principled approximate inference over traces, with exact methods used where they are actually available."

The path to becoming the gold standard is therefore straightforward:

1. keep the current PLP semantics,
2. rename the approximate engines honestly,
3. define an explicit variational family over prompt traces and topology choices,
4. add exact discrete message-passing and elimination algorithms where tractable,
5. expose the local support and scoring interfaces needed for exact inference and BBVI-style training,
6. and validate every approximate engine against exact toy references.

Murphy's message-passing chapter gives the calibration standard. Murphy's variational chapter gives the more realistic research direction. PLP already has the right object. The next step is to make the algorithms and objectives match the mathematics as closely as the abstraction already does.
