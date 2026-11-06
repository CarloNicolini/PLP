"""
PLP guarantee primitive.

Provides a local robustness wrapper that repeats a stochastic
subcomputation and chooses a representative output under a fixed error
budget.

The wrapper is conceptually three orthogonal pieces:

  * ``repeat_select`` -- the "generate N candidates, pick one" loop.
  * :class:`Selector` -- the policy deciding which candidate wins
    (``MajoritySelector`` with optional ``group_by`` for canonical
    equivalence, or ``SPRTSelector`` for judge-driven voting).
  * :func:`required_k_for_eps` / :func:`achieved_eps_from_k` -- honest
    accounting of what eps the SPRT vote margin really buys.

``@guarantee`` wires these together for the common case but the pieces
can be used individually.
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Hashable, Protocol

from ._llm import LLMLike, llm_name
from .handlers import rng_for
from .state import PLPState, TraceEvent


# ---------------------------------------------------------------------------
# Eps <-> k accounting
# ---------------------------------------------------------------------------


def required_k_for_eps(p: float, eps: float) -> int:
    """Minimum SPRT vote margin needed to bound the error below ``eps``.

    Assumes a correct-answer probability ``p`` in (0.5, 1.0) and a target
    failure rate ``eps`` in (0, 1).
    """
    if not 0.5 < p < 1.0:
        raise ValueError(f"p must lie in (0.5, 1.0), got {p}")
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps must lie in (0.0, 1.0), got {eps}")
    return math.ceil(math.log((1 - eps) / eps) / math.log(p / (1 - p)))


def achieved_eps_from_k(p: float, k: int) -> float:
    """Invert :func:`required_k_for_eps`: the eps a margin ``k`` actually buys."""
    if not 0.5 < p < 1.0:
        raise ValueError(f"p must lie in (0.5, 1.0), got {p}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    ratio = (p / (1.0 - p)) ** k
    return 1.0 / (1.0 + ratio)


# Backwards-compatible private alias.
_compute_k_for_eps = required_k_for_eps


# ---------------------------------------------------------------------------
# Selector protocol + built-ins
# ---------------------------------------------------------------------------


class Selector(Protocol):
    """Policy that picks one candidate out of a list of strings.

    The concrete object must expose:

    * ``name``: short identifier recorded in the trace.
    * ``select(candidates, *, rng) -> (index, info)``: ``info`` is a dict
      of diagnostic data (group sizes, SPRT round count, etc.) copied
      into the trace ``meta``.
    """

    name: str

    def select(
        self,
        candidates: list[str],
        *,
        rng: random.Random,
    ) -> tuple[int, dict[str, Any]]: ...


@dataclass
class MajoritySelector:
    """Pick the candidate whose canonical key is most common.

    ``group_by`` is applied to each raw candidate string to obtain a
    hashable key; ties are broken by first-appearance order. This is
    the right primitive when you want to collapse whitespace / casing /
    canonical JSON before voting.
    """

    group_by: Callable[[str], Hashable] = lambda s: s
    name: str = "majority"

    def select(
        self,
        candidates: list[str],
        *,
        rng: random.Random,
    ) -> tuple[int, dict[str, Any]]:
        if not candidates:
            raise ValueError("MajoritySelector needs at least one candidate")

        counts: dict[Hashable, int] = {}
        first_index: dict[Hashable, int] = {}
        for i, cand in enumerate(candidates):
            key = self.group_by(cand)
            counts[key] = counts.get(key, 0) + 1
            first_index.setdefault(key, i)

        winning_key = max(counts.items(), key=lambda kv: (kv[1], -first_index[kv[0]]))[
            0
        ]
        idx = first_index[winning_key]
        info: dict[str, Any] = {
            "group_sizes": dict(counts),
            "winning_key": winning_key,
            "num_unique_groups": len(counts),
        }
        return idx, info


@dataclass
class SPRTSelector:
    """Sequential pairwise voting with a vote-margin stopping rule."""

    judge_fn: Callable[[str, str], str]
    k_target: int
    max_rounds: int = 1000
    name: str = "sprt_discrete"

    def select(
        self,
        candidates: list[str],
        *,
        rng: random.Random,
    ) -> tuple[int, dict[str, Any]]:
        if not candidates:
            raise ValueError("SPRTSelector needs at least one candidate")
        if self.k_target <= 0:
            raise ValueError(f"k_target must be positive, got {self.k_target}")

        if len(candidates) == 1:
            return 0, {
                "k_target": self.k_target,
                "rounds_used": 0,
                "final_scores": [0],
                "stopped_early": True,
            }

        scores = [0] * len(candidates)
        rounds = 0
        stopped = False
        for _ in range(self.max_rounds):
            rounds += 1
            i, j = rng.sample(range(len(candidates)), 2)
            verdict = self.judge_fn(candidates[i], candidates[j])
            if verdict not in ("a", "b", "tie"):
                raise ValueError(
                    f"judge_fn must return 'a', 'b', or 'tie', got {verdict!r}"
                )
            if verdict == "a":
                scores[i] += 1
                scores[j] -= 1
            elif verdict == "b":
                scores[j] += 1
                scores[i] -= 1

            best_idx = max(range(len(candidates)), key=scores.__getitem__)
            other = [scores[t] for t in range(len(candidates)) if t != best_idx]
            second = max(other) if other else float("-inf")
            if scores[best_idx] - second >= self.k_target:
                stopped = True
                break

        best_idx = max(range(len(candidates)), key=scores.__getitem__)
        info: dict[str, Any] = {
            "k_target": self.k_target,
            "rounds_used": rounds,
            "final_scores": list(scores),
            "stopped_early": stopped,
        }
        return best_idx, info


def repeat_select(
    fn: Callable[..., tuple[PLPState, str]],
    state: PLPState,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    n: int,
    selector: Selector,
    independent: bool = True,
    rng: random.Random | None = None,
) -> tuple[PLPState, str, dict[str, Any]]:
    """Run ``fn`` ``n`` times, then pick one candidate via ``selector``.

    Returns ``(new_state, output, info)`` where ``new_state`` contains
    all sub-traces concatenated (so replay still sees each call) and
    ``info`` carries diagnostic data from the selector plus the full
    list of candidate outputs.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if rng is None:
        rng = rng_for("guarantee.sprt")

    candidates: list[str] = []
    candidate_states: list[PLPState] = []
    candidate_traces: list[list[TraceEvent]] = []
    running_state = state

    for _ in range(n):
        base_state = state.copy() if independent else running_state
        base_trace_len = len(base_state.trace)
        s2, out = fn(base_state, *args, **kwargs)
        if not independent:
            running_state = s2
        candidates.append(out)
        candidate_states.append(s2)
        candidate_traces.append(s2.trace[base_trace_len:])

    idx, selector_info = selector.select(candidates, rng=rng)

    new_state = candidate_states[idx].copy()
    new_state.trace = list(state.trace)
    for trace_suffix in candidate_traces:
        new_state.trace.extend(trace_suffix)

    info: dict[str, Any] = {
        "selector": selector.name,
        "selector_info": selector_info,
        "num_calls": len(candidates),
        "candidate_outputs": list(candidates),
        "num_unique_candidates": len(set(candidates)),
        "selected_index": idx,
        "independent_replicas": independent,
    }
    return new_state, candidates[idx], info


# Public @guarantee decorator


def guarantee(
    llm: LLMLike,
    eps: float,
    judge_fn: Callable[[str, str], str] | None = None,
    method: str = "sprt_discrete",
    assumed_p: float = 0.7,
    max_calls: int = 20,
    independent: bool = True,
    group_by: Callable[[str], Hashable] | None = None,
    selector: Selector | None = None,
):
    """
    Decorator for local guarantee wrappers.

    Wraps a PLP subcomputation with repetition + voting, targeting a
    failure probability of ``eps``. Three orthogonal knobs:

    * *repetition*: ``max_calls`` bounds the total calls; the effective
      sample count is ``2 * k_effective + 1`` where ``k_effective`` is
      clamped by ``max_calls // 2``.
    * *selection*: pass ``selector=`` directly for full control.
      Otherwise a selector is built from ``judge_fn`` + ``method`` +
      ``group_by`` (None means byte-equality, matching the historical
      default).
    * *eps bookkeeping*: when the SPRT margin is clamped by
      ``max_calls``, the achieved eps rises above ``eps``. We record
      ``eps_requested``, ``eps_achieved`` and ``clamped`` in the trace
      and warn the caller.

    Args:
        llm: LLM client (for metadata logging).
        eps: Target failure probability (e.g., 1e-3).
        judge_fn: Optional ``(a, b) -> "a"|"b"|"tie"`` comparison.
        method: "sprt_discrete" or "majority".
        assumed_p: Assumed probability of correct output (default 0.7).
        max_calls: Upper bound on repeats.
        independent: If True, each replica starts from a fresh copy of
            the input state.
        group_by: Optional canonicalisation function for majority voting
            (e.g. ``lambda s: s.strip().lower()``).
        selector: Explicit :class:`Selector` instance. If set, overrides
            ``judge_fn`` / ``method`` / ``group_by``.

    Example:
        >>> @guarantee(llm, eps=1e-3, group_by=lambda s: s.strip().lower())
        ... def robust_answer(state, q):
        ...     return propose_answer(state, q)
    """

    def decorator(fn: Callable[..., tuple[PLPState, str]]):
        prim_name = f"guarantee({fn.__name__})"

        @wraps(fn)
        def wrapped(state: PLPState, *args, **kwargs) -> tuple[PLPState, str]:
            # --- eps accounting -------------------------------------------------
            try:
                k_required = required_k_for_eps(assumed_p, eps)
            except ValueError:
                k_required = 3

            k_cap = max(2, max_calls // 2)
            k_effective = min(k_required, k_cap)
            eps_achieved = achieved_eps_from_k(assumed_p, k_effective)
            clamped = k_effective < k_required

            if clamped:
                warnings.warn(
                    f"@guarantee on {fn.__name__}: requested eps={eps:.2e} "
                    f"needed k={k_required} replicas but max_calls={max_calls} "
                    f"caps it at k={k_effective}; achieved eps={eps_achieved:.2e}.",
                    stacklevel=2,
                )

            # --- selector wiring -----------------------------------------------
            chosen_selector: Selector
            if selector is not None:
                chosen_selector = selector
            elif judge_fn is not None and method == "sprt_discrete":
                chosen_selector = SPRTSelector(judge_fn=judge_fn, k_target=k_effective)
            else:
                chosen_selector = MajoritySelector(
                    group_by=group_by if group_by is not None else (lambda s: s)
                )

            # --- repetition loop -----------------------------------------------
            n = min(max_calls, 2 * k_effective + 1)
            rng = rng_for("guarantee.sprt")
            new_state, best, info = repeat_select(
                fn,
                state,
                args,
                kwargs,
                n=n,
                selector=chosen_selector,
                independent=independent,
                rng=rng,
            )

            # --- trace event ----------------------------------------------------
            event_meta: dict[str, Any] = {
                "eps_requested": eps,
                "eps_achieved": eps_achieved,
                "clamped": clamped,
                "method": method,
                "assumed_p": assumed_p,
                "k_required": k_required,
                "k_effective": k_effective,
                "max_calls": max_calls,
                "llm": llm_name(llm),
            }
            event_meta.update(info)
            new_state.trace.append(
                TraceEvent(
                    primitive=prim_name,
                    prompt="(guarantee wrapper)",
                    output=best,
                    meta=event_meta,
                )
            )
            return new_state, best

        return wrapped

    return decorator
