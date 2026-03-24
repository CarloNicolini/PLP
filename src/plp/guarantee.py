"""
PLP guarantee primitive.

Provides a local robustness wrapper that repeats a stochastic subcomputation
and chooses a representative output under a fixed error budget.
"""

from functools import wraps
import math
import random
from typing import Callable

from ._llm import LLMLike, llm_name
from .state import PLPState, TraceEvent


def _compute_k_for_eps(p: float, eps: float) -> int:
    """Convert an error target into a discrete vote margin."""
    if not 0.5 < p < 1.0:
        raise ValueError(f"p must lie in (0.5, 1.0), got {p}")
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps must lie in (0.0, 1.0), got {eps}")
    return math.ceil(math.log((1 - eps) / eps) / math.log(p / (1 - p)))


def _sprt_discrete(
    candidates: list[str],
    judge_fn: Callable[[str, str], str],
    k: int,
    *,
    rng: random.Random | None = None,
    max_rounds: int = 1000,
) -> str:
    """Select a candidate by repeated pairwise judging."""
    if not candidates:
        raise ValueError("No candidates provided")
    if not callable(judge_fn):
        raise TypeError(f"judge_fn must be callable, got {type(judge_fn)}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if rng is None:
        rng = random.Random()

    if len(candidates) == 1:
        return candidates[0]

    scores = [0] * len(candidates)
    for _ in range(max_rounds):
        i, j = rng.sample(range(len(candidates)), 2)
        verdict = judge_fn(candidates[i], candidates[j])
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
        other_scores = [scores[t] for t in range(len(candidates)) if t != best_idx]
        second_best = max(other_scores) if other_scores else float("-inf")
        if scores[best_idx] - second_best >= k:
            return candidates[best_idx]

    best_idx = max(range(len(candidates)), key=scores.__getitem__)
    return candidates[best_idx]


def guarantee(
    llm: LLMLike,
    eps: float,
    judge_fn: Callable[[str, str], str] | None = None,
    method: str = "sprt_discrete",
    assumed_p: float = 0.7,
    max_calls: int = 20,
):
    """
    Decorator for local guarantee wrappers.

    Wraps a PLP subcomputation to ensure its failure probability is
    bounded by eps, using either SPRT-based voting or simple majority.

    The wrapped function should return (state, output_text). This decorator:
    - Repeatedly calls the wrapped function up to max_calls times
    - If judge_fn is provided and method == "sprt_discrete", uses SPRT voting
    - Otherwise, falls back to simple majority over identical strings
    - Returns (new_state, best_output)

    Args:
        llm: LLM client (for metadata logging)
        eps: Target failure probability (e.g., 1e-3)
        judge_fn: Optional (a, b) -> "a"|"b"|"tie" comparison function
        method: Inference method ("sprt_discrete" or "majority")
        assumed_p: Assumed probability of correct output (default: 0.7)
        max_calls: Maximum number of calls to wrapped function

    Example:
        >>> @guarantee(llm, eps=1e-3, judge_fn=my_judge)
        ... def robust_arithmetic(state: PLPState, problem: str) -> tuple[PLPState, str]:
        ...     return arithmetic_solver(state, problem)
    """

    def decorator(fn: Callable[..., tuple[PLPState, str]]):
        prim_name = f"guarantee({fn.__name__})"

        @wraps(fn)
        def wrapped(state: PLPState, *args, **kwargs) -> tuple[PLPState, str]:
            # Compute k from eps and assumed_p
            try:
                k = _compute_k_for_eps(assumed_p, eps)
            except ValueError:
                k = 3  # fallback

            # Clamp k by max_calls
            k = min(k, max_calls // 2)
            k = max(k, 2)

            # Collect candidates
            num_samples = min(max_calls, 2 * k + 1)
            candidates: list[str] = []
            last_state = state

            for _ in range(num_samples):
                s2, out = fn(last_state, *args, **kwargs)
                last_state = s2
                candidates.append(out)

            # Select best candidate
            if judge_fn is not None and method == "sprt_discrete":
                best = _sprt_discrete(candidates, judge_fn, k)
            else:
                # Majority vote on identical strings
                counts: dict[str, int] = {}
                for c in candidates:
                    counts[c] = counts.get(c, 0) + 1
                best = max(counts.items(), key=lambda kv: kv[1])[0]

            # Log the guarantee wrapper
            new_state = last_state.copy()
            event = TraceEvent(
                primitive=prim_name,
                prompt="(guarantee wrapper)",
                output=best,
                meta={
                    "eps": eps,
                    "method": method,
                    "k": k,
                    "num_calls": len(candidates),
                    "llm": llm_name(llm),
                },
            )
            new_state.trace.append(event)

            return new_state, best

        return wrapped

    return decorator

