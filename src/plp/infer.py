"""
PLP Inference Engines.

Provides inference algorithms for PLP:
- run_map: Naive MAP by multiple runs
- sprt_discrete: First-to-ahead-by-k voting over candidate strings
"""

import random
from typing import Callable, Any

from .state import PLPState
from .client import LLMClient


class InferEngine:
    """
    Collection of inference kernels for PLP.

    These algorithms approximate MAP or posterior solutions by:
    - Running the model multiple times (run_map)
    - Voting among discrete candidates (sprt_discrete)

    Example:
        >>> engine = InferEngine(llm)
        >>> best_state, best_output = engine.run_map(model_fn, initial_state, runs=5)
    """

    def __init__(self, llm: LLMClient):
        """
        Initialize the inference engine.

        Args:
            llm: LLM client to use for any inference-time LLM calls
        """
        self.llm = llm

    def run_map(
        self,
        model_fn: Callable[[PLPState, Any], tuple[PLPState, str]],
        init_state: PLPState,
        runs: int,
        *args,
        **kwargs,
    ) -> tuple[PLPState, str]:
        """
        Naive MAP over full executions.

        Re-runs model_fn from init_state 'runs' times (the model_fn is
        expected to be stochastic via LLM calls), collects (final_state, output),
        and returns the one with maximum log_weight.

        Args:
            model_fn: PLP model function returning (state, output)
            init_state: Initial state to start each run from
            runs: Number of independent runs
            *args: Positional arguments passed to model_fn
            **kwargs: Keyword arguments passed to model_fn

        Returns:
            Tuple of (best_state, best_output) with highest log_weight
        """
        best_state: PLPState | None = None
        best_out: str | None = None
        best_weight = float("-inf")

        for _ in range(runs):
            state_copy = init_state.copy()
            final_state, out = model_fn(state_copy, *args, **kwargs)

            if final_state.log_weight > best_weight:
                best_weight = final_state.log_weight
                best_state = final_state
                best_out = out

        if best_state is None:
            # All runs failed (log_weight = -inf for all)
            return init_state.copy(), ""

        return best_state, best_out

    @staticmethod
    def sprt_discrete(
        candidates: list[str],
        judge_fn: Callable[[str, str], str],
        k: int,
        rng: random.Random | None = None,
        max_rounds: int = 1000,
    ) -> str:
        """
        SPRT-style first-to-ahead-by-k voting over candidate strings.

        Given a set of candidate strings and a pairwise judge function,
        repeatedly samples pairs and updates scores until one candidate
        is ahead of all others by at least k votes.

        This implements the core inference kernel from the "Million Steps"
        paper: if the judge prefers the correct candidate with probability
        p > 0.5, the probability of selecting the correct candidate is:

            p_win = 1 / (1 + (1-p) / p^k)

        Args:
            candidates: List of candidate strings to vote among
            judge_fn: (a, b) -> "a" | "b" | "tie" comparison function
            k: Vote margin threshold for stopping
            rng: Optional random number generator for reproducibility
            max_rounds: Safety limit on number of comparison rounds

        Returns:
            Winning candidate string

        Example:
            >>> def deterministic_judge(a, b):
            ...     a_sorted = sorted(a.split())
            ...     b_sorted = sorted(b.split())
            ...     if a_sorted == ["1", "2", "3"] and b_sorted != ["1", "2", "3"]:
            ...         return "a"
            ...     elif b_sorted == ["1", "2", "3"]:
            ...         return "b"
            ...     return "tie"
            ...
            >>> winner = InferEngine.sprt_discrete(
            ...     candidates=["1 2 3", "3 2 1", "2 1 3"],
            ...     judge_fn=deterministic_judge,
            ...     k=3
            ... )
        """
        if rng is None:
            rng = random.Random()

        n = len(candidates)
        if n == 0:
            raise ValueError("No candidates provided")
        if n == 1:
            return candidates[0]

        scores = [0] * n

        for _ in range(max_rounds):
            # Sample a random pair
            i, j = rng.sample(range(n), 2)
            a, b = candidates[i], candidates[j]

            verdict = judge_fn(a, b)

            if verdict == "a":
                scores[i] += 1
                scores[j] -= 1
            elif verdict == "b":
                scores[j] += 1
                scores[i] -= 1
            # "tie" -> no score change

            # Check stopping condition
            best_idx = max(range(n), key=lambda t: scores[t])
            other_scores = [scores[t] for t in range(n) if t != best_idx]
            second_best = max(other_scores) if other_scores else float("-inf")

            if scores[best_idx] - second_best >= k:
                return candidates[best_idx]

        # Max rounds reached, return current leader
        best_idx = max(range(n), key=lambda t: scores[t])
        return candidates[best_idx]


def compute_k_for_eps(p: float, eps: float) -> int:
    """
    Compute the vote margin k needed to achieve target error probability eps.

    From the SPRT analysis, p_win = 1 / (1 + ((1-p)/p)^k).
    Error probability = 1 - p_win = ((1-p)/p)^k / (1 + ((1-p)/p)^k)

    For small error (eps << 1), this approximates to:
        ((1-p)/p)^k ≈ eps

    Taking logs:
        k * log((1-p)/p) = log(eps)
        k = log(eps) / log((1-p)/p)

    Since p > 0.5, we have (1-p)/p < 1, so log((1-p)/p) < 0.
    Also log(eps) < 0 for eps < 1, so k is positive.

    Args:
        p: Probability that judge prefers correct candidate (must be > 0.5)
        eps: Target error probability

    Returns:
        Minimum k value (as integer, ceiling)
    """
    import math

    if p <= 0.5:
        raise ValueError(f"p must be > 0.5, got {p}")
    if eps <= 0 or eps >= 1:
        raise ValueError(f"eps must be in (0, 1), got {eps}")

    # k = log(eps) / log((1-p)/p)
    # Both logs are negative, so ratio is positive
    odds_ratio = (1 - p) / p  # < 1 when p > 0.5
    k_float = math.log(eps) / math.log(odds_ratio)

    return max(1, int(math.ceil(k_float)))
