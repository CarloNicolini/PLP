"""
PLP Inference Engines.

Provides inference algorithms for PLP:
- run_map: Naive MAP by multiple runs
- sprt_discrete: First-to-ahead-by-k voting over candidate strings

TextGrad connection:
- TextGrad (Yuksekgonul et al., 2024) frames textual gradients as the
  optimization primitive for compound AI systems.
- PLP treats those same textual gradients as *signals* that can be fed into
  broader inference loops. `SVGDEngine` exposes Stein-style updates whose
  drive functions are often implemented with `textgrad_step`, while
  `sprt_discrete` lets textual judges vote over candidates produced by
  text-first sites.
"""

import random
from dataclasses import dataclass
from typing import Callable, Any, Sequence

from .state import PLPState


class InferEngine:
    """
    Collection of inference kernels for PLP.

    These algorithms approximate MAP or posterior solutions by:
    - Running the model multiple times (run_map)
    - Voting among discrete candidates (sprt_discrete)
    """

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

        Raises:
            ValueError: If candidates is empty or k <= 0
            TypeError: If judge_fn is not callable

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
        # Validate inputs
        if not candidates:
            raise ValueError("No candidates provided")
        if not callable(judge_fn):
            raise TypeError(f"judge_fn must be callable, got {type(judge_fn)}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        if rng is None:
            rng = random.Random()

        n = len(candidates)
        if n == 1:
            return candidates[0]

        scores = [0] * n

        for _ in range(max_rounds):
            # Sample a random pair
            i, j = rng.sample(range(n), 2)
            a, b = candidates[i], candidates[j]

            verdict = judge_fn(a, b)

            # Validate judge output
            if verdict not in ("a", "b", "tie"):
                raise ValueError(
                    f"judge_fn must return 'a', 'b', or 'tie', got {verdict!r}"
                )

            # Update scores based on verdict
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


@dataclass
class SVGDParticle:
    """
    Container for a single SVGD particle.

    Attributes:
        value: User-defined payload (e.g., prompt string, dict of fields, etc.).
        state: The PLPState associated with this particle. Each step clones and
            updates the state so gradient calculations remain traceable.
    """

    value: Any
    state: PLPState


DriveFn = Callable[
    [SVGDParticle, int], tuple[SVGDParticle, Any, dict[str, Any]]
]
RepulsionFn = Callable[
    [SVGDParticle, Sequence[SVGDParticle], int],
    tuple[SVGDParticle, Any, dict[str, Any]],
]
UpdateFn = Callable[
    [
        SVGDParticle,
        Any,
        Any,
        dict[str, Any],
        dict[str, Any],
        float,
        int,
    ],
    tuple[SVGDParticle, dict[str, Any]],
]


class SVGDEngine:
    """
    Generic Stein Variational Gradient Descent (SVGD) driver.

    Users supply three callables:
        - drive_fn:     Computes the TextGrad-style "drive" force (∇ log p).
        - repulsion_fn: Computes the kernel repulsion term given all particles.
        - update_fn:    Applies both forces to produce the refined particle.

    Each callable receives and returns SVGDParticle instances so PLP states and
    traces remain accessible (allowing factor/sample primitives inside).
    In practice, the drive_fn is often implemented with `textgrad_step` or a
    related textual gradient oracle, making this class the bridge between the
    TextGrad computation-graph metaphor and probabilistic inference kernels.
    The repulsion_fn can likewise query `text_kernel_constraint` or overlap
    judges to express SVGD's kernel forces in natural language.
    """

    def __init__(self, diversity_force: float = 0.7):
        if diversity_force < 0:
            raise ValueError(
                f"diversity_force must be non-negative, got {diversity_force}"
            )
        self.diversity_force = diversity_force

    def step(
        self,
        particles: Sequence[SVGDParticle],
        drive_fn: DriveFn,
        repulsion_fn: RepulsionFn,
        update_fn: UpdateFn,
    ) -> tuple[list[SVGDParticle], list[dict[str, Any]]]:
        """
        Run a single SVGD sweep over the particles.

        Args:
            particles: Current list/tuple of SVGDParticle objects.
            drive_fn: Callable returning (particle, drive_signal, metadata).
            repulsion_fn: Callable returning (particle, repulsion_signal, metadata).
            update_fn: Callable applying both signals and returning
                (updated_particle, diagnostics).

        Returns:
            Tuple (new_particles, diagnostics_per_particle).
        """
        new_particles: list[SVGDParticle] = []
        diagnostics: list[dict[str, Any]] = []

        base_particles = list(particles)
        if not base_particles:
            return new_particles, diagnostics

        for idx, particle in enumerate(base_particles):
            working_state = particle.state.copy()
            working_particle = SVGDParticle(value=particle.value, state=working_state)

            working_particle, drive_signal, drive_meta = drive_fn(
                working_particle, idx
            )
            working_particle, repulsion_signal, repulsion_meta = repulsion_fn(
                working_particle, base_particles, idx
            )

            updated_particle, update_meta = update_fn(
                working_particle,
                drive_signal,
                repulsion_signal,
                drive_meta,
                repulsion_meta,
                self.diversity_force,
                idx,
            )
            new_particles.append(updated_particle)

            diag = {
                "index": idx,
                "drive": drive_meta,
                "repulsion": repulsion_meta,
                "update": update_meta,
            }
            diagnostics.append(diag)

        return new_particles, diagnostics

    def run(
        self,
        particles: Sequence[SVGDParticle],
        drive_fn: DriveFn,
        repulsion_fn: RepulsionFn,
        update_fn: UpdateFn,
        iterations: int,
    ) -> tuple[list[SVGDParticle], list[list[dict[str, Any]]]]:
        """
        Convenience helper to run multiple SVGD iterations.

        Returns:
            (final_particles, diagnostics_history) where diagnostics_history
            is a list with one diagnostics entry per iteration.
        """
        if iterations <= 0:
            raise ValueError(f"iterations must be positive, got {iterations}")

        current = list(particles)
        history: list[list[dict[str, Any]]] = []

        for _ in range(iterations):
            current, diagnostics = self.step(
                current, drive_fn=drive_fn, repulsion_fn=repulsion_fn, update_fn=update_fn
            )
            history.append(diagnostics)

        return current, history


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

    Raises:
        ValueError: If p <= 0.5 or eps not in (0, 1)
        TypeError: If p or eps are not numeric
    """
    import math

    # Type validation
    if not isinstance(p, (int, float)):
        raise TypeError(f"p must be numeric, got {type(p)}")
    if not isinstance(eps, (int, float)):
        raise TypeError(f"eps must be numeric, got {type(eps)}")

    # Value validation
    if p <= 0.5:
        raise ValueError(f"p must be > 0.5, got {p}")
    if eps <= 0 or eps >= 1:
        raise ValueError(f"eps must be in (0, 1), got {eps}")

    # k = log(eps) / log((1-p)/p)
    # Both logs are negative, so ratio is positive
    odds_ratio = (1 - p) / p  # < 1 when p > 0.5
    k_float = math.log(eps) / math.log(odds_ratio)

    return max(1, int(math.ceil(k_float)))
