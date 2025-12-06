"""
Tests for PLP inference engines.

Tests SPRT discrete voting and MAP inference.
"""

import pytest
import random

from plp import InferEngine, PLPState, LLMClient, compute_k_for_eps


class TestSPRTDiscrete:
    """Tests for the SPRT-style first-to-ahead-by-k voting."""

    def test_sprt_single_candidate(self):
        """Single candidate should return immediately."""
        candidates = ["only_option"]

        def dummy_judge(a: str, b: str) -> str:
            return "tie"

        result = InferEngine.sprt_discrete(candidates, dummy_judge, k=3)
        assert result == "only_option"

    def test_sprt_with_deterministic_judge(self):
        """Test SPRT with a deterministic judge that always prefers correct answer."""
        candidates = ["wrong1", "correct", "wrong2"]

        def judge(a: str, b: str) -> str:
            if a == "correct" and b != "correct":
                return "a"
            elif b == "correct" and a != "correct":
                return "b"
            return "tie"

        # Run multiple times to verify consistency
        for _ in range(5):
            result = InferEngine.sprt_discrete(candidates, judge, k=3)
            assert result == "correct"

    def test_sprt_with_tied_candidates(self):
        """Test SPRT when all candidates are equivalent."""
        candidates = ["a", "b", "c"]

        def always_tie(a: str, b: str) -> str:
            return "tie"

        # Should eventually return something (hits max_rounds)
        result = InferEngine.sprt_discrete(candidates, always_tie, k=3, max_rounds=50)
        assert result in candidates

    def test_sprt_reproducibility_with_seed(self):
        """Test that SPRT is reproducible with a fixed seed."""
        candidates = ["a", "b", "c"]

        def random_judge(a: str, b: str) -> str:
            # Use a simple deterministic rule
            return "a" if a < b else "b" if b < a else "tie"

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        result1 = InferEngine.sprt_discrete(candidates, random_judge, k=2, rng=rng1)
        result2 = InferEngine.sprt_discrete(candidates, random_judge, k=2, rng=rng2)

        assert result1 == result2

    def test_sprt_sorting_correctness(self):
        """Test SPRT for selecting correctly sorted list."""
        # Candidates for sorting [3, 1, 2]
        candidates = [
            "1 2 3",  # correct
            "3 1 2",  # wrong
            "2 3 1",  # wrong
            "1 3 2",  # wrong
        ]

        def is_sorted(s: str) -> bool:
            nums = [int(x) for x in s.split()]
            return nums == sorted(nums)

        def judge(a: str, b: str) -> str:
            a_ok = is_sorted(a)
            b_ok = is_sorted(b)
            if a_ok and not b_ok:
                return "a"
            elif b_ok and not a_ok:
                return "b"
            return "tie"

        result = InferEngine.sprt_discrete(candidates, judge, k=3)
        assert result == "1 2 3"


class TestComputeKForEps:
    """Tests for the k computation from epsilon."""

    def test_compute_k_basic(self):
        """Test basic k computation."""
        # p=0.7, eps=0.01 should give reasonable k
        k = compute_k_for_eps(0.7, 0.01)
        assert k >= 1
        assert k <= 20  # Reasonable bound

    def test_compute_k_higher_p(self):
        """Higher p should require lower k for same eps."""
        k_low_p = compute_k_for_eps(0.6, 0.01)
        k_high_p = compute_k_for_eps(0.9, 0.01)

        # Higher p needs fewer votes
        assert k_high_p < k_low_p

    def test_compute_k_lower_eps(self):
        """Lower eps should require higher k."""
        k_low_eps = compute_k_for_eps(0.7, 0.001)
        k_high_eps = compute_k_for_eps(0.7, 0.1)

        # Lower eps needs more votes
        assert k_low_eps > k_high_eps

    def test_compute_k_invalid_p(self):
        """Should raise for p <= 0.5."""
        with pytest.raises(ValueError):
            compute_k_for_eps(0.5, 0.01)
        with pytest.raises(ValueError):
            compute_k_for_eps(0.3, 0.01)

    def test_compute_k_invalid_eps(self):
        """Should raise for eps out of (0, 1)."""
        with pytest.raises(ValueError):
            compute_k_for_eps(0.7, 0.0)
        with pytest.raises(ValueError):
            compute_k_for_eps(0.7, 1.0)
        with pytest.raises(ValueError):
            compute_k_for_eps(0.7, -0.1)


class TestRunMAP:
    """Tests for the naive MAP inference."""

    def test_run_map_selects_highest_weight(self, llm: LLMClient):
        """Test that run_map returns trace with highest log_weight."""
        engine = InferEngine(llm)

        # Track which run we're on
        run_counter = {"count": 0}

        def model_fn(state: PLPState) -> tuple[PLPState, str]:
            run_counter["count"] += 1
            # Assign different weights based on run number
            new_state = state.copy()
            if run_counter["count"] == 2:
                # Second run has highest weight
                new_state.log_weight = 10.0
                return new_state, "best"
            else:
                new_state.log_weight = 1.0
                return new_state, "other"

        best_state, best_out = engine.run_map(model_fn, PLPState(), runs=3)

        assert best_out == "best"
        assert best_state.log_weight == 10.0

    def test_run_map_with_all_invalid(self, llm: LLMClient):
        """Test run_map when all runs produce invalid traces."""
        engine = InferEngine(llm)

        def always_fail(state: PLPState) -> tuple[PLPState, str]:
            new_state = state.copy()
            new_state.log_weight = float("-inf")
            return new_state, "failed"

        state, out = engine.run_map(always_fail, PLPState(), runs=3)

        # Should return empty result when all fail
        assert out == ""
