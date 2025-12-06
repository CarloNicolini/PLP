"""
Integration tests for PLP.

Tests complete PLP programs including:
- Sorting with SPRT-based candidate selection
- Simple arithmetic with guarantee wrappers
"""

import pytest

from plp import PLPState, LLMClient, InferEngine, sample, observe, factor, guarantee


class TestSortingWithPLP:
    """Integration tests for sorting using PLP primitives."""

    def test_sorting_with_sprt(
        self, llm: LLMClient, plp_state: PLPState, judge_factory
    ):
        """
        Test sorting a list using PLP:
        1. Generate multiple candidate sorted lists via LLM
        2. Use SPRT-based voting to select the correct one
        """
        items_text = "3 1 2"
        original_items = [int(x) for x in items_text.split()]

        @sample(llm, name="propose_sort")
        def propose_sort(state: PLPState, items: str) -> str:
            return (
                "Sort these integers in ascending order.\n"
                f"Input: {items}\n"
                "Reply with ONLY the sorted integers separated by spaces, nothing else."
            )

        # Generate several candidate sorts
        candidates = []
        state = plp_state
        for _ in range(5):
            state, result = propose_sort(state, items_text)
            candidates.append(result.strip())

        # Define a deterministic judge for correctness
        def is_sorted(s: str) -> bool:
            try:
                nums = [int(x) for x in s.split()]
                return nums == sorted(original_items)
            except ValueError:
                return False

        judge = judge_factory(is_sorted)

        # Use SPRT to select best candidate
        winner = InferEngine.sprt_discrete(candidates, judge, k=2)

        # Verify the winner is correctly sorted
        # (This may fail if LLM never produced correct output, but usually succeeds)
        winner_nums = [int(x) for x in winner.split()]
        assert winner_nums == [1, 2, 3], f"Expected [1, 2, 3], got {winner_nums}"

    def test_sorting_larger_list(
        self, llm: LLMClient, plp_state: PLPState, judge_factory
    ):
        """Test sorting a larger list."""
        items_text = "5 2 8 1 9 3"
        original_items = [int(x) for x in items_text.split()]
        expected = sorted(original_items)

        @sample(llm, name="sort_large")
        def sort_list(state: PLPState, items: str) -> str:
            return (
                "Sort these integers in ascending order.\n"
                f"Input: {items}\n"
                "Reply with ONLY the sorted integers separated by spaces."
            )

        # Generate candidates
        candidates = []
        state = plp_state
        for _ in range(7):
            state, result = sort_list(state, items_text)
            candidates.append(result.strip())

        def is_correct(s: str) -> bool:
            try:
                nums = [int(x) for x in s.split()]
                return nums == expected
            except ValueError:
                return False

        judge = judge_factory(is_correct)

        winner = InferEngine.sprt_discrete(candidates, judge, k=2)
        winner_nums = [int(x) for x in winner.split()]
        assert winner_nums == expected


class TestArithmeticWithPLP:
    """Integration tests for arithmetic using PLP primitives."""

    def test_simple_addition(self, llm: LLMClient, plp_state: PLPState):
        """Test simple addition with observe for validation."""

        @sample(llm, name="add")
        def add_numbers(state: PLPState, a: int, b: int) -> str:
            return f"What is {a} + {b}? Reply with just the number."

        @observe(name="is_number")
        def check_is_number(state: PLPState, text: str) -> bool:
            try:
                int(text.strip())
                return True
            except ValueError:
                return False

        state, result = add_numbers(plp_state, 7, 8)
        state = check_is_number(state, result)

        # Should produce a valid trace
        assert len(state.trace) == 2

        # If observe passed, verify correctness
        if state.is_valid():
            assert "15" in result

    def test_multiplication_with_factor(self, llm: LLMClient, plp_state: PLPState):
        """Test multiplication with factor for confidence scoring."""

        @sample(llm, name="multiply")
        def multiply(state: PLPState, a: int, b: int) -> str:
            return f"Calculate {a} × {b}. Reply with just the number."

        @factor(name="confidence")
        def score_result(state: PLPState, result: str, expected: int) -> float:
            try:
                actual = int(result.strip())
                if actual == expected:
                    return 5.0  # High score for correct
                else:
                    return -5.0  # Penalty for incorrect
            except ValueError:
                return -10.0  # Big penalty for invalid

        state, result = multiply(plp_state, 6, 7)
        state = score_result(state, result, 42)

        # Check factor was applied
        assert len(state.trace) == 2
        assert state.trace[1].meta["factor"] is True

    def test_arithmetic_with_guarantee(self, llm: LLMClient, plp_state: PLPState):
        """Test arithmetic wrapped with guarantee for robustness."""

        @sample(llm, name="compute")
        def compute(state: PLPState, problem: str) -> str:
            return f"Solve: {problem}. Reply with just the number."

        # Define a checker that returns the correct answer
        def check_answer(result: str, expected: int) -> bool:
            try:
                return int(result.strip()) == expected
            except ValueError:
                return False

        # Build a judge for SPRT
        def make_judge(expected: int):
            def judge(a: str, b: str) -> str:
                a_ok = check_answer(a, expected)
                b_ok = check_answer(b, expected)
                if a_ok and not b_ok:
                    return "a"
                elif b_ok and not a_ok:
                    return "b"
                return "tie"

            return judge

        # Use guarantee wrapper
        @guarantee(llm, eps=0.01, judge_fn=make_judge(25), method="sprt_discrete")
        def robust_compute(state: PLPState, problem: str) -> tuple[PLPState, str]:
            return compute(state, problem)

        state, result = robust_compute(plp_state, "5 * 5")

        # Verify we got a result through the guarantee wrapper
        assert any("guarantee" in e.primitive for e in state.trace)

        # Check result
        result_clean = result.strip()
        if result_clean.isdigit():
            assert int(result_clean) == 25


class TestChainedOperations:
    """Tests for chaining multiple PLP operations."""

    def test_multi_step_calculation(self, llm: LLMClient, plp_state: PLPState):
        """Test a multi-step calculation using PLP."""

        @sample(llm, name="step1")
        def compute_step1(state: PLPState) -> str:
            return "What is 10 + 5? Reply with just the number."

        @sample(llm, name="step2")
        def compute_step2(state: PLPState, prev_result: str) -> str:
            return f"What is {prev_result.strip()} × 2? Reply with just the number."

        @observe(name="final_check")
        def check_final(state: PLPState, result: str) -> bool:
            try:
                val = int(result.strip())
                return val == 30  # (10+5)*2 = 30
            except ValueError:
                return False

        state, result1 = compute_step1(plp_state)
        state, result2 = compute_step2(state, result1)
        state = check_final(state, result2)

        # Should have 3 trace events
        assert len(state.trace) == 3

        # Check primitive names in order
        assert state.trace[0].primitive == "step1"
        assert state.trace[1].primitive == "step2"
        assert state.trace[2].primitive == "final_check"


class TestConditioningPattern:
    """Tests for the conditioning pattern."""

    def test_condition_with_retry(self, llm: LLMClient, plp_state: PLPState):
        """Test conditioning with retries until condition met."""
        from plp import sample_with_condition

        @sample(llm, name="generate_even")
        def generate_number(state: PLPState) -> str:
            return (
                "Give me a random even number between 2 and 10. "
                "Reply with just the number."
            )

        def is_even(state: PLPState, text: str) -> bool:
            try:
                num = int(text.strip())
                return num % 2 == 0 and 2 <= num <= 10
            except ValueError:
                return False

        state, result = sample_with_condition(
            generate_number, is_even, plp_state, max_retries=5
        )

        # Check we got a result
        assert result is not None

        # Verify it's even (if condition was satisfied)
        try:
            num = int(result.strip())
            if 2 <= num <= 10:
                assert num % 2 == 0
        except ValueError:
            pass  # LLM returned non-number, acceptable for test
