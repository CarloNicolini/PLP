"""
Tests for PLP core primitives: sample, observe, factor.

These tests use real LLM calls via the fixtures in conftest.py.
"""

import pytest

from plp import PLPState, sample, observe, factor, LLMClient


class TestSamplePrimitive:
    """Tests for the @sample decorator."""

    def test_sample_basic_arithmetic(self, llm: LLMClient, plp_state: PLPState):
        """Test that sample correctly calls LLM and returns result."""

        @sample(llm, name="arithmetic")
        def solve_addition(state: PLPState, a: int, b: int) -> str:
            return f"What is {a} + {b}? Reply with just the number, nothing else."

        state, result = solve_addition(plp_state, 2, 2)

        # Check that we got a response
        assert result is not None
        assert len(result) > 0

        # Check state was updated with trace
        assert len(state.trace) == 1
        assert state.trace[0].primitive == "arithmetic"
        assert "2 + 2" in state.trace[0].prompt

        # Should parse as a number (likely "4" but LLM might add text)
        # Just check that the response contains "4"
        assert "4" in result

    def test_sample_preserves_log_weight(self, llm: LLMClient, plp_state: PLPState):
        """Test that sample doesn't modify log_weight."""

        @sample(llm, name="test")
        def simple_prompt(state: PLPState) -> str:
            return "Say hello"

        initial_weight = plp_state.log_weight
        state, _ = simple_prompt(plp_state)

        assert state.log_weight == initial_weight

    def test_sample_with_update_state(self, llm: LLMClient, plp_state: PLPState):
        """Test sample with custom state update callback."""

        def store_answer(state: PLPState, output: str):
            state.vars["answer"] = output

        @sample(llm, name="math", update_state=store_answer)
        def solve(state: PLPState) -> str:
            return "What is 5 + 5? Reply with just the number."

        state, result = solve(plp_state)

        assert "answer" in state.vars
        assert state.vars["answer"] == result


class TestObservePrimitive:
    """Tests for the @observe decorator."""

    def test_observe_passing(self, plp_state: PLPState):
        """Test that passing observe keeps state valid."""

        @observe(name="always_true")
        def always_passes(state: PLPState) -> bool:
            return True

        state = always_passes(plp_state)

        assert state.is_valid()
        assert state.log_weight == 0.0
        assert len(state.trace) == 1
        assert state.trace[0].meta["passed"] is True

    def test_observe_failing(self, plp_state: PLPState):
        """Test that failing observe sets log_weight to -inf."""

        @observe(name="always_false")
        def always_fails(state: PLPState) -> bool:
            return False

        state = always_fails(plp_state)

        assert not state.is_valid()
        assert state.log_weight == float("-inf")
        assert state.trace[0].meta["passed"] is False

    def test_observe_with_condition(self, plp_state: PLPState):
        """Test observe with a real condition check."""

        @observe(name="is_number")
        def is_valid_number(state: PLPState, text: str) -> bool:
            try:
                int(text.strip())
                return True
            except ValueError:
                return False

        # Valid number
        state1 = is_valid_number(plp_state, "42")
        assert state1.is_valid()

        # Invalid number
        state2 = is_valid_number(plp_state, "hello")
        assert not state2.is_valid()


class TestFactorPrimitive:
    """Tests for the @factor decorator."""

    def test_factor_positive(self, plp_state: PLPState):
        """Test that positive factor increases log_weight."""

        @factor(name="bonus")
        def add_bonus(state: PLPState) -> float:
            return 1.0

        state = add_bonus(plp_state)

        assert state.log_weight == 1.0
        assert len(state.trace) == 1
        assert state.trace[0].meta["log_factor"] == 1.0

    def test_factor_negative(self, plp_state: PLPState):
        """Test that negative factor decreases log_weight."""

        @factor(name="penalty")
        def add_penalty(state: PLPState) -> float:
            return -2.0

        state = add_penalty(plp_state)

        assert state.log_weight == -2.0

    def test_factor_accumulation(self, plp_state: PLPState):
        """Test that multiple factors accumulate correctly."""

        @factor(name="f1")
        def factor1(state: PLPState) -> float:
            return 1.5

        @factor(name="f2")
        def factor2(state: PLPState) -> float:
            return 2.5

        state = factor1(plp_state)
        state = factor2(state)

        assert state.log_weight == 4.0
        assert len(state.trace) == 2

    def test_factor_with_input(self, plp_state: PLPState):
        """Test factor that depends on input."""

        @factor(name="length_penalty")
        def length_penalty(state: PLPState, text: str) -> float:
            # Penalize texts longer than 10 chars
            excess = max(0, len(text) - 10)
            return -0.1 * excess

        state1 = length_penalty(plp_state, "short")
        assert state1.log_weight == 0.0  # No penalty

        state2 = length_penalty(plp_state, "this is a longer text")
        assert state2.log_weight < 0.0  # Some penalty


class TestPrimitiveChaining:
    """Tests for chaining multiple primitives."""

    def test_sample_then_observe(self, llm: LLMClient, plp_state: PLPState):
        """Test chaining sample followed by observe."""

        @sample(llm, name="generate_number")
        def generate(state: PLPState) -> str:
            return "Give me a single digit number (0-9). Reply with just the digit."

        @observe(name="is_digit")
        def check_digit(state: PLPState, text: str) -> bool:
            stripped = text.strip()
            return len(stripped) == 1 and stripped.isdigit()

        state, result = generate(plp_state)
        # Note: This might fail sometimes due to LLM variability
        # Just check the flow works
        final_state = check_digit(state, result)

        assert len(final_state.trace) == 2
        assert final_state.trace[0].primitive == "generate_number"
        assert final_state.trace[1].primitive == "is_digit"

    def test_sample_factor_observe_chain(self, llm: LLMClient, plp_state: PLPState):
        """Test full chain: sample -> factor -> observe."""

        @sample(llm, name="solve")
        def solve(state: PLPState) -> str:
            return "What is 10 - 3? Reply with just the number."

        @factor(name="prefer_short")
        def prefer_short(state: PLPState, text: str) -> float:
            return -0.1 * len(text)

        @observe(name="has_digit")
        def has_digit(state: PLPState, text: str) -> bool:
            return any(c.isdigit() for c in text)

        state, result = solve(plp_state)
        state = prefer_short(state, result)
        state = has_digit(state, result)

        assert len(state.trace) == 3
        # log_weight should be negative (length penalty)
        assert state.log_weight < 0
