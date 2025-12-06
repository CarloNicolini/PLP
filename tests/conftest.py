"""
Pytest configuration and fixtures for PLP tests.

Loads .env file and provides shared fixtures for LLM client and state.
"""

import pytest
from dotenv import load_dotenv
from typing import Callable

from plp import LLMClient, PLPState


# Load environment variables from .env file
load_dotenv()


@pytest.fixture
def llm() -> LLMClient:
    """
    Provide a configured LLM client for tests.

    Uses gpt-4.1-mini as specified in requirements.
    """
    return LLMClient(model="gpt-4.1-mini", temperature=0.7)


@pytest.fixture
def llm_deterministic() -> LLMClient:
    """
    Provide a more deterministic LLM client (temperature=0.1).

    Useful for tests that expect more consistent outputs.
    """
    return LLMClient(model="gpt-4.1-mini", temperature=0.1)


@pytest.fixture
def plp_state() -> PLPState:
    """Provide a fresh PLP state for tests."""
    return PLPState()


@pytest.fixture
def judge_factory() -> Callable:
    """
    Factory for creating judge functions with a correctness predicate.

    Returns a function that takes a predicate and returns a judge function.

    Example:
        >>> def judge_fn(a, b):
        ...     return judge_factory(lambda x: x == "correct")(a, b)
    """

    def create_judge(is_correct: Callable[[str], bool]) -> Callable[[str, str], str]:
        """
        Create a judge function from a correctness predicate.

        Args:
            is_correct: Function that returns True if a candidate is correct

        Returns:
            Judge function (a, b) -> "a" | "b" | "tie"
        """

        def judge(a: str, b: str) -> str:
            a_ok = is_correct(a)
            b_ok = is_correct(b)
            if a_ok and not b_ok:
                return "a"
            elif b_ok and not a_ok:
                return "b"
            return "tie"

        return judge

    return create_judge
