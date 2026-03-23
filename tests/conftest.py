"""
Pytest configuration and fixtures for PLP tests.

Loads .env file and provides shared fixtures for LLM client and state.
"""

import pytest
from dotenv import load_dotenv
from plp import ChatOpenAI, PLPState


# Load environment variables from .env file
load_dotenv()


@pytest.fixture
def llm() -> ChatOpenAI:
    """
    Provide a configured LLM client for tests.

    Uses gpt-4.1-mini as specified in requirements.
    """
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)


@pytest.fixture
def llm_deterministic() -> ChatOpenAI:
    """
    Provide a more deterministic LLM client (temperature=0.1).

    Useful for tests that expect more consistent outputs.
    """
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)


@pytest.fixture
def plp_state() -> PLPState:
    """Provide a fresh PLP state for tests."""
    return PLPState()
