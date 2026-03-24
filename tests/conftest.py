"""Pytest fixtures for PLP."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from plp import PLPChatClient, PLPState

TEST_MODEL = "openai/gpt-4.1-mini"


def _load_test_env() -> None:
    """Load local environment variables for live LLM tests."""
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env", override=False)
    load_dotenv(override=False)


@pytest.fixture(scope="session")
def llm() -> PLPChatClient:
    """Provide a live OpenAI-compatible client for tests."""
    _load_test_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is required for live LLM tests.")

    return PLPChatClient(
        model=TEST_MODEL,
        temperature=0.0,
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )


@pytest.fixture
def plp_state() -> PLPState:
    """Provide a fresh PLP state for tests."""
    return PLPState()
