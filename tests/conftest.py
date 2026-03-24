"""Pytest fixtures for PLP."""

from types import SimpleNamespace

import pytest

from plp import PLPState


class MockLLM:
    model = "mock-model"
    temperature = 0.0

    def __init__(self, content: str = "ok") -> None:
        self.content = content

    def invoke(self, messages):
        return SimpleNamespace(content=self.content)

    async def ainvoke(self, messages):
        return SimpleNamespace(content=self.content)


@pytest.fixture
def llm() -> MockLLM:
    """Provide a deterministic LLM-like object for tests."""
    return MockLLM()


@pytest.fixture
def plp_state() -> PLPState:
    """Provide a fresh PLP state for tests."""
    return PLPState()
