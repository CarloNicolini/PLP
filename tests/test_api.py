"""
Tests for PLP High-Level API.
"""

import pytest
from unittest.mock import MagicMock, patch

import plp
from plp import PLPState


@pytest.fixture
def mock_llm():
    with patch("plp.api.LLMClient") as mock:
        client_instance = mock.return_value
        client_instance.invoke.return_value = "Mock response"
        yield client_instance


def test_configure():
    """Test global configuration."""
    plp.configure(model="test-model", temperature=0.1)
    config = plp.get_config()
    assert config.model == "test-model"
    assert config.temperature == 0.1

    # Reset
    plp.configure(model="gpt-4.1-mini", temperature=0.7)


def test_model_context():
    """Test model context manager."""
    plp.configure(model="global", temperature=0.5)

    with plp.model("local", temperature=0.1) as cfg:
        assert cfg.model == "local"
        assert cfg.temperature == 0.1
        assert plp.get_config().model == "local"

    # Should revert
    assert plp.get_config().model == "global"
    assert plp.get_config().temperature == 0.5


def test_sample_simple(mock_llm):
    """Test simple sample function."""
    plp.configure(model="gpt-4-test")

    resp = plp.sample_fn("Hello")

    assert resp == "Mock response"
    # Verify LLM created with correct config
    # (We can't easily check constructor args with this mock setup,
    # but we can check invoke was called)
    mock_llm.invoke.assert_called_once()


def test_sampler_decorator(mock_llm):
    """Test sampler decorator."""

    @plp.sampler(name="test_op")
    def my_op(x: str) -> str:
        return f"Prompt: {x}"

    # Direct call
    res = my_op("input")
    assert res == "Mock response"

    # With state return
    res, state = my_op("input", return_state=True)
    assert res == "Mock response"
    assert isinstance(state, PLPState)
    # Trace should contain event
    assert len(state.trace) == 1
    assert state.trace[0].primitive == "test_op"


def test_vote_shortcut():
    """Test vote shortcut with mocked engine."""
    with patch("plp.api.InferEngine.sprt_discrete") as mock_sprt:
        mock_sprt.return_value = "winner"

        res = plp.vote(["a", "b"], k=5)

        assert res == "winner"
        mock_sprt.assert_called_once()
        call_kwargs = mock_sprt.call_args[1]
        assert call_kwargs["k"] == 5
        assert call_kwargs["candidates"] == ["a", "b"]


def test_infer_builder():
    """Test fluent builder API."""
    builder = plp.infer().with_k(10).with_p(0.9)

    with patch("plp.api.vote") as mock_vote:
        builder.vote(["x", "y"])

        mock_vote.assert_called_once()
        call_kwargs = mock_vote.call_args[1]
        assert call_kwargs["k"] == 10
        assert call_kwargs["p"] == 0.9
