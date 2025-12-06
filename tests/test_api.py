"""
Tests for PLP High-Level API.
"""

import plp
from plp import PLPState


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


def test_sample_simple(llm):
    """Test simple sample function."""
    plp.configure(model="gpt-4-test")

    resp = plp.sample_fn("Hello")

    assert resp == "Mock response"
    # Verify LLM created with correct config
    # (We can't easily check constructor args with this mock setup,
    # but we can check invoke was called)
    llm.invoke.assert_called_once()


def test_sampler_decorator(llm):
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


def test_infer_builder():
    """Test fluent builder API."""
    builder = plp.infer().with_k(10).with_p(0.9)
    vote = builder.vote(["x", "y"])
    return vote
