"""
Tests for Tier 3 PLP operators: guide and agent.
"""

import pytest
from plp import (
    PLPState,
    LLMClient,
    # Guide
    guide,
    register_guide,
    get_guide,
    has_guide,
    clear_guides,
    GuidedSampler,
    MetaGuide,
    # Agent
    agent,
    register_agent,
    get_agent,
    create_agent,
    list_agents,
    clear_agents,
    BaseAgent,
    DecomposerAgent,
    AggregatorAgent,
    CriticAgent,
    JudgeAgent,
)


class TestGuide:
    """Tests for guide operator."""

    def setup_method(self):
        """Clear guides before each test."""
        clear_guides()

    def test_register_guide(self):
        """Test guide registration."""

        @register_guide("my_model")
        def my_guide(state, x):
            return state, "guided_output"

        assert has_guide("my_model")
        spec = get_guide("my_model")
        assert spec is not None
        assert spec.model_name == "my_model"

    def test_guide_decorator(self):
        """Test @guide decorator."""

        @guide(for_model="solver")
        def solver_guide(state: PLPState, problem: str):
            return state.copy(), f"quick_solution({problem})"

        assert has_guide("solver")

        # Use the guide
        state, out = solver_guide(PLPState(), "test")
        assert "quick_solution" in out

    def test_guided_sampler_fallback(self, llm: LLMClient):
        """Test GuidedSampler falls back to model when no guide."""
        sampler = GuidedSampler(llm)

        def model_fn(state: PLPState):
            return state.copy(), "model_output"

        state, out = sampler.importance_sample("unknown", model_fn, PLPState())
        assert out == "model_output"

    def test_meta_guide(self, llm: LLMClient):
        """Test MetaGuide learns from history."""
        meta = MetaGuide(llm)

        # Record some outcomes
        meta.record_outcome("solver", "approach A", True)
        meta.record_outcome("solver", "approach B", False)
        meta.record_outcome("solver", "approach A variant", True)

        # Improved prompt should include successes
        improved = meta.get_improved_prompt("solver", "Solve this")
        assert "Successful approaches" in improved


class TestAgent:
    """Tests for agent operator."""

    def setup_method(self):
        """Note: built-in agents are registered at import time."""
        pass

    def test_builtin_agents_registered(self):
        """Test that built-in agents are registered."""
        agents = list_agents()
        assert "decomposer" in agents
        assert "aggregator" in agents
        assert "critic" in agents
        assert "judge" in agents

    def test_get_agent_spec(self):
        """Test getting agent specifications."""
        spec = get_agent("decomposer")
        assert spec is not None
        assert spec.reliability == 0.8
        assert "subtasks" in spec.description.lower()

    def test_create_agent(self, llm: LLMClient):
        """Test creating agent instance."""
        decomposer = create_agent("decomposer", llm=llm)
        assert decomposer is not None
        assert decomposer.llm == llm

    def test_decomposer_agent(self, llm: LLMClient):
        """Test DecomposerAgent breaks down tasks."""
        decomposer = DecomposerAgent(llm)

        state, subtasks = decomposer(
            PLPState(), "Write a Python function to sort a list"
        )

        assert isinstance(subtasks, list)
        assert len(subtasks) > 0
        assert len(state.trace) > 0
        assert "decomposer" in state.trace[-1].meta.get("agent", "")

    def test_aggregator_agent(self, llm: LLMClient):
        """Test AggregatorAgent combines results."""
        aggregator = AggregatorAgent(llm)

        results = ["Step 1 result", "Step 2 result", "Step 3 result"]

        state, synthesis = aggregator(PLPState(), results, "Complete the task")

        assert isinstance(synthesis, str)
        assert len(synthesis) > 0

    def test_critic_agent(self, llm: LLMClient):
        """Test CriticAgent evaluates output."""
        critic = CriticAgent(llm)

        state, critique = critic(PLPState(), "This is a test output", "clarity")

        assert isinstance(critique, dict)
        assert "score" in critique
        assert 1 <= critique["score"] <= 10
        assert "feedback" in critique

    def test_judge_agent(self, llm: LLMClient):
        """Test JudgeAgent compares candidates."""
        judge = JudgeAgent(llm)

        state, verdict = judge(
            PLPState(),
            "1 2 3",  # Correct sorted list
            "3 1 2",  # Wrong order
            "sorted order correctness",
        )

        assert verdict in ["a", "b", "tie"]

    def test_judge_agent_as_judge_fn(self, llm: LLMClient):
        """Test JudgeAgent can be used with SPRT."""
        judge = JudgeAgent(llm)
        judge_fn = judge.as_judge_fn("correctness")

        # Should be callable with two strings
        result = judge_fn("correct answer", "wrong answer")
        assert result in ["a", "b", "tie"]

    def test_functional_agent_decorator(self, llm: LLMClient):
        """Test @agent decorator for functions."""

        @agent("custom_solver", llm=llm, reliability=0.9)
        def my_solver(state: PLPState, problem: str):
            response = llm.invoke(f"Solve: {problem}")
            return state.copy(), response

        assert "custom_solver" in list_agents()

        state, result = my_solver(PLPState(), "2+2")
        assert result is not None

    def test_agent_call_tracking(self, llm: LLMClient):
        """Test that agent calls are tracked."""
        decomposer = DecomposerAgent(llm)

        initial_count = decomposer._call_count
        decomposer(PLPState(), "Task 1")
        decomposer(PLPState(), "Task 2")

        assert decomposer._call_count == initial_count + 2


class TestAgentIntegration:
    """Integration tests combining agents."""

    def test_decompose_and_aggregate(self, llm: LLMClient):
        """Test full decompose-aggregate workflow."""
        decomposer = DecomposerAgent(llm)
        aggregator = AggregatorAgent(llm)

        task = "Explain how to make tea"

        # Decompose
        state1, subtasks = decomposer(PLPState(), task)
        assert len(subtasks) > 0

        # Simulate subtask results
        results = [f"Result for: {st}" for st in subtasks[:3]]

        # Aggregate
        state2, synthesis = aggregator(state1, results, task)
        assert len(synthesis) > 0

        # Check traces
        assert len(state2.trace) >= 2

    def test_critic_feedback_loop(self, llm: LLMClient):
        """Test using critic for quality control."""
        critic = CriticAgent(llm)

        output = "The answer is 42."
        state, critique = critic(PLPState(), output, "correctness and explanation")

        score = critique["score"]

        # If low score, we could regenerate
        if score < 5:
            # In a real system, would regenerate here
            pass

        assert isinstance(score, int)
