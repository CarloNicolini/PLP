"""
PLP Agent Module.

Reusable agent components with their own LLM client, prompts,
and reliability profile. Analogous to Pyro's pyro.module.

Agents are named black boxes with known input/output interfaces
that can be composed into larger PLP programs.
"""

from dataclasses import dataclass
from typing import Callable, Any, Protocol
from functools import wraps
import time

from ._llm import LLMLike, invoke_llm
from .state import PLPState, TraceEvent


class AgentProtocol(Protocol):
    """Protocol that all PLP agents must implement."""

    @property
    def name(self) -> str:
        """Agent's unique identifier."""
        ...

    def __call__(self, state: PLPState, *args, **kwargs) -> tuple[PLPState, Any]:
        """Execute the agent."""
        ...


@dataclass
class AgentSpec:
    """
    Specification for a registered agent.

    Attributes:
        name: Unique identifier
        agent_class: The agent class or factory
        llm: Default LLM client for this agent
        reliability: Expected accuracy (p > 0.5 assumption)
        timeout: Maximum execution time in seconds
        description: Human-readable description
    """

    name: str
    agent_class: type | Callable
    llm: LLMLike | None = None
    reliability: float = 0.7
    timeout: float | None = None
    description: str = ""
    call_count: int = 0
    success_count: int = 0

    @property
    def observed_reliability(self) -> float:
        """Empirical reliability from call history."""
        if self.call_count == 0:
            return self.reliability
        return self.success_count / self.call_count


# Global agent registry
_agents: dict[str, AgentSpec] = {}


def register_agent(
    name: str,
    llm: LLMLike | None = None,
    reliability: float = 0.7,
    timeout: float | None = None,
    description: str = "",
):
    """
    Decorator to register a class as a PLP agent.

    Args:
        name: Unique agent identifier
        llm: Default LLM client
        reliability: Expected accuracy (for guarantee calculations)
        timeout: Maximum execution time
        description: Human-readable description

    Example:
        >>> @register_agent("decomposer", llm=decomp_llm, reliability=0.8)
        ... class DecomposerAgent:
        ...     def __call__(self, state, task):
        ...         return self.decompose(state, task)
        ...
        ...     def decompose(self, state, task):
        ...         # Implementation
        ...         pass
    """

    def decorator(cls: type):
        spec = AgentSpec(
            name=name,
            agent_class=cls,
            llm=llm,
            reliability=reliability,
            timeout=timeout,
            description=description,
        )
        _agents[name] = spec

        # Add metadata to class
        cls._plp_agent_name = name
        cls._plp_agent_spec = spec

        return cls

    return decorator


def get_agent(name: str) -> AgentSpec | None:
    """Get agent spec by name."""
    return _agents.get(name)


def create_agent(name: str, llm: LLMLike | None = None) -> Any:
    """
    Instantiate a registered agent.

    Args:
        name: Agent name from registry
        llm: Optional LLM override

    Returns:
        Instantiated agent
    """
    spec = _agents.get(name)
    if spec is None:
        raise ValueError(f"No agent registered with name: {name}")

    instance = spec.agent_class()
    if hasattr(instance, "llm") and llm is not None:
        instance.llm = llm
    elif hasattr(instance, "llm") and spec.llm is not None:
        instance.llm = spec.llm

    return instance


def list_agents() -> list[str]:
    """List all registered agent names."""
    return list(_agents.keys())


def clear_agents():
    """Clear agent registry."""
    _agents.clear()


class BaseAgent:
    """
    Base class for PLP agents.

    Provides common functionality like tracing, timing,
    and reliability tracking.
    """

    def __init__(self, llm: LLMLike | None = None):
        self.llm = llm
        self._call_count = 0
        self._total_time = 0.0

    @property
    def name(self) -> str:
        return getattr(self, "_plp_agent_name", self.__class__.__name__)

    def __call__(self, state: PLPState, *args, **kwargs) -> tuple[PLPState, Any]:
        """Execute agent with timing and tracing."""
        start = time.time()
        self._call_count += 1

        try:
            result = self.run(state, *args, **kwargs)
            elapsed = time.time() - start
            self._total_time += elapsed

            # Update registry stats
            spec = get_agent(self.name)
            if spec:
                spec.call_count += 1

            return result
        except Exception:
            elapsed = time.time() - start
            self._total_time += elapsed
            raise

    def run(self, state: PLPState, *args, **kwargs) -> tuple[PLPState, Any]:
        """Override this in subclasses."""
        raise NotImplementedError

    @property
    def avg_time(self) -> float:
        """Average execution time."""
        if self._call_count == 0:
            return 0.0
        return self._total_time / self._call_count


@register_agent("decomposer", reliability=0.8, description="Breaks tasks into subtasks")
class DecomposerAgent(BaseAgent):
    """
    Agent that decomposes complex tasks into subtasks.

    This is a fundamental agent for divide-and-conquer strategies.
    """

    def run(
        self, state: PLPState, task: str, max_subtasks: int = 5
    ) -> tuple[PLPState, list[str]]:
        if self.llm is None:
            raise ValueError("DecomposerAgent requires an LLM")

        prompt = (
            f"Break down this task into {max_subtasks} or fewer clear subtasks:\n"
            f"Task: {task}\n\n"
            "List each subtask on a new line, numbered 1., 2., etc."
        )

        response = invoke_llm(self.llm, prompt)

        # Parse subtasks
        lines = response.strip().split("\n")
        subtasks = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove numbering
                clean = line.lstrip("0123456789.-) ").strip()
                if clean:
                    subtasks.append(clean)

        new_state = state.copy()
        new_state.trace.append(
            TraceEvent(
                primitive=f"agent({self.name})",
                prompt=prompt,
                output=str(subtasks),
                meta={"agent": self.name, "n_subtasks": len(subtasks)},
            )
        )

        return new_state, subtasks


@register_agent(
    "aggregator", reliability=0.85, description="Combines results into synthesis"
)
class AggregatorAgent(BaseAgent):
    """
    Agent that combines/synthesizes multiple results.

    Partner to DecomposerAgent - aggregates subtask results.
    """

    def run(
        self, state: PLPState, results: list[str], goal: str
    ) -> tuple[PLPState, str]:
        if self.llm is None:
            raise ValueError("AggregatorAgent requires an LLM")

        results_text = "\n".join([f"- {r}" for r in results])

        prompt = (
            f"Combine these results to achieve the goal:\n\n"
            f"Goal: {goal}\n\n"
            f"Results:\n{results_text}\n\n"
            "Provide a synthesized answer:"
        )

        response = invoke_llm(self.llm, prompt)

        new_state = state.copy()
        new_state.trace.append(
            TraceEvent(
                primitive=f"agent({self.name})",
                prompt=prompt,
                output=response,
                meta={"agent": self.name, "n_results": len(results)},
            )
        )

        return new_state, response


@register_agent(
    "critic", reliability=0.75, description="Evaluates and critiques outputs"
)
class CriticAgent(BaseAgent):
    """
    Agent that critiques/evaluates outputs.

    Useful for self-reflection and quality control.
    """

    def run(
        self,
        state: PLPState,
        output: str,
        criteria: str = "correctness, clarity, completeness",
    ) -> tuple[PLPState, dict[str, Any]]:
        if self.llm is None:
            raise ValueError("CriticAgent requires an LLM")

        prompt = (
            f"Evaluate this output based on: {criteria}\n\n"
            f"Output:\n{output}\n\n"
            "Provide:\n"
            "1. Score (1-10)\n"
            "2. Strengths\n"
            "3. Weaknesses\n"
            "4. Suggested improvements"
        )

        response = invoke_llm(self.llm, prompt)

        # Try to extract score
        score = 5  # default
        for line in response.split("\n"):
            if "score" in line.lower() or line.strip().startswith("1"):
                nums = [int(c) for c in line if c.isdigit()]
                if nums:
                    score = min(10, max(1, nums[0]))
                    break

        critique = {"score": score, "feedback": response, "criteria": criteria}

        new_state = state.copy()
        new_state.trace.append(
            TraceEvent(
                primitive=f"agent({self.name})",
                prompt=prompt,
                output=response,
                meta={"agent": self.name, "score": score},
            )
        )

        return new_state, critique


@register_agent("judge", reliability=0.8, description="Compares and ranks candidates")
class JudgeAgent(BaseAgent):
    """
    Agent that judges/compares between candidates.

    Useful for SPRT-style voting and tournament selection.
    """

    def run(
        self,
        state: PLPState,
        candidate_a: str,
        candidate_b: str,
        criterion: str = "overall quality",
    ) -> tuple[PLPState, str]:
        if self.llm is None:
            raise ValueError("JudgeAgent requires an LLM")

        prompt = (
            f"Compare these two candidates based on: {criterion}\n\n"
            f"Candidate A:\n{candidate_a}\n\n"
            f"Candidate B:\n{candidate_b}\n\n"
            "Which is better? Reply with exactly 'A', 'B', or 'TIE'."
        )

        response = invoke_llm(self.llm, prompt).strip().upper()

        # Normalize response
        if "A" in response and "B" not in response:
            verdict = "a"
        elif "B" in response and "A" not in response:
            verdict = "b"
        elif "TIE" in response:
            verdict = "tie"
        else:
            # Ambiguous, default to tie
            verdict = "tie"

        new_state = state.copy()
        new_state.trace.append(
            TraceEvent(
                primitive=f"agent({self.name})",
                prompt=prompt,
                output=verdict,
                meta={"agent": self.name, "criterion": criterion},
            )
        )

        return new_state, verdict

    def as_judge_fn(
        self, criterion: str = "overall quality"
    ) -> Callable[[str, str], str]:
        """
        Return a judge function compatible with SPRT.

        Example:
            >>> judge = JudgeAgent(llm)
            >>> judge_fn = judge.as_judge_fn("correctness")
            >>> winner = InferEngine.sprt_discrete(candidates, judge_fn, k=3)
        """

        def judge_fn(a: str, b: str) -> str:
            state = PLPState()
            _, verdict = self.run(state, a, b, criterion)
            return verdict

        return judge_fn


def agent(name: str, llm: LLMLike | None = None, reliability: float = 0.7):
    """
    Decorator to create a simple agent from a function.

    Args:
        name: Agent identifier
        llm: Optional LLM client
        reliability: Expected accuracy

    Example:
        >>> @agent("solver", llm=llm, reliability=0.75)
        ... def solve_math(state, problem):
        ...     prompt = f"Solve: {problem}"
        ...     # llm is available via closure
        ...     response = llm.invoke(prompt)
        ...     return state.copy(), response
    """

    def decorator(fn: Callable[..., tuple[PLPState, Any]]):
        @wraps(fn)
        def wrapped(state: PLPState, *args, **kwargs) -> tuple[PLPState, Any]:
            result_state, result = fn(state, *args, **kwargs)

            # Add agent trace
            new_state = result_state.copy()
            new_state.trace.append(
                TraceEvent(
                    primitive=f"agent({name})",
                    prompt="(functional agent)",
                    output=str(result)[:200],
                    meta={"agent": name, "reliability": reliability},
                )
            )

            return new_state, result

        # Register in global registry
        spec = AgentSpec(
            name=name, agent_class=lambda: wrapped, llm=llm, reliability=reliability
        )
        _agents[name] = spec

        return wrapped

    return decorator
