"""
PLP State Management.

Core state dataclasses for Probabilistic Language Programming:
- TraceEvent: logs primitive invocations
- PLPState: holds NL variables, log_weight, and trace
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    """
    A single event in the PLP execution trace.

    Attributes:
        primitive: Name of the primitive that produced this event
        prompt: The prompt string sent to the LLM (if applicable)
        output: The output string received from the LLM or checker
        meta: Additional metadata (llm name, factor value, etc.)
    """

    primitive: str
    prompt: str
    output: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PLPState:
    """
    PLP runtime state.

    All user-visible state is stored as natural language strings in `vars`.
    The `log_weight` accumulates factor contributions for inference.
    The `trace` records all primitive invocations for debugging/analysis.

    Attributes:
        vars: Mapping from variable names to natural-language strings
        log_weight: Accumulated log-factor weight for this trace
        trace: List of TraceEvents describing execution history
    """

    vars: dict[str, str] = field(default_factory=dict)
    log_weight: float = 0.0
    trace: list[TraceEvent] = field(default_factory=list)

    def copy(self) -> "PLPState":
        """Create a shallow copy of this state."""
        return PLPState(
            vars=dict(self.vars), log_weight=self.log_weight, trace=list(self.trace)
        )

    def is_valid(self) -> bool:
        """Check if this trace is valid (log_weight != -inf)."""
        return self.log_weight != float("-inf")
