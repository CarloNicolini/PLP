"""
PLP Handlers.

Program transformers for trace manipulation, replay, blocking, and interventions.
Inspired by Pyro's effect handlers.
"""

from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, TypeVar

from .state import TraceEvent

T = TypeVar("T", bound="HandlerContext")


@dataclass
class HandlerContext:
    """Base context for handlers."""

    name: str
    active: bool = True


# Handler contexts are task-local so concurrent async calls do not interfere.
_handler_stacks: ContextVar[dict[str, tuple[Any, ...]]] = ContextVar(
    "plp_handler_stacks", default={}
)


def _push_handler(handler_type: str, context: T) -> T:
    """Push a handler context onto the stack."""
    stacks = dict(_handler_stacks.get())
    current = stacks.get(handler_type, ())
    stacks[handler_type] = (*current, context)
    _handler_stacks.set(stacks)
    return context


def _pop_handler(handler_type: str) -> Any:
    """Pop a handler context from the stack."""
    stacks = dict(_handler_stacks.get())
    current = stacks.get(handler_type, ())
    if not current:
        return None

    popped = current[-1]
    if len(current) == 1:
        stacks.pop(handler_type, None)
    else:
        stacks[handler_type] = current[:-1]
    _handler_stacks.set(stacks)
    return popped


def _get_handler(handler_type: str) -> Any:
    """Get the active handler context, if any."""
    current = _handler_stacks.get().get(handler_type, ())
    return current[-1] if current else None


@contextmanager
def _generic_handler(handler_type: str, context: T) -> Iterator[T]:
    """Generic context manager for handler contexts."""
    _push_handler(handler_type, context)
    try:
        yield context
    finally:
        _pop_handler(handler_type)


# ============================================================
# Replay Handler
# ============================================================


@dataclass
class ReplayContext(HandlerContext):
    """
    Context for replaying previous traces.

    Attributes:
        trace: Previous trace to replay from
        site_outputs: Mapping of site names to their recorded outputs
        replay_count: Number of sites successfully replayed
    """

    trace: list[TraceEvent] = field(default_factory=list)
    site_outputs: dict[str, str] = field(default_factory=dict)
    replay_count: int = 0

    def get_cached_output(self, site_name: str) -> str | None:
        """Get cached output for a site, if available."""
        return self.site_outputs.get(site_name)


def get_replay_context() -> ReplayContext | None:
    """Get the active replay context, if any."""
    return _get_handler("replay")


@contextmanager
def replay(trace: list[TraceEvent]) -> Iterator[ReplayContext]:
    """
    Context manager to replay sample sites from a previous trace.

    Within this context, sample primitives will check if their site
    exists in the recorded trace and return the cached output instead
    of calling the LLM.

    Args:
        trace: Previous trace (list of TraceEvents) to replay from

    Yields:
        ReplayContext with replay statistics

    Example:
        >>> # First run - records trace
        >>> state1, out1 = my_model(PLPState())
        >>>
        >>> # Replay - reuses cached outputs
        >>> with replay(state1.trace) as r:
        ...     state2, out2 = my_model(PLPState())
        ...     print(f"Replayed {r.replay_count} sites")
    """
    # Build site_outputs map from trace
    site_outputs = {}
    for event in trace:
        if event.primitive not in ["(observe)", "(factor)"]:
            site_outputs[event.primitive] = event.output

    ctx = ReplayContext(name="replay", trace=trace, site_outputs=site_outputs)

    with _generic_handler("replay", ctx) as context:
        yield context


# ============================================================
# Block Handler
# ============================================================


@dataclass
class BlockContext(HandlerContext):
    """
    Context for blocking specific sites.

    Attributes:
        blocked_sites: Set of site names to block
        block_count: Number of sites blocked
    """

    blocked_sites: set[str] = field(default_factory=set)
    block_count: int = 0

    def is_blocked(self, site_name: str) -> bool:
        """Check if a site is blocked."""
        return site_name in self.blocked_sites


def get_block_context() -> BlockContext | None:
    """Get the active block context, if any."""
    return _get_handler("block")


@contextmanager
def block(sites: list[str] | set[str]) -> Iterator[BlockContext]:
    """
    Context manager to block specific sites during inference.

    Blocked sites will not execute their normal logic - sample sites
    return empty strings, observe/factor sites are skipped.

    Args:
        sites: Collection of site names to block

    Yields:
        BlockContext with block statistics

    Example:
        >>> # Block stylistic factors during core inference
        >>> with block(["style_score", "brevity_penalty"]):
        ...     state, out = my_model(PLPState())
    """
    ctx = BlockContext(name="block", blocked_sites=set(sites))

    with _generic_handler("block", ctx) as context:
        yield context


# ============================================================
# Do/Intervention Handler
# ============================================================


@dataclass
class DoContext(HandlerContext):
    """
    Context for causal interventions.

    Attributes:
        interventions: Mapping of site names to forced values
        intervention_count: Number of interventions applied
    """

    interventions: dict[str, str] = field(default_factory=dict)
    intervention_count: int = 0

    def get_intervention(self, site_name: str) -> str | None:
        """Get intervention value for a site, if any."""
        return self.interventions.get(site_name)


def get_do_context() -> DoContext | None:
    """Get the active do/intervention context, if any."""
    return _get_handler("do")


@contextmanager
def do(interventions: dict[str, str]) -> Iterator[DoContext]:
    """
    Context manager for causal interventions (do-operator).

    Forces specific sample sites to return predetermined values,
    ignoring the normal generative process. Useful for counterfactual
    analysis and debugging.

    Args:
        interventions: Mapping of site names to forced output values

    Yields:
        DoContext with intervention statistics

    Example:
        >>> # Force decomposition to use specific approach
        >>> with do({"decomposition": "Use dynamic programming"}):
        ...     state, plan = planning_agent(PLPState())
        ...     # decomposition site will return "Use dynamic programming"
    """
    ctx = DoContext(name="do", interventions=interventions)

    with _generic_handler("do", ctx) as context:
        yield context


# ============================================================
# Condition Handler (pin values as observations)
# ============================================================


@dataclass
class ConditionContext(HandlerContext):
    """
    Context for conditioning on observed values.

    Similar to do(), but semantically represents observations
    rather than interventions.
    """

    observations: dict[str, str] = field(default_factory=dict)
    condition_count: int = 0


def get_condition_context() -> ConditionContext | None:
    """Get the active condition context, if any."""
    return _get_handler("condition")


@contextmanager
def condition(observations: dict[str, str]) -> Iterator[ConditionContext]:
    """
    Context manager for conditioning on observed values.

    Similar to do(), but semantically represents that we're
    conditioning the model on these observations rather than
    intervening.

    Args:
        observations: Mapping of site names to observed values

    Yields:
        ConditionContext
    """
    ctx = ConditionContext(name="condition", observations=observations)

    with _generic_handler("condition", ctx) as context:
        yield context


# ============================================================
# Helper to check handlers in sample primitive
# ============================================================


def check_handlers(site_name: str) -> tuple[str | None, str]:
    """
    Check all active handlers for a sample site.

    Returns:
        Tuple of (cached_output_or_none, handler_used)
        If cached_output is not None, use it instead of calling LLM.
    """
    # Check do context first (interventions take precedence)
    do_ctx = get_do_context()
    if do_ctx:
        intervention = do_ctx.get_intervention(site_name)
        if intervention is not None:
            do_ctx.intervention_count += 1
            return intervention, "do"

    # Check condition context
    cond_ctx = get_condition_context()
    if cond_ctx:
        obs = cond_ctx.observations.get(site_name)
        if obs is not None:
            cond_ctx.condition_count += 1
            return obs, "condition"

    # Check replay context
    replay_ctx = get_replay_context()
    if replay_ctx:
        cached = replay_ctx.get_cached_output(site_name)
        if cached is not None:
            replay_ctx.replay_count += 1
            return cached, "replay"

    # Check block context
    block_ctx = get_block_context()
    if block_ctx and block_ctx.is_blocked(site_name):
        block_ctx.block_count += 1
        return "", "block"

    return None, "none"
