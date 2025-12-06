"""
PLP Deterministic Operator.

Named derived quantities that are recorded in the trace
but don't involve stochastic sampling.
"""

from typing import Callable, Any
from functools import wraps

from .state import PLPState, TraceEvent


def deterministic(name: str | None = None):
    """
    Decorator for deterministic derived quantities.

    The wrapped function performs a pure transformation on the state/inputs
    and its result is recorded in the trace. No LLM calls, no randomness.

    Useful for:
    - Recording summaries of longer texts
    - Canonicalizing candidate answers
    - Extracting structured data from NL
    - Naming intermediate computations for debugging

    Args:
        name: Name for this deterministic site (defaults to function name)

    Example:
        >>> @deterministic(name="summary")
        ... def extract_summary(state: PLPState, long_text: str) -> str:
        ...     return long_text[:200] + "..."
        ...
        >>> state, summary = extract_summary(PLPState(), my_long_text)
    """

    def decorator(fn: Callable[..., Any]):
        site_name = name or fn.__name__

        @wraps(fn)
        def wrapped(state: PLPState, *args, **kwargs) -> tuple[PLPState, Any]:
            result = fn(state, *args, **kwargs)

            new_state = state.copy()
            event = TraceEvent(
                primitive=site_name,
                prompt="(deterministic)",
                output=str(result) if not isinstance(result, str) else result,
                meta={"deterministic": True},
            )
            new_state.trace.append(event)

            return new_state, result

        return wrapped

    return decorator


def store(state: PLPState, name: str, value: str) -> PLPState:
    """
    Store a named value in state.vars and record in trace.

    Simple helper for explicitly naming intermediate NL values.

    Args:
        state: Current PLP state
        name: Variable name
        value: Value to store (string)

    Returns:
        New state with value stored

    Example:
        >>> state = store(state, "plan", "First do X, then Y")
    """
    new_state = state.copy()
    new_state.vars[name] = value
    new_state.trace.append(
        TraceEvent(
            primitive=f"store({name})",
            prompt="(deterministic store)",
            output=value,
            meta={"deterministic": True, "var_name": name},
        )
    )
    return new_state


def summarize(
    state: PLPState, content: str, max_length: int = 500, var_name: str | None = None
) -> tuple[PLPState, str]:
    """
    Create a truncated summary (deterministic, no LLM).

    For LLM-based summarization, use a @sample decorator instead.

    Args:
        state: Current PLP state
        content: Content to summarize
        max_length: Maximum length
        var_name: Optional name to store in state.vars

    Returns:
        Tuple of (new_state, summary_string)
    """
    if len(content) <= max_length:
        summary = content
    else:
        summary = content[: max_length - 3] + "..."

    new_state = state.copy()
    if var_name:
        new_state.vars[var_name] = summary

    new_state.trace.append(
        TraceEvent(
            primitive="summarize",
            prompt=f"(deterministic summarize, max_length={max_length})",
            output=summary,
            meta={
                "deterministic": True,
                "original_length": len(content),
                "summary_length": len(summary),
            },
        )
    )

    return new_state, summary


def canonicalize(
    state: PLPState,
    text: str,
    strip: bool = True,
    lowercase: bool = False,
    var_name: str | None = None,
) -> tuple[PLPState, str]:
    """
    Canonicalize text (deterministic normalization).

    Args:
        state: Current PLP state
        text: Text to canonicalize
        strip: Whether to strip whitespace
        lowercase: Whether to lowercase
        var_name: Optional name to store in state.vars

    Returns:
        Tuple of (new_state, canonicalized_text)
    """
    result = text
    if strip:
        result = result.strip()
    if lowercase:
        result = result.lower()

    new_state = state.copy()
    if var_name:
        new_state.vars[var_name] = result

    new_state.trace.append(
        TraceEvent(
            primitive="canonicalize",
            prompt=f"(deterministic canonicalize)",
            output=result,
            meta={"deterministic": True, "strip": strip, "lowercase": lowercase},
        )
    )

    return new_state, result
