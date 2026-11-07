"""
PLP Core Primitives.

Decorators for the fundamental PLP operations:
- sample / asample: LLM-based stochastic generation (sync ``invoke`` / async ``ainvoke``)
- observe: Hard 0/1 constraints
- factor: Soft log-weight scoring (with scale/mask support)
"""

from functools import wraps
from typing import Any, Callable

from ._llm import AsyncLLMLike, LLMLike
from .runtime import ensure_async_llm, ensure_llm, arun_prompt, run_prompt
from .state import PLPState, TraceEvent


def sample(
    llm: LLMLike,
    name: str | None = None,
    update_state: Callable[[PLPState, str], None] | None = None,
    system_prompt: str | None = None,
    parents: list[str] | tuple[str, ...] = (),
):
    """
    Decorator for LLM-based sample primitives.

    The wrapped function must have signature
    ``fn(state: PLPState, *args, **kwargs) -> str``. It returns the prompt string to send to the LLM.

    This decorator:
        - Checks for active handlers (replay, block, do, condition)
        - If no handler intervenes, builds prompt and calls LLM
        - Logs a TraceEvent
        - Optionally updates the state via update_state(state, output)
        - Returns (new_state, output)

    Args:
        llm: The LLM client to use for generation
        name: Optional name for this primitive (defaults to function name)
        update_state: Optional callback to update state.vars with the output
        system_prompt: Optional system message prepended to the prompt
        parents: Names of other sample sites this one reads from state.vars.
            Declares the directed edges in the ChainGraph. Empty by default
            (a root site). Cyclic declarations are permitted — they induce
            a Case-B (MRF) graph requiring Gibbs iteration.

    Raises:
        TypeError: If llm does not provide an 'invoke' method
        TypeError: If update_state is not callable

    Example:
        >>> @sample(llm, name="arithmetic", parents=["reasoning"])
        ... def solve_arithmetic(state: PLPState, problem: str) -> str:
        ...     r = state.vars.get("reasoning", "")
        ...     return f"Given {r}, answer: {problem}"
        ...
        >>> state, answer = solve_arithmetic(PLPState(), "2+2")
    """
    ensure_llm(llm)
    if update_state is not None and not callable(update_state):
        raise TypeError(f"update_state must be callable, got {type(update_state)}")

    parents_tuple = tuple(parents)

    def decorator(fn: Callable[..., str]):
        prim_name = name or fn.__name__

        @wraps(fn)
        def wrapped(state: PLPState, *args, **kwargs) -> tuple[PLPState, str]:
            prompt = fn(state, *args, **kwargs)
            new_state, output = run_prompt(
                llm,
                state,
                name=prim_name,
                prompt=prompt,
                system_prompt=system_prompt,
            )

            if update_state is not None:
                update_state(new_state, output)

            return new_state, output

        # Attach graph metadata so ChainGraph builders can introspect.
        wrapped.__plp_site_name__ = prim_name  # type: ignore[attr-defined]
        wrapped.__plp_parents__ = parents_tuple  # type: ignore[attr-defined]
        wrapped.__plp_kind__ = "sample"  # type: ignore[attr-defined]
        return wrapped

    return decorator


def asample(
    llm: AsyncLLMLike | Any,
    name: str | None = None,
    update_state: Callable[[PLPState, str], None] | None = None,
    system_prompt: str | None = None,
    parents: list[str] | tuple[str, ...] = (),
):
    """
    Async decorator for LLM-based sample primitives (uses ``ainvoke``).

    Same contract as :func:`sample`: the wrapped function returns the prompt
    string; the wrapper returns ``(PLPState, str)`` but is an ``async def`` so
    it composes with :func:`plp.plate_amap` and other async plate helpers.

    Args:
        llm: Object with ``async def ainvoke(messages) -> response``
        name: Optional site name (defaults to fn.__name__)
        update_state: Optional callback to update state.vars with the output
        system_prompt: Optional system message
        parents: Names of other sample sites this one reads from state.vars.
            See :func:`sample` for details. Cyclic parent declarations across
            sites produce a Case-B (MRF) graph.

    Raises:
        TypeError: If ``llm`` does not provide a callable ``ainvoke``
        TypeError: If ``update_state`` is not callable
    """
    ensure_async_llm(llm)
    if update_state is not None and not callable(update_state):
        raise TypeError(f"update_state must be callable, got {type(update_state)}")

    parents_tuple = tuple(parents)

    def decorator(fn: Callable[..., str]):
        prim_name = name or fn.__name__

        @wraps(fn)
        async def wrapped(state: PLPState, *args, **kwargs) -> tuple[PLPState, str]:
            prompt = fn(state, *args, **kwargs)
            new_state, output = await arun_prompt(
                llm,
                state,
                name=prim_name,
                prompt=prompt,
                system_prompt=system_prompt,
            )

            if update_state is not None:
                update_state(new_state, output)

            return new_state, output

        wrapped.__plp_site_name__ = prim_name  # type: ignore[attr-defined]
        wrapped.__plp_parents__ = parents_tuple  # type: ignore[attr-defined]
        wrapped.__plp_kind__ = "asample"  # type: ignore[attr-defined]
        return wrapped

    return decorator


def observe(name: str | None = None):
    """
    Decorator for hard constraints (observe).

    The wrapped function must have signature
    ``fn(state: PLPState, *args, **kwargs) -> bool``.

    This decorator:
        - Calls fn to get a Boolean
        - If False, sets log_weight = -inf (kills this trace)
        - Logs a TraceEvent
        - Returns the new state

    Use sparingly: this is a true 0/1 constraint at the model level.
    For softer constraints, use @factor instead.

    Args:
        name: Optional name for this observation (defaults to function name)

    Example:
        >>> @observe(name="check_format")
        ... def is_valid_number(state: PLPState, text: str) -> bool:
        ...     try:
        ...         int(text.strip())
        ...         return True
        ...     except ValueError:
        ...         return False
    """

    def decorator(fn: Callable[..., bool]):
        obs_name = name or fn.__name__

        @wraps(fn)
        def wrapped(state: PLPState, *args, **kwargs) -> PLPState:
            from .handlers import is_blocked, record_blocked_event

            if is_blocked(obs_name):
                return record_blocked_event(
                    state,
                    obs_name,
                    "observe",
                    output="skipped",
                    extra_meta={"passed": True},
                )

            ok = fn(state, *args, **kwargs)

            new_state = state.copy()
            if not ok:
                new_state.log_weight = float("-inf")

            event = TraceEvent(
                primitive=obs_name,
                prompt="(observe)",
                output=str(ok),
                meta={"observe": True, "passed": ok},
            )
            new_state.trace.append(event)
            return new_state

        return wrapped

    return decorator


def factor(
    name: str | None = None,
    scale: float = 1.0,
    mask: Callable[[PLPState], bool] | None = None,
    parents: list[str] | tuple[str, ...] | None = None,
):
    """
    Decorator for soft factors (log-weights).

    The wrapped function must have signature
    ``fn(state: PLPState, *args, **kwargs) -> float``.

    It should return a log-factor that will be scaled and added to log_weight.
    Positive values favor this trace; negative values penalize it.

    Args:
        name: Optional name for this factor (defaults to function name)
        scale: Multiplier for the log-factor (default 1.0)
        mask: Optional predicate - if returns False, factor is skipped
        parents: Names of sample sites this factor reads. These form the
            clique of the factor in the chain graph. If None, parents are
            inferred from the function's signature (all parameters after
            ``state``), matching the legacy FactorGraph behaviour.

    Raises:
        TypeError: If mask is not callable
        ValueError: If scale is not a finite number

    Example:
        >>> @factor(name="coherence", scale=0.1, parents=["text"])
        ... def score_coherence(state: PLPState, text: str) -> float:
        ...     return some_coherence_score(text)
    """
    if mask is not None and not callable(mask):
        raise TypeError(f"mask must be callable, got {type(mask)}")
    if not isinstance(scale, (int, float)) or not (
        -float("inf") < scale < float("inf")
    ):
        raise ValueError(f"scale must be a finite number, got {scale}")

    def decorator(fn: Callable[..., float]):
        fac_name = name or fn.__name__
        inferred = False
        if parents is None:
            import inspect
            sig = inspect.signature(fn)
            params = list(sig.parameters)
            if params and params[0] == "state":
                params = params[1:]
            parents_tuple = tuple(params)
            inferred = True
        else:
            parents_tuple = tuple(parents)

        @wraps(fn)
        def wrapped(state: PLPState, *args, **kwargs) -> PLPState:
            from .handlers import is_blocked, record_blocked_event

            if is_blocked(fac_name):
                return record_blocked_event(
                    state, fac_name, "factor", output="0.0",
                )

            if mask is not None and not mask(state):
                new_state = state.copy()
                new_state.trace.append(
                    TraceEvent(
                        primitive=fac_name,
                        prompt="(factor - masked)",
                        output="0.0",
                        meta={"factor": True, "log_factor": 0.0, "masked": True},
                    )
                )
                return new_state

            raw_logf = fn(state, *args, **kwargs)
            scaled_logf = raw_logf * scale

            new_state = state.copy()
            new_state.log_weight += scaled_logf

            event = TraceEvent(
                primitive=fac_name,
                prompt="(factor)",
                output=str(scaled_logf),
                meta={
                    "factor": True,
                    "log_factor": scaled_logf,
                    "raw_log_factor": raw_logf,
                    "scale": scale,
                },
            )
            new_state.trace.append(event)
            return new_state

        wrapped.__plp_site_name__ = fac_name  # type: ignore[attr-defined]
        wrapped.__plp_parents__ = parents_tuple  # type: ignore[attr-defined]
        wrapped.__plp_parents_inferred__ = inferred  # type: ignore[attr-defined]
        wrapped.__plp_kind__ = "factor"  # type: ignore[attr-defined]
        return wrapped

    return decorator
