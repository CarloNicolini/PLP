"""
PLP Conditioning Helpers.

Implements approximate conditioning via rejection sampling.
"""

from typing import Callable, Any

from .state import PLPState


def sample_with_condition(
    sample_fn: Callable[[PLPState, Any], tuple[PLPState, str]],
    condition_fn: Callable[[PLPState, str], bool],
    state: PLPState,
    *args,
    max_retries: int = 5,
    **kwargs,
) -> tuple[PLPState, str]:
    """
    Helper to implement approximate conditioning on a property φ.

    Repeatedly calls sample_fn until condition_fn returns True, or
    max_retries is reached. This implements rejection sampling over
    the LLM's output distribution.

    Args:
        sample_fn: A @sample-wrapped primitive, returning (state, output_text)
        condition_fn: (state_after_sample, output_text) -> bool
        state: Initial PLP state
        *args: Positional arguments passed to sample_fn
        max_retries: Maximum number of sampling attempts (default: 5)
        **kwargs: Keyword arguments passed to sample_fn

    Returns:
        Tuple of (final_state, output_text). If no sample satisfied the
        condition, returns the last sample (user should add observe/factor).

    Example:
        >>> def is_even_number(state, text):
        ...     try:
        ...         return int(text.strip()) % 2 == 0
        ...     except ValueError:
        ...         return False
        ...
        >>> state, answer = sample_with_condition(
        ...     generate_number_primitive,
        ...     is_even_number,
        ...     initial_state,
        ...     "Give me a number between 1 and 10",
        ...     max_retries=10
        ... )
    """
    last_state = state
    last_out = ""

    for attempt in range(max_retries):
        s2, out = sample_fn(last_state, *args, **kwargs)
        last_state, last_out = s2, out

        if condition_fn(last_state, last_out):
            return last_state, last_out

    # Return last sample even if condition not met
    return last_state, last_out


async def asample_with_condition(
    sample_fn: Callable[[PLPState, Any], tuple[PLPState, str]],
    condition_fn: Callable[[PLPState, str], bool],
    state: PLPState,
    *args,
    max_retries: int = 5,
    **kwargs,
) -> tuple[PLPState, str]:
    """
    Async version of sample_with_condition.

    Note: sample_fn should be an async function returning (state, output_text).
    """
    last_state = state
    last_out = ""

    for attempt in range(max_retries):
        s2, out = await sample_fn(last_state, *args, **kwargs)
        last_state, last_out = s2, out

        if condition_fn(last_state, last_out):
            return last_state, last_out

    return last_state, last_out
