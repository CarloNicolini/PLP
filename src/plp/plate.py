"""
PLP Plate Operator.

Provides plate context manager for independent, identically distributed
repetition - matching NumPyro's plate syntax.

Usage:
    with plp.plate("candidates", 5) as indices:
        for i in indices:
            state, candidate = propose_solution(state, problem)

Concurrency:
    The global ``_plate_stack`` is not safe for concurrent *entry/exit* of
    ``plate()`` / ``plate_stack()`` from multiple threads. Parallel helpers
    (threaded: ``vectorized_sample(..., max_workers=...)``, ``plate_map(...)``;
    async: ``avectorized_sample(...)``, ``plate_amap(...)``) run the per-index
    body concurrently while a single plate context is active on the spawning
    thread/task; that pattern is intended for I/O-bound work (e.g. LLM calls).
    Do not nest new ``plate()`` contexts inside worker threads unless you
    serialize those sections.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Any, Awaitable, Iterator

from .state import PLPState, TraceEvent


@dataclass
class PlateContext:
    """
    Captures the context of a plate execution.

    Attributes:
        name: Identifier for this plate
        size: Number of independent repetitions (full plate size)
        subsample_size: Size of mini-batch (if subsampling)
        dim: Dimension for plate (for compatibility, -1 default)
        indices: Indices to iterate over
        results: Collected results from plate iterations
    """

    name: str
    size: int
    subsample_size: int | None = None
    dim: int = -1
    indices: list[int] = field(default_factory=list)
    results: list[Any] = field(default_factory=list)
    states: list[PLPState] = field(default_factory=list)

    def __iter__(self):
        """Iterate over plate indices."""
        return iter(self.indices)

    def __len__(self):
        """Return number of indices."""
        return len(self.indices)

    def add_result(self, state: PLPState, result: Any):
        """Store a result from one plate iteration."""
        self.states.append(state)
        self.results.append(result)

    @property
    def scale(self) -> float:
        """
        Scale factor for ELBO computation when subsampling.

        Returns size / subsample_size, or 1.0 if no subsampling.
        """
        if self.subsample_size is None:
            return 1.0
        return self.size / self.subsample_size


# Stack of active plates (for nested plates)
_plate_stack: list[PlateContext] = []


def get_active_plate() -> PlateContext | None:
    """Get the currently active plate context, if any."""
    return _plate_stack[-1] if _plate_stack else None


def get_plate_stack() -> list[PlateContext]:
    """Get the full stack of active plates."""
    return _plate_stack.copy()


@contextmanager
def plate(
    name: str, size: int, subsample_size: int | None = None, dim: int = -1
) -> Iterator[PlateContext]:
    """
    Context manager for independent parallel repetition.

    Matches NumPyro's plate syntax. Within a plate, you iterate over
    indices to run independent samples.

    Args:
        name: Identifier for this plate (e.g., "candidates", "data")
        size: Total size of the plate
        subsample_size: Optional mini-batch size for subsampling
        dim: Dimension for plate (for tensor compatibility, default -1)

    Yields:
        PlateContext that can be iterated for indices

    Example (NumPyro-style):
        >>> with plp.plate("candidates", 5) as p:
        ...     candidates = []
        ...     for i in p:
        ...         state, candidate = propose(state, problem)
        ...         candidates.append(candidate)
        ...         p.add_result(state, candidate)

    Example (subsampling):
        >>> with plp.plate("data", len(dataset), subsample_size=32) as p:
        ...     batch = [dataset[i] for i in p]
        ...     # Process mini-batch
    """
    # Determine indices
    if subsample_size is not None and subsample_size < size:
        import random

        indices = random.sample(range(size), subsample_size)
    else:
        indices = list(range(size))

    ctx = PlateContext(
        name=name, size=size, subsample_size=subsample_size, dim=dim, indices=indices
    )

    _plate_stack.append(ctx)
    try:
        yield ctx
    finally:
        _plate_stack.pop()


def vectorized_sample(
    sample_fn: Callable[[PLPState, int], tuple[PLPState, Any]],
    init_state: PLPState,
    plate_ctx: PlateContext,
    *,
    max_workers: int | None = None,
) -> tuple[list[PLPState], list[Any]]:
    """
    Run sample_fn for each index in the plate.

    This is a helper that vectorizes sampling across the plate.

    Args:
        sample_fn: Function taking (state, index) returning (state, output)
        init_state: Initial state (copied for each run)
        plate_ctx: Active plate context
        max_workers: If > 1, run each index in a thread pool (order of results
            matches ``plate_ctx.indices``). Use for I/O-bound ``sample_fn`` (e.g.
            remote LLM calls). ``None`` or ``1`` keeps the original sequential
            loop.

    Returns:
        Tuple of (list of states, list of outputs)
    """
    indices = list(plate_ctx.indices)

    def run_one(i: int) -> tuple[PLPState, Any]:
        state_copy = init_state.copy()
        return sample_fn(state_copy, i)

    if max_workers is None or max_workers <= 1:
        states: list[PLPState] = []
        outputs: list[Any] = []
        for i in indices:
            final_state, output = run_one(i)
            states.append(final_state)
            outputs.append(output)
            plate_ctx.add_result(final_state, output)
        return states, outputs

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(run_one, indices))

    states = [s for s, _ in results]
    outputs = [o for _, o in results]
    for fs, out in zip(states, outputs):
        plate_ctx.add_result(fs, out)

    return states, outputs


async def avectorized_sample(
    sample_fn: Callable[[PLPState, int], Awaitable[tuple[PLPState, Any]]],
    init_state: PLPState,
    plate_ctx: PlateContext,
    *,
    max_concurrency: int | None = None,
) -> tuple[list[PLPState], list[Any]]:
    """
    Async variant of :func:`vectorized_sample`.

    Args:
        sample_fn: Async function taking (state, index) and returning
            (state, output).
        init_state: Initial state (copied for each run).
        plate_ctx: Active plate context.
        max_concurrency: Maximum number of in-flight tasks. ``None`` means no
            explicit cap other than the plate size.

    Returns:
        Tuple of (list of states, list of outputs) ordered by ``plate_ctx.indices``.
    """
    indices = list(plate_ctx.indices)
    if not indices:
        return [], []

    if max_concurrency is not None and max_concurrency <= 0:
        raise ValueError("max_concurrency must be > 0 when provided")

    limit = max_concurrency if max_concurrency is not None else len(indices)
    semaphore = asyncio.Semaphore(limit)
    results: list[tuple[PLPState, Any] | None] = [None] * len(indices)

    async def run_one(pos: int, i: int) -> None:
        async with semaphore:
            state_copy = init_state.copy()
            results[pos] = await sample_fn(state_copy, i)

    await asyncio.gather(*(run_one(pos, i) for pos, i in enumerate(indices)))

    states: list[PLPState] = []
    outputs: list[Any] = []
    for result in results:
        if result is None:
            raise RuntimeError("Missing async plate result")
        final_state, output = result
        states.append(final_state)
        outputs.append(output)
        plate_ctx.add_result(final_state, output)

    return states, outputs


# ============================================================
# Convenience functions (keep for backwards compatibility)
# ============================================================


def plate_map(
    name: str,
    size: int,
    fn: Callable[[PLPState], tuple[PLPState, Any]],
    init_state: PLPState | None = None,
    *,
    max_workers: int | None = None,
) -> tuple[list[PLPState], list[Any]]:
    """
    Run a function N times independently, collecting results.

    Convenience wrapper around plate context manager.

    Args:
        name: Identifier for this plate
        size: Number of independent runs
        fn: Function taking state and returning (new_state, output)
        init_state: Initial state (defaults to fresh PLPState)
        max_workers: If > 1, run iterations concurrently via a thread pool
            (output order matches plate indices). Intended for I/O-bound ``fn``.

    Returns:
        Tuple of (list of final states, list of outputs)

    Example:
        >>> states, candidates = plate_map("candidates", 5, propose_fn)
    """
    if init_state is None:
        init_state = PLPState()

    states: list[PLPState] = []
    outputs: list[Any] = []

    with plate(name, size) as p:
        if max_workers is None or max_workers <= 1:
            for _ in p:
                state_copy = init_state.copy()
                final_state, output = fn(state_copy)
                states.append(final_state)
                outputs.append(output)
                p.add_result(final_state, output)
        else:

            def run_one(_i: int) -> tuple[PLPState, Any]:
                state_copy = init_state.copy()
                return fn(state_copy)

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                results = list(pool.map(run_one, p.indices))
            for final_state, output in results:
                states.append(final_state)
                outputs.append(output)
                p.add_result(final_state, output)

    return states, outputs


def plate_reduce(
    name: str,
    size: int,
    fn: Callable[[PLPState], tuple[PLPState, Any]],
    reducer: Callable[[list[Any]], Any],
    init_state: PLPState | None = None,
    *,
    max_workers: int | None = None,
) -> tuple[PLPState, Any]:
    """
    Run fn N times and reduce the outputs to a single result.

    Useful for ensemble patterns where you want a single output.

    Args:
        name: Identifier for this plate
        size: Number of independent runs
        fn: Function taking state and returning (new_state, output)
        reducer: Function to combine N outputs into one
        init_state: Initial state
        max_workers: Passed to :func:`plate_map` for parallel runs.

    Returns:
        Tuple of (combined state with plate event, reduced output)

    Example:
        >>> # Majority vote
        >>> state, best = plate_reduce(
        ...     "ensemble", 5, solve_problem,
        ...     reducer=lambda outs: max(set(outs), key=outs.count)
        ... )
    """
    states, outputs = plate_map(name, size, fn, init_state, max_workers=max_workers)

    reduced = reducer(outputs)

    # Create a combined state with plate trace event
    combined_state = states[-1].copy() if states else PLPState()
    combined_state.trace.append(
        TraceEvent(
            primitive=f"plate_reduce({name})",
            prompt=f"(plate: {size} runs)",
            output=str(reduced),
            meta={"plate_name": name, "plate_size": size, "all_outputs": outputs},
        )
    )

    return combined_state, reduced


async def plate_amap(
    name: str,
    size: int,
    fn: Callable[[PLPState], Awaitable[tuple[PLPState, Any]]],
    init_state: PLPState | None = None,
    *,
    max_concurrency: int | None = None,
) -> tuple[list[PLPState], list[Any]]:
    """
    Async variant of :func:`plate_map` for concurrent async LLM calls.

    Args:
        name: Identifier for this plate.
        size: Number of independent runs.
        fn: Async function taking a state and returning (new_state, output).
        init_state: Initial state (defaults to fresh PLPState).
        max_concurrency: Maximum number of in-flight tasks. ``None`` means no
            explicit cap other than the plate size.

    Returns:
        Tuple of (list of final states, list of outputs), in index order.
    """
    if init_state is None:
        init_state = PLPState()

    with plate(name, size) as p:

        async def run_one(state: PLPState, _i: int) -> tuple[PLPState, Any]:
            return await fn(state)

        return await avectorized_sample(
            sample_fn=run_one,
            init_state=init_state,
            plate_ctx=p,
            max_concurrency=max_concurrency,
        )


async def plate_areduce(
    name: str,
    size: int,
    fn: Callable[[PLPState], Awaitable[tuple[PLPState, Any]]],
    reducer: Callable[[list[Any]], Any],
    init_state: PLPState | None = None,
    *,
    max_concurrency: int | None = None,
) -> tuple[PLPState, Any]:
    """
    Async variant of :func:`plate_reduce`.

    Args:
        name: Identifier for this plate.
        size: Number of independent runs.
        fn: Async function taking state and returning (new_state, output).
        reducer: Function combining outputs into one result.
        init_state: Initial state.
        max_concurrency: Passed to :func:`plate_amap`.

    Returns:
        Tuple of (combined state with plate event, reduced output).
    """
    states, outputs = await plate_amap(
        name=name,
        size=size,
        fn=fn,
        init_state=init_state,
        max_concurrency=max_concurrency,
    )

    reduced = reducer(outputs)

    combined_state = states[-1].copy() if states else PLPState()
    combined_state.trace.append(
        TraceEvent(
            primitive=f"plate_areduce({name})",
            prompt=f"(plate: {size} runs)",
            output=str(reduced),
            meta={"plate_name": name, "plate_size": size, "all_outputs": outputs},
        )
    )

    return combined_state, reduced


# ============================================================
# plate_stack for nested plates (NumPyro compatibility)
# ============================================================


@contextmanager
def plate_stack(prefix: str, sizes: list[int]) -> Iterator[list[PlateContext]]:
    """
    Create a stack of nested plates.

    Convenience for common patterns with multiple plates.

    Args:
        prefix: Base name for plates (will be prefix_0, prefix_1, etc.)
        sizes: List of sizes for each plate

    Yields:
        List of PlateContext objects

    Example:
        >>> with plate_stack("batch", [10, 5]) as plates:
        ...     # plates[0] is plate("batch_0", 10)
        ...     # plates[1] is plate("batch_1", 5)
    """
    contexts = []

    for i, size in enumerate(sizes):
        ctx = PlateContext(name=f"{prefix}_{i}", size=size, indices=list(range(size)))
        contexts.append(ctx)
        _plate_stack.append(ctx)

    try:
        yield contexts
    finally:
        for _ in sizes:
            _plate_stack.pop()
