"""
PLP High-Level API.

Provides a simplified, Pythonic interface for Probabilistic Language Programming.
Hides the complexity of PLPState and LLMClient for common use cases.
"""

from typing import Callable, Any, Literal, Optional
from functools import wraps

from .state import PLPState
from .client import LLMClient
from .primitives import sample as _sample_primitive
from .infer import InferEngine, compute_k_for_eps
from ._config import get_llm, get_config


# ============================================================
# Simplified Sampling
# ============================================================


def sample(
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    **kwargs,
) -> str:
    """
    One-liner LLM sampling using global configuration.

    Args:
        prompt: The prompt string to send to the LLM
        system: Optional system prompt (overrides global default)
        temperature: Optional temperature override
        model: Optional model override
        **kwargs: Additional arguments passed to LLM

    Returns:
        Generated text response

    Example:
        >>> import plp
        >>> plp.configure(model="gpt-4.1-mini")
        >>> answer = plp.sample("What is 2+2?")
    """
    # Get configuration
    config = get_config()
    sys_prompt = system if system is not None else config.system_prompt

    # Get LLM instance
    client_kwargs = {
        "model": model if model is not None else config.model,
        "temperature": temperature if temperature is not None else config.temperature,
        "base_url": config.base_url,
        **config.additional_kwargs,
        **kwargs,
    }

    if config.api_key is not None:
        client_kwargs["api_key"] = config.api_key

    llm_client = LLMClient(**client_kwargs)

    return llm_client.invoke(prompt, system_prompt=sys_prompt)


def sampler(
    name: Optional[str] = None,
    system: Optional[str] = None,
    update_vars: Optional[dict[str, str]] = None,
):
    """
    Decorator for sample sites that hides PLPState.

    Wraps a function that returns a prompt string. The wrapper handles
    state creation/passing internally.

    Args:
        name: Name of the sample site (defaults to function name)
        system: Optional system prompt for this site
        update_vars: Optional dict to map output to state variables

    Example:
        >>> @plp.sampler(name="solver")
        ... def solve(problem: str) -> str:
        ...     return f"Solve: {problem}"
        ...
        >>> result = solve("2+2")
    """

    def decorator(fn: Callable[..., str]):
        site_name = name or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            # Check if called with return_state=True
            return_state = kwargs.pop("return_state", False)

            # Check if state is passed explicitly as first arg
            # This allows mixing high-level and low-level styles
            if args and isinstance(args[0], PLPState):
                state = args[0]
                fn_args = args[1:]
            else:
                state = PLPState()
                fn_args = args

            # Get LLM from global config
            # We create a new client for each call to respect current config
            # but get_llm caches the underlying ChatOpenAI instance
            config = get_config()
            sys_prompt = system if system is not None else config.system_prompt

            # Create a temporary client wrapper that uses our cached LLM
            client_kwargs = {
                "model": config.model,
                "temperature": config.temperature,
                "base_url": config.base_url,
                **config.additional_kwargs,
            }
            if config.api_key is not None:
                client_kwargs["api_key"] = config.api_key

            llm = LLMClient(**client_kwargs)

            # Define the prompt generator
            def prompt_gen(s: PLPState, *a, **k) -> str:
                # If system prompt is set, we prepend it to the prompt
                # This is a bit of a hack since low-level sample doesn't support system prompt natively yet
                # ideally we'd pass it to invoke, but the primitive calls invoke(prompt)
                # So we bake it into the prompt or rely on the client update
                p = fn(*a, **k)
                # We can't easily pass system prompt to the primitive's invoke call
                # without changing primitives.py.
                # For now, let's assume the user handles system prompt in the prompt
                # OR we update LLMClient to handle it if we could pass it.
                return p

            # Use the low-level primitive
            # We need to monkey-patch the client's invoke if we want system prompt support
            # inside the primitive without changing primitive signature
            if sys_prompt:
                original_invoke = llm.invoke
                llm.invoke = lambda p: original_invoke(p, system_prompt=sys_prompt)

            # Create the primitive
            prim = _sample_primitive(llm, name=site_name)(prompt_gen)

            # Execute
            new_state, output = prim(state, *fn_args, **kwargs)

            # Update vars if requested
            if update_vars:
                # TODO: Implement var updates
                pass

            if return_state:
                return output, new_state
            return output

        return wrapper

    return decorator


# ============================================================
# Inference Shortcuts
# ============================================================


def vote(
    candidates: list[str],
    judge: Callable[[str, str], str] | str | None = None,
    k: int = 3,
    p: float = 0.7,
    eps: Optional[float] = None,
    max_rounds: int = 1000,
) -> str:
    """
    SPRT voting over candidates.

    Args:
        candidates: List of candidate strings
        judge: Comparison function or prompt template.
               If callable: (a, b) -> "a" | "b" | "tie"
               If string: Template with {a} and {b} placeholders
               If None: Uses default comparison prompt
        k: Vote margin (auto-computed if eps provided)
        p: Assumed judge accuracy (used if eps provided)
        eps: Target error probability (overrides k)
        max_rounds: Safety limit on rounds

    Returns:
        Winning candidate
    """
    # Calculate k from eps if provided
    if eps is not None:
        k = compute_k_for_eps(p, eps)

    # Normalize judge
    if judge is None:
        judge_fn = _default_judge
    elif isinstance(judge, str):
        judge_fn = _create_llm_judge(judge)
    else:
        judge_fn = judge

    return InferEngine.sprt_discrete(
        candidates=candidates, judge_fn=judge_fn, k=k, max_rounds=max_rounds
    )


def map(
    fn: Callable,
    inputs: list[Any],
    runs: int = 5,
    select: Literal["best", "first_valid", "all"] = "best",
) -> Any:
    """
    Run function multiple times and select best result.

    Args:
        fn: Function to run (must be compatible with PLP state if using 'best')
        inputs: List of inputs to run on (or list of None for simple repetition)
        runs: Number of runs per input
        select: Selection strategy

    Returns:
        Selected result(s)
    """
    # TODO: Implement full map logic
    # For now, simple repetition wrapper
    results = []
    for inp in inputs:
        for _ in range(runs):
            if inp is None:
                res = fn()
            else:
                res = fn(inp)
            results.append(res)

    if select == "best":
        # Needs a way to score results - for now just return last
        return results[-1] if results else None
    return results


def guarantee(
    eps: float = 0.01,
    p: float = 0.7,
    checker: Optional[Callable[[str], bool]] = None,
):
    """
    Decorator to wrap a function with reliability guarantees.

    Args:
        eps: Target error probability
        p: Assumed generator accuracy
        checker: Optional correctness checker
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # Calculate required redundancy
            # This is a placeholder for the full guarantee logic
            # which would involve retries/ensembles
            return fn(*args, **kwargs)

        return wrapper

    return decorator


# ============================================================
# Inference Builder
# ============================================================


class InferBuilder:
    """Fluent builder for inference configuration."""

    def __init__(self):
        self._k = 3
        self._eps = None
        self._p = 0.7
        self._runs = 5
        self._judge = None

    def with_k(self, k: int) -> "InferBuilder":
        self._k = k
        self._eps = None  # Clear eps if k is set manually
        return self

    def with_eps(self, eps: float) -> "InferBuilder":
        self._eps = eps
        return self

    def with_p(self, p: float) -> "InferBuilder":
        self._p = p
        return self

    def with_runs(self, runs: int) -> "InferBuilder":
        self._runs = runs
        return self

    def vote(self, candidates: list[str], judge=None) -> str:
        """Execute SPRT voting with current settings."""
        return vote(
            candidates, judge=judge or self._judge, k=self._k, p=self._p, eps=self._eps
        )

    def map(self, fn: Callable, inputs: list) -> Any:
        """Execute MAP inference with current settings."""
        return map(fn, inputs, runs=self._runs)


def infer() -> InferBuilder:
    """Create inference builder for fluent API."""
    return InferBuilder()


# ============================================================
# Internal Helpers
# ============================================================


def _default_judge(a: str, b: str) -> str:
    """Default LLM-based judge."""
    prompt = f"Compare these two options:\nA: {a}\nB: {b}\n\nWhich is better? Reply 'A' or 'B'."
    resp = sample(prompt, temperature=0.1)
    if "A" in resp and "B" not in resp:
        return "a"
    if "B" in resp and "A" not in resp:
        return "b"
    return "tie"


def _create_llm_judge(template: str) -> Callable[[str, str], str]:
    """Create a judge function from a prompt template."""

    def judge(a: str, b: str) -> str:
        prompt = template.format(a=a, b=b)
        resp = sample(prompt, temperature=0.1)
        # Simple heuristic parsing
        lower = resp.lower()
        if "a" in lower and "b" not in lower:
            return "a"
        if "b" in lower and "a" not in lower:
            return "b"
        # If both or neither, try to parse strict start
        if resp.strip().lower().startswith("a"):
            return "a"
        if resp.strip().lower().startswith("b"):
            return "b"
        return "tie"

    return judge
