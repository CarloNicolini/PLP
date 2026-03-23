"""
PLP High-Level API.

Provides ergonomic helpers on top of the core primitives:
- `sample`: one-off prompt using global config or a supplied LLM
- `sampler`: decorator for global-config-backed sample sites
- `vote`: SPRT inference shortcut
"""

from typing import Callable, Any, Literal, Optional
from functools import wraps

from .state import PLPState
from .infer import InferEngine, compute_k_for_eps
from ._config import get_config
from ._llm import LLMLike
from .runtime import run_prompt
from .session import _resolve_llm


# ============================================================
# Simplified Sampling
# ============================================================


def prompt(
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    llm: LLMLike | None = None,
    name: Optional[str] = None,
    return_state: bool = False,
    **llm_kwargs,
) -> str | tuple[str, PLPState]:
    """
    One-liner LLM sampling using global configuration.

    Args:
        prompt: The prompt string to send to the LLM
        system: Optional system prompt (overrides global default)
        temperature: Optional temperature override
        model: Optional model override
        llm: Optional explicit LLM client
        name: Optional trace site name (defaults to "plp.prompt")
        return_state: If True, also return the PLPState after sampling
        llm_kwargs: Additional args forwarded to ChatOpenAI

    Returns:
        Generated text response (and optionally the PLPState)

    Example:
        >>> import plp
        >>> plp.configure(model="gpt-4.1-mini")
        >>> answer = plp.prompt("What is 2+2?")
    """
    state = PLPState()
    llm_client = _resolve_llm(
        llm,
        model=model,
        temperature=temperature,
        **llm_kwargs,
    )
    config = get_config()
    sys_prompt = system if system is not None else config.system_prompt
    site_name = name or "plp.prompt"

    new_state, output = run_prompt(
        llm_client,
        state,
        name=site_name,
        prompt=prompt,
        system_prompt=sys_prompt,
    )
    if return_state:
        return output, new_state
    return output


def sampler(
    name: Optional[str] = None,
    system: Optional[str] = None,
    *,
    llm: LLMLike | None = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    **llm_kwargs,
) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """
    Decorator for sample sites that hides PLPState.

    Wraps a function that returns a prompt string. The wrapper handles
    state creation/passing internally.

    Args:
        name: Name of the sample site (defaults to function name)
        system: Optional system prompt for this site
        llm: Optional explicit LLM client
        model: Optional model override (when no llm supplied)
        temperature: Optional temperature override
        llm_kwargs: Additional ChatOpenAI kwargs

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
            return_state = kwargs.pop("return_state", False)

            if args and isinstance(args[0], PLPState):
                state = args[0]
                call_args = args[1:]
            else:
                state = PLPState()
                call_args = args

            prompt = fn(*call_args, **kwargs)

            if not isinstance(prompt, str):
                raise TypeError(
                    f"@sampler functions must return a string prompt, got {type(prompt)}"
                )

            llm_client = _resolve_llm(
                llm,
                model=model,
                temperature=temperature,
                **llm_kwargs,
            )
            config = get_config()
            sys_prompt = system if system is not None else config.system_prompt

            new_state, output = run_prompt(
                llm_client,
                state,
                name=site_name,
                prompt=prompt,
                system_prompt=sys_prompt,
            )

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


def _parse_judge_response(response: str) -> str:
    """
    Parse LLM response to extract judge verdict.

    Tries multiple heuristics to extract "a", "b", or "tie" from response.

    Args:
        response: LLM response string

    Returns:
        "a", "b", or "tie"
    """
    lower = response.lower()

    # Check for explicit mentions
    if "a" in lower and "b" not in lower:
        return "a"
    if "b" in lower and "a" not in lower:
        return "b"

    # Check for strict start
    stripped = response.strip().lower()
    if stripped.startswith("a"):
        return "a"
    if stripped.startswith("b"):
        return "b"

    return "tie"


def _default_judge(a: str, b: str) -> str:
    """Default LLM-based judge."""
    message = f"Compare these two options:\nA: {a}\nB: {b}\n\nWhich is better? Reply 'A' or 'B'."
    resp = prompt(message, temperature=0.1)
    return _parse_judge_response(resp)


def _create_llm_judge(template: str) -> Callable[[str, str], str]:
    """Create a judge function from a prompt template."""

    def judge(a: str, b: str) -> str:
        message = template.format(a=a, b=b)
        resp = prompt(message, temperature=0.1)
        return _parse_judge_response(resp)

    return judge
