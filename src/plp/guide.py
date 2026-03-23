"""
PLP Guide Operator.

Guides are secondary PLP programs that propose better samples for inference.
Inspired by Pyro's guide programs for variational inference.

A guide provides an importance proposal distribution that can make sampling
more efficient for hard subproblems.
"""

from dataclasses import dataclass
from typing import Callable, Any
from functools import wraps

from ._llm import LLMLike, invoke_llm
from .state import PLPState, TraceEvent


@dataclass
class GuideSpec:
    """
    Specification for a guide program.

    Attributes:
        name: Unique identifier for this guide
        model_name: Name of the model this guide is for
        guide_fn: The guide function
        llm: LLM client for the guide (may differ from model)
        weight: Importance weight multiplier for proposals
    """

    name: str
    model_name: str
    guide_fn: Callable[..., tuple[PLPState, Any]]
    llm: LLMLike | None = None
    weight: float = 1.0


# Global registry of guides
_guides: dict[str, GuideSpec] = {}


def register_guide(
    model_name: str,
    name: str | None = None,
    llm: LLMLike | None = None,
    weight: float = 1.0,
):
    """
    Decorator to register a guide for a model.

    A guide is a proposal distribution that suggests better samples
    for a given model. Used in importance sampling and variational
    inference patterns.

    Args:
        model_name: Name of the model this guide proposes for
        name: Optional name for the guide (defaults to function name)
        llm: Optional specialized LLM for the guide
        weight: Importance weight multiplier

    Example:
        >>> # Model: generic planner
        >>> @sample(llm, name="plan")
        ... def plan_generic(state, goal):
        ...     return f"Create a plan for: {goal}"
        ...
        >>> # Guide: specialized proposal
        >>> @register_guide("plan", llm=specialized_llm)
        ... def plan_guide(state, goal):
        ...     return efficient_planning_prompt(state, goal)
    """

    def decorator(fn: Callable[..., tuple[PLPState, Any]]):
        guide_name = name or fn.__name__

        spec = GuideSpec(
            name=guide_name, model_name=model_name, guide_fn=fn, llm=llm, weight=weight
        )
        _guides[model_name] = spec

        return fn

    return decorator


def get_guide(model_name: str) -> GuideSpec | None:
    """Get registered guide for a model, if any."""
    return _guides.get(model_name)


def has_guide(model_name: str) -> bool:
    """Check if a guide exists for this model."""
    return model_name in _guides


def clear_guides():
    """Clear all registered guides."""
    _guides.clear()


class GuidedSampler:
    """
    Importance sampler that uses guide proposals.

    Combines model and guide to improve sampling efficiency:
    - Uses guide to propose candidates
    - Weights samples by model/guide likelihood ratio
    - Returns weighted samples for inference
    """

    def __init__(self, model_llm: LLMLike, guide_llm: LLMLike | None = None):
        """
        Initialize guided sampler.

        Args:
            model_llm: LLM for the model (target distribution)
            guide_llm: LLM for the guide (proposal distribution)
        """
        self.model_llm = model_llm
        self.guide_llm = guide_llm or model_llm

    def sample_with_guide(
        self,
        model_fn: Callable[[PLPState], tuple[PLPState, str]],
        guide_fn: Callable[[PLPState], tuple[PLPState, str]],
        state: PLPState,
        n_samples: int = 1,
    ) -> list[tuple[PLPState, str, float]]:
        """
        Sample using guide as proposal distribution.

        Args:
            model_fn: The model's sample function (uses model_llm internally)
            guide_fn: The guide's proposal function
            state: Initial state
            n_samples: Number of samples to draw

        Returns:
            List of (state, output, importance_weight) tuples
        """
        samples = []

        for _ in range(n_samples):
            # Propose from guide
            guide_state, proposal = guide_fn(state.copy())

            # For now, use uniform weights (true importance weighting
            # would require likelihood estimation)
            importance_weight = 1.0

            samples.append((guide_state, proposal, importance_weight))

        return samples

    def importance_sample(
        self,
        model_name: str,
        model_fn: Callable[[PLPState], tuple[PLPState, str]],
        state: PLPState,
        n_samples: int = 5,
    ) -> tuple[PLPState, str]:
        """
        Importance sample using registered guide if available.

        Args:
            model_name: Name to look up guide
            model_fn: Model sampling function
            state: Initial state
            n_samples: Number of proposal samples

        Returns:
            Best (state, output) based on importance weights
        """
        guide_spec = get_guide(model_name)

        if guide_spec is None:
            # No guide, just sample from model once
            return model_fn(state)

        # Sample from guide
        samples = self.sample_with_guide(
            model_fn, guide_spec.guide_fn, state, n_samples
        )

        # Pick best by weight (for now, just first)
        if samples:
            best = max(samples, key=lambda x: x[2])
            return best[0], best[1]

        # Fallback to model
        return model_fn(state)


def guide(
    for_model: str, llm: LLMLike | None = None, temperature: float | None = None
):
    """
    Decorator to create a guide for a specific model sample site.

    The guide provides an efficient proposal distribution that
    can be used in importance sampling.

    Args:
        for_model: Name of the sample site this guides
        llm: Optional specialized LLM (e.g., faster/cheaper model)
        temperature: Optional temperature override

    Example:
        >>> @sample(llm, name="decompose")
        ... def decompose_task(state, task):
        ...     return f"Break down: {task}"
        ...
        >>> @guide(for_model="decompose", llm=fast_llm)
        ... def decompose_guide(state, task):
        ...     # More structured prompt for efficiency
        ...     return (state, structured_decomposition(task))
    """

    def decorator(fn: Callable[..., tuple[PLPState, Any]]):
        guide_name = f"guide_{fn.__name__}"

        @wraps(fn)
        def wrapped(*args, **kwargs) -> tuple[PLPState, Any]:
            return fn(*args, **kwargs)

        # Register the guide
        spec = GuideSpec(
            name=guide_name, model_name=for_model, guide_fn=wrapped, llm=llm, weight=1.0
        )
        _guides[for_model] = spec

        return wrapped

    return decorator


@dataclass
class VariationalState:
    """
    State for variational inference.

    Tracks both the model state and guide-related quantities
    for variational optimization (e.g., ELBO estimation).
    """

    model_state: PLPState
    guide_state: PLPState
    elbo_estimate: float = 0.0
    kl_divergence: float = 0.0
    n_samples: int = 0

    def update_elbo(self, log_p: float, log_q: float):
        """Update ELBO estimate with new sample."""
        self.n_samples += 1
        # Running average ELBO = E[log p - log q]
        new_term = log_p - log_q
        self.elbo_estimate += (new_term - self.elbo_estimate) / self.n_samples
        self.kl_divergence = -self.elbo_estimate  # KL = -ELBO when E[log p(x)] cancels


class MetaGuide:
    """
    Meta-agent that dynamically improves guides based on outcomes.

    This is the "guide that learns to guide" - it observes which
    proposals lead to successful outcomes and adapts.
    """

    def __init__(self, llm: LLMLike):
        self.llm = llm
        self.success_history: list[
            tuple[str, str, bool]
        ] = []  # (model, proposal, success)
        self.strategy_prompt: str = (
            "Propose effective solutions based on past successes."
        )

    def record_outcome(self, model_name: str, proposal: str, success: bool):
        """Record whether a proposal was successful."""
        self.success_history.append((model_name, proposal, success))

    def get_improved_prompt(self, model_name: str, base_prompt: str) -> str:
        """Get improved prompt based on success history."""
        successes = [h for h in self.success_history if h[0] == model_name and h[2]]

        if not successes:
            return base_prompt

        # Add successful examples to prompt
        examples = "\n".join([f"- {s[1][:100]}" for s in successes[-3:]])
        return f"{base_prompt}\n\nSuccessful approaches:\n{examples}"

    def propose(
        self, state: PLPState, model_name: str, task: str
    ) -> tuple[PLPState, str]:
        """Generate a proposal using learned strategies."""
        improved_prompt = self.get_improved_prompt(
            model_name, f"Propose a solution for: {task}"
        )

        response = invoke_llm(self.llm, improved_prompt)

        new_state = state.copy()
        new_state.trace.append(
            TraceEvent(
                primitive=f"meta_guide({model_name})",
                prompt=improved_prompt,
                output=response,
                meta={"meta_guide": True, "history_size": len(self.success_history)},
            )
        )

        return new_state, response
