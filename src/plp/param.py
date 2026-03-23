"""
PLP Param Operator.

Learnable parameters for PLP - both numeric hyperparameters
and text-based prompt snippets that can be tuned.
"""

from dataclasses import dataclass, field
from typing import Callable
import random


@dataclass
class Param:
    """
    A learnable numeric parameter.

    Attributes:
        name: Unique identifier
        value: Current value
        default: Default value
        bounds: Optional (min, max) bounds
        description: Human-readable description
    """

    name: str
    value: float
    default: float
    bounds: tuple[float, float] | None = None
    description: str = ""

    def get(self) -> float:
        """Get current value."""
        return self.value

    def set(self, new_value: float):
        """Set value, respecting bounds if any."""
        if self.bounds:
            new_value = max(self.bounds[0], min(self.bounds[1], new_value))
        self.value = new_value

    def reset(self):
        """Reset to default value."""
        self.value = self.default


@dataclass
class TextParam:
    """
    A learnable text parameter (prompt snippet).

    Attributes:
        name: Unique identifier
        value: Current text value
        default: Default text value
        candidates: Optional list of alternative values to explore
        description: Human-readable description
    """

    name: str
    value: str
    default: str
    candidates: list[str] = field(default_factory=list)
    description: str = ""

    def get(self) -> str:
        """Get current value."""
        return self.value

    def set(self, new_value: str):
        """Set value."""
        self.value = new_value

    def reset(self):
        """Reset to default value."""
        self.value = self.default

    def add_candidate(self, text: str):
        """Add a candidate alternative."""
        if text not in self.candidates:
            self.candidates.append(text)


class ParamRegistry:
    """
    Global registry for PLP parameters.

    Tracks all numeric and text parameters, enabling:
    - Hyperparameter optimization
    - Prompt tuning
    - Configuration management
    """

    def __init__(self):
        self._params: dict[str, Param] = {}
        self._text_params: dict[str, TextParam] = {}

    def register_param(
        self,
        name: str,
        default: float,
        bounds: tuple[float, float] | None = None,
        description: str = "",
    ) -> Param:
        """Register a numeric parameter."""
        param = Param(
            name=name,
            value=default,
            default=default,
            bounds=bounds,
            description=description,
        )
        self._params[name] = param
        return param

    def register_text_param(
        self,
        name: str,
        default: str,
        candidates: list[str] | None = None,
        description: str = "",
    ) -> TextParam:
        """Register a text parameter."""
        param = TextParam(
            name=name,
            value=default,
            default=default,
            candidates=candidates or [],
            description=description,
        )
        self._text_params[name] = param
        return param

    def get_param(self, name: str) -> float:
        """Get numeric parameter value."""
        return self._params[name].get()

    def get_text_param(self, name: str) -> str:
        """Get text parameter value."""
        return self._text_params[name].get()

    def set_param(self, name: str, value: float):
        """Set numeric parameter value."""
        self._params[name].set(value)

    def set_text_param(self, name: str, value: str):
        """Set text parameter value."""
        self._text_params[name].set(value)

    def reset_all(self):
        """Reset all parameters to defaults."""
        for p in self._params.values():
            p.reset()
        for tp in self._text_params.values():
            tp.reset()

    def get_all_params(self) -> dict[str, float]:
        """Get all numeric parameter values."""
        return {name: p.get() for name, p in self._params.items()}

    def get_all_text_params(self) -> dict[str, str]:
        """Get all text parameter values."""
        return {name: tp.get() for name, tp in self._text_params.items()}

    def random_sample(self, param_name: str) -> float:
        """Sample a random value within bounds for a numeric param."""
        param = self._params[param_name]
        if param.bounds:
            return random.uniform(param.bounds[0], param.bounds[1])
        else:
            # Default: ±50% of current value
            return param.value * random.uniform(0.5, 1.5)

    def random_text_sample(self, param_name: str) -> str:
        """Sample a random candidate for a text param."""
        param = self._text_params[param_name]
        if param.candidates:
            return random.choice(param.candidates)
        return param.value


# Global registry instance
_registry = ParamRegistry()


def param(
    name: str,
    default: float,
    bounds: tuple[float, float] | None = None,
    description: str = "",
) -> Param:
    """
    Register and return a learnable numeric parameter.

    Args:
        name: Unique identifier
        default: Default value
        bounds: Optional (min, max) bounds
        description: Human-readable description

    Returns:
        Param object

    Example:
        >>> temperature = param("temperature", 0.7, bounds=(0.1, 1.5))
        >>> llm = ChatOpenAI(temperature=temperature.get())
    """
    return _registry.register_param(name, default, bounds, description)


def text_param(
    name: str, default: str, candidates: list[str] | None = None, description: str = ""
) -> TextParam:
    """
    Register and return a learnable text parameter.

    Args:
        name: Unique identifier
        default: Default text value
        candidates: Alternative values to explore
        description: Human-readable description

    Returns:
        TextParam object

    Example:
        >>> reasoning_style = text_param(
        ...     "reasoning_style",
        ...     default="Think step by step",
        ...     candidates=[
        ...         "Think step by step",
        ...         "Let's work through this carefully",
        ...         "Break this down into parts"
        ...     ]
        ... )
        >>> prompt = f"{reasoning_style.get()}. Solve: {problem}"
    """
    return _registry.register_text_param(name, default, candidates, description)


def get_registry() -> ParamRegistry:
    """Get the global parameter registry."""
    return _registry


def optimize_params(
    objective_fn: Callable[[], float], param_names: list[str], n_trials: int = 20
) -> dict[str, float]:
    """
    Simple random search optimization for numeric parameters.

    Args:
        objective_fn: Function returning a score to maximize
        param_names: List of parameter names to optimize
        n_trials: Number of random trials

    Returns:
        Best parameter values found
    """
    best_score = float("-inf")
    best_params = {name: _registry.get_param(name) for name in param_names}

    for _ in range(n_trials):
        # Random sample for each param
        trial_params = {}
        for name in param_names:
            new_value = _registry.random_sample(name)
            _registry.set_param(name, new_value)
            trial_params[name] = new_value

        # Evaluate
        score = objective_fn()

        if score > best_score:
            best_score = score
            best_params = trial_params.copy()

    # Restore best params
    for name, value in best_params.items():
        _registry.set_param(name, value)

    return best_params
