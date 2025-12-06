"""
PLP - Probabilistic Language Programming

A Python framework for probabilistic programming with LLMs as stochastic kernels.

Core primitives:
    - sample: LLM-based stochastic generation
    - observe: Hard 0/1 constraints
    - factor: Soft log-weight scoring (with scale/mask)
    - guarantee: Local robustness wrappers

Inference engines:
    - InferEngine.run_map: Naive MAP by multiple runs
    - InferEngine.sprt_discrete: First-to-ahead-by-k voting

Advanced operators (Pyro-inspired):
    - plate, plate_map, plate_reduce: Independent parallel repetition
    - replay, block, do, condition: Program handlers
    - param, text_param: Learnable parameters
    - deterministic, store: Named derived quantities
    - guide, GuidedSampler: Variational proposal programs
    - agent, BaseAgent, DecomposerAgent, etc.: Reusable agent modules

State management:
    - PLPState: Holds NL variables, log_weight, and execution trace
    - TraceEvent: Logs individual primitive invocations
"""

from .state import PLPState, TraceEvent
from .client import LLMClient
from .primitives import sample, observe, factor
from .condition import sample_with_condition, asample_with_condition
from .infer import InferEngine, compute_k_for_eps
from .guarantee import guarantee, async_guarantee

# High-level API
from ._config import configure, model, get_config
from .api import (
    sample as sample_fn,
    sampler,
    vote,
    map as map_fn,
    guarantee as guarantee_fn,
    infer,
    InferBuilder,
)

# Phase 2: Advanced operators
from .plate import (
    plate,
    plate_map,
    aplate_map,
    plate_reduce,
    plate_stack,
    PlateContext,
    get_active_plate,
    get_plate_stack,
    vectorized_sample,
)
from .handlers import replay, block, do, condition, check_handlers
from .param import param, text_param, get_registry, Param, TextParam, ParamRegistry
from .deterministic import deterministic, store, summarize, canonicalize

# Tier 3: Guide and Agent
from .guide import (
    guide,
    register_guide,
    get_guide,
    has_guide,
    clear_guides,
    GuidedSampler,
    MetaGuide,
    GuideSpec,
    VariationalState,
)
from .agent import (
    agent,
    register_agent,
    get_agent,
    create_agent,
    list_agents,
    clear_agents,
    BaseAgent,
    AgentSpec,
    DecomposerAgent,
    AggregatorAgent,
    CriticAgent,
    JudgeAgent,
)

__all__ = [
    # State
    "PLPState",
    "TraceEvent",
    # Client
    "LLMClient",
    # Primitives
    "sample",
    "observe",
    "factor",
    # Conditioning
    "sample_with_condition",
    "asample_with_condition",
    # Inference
    "InferEngine",
    "compute_k_for_eps",
    # Guarantee
    "guarantee",
    "async_guarantee",
    # Plate
    "plate",
    "plate_map",
    "aplate_map",
    "plate_reduce",
    "plate_stack",
    "PlateContext",
    "get_active_plate",
    "get_plate_stack",
    "vectorized_sample",
    # Handlers
    "replay",
    "block",
    "do",
    "condition",
    "check_handlers",
    # Params
    "param",
    "text_param",
    "get_registry",
    "Param",
    "TextParam",
    "ParamRegistry",
    # Deterministic
    "deterministic",
    "store",
    "summarize",
    "canonicalize",
    # Guide
    "guide",
    "register_guide",
    "get_guide",
    "has_guide",
    "clear_guides",
    "GuidedSampler",
    "MetaGuide",
    "GuideSpec",
    "VariationalState",
    # Agent
    "agent",
    "register_agent",
    "get_agent",
    "create_agent",
    "list_agents",
    "clear_agents",
    "BaseAgent",
    "AgentSpec",
    "DecomposerAgent",
    "AggregatorAgent",
    "CriticAgent",
    "JudgeAgent",
    # High-level API
    "configure",
    "model",
    "get_config",
    "sample_fn",
    "sampler",
    "vote",
    "map_fn",
    "guarantee_fn",
    "infer",
    "InferBuilder",
]

__version__ = "0.3.0"
