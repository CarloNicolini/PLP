"""
PLP - Probabilistic Language Programming

A Python framework for probabilistic programming with LLMs as stochastic kernels.

TextGrad framing:
    - TextGrad (Yuksekgonul et al., 2024) treats compound AI systems as
      computation graphs whose nodes receive textual gradients.
    - PLP wraps those nodes in probabilistic-programming structure so every
      textual gradient, loss, or refinement step is logged as a TraceEvent with
      a log-weight. That makes TextGrad-style steps compose with observes,
      factors, plates, and inference handlers.
    - Textual gradients become first-class primitives via `textgrad_step`,
      `text_transition`, `text_kernel_constraint`, and the SVGD drivers.

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
from langchain_openai import ChatOpenAI
from .primitives import sample, observe, factor
from .condition import sample_with_condition
from .infer import InferEngine, SVGDEngine, SVGDParticle, compute_k_for_eps
from .guarantee import guarantee
from .textppl import (
    text_sample,
    text_score,
    text_observe,
    length_penalty,
    text_transition,
    textgrad_step,
    text_factor,
    text_overlap,
    apply_overlap_penalty,
    text_kernel_constraint,
    LLMJudgeResult,
    OverlapResult,
    KernelAdvice,
    KernelConstraint,
    text_sampler,
    tool,
    get_tool,
    list_tools,
    agent_sampler,
    cot_sampler,
    reflection_sampler,
    self_consistency_sampler,
    TextInferenceResult,
    TextSite,
    make_text_site,
    infer_rejection,
    infer_ensemble_map,
)

# High-level API
from ._config import configure, model, get_config
from .session import session, Session
from .api import (
    prompt,
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
    plate_amap,
    plate_reduce,
    plate_areduce,
    plate_stack,
    PlateContext,
    get_active_plate,
    get_plate_stack,
    vectorized_sample,
    avectorized_sample,
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
    "ChatOpenAI",
    # Sessions
    "session",
    "Session",
    # Primitives
    "sample",
    "observe",
    "factor",
    # Conditioning
    "sample_with_condition",
    # Inference
    "InferEngine",
    "SVGDEngine",
    "SVGDParticle",
    "compute_k_for_eps",
    # Guarantee
    "guarantee",
    # Text-first primitives
    "text_sample",
    "text_score",
    "text_observe",
    "length_penalty",
    "text_transition",
    "textgrad_step",
    "text_factor",
    "text_overlap",
    "apply_overlap_penalty",
    "text_kernel_constraint",
    "LLMJudgeResult",
    "OverlapResult",
    "KernelAdvice",
    "KernelConstraint",
    "text_sampler",
    "tool",
    "get_tool",
    "list_tools",
    "agent_sampler",
    "cot_sampler",
    "reflection_sampler",
    "self_consistency_sampler",
    "TextInferenceResult",
    "TextSite",
    "make_text_site",
    "infer_rejection",
    "infer_ensemble_map",
    # Plate
    "plate",
    "plate_map",
    "plate_amap",
    "plate_reduce",
    "plate_areduce",
    "plate_stack",
    "PlateContext",
    "get_active_plate",
    "get_plate_stack",
    "vectorized_sample",
    "avectorized_sample",
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
    "prompt",
    "sampler",
    "vote",
    "map_fn",
    "guarantee_fn",
    "infer",
    "InferBuilder",
]

__version__ = "0.3.0"
