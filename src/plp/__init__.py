"""
PLP - Probabilistic Language Programming.

This package exposes the small semantics-first core used throughout the paper:
trace/state objects, primitive operations, plate combinators, and minimal
LLM client utilities.
"""

from ._llm import AsyncLLMLike, LLMLike, ainvoke_llm, invoke_llm
from .client import (
    ChatOpenAI,
    HuggingFaceLocalBackend,
    LocalHuggingFaceClient,
    PLPChatClient,
    normalize_chat_messages,
)
from .guarantee import guarantee
from .handlers import block, condition, do, replay, seed
from .plate import (
    PlateContext,
    PlateRunner,
    plate,
    plate_amap,
    plate_areduce,
    plate_map,
    plate_reduce,
)

from .inference import ChainGraph, choose_engine
from .primitives import asample, factor, observe, sample
from .state import PLPState, TraceEvent

__version__ = "0.3.0"

__all__ = [
    "AsyncLLMLike",
    "ChainGraph",
    "ChatOpenAI",
    "HuggingFaceLocalBackend",
    "LLMLike",
    "LocalHuggingFaceClient",
    "PLPChatClient",
    "PLPState",
    "PlateContext",
    "PlateRunner",
    "TraceEvent",
    "ainvoke_llm",
    "asample",
    "block",
    "choose_engine",
    "condition",
    "do",
    "factor",
    "guarantee",
    "invoke_llm",
    "normalize_chat_messages",
    "observe",
    "plate",
    "plate_amap",
    "plate_areduce",
    "plate_map",
    "plate_reduce",
    "replay",
    "sample",
    "seed",
]
