"""
PLP - Probabilistic Language Programming.

This package exposes the small semantics-first core used throughout the paper:
trace/state objects, primitive operations, plate combinators, and minimal
LLM client utilities.
"""

from ._llm import AsyncLLMLike, LLMLike, ainvoke_llm, invoke_llm
from .client import ChatOpenAI, PLPChatClient, normalize_chat_messages
from .guarantee import guarantee
from .plate import (
    PlateContext,
    PlateRunner,
    plate,
    plate_amap,
    plate_areduce,
    plate_map,
    plate_reduce,
)
from .primitives import asample, factor, observe, sample
from .state import PLPState, TraceEvent

__version__ = "0.3.0"

__all__ = [
    "AsyncLLMLike",
    "ChatOpenAI",
    "LLMLike",
    "PLPChatClient",
    "PLPState",
    "PlateContext",
    "PlateRunner",
    "TraceEvent",
    "ainvoke_llm",
    "asample",
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
    "sample",
]
