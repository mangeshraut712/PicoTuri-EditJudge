"""
Runtime Module
Real-time batch processing and session management for PicoTuri-EditJudge
"""

from .batcher import AdaptiveMicroBatcher, BatchStrategy, BatchRequest, BatchMetrics
from .engine import RuntimeEngine, SessionPool, SessionInfo, InferenceRequest, InferenceResult

__all__ = [
    # Batcher components
    "AdaptiveMicroBatcher",
    "BatchStrategy",
    "BatchRequest",
    "BatchMetrics",

    # Engine components
    "RuntimeEngine",
    "SessionPool",
    "SessionInfo",
    "InferenceRequest",
    "InferenceResult",
]

# Version info
__version__ = "0.2.0"
__author__ = "PicoTuri Team"
__description__ = "Real-time batch processing and session management for EditJudge"
