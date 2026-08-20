"""Speech-to-Text provider implementations and abstractions."""

from app.providers.stt.base import STTProvider, STTResult
from app.providers.stt.mock import MockSTTProvider
from app.providers.stt.sarvam import SarvamSTTProvider

__all__ = [
    "STTProvider",
    "STTResult",
    "MockSTTProvider",
    "SarvamSTTProvider",
]
