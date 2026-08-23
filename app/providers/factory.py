"""Centralized Provider Factory & Registry with Lazy Initialization."""

import logging
import threading
from typing import Any, Dict, Optional

from app.config import get_settings
from app.generation.base import GenerationProvider
from app.providers.exceptions import InvalidRequestError
from app.providers.llm.gemini import GeminiLLMProvider
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.openai import OpenAILLMProvider
from app.providers.llm.sarvam import SarvamLLMProvider
from app.providers.stt.base import STTProvider
from app.providers.stt.mock import MockSTTProvider
from app.providers.stt.sarvam import SarvamSTTProvider
from app.providers.tts.base import TTSProvider
from app.providers.tts.mock import MockTTSProvider
from app.providers.tts.sarvam import SarvamTTSProvider
from app.providers.vector.base import VectorStoreProvider
from app.providers.vector.qdrant import QdrantVectorProvider

logger = logging.getLogger(__name__)

# Singletons and thread lock for lazy initialization
_LOCK = threading.Lock()
_STT_INSTANCES: Dict[str, STTProvider] = {}
_TTS_INSTANCES: Dict[str, TTSProvider] = {}
_LLM_INSTANCES: Dict[str, GenerationProvider] = {}
_VECTOR_INSTANCES: Dict[str, VectorStoreProvider] = {}


def get_stt_provider(provider_name: Optional[str] = None, **kwargs: Any) -> STTProvider:
    """Resolve and lazily instantiate a Speech-to-Text provider.

    Args:
        provider_name: 'mock', 'sarvam', or None (uses Settings.STT_PROVIDER).
        **kwargs: Additional parameters passed to provider constructor.

    Returns:
        STTProvider: Instantiated STT provider.
    """
    settings = get_settings()
    target_name = (provider_name or settings.STT_PROVIDER).lower().strip()

    cache_key = f"{target_name}_{sorted(kwargs.items())}"
    with _LOCK:
        if cache_key in _STT_INSTANCES:
            return _STT_INSTANCES[cache_key]

        if target_name == "mock":
            inst = MockSTTProvider(**kwargs)
        elif target_name == "sarvam":
            inst = SarvamSTTProvider(**kwargs)
        else:
            raise InvalidRequestError(
                provider_name=target_name,
                message=f"Unsupported STT provider '{target_name}'. Supported: 'mock', 'sarvam'.",
            )

        _STT_INSTANCES[cache_key] = inst
        return inst


def get_tts_provider(provider_name: Optional[str] = None, **kwargs: Any) -> TTSProvider:
    """Resolve and lazily instantiate a Text-to-Speech provider.

    Args:
        provider_name: 'mock', 'sarvam', or None (uses Settings.TTS_PROVIDER).
        **kwargs: Additional parameters passed to provider constructor.

    Returns:
        TTSProvider: Instantiated TTS provider.
    """
    settings = get_settings()
    target_name = (provider_name or settings.TTS_PROVIDER).lower().strip()

    cache_key = f"{target_name}_{sorted(kwargs.items())}"
    with _LOCK:
        if cache_key in _TTS_INSTANCES:
            return _TTS_INSTANCES[cache_key]

        if target_name == "mock":
            inst = MockTTSProvider(**kwargs)
        elif target_name == "sarvam":
            inst = SarvamTTSProvider(**kwargs)
        else:
            raise InvalidRequestError(
                provider_name=target_name,
                message=f"Unsupported TTS provider '{target_name}'. Supported: 'mock', 'sarvam'.",
            )

        _TTS_INSTANCES[cache_key] = inst
        return inst


def get_llm_provider(provider_name: Optional[str] = None, **kwargs: Any) -> GenerationProvider:
    """Resolve and lazily instantiate an LLM generation provider.

    Args:
        provider_name: 'gemini', 'mock', 'sarvam', 'openai', or None (uses Settings.LLM_PROVIDER).
        **kwargs: Additional parameters passed to provider constructor.

    Returns:
        GenerationProvider: Instantiated LLM provider.
    """
    settings = get_settings()
    target_name = (provider_name or settings.LLM_PROVIDER).lower().strip()

    cache_key = f"{target_name}_{sorted(kwargs.items())}"
    with _LOCK:
        if cache_key in _LLM_INSTANCES:
            return _LLM_INSTANCES[cache_key]

        if target_name == "mock":
            inst = MockLLMProvider(**kwargs)
        elif target_name in ("gemini", "google"):
            inst = GeminiLLMProvider(**kwargs)
        elif target_name == "sarvam":
            inst = SarvamLLMProvider(**kwargs)
        elif target_name in ("openai", "gpt"):
            inst = OpenAILLMProvider(**kwargs)
        else:
            raise InvalidRequestError(
                provider_name=target_name,
                message=f"Unsupported LLM provider '{target_name}'. Supported: 'gemini', 'mock', 'sarvam', 'openai'.",
            )

        _LLM_INSTANCES[cache_key] = inst
        return inst


def get_vector_provider(provider_name: Optional[str] = None, **kwargs: Any) -> VectorStoreProvider:
    """Resolve and lazily instantiate a Vector Store provider.

    Args:
        provider_name: 'qdrant_local', 'qdrant_cloud', 'memory', or None (uses Settings.VECTOR_PROVIDER).
        **kwargs: Additional parameters passed to provider constructor.

    Returns:
        VectorStoreProvider: Instantiated Vector provider.
    """
    settings = get_settings()
    target_name = (provider_name or settings.VECTOR_PROVIDER).lower().strip()

    cache_key = f"{target_name}_{sorted(kwargs.items())}"
    with _LOCK:
        if cache_key in _VECTOR_INSTANCES:
            return _VECTOR_INSTANCES[cache_key]

        if target_name in ("qdrant_local", "qdrant_cloud", "memory"):
            inst = QdrantVectorProvider(provider_type=target_name, **kwargs)
        else:
            raise InvalidRequestError(
                provider_name=target_name,
                message=f"Unsupported Vector provider '{target_name}'. Supported: 'qdrant_local', 'qdrant_cloud', 'memory'.",
            )

        _VECTOR_INSTANCES[cache_key] = inst
        return inst


def get_active_providers_info() -> Dict[str, str]:
    """Return active provider names for health telemetry and UI status display."""
    settings = get_settings()
    return {
        "stt": f"{settings.STT_PROVIDER} ({settings.SARVAM_STT_MODEL})",
        "llm": f"{settings.LLM_PROVIDER} ({settings.GEMINI_MODEL if settings.LLM_PROVIDER == 'gemini' else settings.LLM_MODEL_NAME})",
        "tts": f"{settings.TTS_PROVIDER} ({settings.SARVAM_TTS_MODEL})",
        "vector_db": "qdrant" if "qdrant" in settings.VECTOR_PROVIDER else settings.VECTOR_PROVIDER,
        "embedding": settings.EMBEDDING_MODEL,
        "reranker": settings.RERANKER_MODEL,
    }
