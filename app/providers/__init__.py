"""Production Provider Architecture for HH Goa 2026 Voice-Enabled RAG."""

from app.providers.base import BaseProvider
from app.providers.exceptions import (
    InvalidAPIKeyError,
    InvalidRequestError,
    MalformedResponseError,
    MissingAPIKeyError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    UnsupportedLanguageError,
    mask_credential,
)
from app.providers.factory import (
    get_llm_provider,
    get_stt_provider,
    get_tts_provider,
    get_vector_provider,
)
from app.providers.llm import (
    GenerationProvider,
    LLMProvider,
    MockGenerationProvider,
    MockLLMProvider,
    OpenAILLMProvider,
    SarvamLLMProvider,
)
from app.providers.stt import (
    MockSTTProvider,
    SarvamSTTProvider,
    STTProvider,
    STTResult,
)
from app.providers.tts import (
    MockTTSProvider,
    SarvamTTSProvider,
    TTSProvider,
    TTSResult,
)
from app.providers.vector import (
    QdrantVectorProvider,
    VectorStoreProvider,
)

__all__ = [
    "BaseProvider",
    "STTProvider",
    "STTResult",
    "MockSTTProvider",
    "SarvamSTTProvider",
    "TTSProvider",
    "TTSResult",
    "MockTTSProvider",
    "SarvamTTSProvider",
    "GenerationProvider",
    "LLMProvider",
    "MockGenerationProvider",
    "MockLLMProvider",
    "SarvamLLMProvider",
    "OpenAILLMProvider",
    "VectorStoreProvider",
    "QdrantVectorProvider",
    "get_stt_provider",
    "get_tts_provider",
    "get_llm_provider",
    "get_vector_provider",
    "ProviderError",
    "MissingAPIKeyError",
    "InvalidAPIKeyError",
    "ProviderTimeoutError",
    "ProviderConnectionError",
    "ProviderRateLimitError",
    "MalformedResponseError",
    "UnsupportedLanguageError",
    "InvalidRequestError",
    "mask_credential",
]
