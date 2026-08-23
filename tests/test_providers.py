"""Comprehensive tests for Production Provider Architecture (Phase 6.1)."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings, get_settings
from app.main import app
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
    MockGenerationProvider,
    MockLLMProvider,
    OpenAILLMProvider,
    SarvamLLMProvider,
)
from app.providers.stt import (
    MockSTTProvider,
    SarvamSTTProvider,
    STTResult,
)
from app.providers.tts import (
    MockTTSProvider,
    SarvamTTSProvider,
    TTSResult,
)
from app.providers.vector import (
    QdrantVectorProvider,
)


# --- 1. Credential Masking & Exception Tests ---

def test_mask_credential_utility():
    """Verify API keys are masked safely to prevent credential exposure in logs."""
    assert mask_credential(None) == "[NOT_SET]"
    assert mask_credential("") == "[NOT_SET]"
    assert mask_credential("123") == "[REDACTED]"
    assert mask_credential("sk-abcdef123456") == "***3456"
    assert mask_credential("sarvam_live_key_xyz9876", visible_chars=4) == "***9876"


def test_provider_exceptions_no_leakage():
    """Verify custom provider exceptions format clean messages without raw credentials."""
    err_missing = MissingAPIKeyError(provider_name="sarvam", key_name="SARVAM_API_KEY")
    assert err_missing.status_code == 401
    assert "SARVAM_API_KEY" in str(err_missing)
    assert "[SARVAM]" in str(err_missing)

    err_invalid = InvalidAPIKeyError(provider_name="openai", key_name="OPENAI_API_KEY")
    assert err_invalid.status_code == 403

    err_timeout = ProviderTimeoutError(provider_name="sarvam", timeout_seconds=15.0)
    assert err_timeout.status_code == 504
    assert "15.0s" in str(err_timeout)

    err_conn = ProviderConnectionError(provider_name="qdrant_cloud", endpoint="https://my-cluster.qdrant.io")
    assert err_conn.status_code == 502
    assert "https://my-cluster.qdrant.io" in str(err_conn)

    err_rate = ProviderRateLimitError(provider_name="sarvam", retry_after=30)
    assert err_rate.status_code == 429
    assert "30 seconds" in str(err_rate)

    err_lang = UnsupportedLanguageError(provider_name="sarvam", language="xx-YY")
    assert err_lang.status_code == 400
    assert "xx-YY" in str(err_lang)

    err_malformed = MalformedResponseError(provider_name="sarvam", message="JSON decode failed")
    assert err_malformed.status_code == 502


# --- 2. Production Provider Initialization & Validation Tests ---

def test_sarvam_stt_initialization_validation(monkeypatch):
    """Verify SarvamSTTProvider validates SARVAM_API_KEY upon instantiation."""
    monkeypatch.setenv("SARVAM_API_KEY", "")
    get_settings.cache_clear()

    with pytest.raises(MissingAPIKeyError, match="SARVAM_API_KEY"):
        SarvamSTTProvider(api_key=None)

    # When key is provided, instantiation succeeds
    provider = SarvamSTTProvider(api_key="mock_valid_sarvam_key")
    assert provider.provider_name == "sarvam"
    assert provider.api_key == "mock_valid_sarvam_key"


def test_sarvam_tts_initialization_validation(monkeypatch):
    """Verify SarvamTTSProvider validates SARVAM_API_KEY upon instantiation."""
    monkeypatch.setenv("SARVAM_API_KEY", "")
    get_settings.cache_clear()

    with pytest.raises(MissingAPIKeyError, match="SARVAM_API_KEY"):
        SarvamTTSProvider(api_key=None)

    provider = SarvamTTSProvider(api_key="mock_valid_sarvam_key")
    assert provider.provider_name == "sarvam"
    assert provider.api_key == "mock_valid_sarvam_key"


def test_sarvam_llm_initialization_validation(monkeypatch):
    """Verify SarvamLLMProvider validates SARVAM_API_KEY upon instantiation."""
    monkeypatch.setenv("SARVAM_API_KEY", "")
    get_settings.cache_clear()

    with pytest.raises(MissingAPIKeyError, match="SARVAM_API_KEY"):
        SarvamLLMProvider(api_key=None)

    provider = SarvamLLMProvider(api_key="mock_valid_sarvam_key")
    assert provider.provider_name == "sarvam"
    assert provider.api_key == "mock_valid_sarvam_key"


def test_gemini_llm_initialization_validation(monkeypatch):
    """Verify GeminiLLMProvider validates GEMINI_API_KEY upon instantiation."""
    from app.providers.llm import GeminiLLMProvider
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("LLM_API_KEY", "")
    get_settings.cache_clear()

    with pytest.raises(MissingAPIKeyError, match="GEMINI_API_KEY"):
        GeminiLLMProvider(api_key=None)

    provider = GeminiLLMProvider(api_key="mock_valid_gemini_key")
    assert provider.provider_name == "gemini"
    assert provider.api_key == "mock_valid_gemini_key"


def test_openai_llm_initialization_validation(monkeypatch):
    """Verify OpenAILLMProvider validates OPENAI_API_KEY / LLM_API_KEY upon instantiation."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("LLM_API_KEY", "")
    get_settings.cache_clear()

    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        OpenAILLMProvider(api_key=None)

    provider = OpenAILLMProvider(api_key="mock_valid_openai_key")
    assert provider.provider_name == "openai"
    assert provider.api_key == "mock_valid_openai_key"


def test_qdrant_cloud_initialization_validation(monkeypatch):
    """Verify QdrantVectorProvider validates VECTOR_DB_URL when qdrant_cloud is requested."""
    monkeypatch.setenv("VECTOR_DB_URL", "")
    get_settings.cache_clear()

    with pytest.raises(MissingAPIKeyError, match="VECTOR_DB_URL"):
        QdrantVectorProvider(provider_type="qdrant_cloud", url=None)


# --- 3. Mock Provider Functionality Tests ---

@pytest.mark.asyncio
async def test_mock_stt_transcription():
    """Verify MockSTTProvider transcribe returns structured STTResult."""
    provider = MockSTTProvider(default_text="गोवा की राजधानी पणजी है।", default_language="hi")
    dummy_audio = b"\x00\x00" * 16000  # 1 second of 16kHz 16-bit audio
    result = await provider.transcribe(audio_bytes=dummy_audio, language="hi")

    assert isinstance(result, STTResult)
    assert result.text == "गोवा की राजधानी पणजी है।"
    assert result.language == "hi"
    assert result.confidence == 0.98
    assert result.duration_s > 0
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_mock_tts_synthesis():
    """Verify MockTTSProvider synthesize returns structured TTSResult."""
    provider = MockTTSProvider(default_sample_rate=24000)
    result = await provider.synthesize(text="नमस्ते, गोवा में आपका स्वागत है।", language="hi")

    assert isinstance(result, TTSResult)
    assert len(result.audio_bytes) > 0
    assert result.audio_format == "wav"
    assert result.sample_rate == 24000
    assert result.duration_s > 0
    assert result.latency_ms >= 0


def test_in_memory_qdrant_vector_provider():
    """Verify QdrantVectorProvider in-memory mode operates cleanly."""
    provider = QdrantVectorProvider(provider_type="memory")
    assert provider.provider_name == "memory"
    count = provider.count("test_collection")
    assert count == 0


# --- 4. Central Factory Tests ---

def test_provider_factory_resolution():
    """Verify get_*_provider factory resolves configured and named providers."""
    stt_p = get_stt_provider("mock")
    assert isinstance(stt_p, MockSTTProvider)

    tts_p = get_tts_provider("mock")
    assert isinstance(tts_p, MockTTSProvider)

    llm_p = get_llm_provider("mock")
    assert isinstance(llm_p, MockLLMProvider)

    vec_p = get_vector_provider("memory")
    assert isinstance(vec_p, QdrantVectorProvider)


def test_provider_factory_unsupported_rejection():
    """Verify provider factories raise InvalidRequestError on unsupported provider names."""
    with pytest.raises(InvalidRequestError, match="Unsupported STT provider"):
        get_stt_provider("invalid_stt_xyz")

    with pytest.raises(InvalidRequestError, match="Unsupported TTS provider"):
        get_tts_provider("invalid_tts_xyz")

    with pytest.raises(InvalidRequestError, match="Unsupported LLM provider"):
        get_llm_provider("invalid_llm_xyz")

    with pytest.raises(InvalidRequestError, match="Unsupported Vector provider"):
        get_vector_provider("invalid_vector_xyz")


# --- 5. Lazy Startup & Health Check Independence ---

@pytest.mark.asyncio
async def test_health_check_fast_and_independent_of_credentials(monkeypatch):
    """Verify GET /health executes instantly without requiring any external provider keys."""
    # Wipe external credentials in test environment
    monkeypatch.setenv("SARVAM_API_KEY", "")
    monkeypatch.setenv("VECTOR_DB_URL", "")
    monkeypatch.setenv("LLM_API_KEY", "")
    get_settings.cache_clear()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
