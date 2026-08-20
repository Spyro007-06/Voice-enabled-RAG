"""Phase 6.12 — Production Provider Integration, Secret Isolation & Failure Resilience Tests."""

import io
import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient
import httpx

from app.config import Settings, get_settings, validate_production_config
from app.main import app
from app.observability.security import sanitize_output_text
from app.observability.security_middleware import get_rate_limiter
from app.providers.exceptions import (
    InvalidAPIKeyError,
    MissingAPIKeyError,
    ProviderConnectionError,
    ProviderTimeoutError,
    mask_credential,
)
from app.providers.factory import (
    get_llm_provider,
    get_stt_provider,
    get_tts_provider,
    get_vector_provider,
)
from app.providers.llm.sarvam import SarvamLLMProvider
from app.providers.stt.sarvam import SarvamSTTProvider
from app.providers.tts.sarvam import SarvamTTSProvider
from app.retrieval.models import RetrievalResult
from app.retrieval.service import get_retrieval_service

# Valid audio headers for multi-format tests
VALID_WAV_HEADER = (
    b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00"
    b"\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
)
VALID_MP3_HEADER = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 32
VALID_OGG_HEADER = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00" + b"\x00" * 32
VALID_WEBM_HEADER = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01" + b"\x00" * 32
VALID_FLAC_HEADER = b"fLaC\x00\x00\x00\x22" + b"\x00" * 32
VALID_M4A_HEADER = b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00" + b"\x00" * 32


@pytest.fixture
def client():
    get_rate_limiter().reset()
    with TestClient(app) as test_client:
        yield test_client
    get_rate_limiter().reset()


# =============================================================================
# 1. PROVIDER FACTORY & ENVIRONMENT SEPARATION TESTS
# =============================================================================

class TestProviderFactoryAndConfiguration:
    """Test suite verifying dynamic provider resolution and environment-driven configuration."""

    def test_mock_provider_resolution(self):
        """Verify default mock provider instantiation in development mode."""
        stt = get_stt_provider("mock")
        tts = get_tts_provider("mock")
        llm = get_llm_provider("mock")
        vec = get_vector_provider("memory")

        assert stt.provider_name == "mock"
        assert tts.provider_name == "mock"
        assert llm.provider_name == "mock"
        assert vec.provider_name == "memory"

    def test_sarvam_provider_requires_api_key(self):
        """Verify Sarvam providers raise MissingAPIKeyError when API key is missing."""
        mock_empty_settings = MagicMock()
        mock_empty_settings.SARVAM_API_KEY = None
        mock_empty_settings.SARVAM_BASE_URL = "https://api.sarvam.ai"
        mock_empty_settings.SARVAM_STT_MODEL = "saarika:v2"
        mock_empty_settings.SARVAM_TTS_MODEL = "bulbul:v1"
        mock_empty_settings.SARVAM_LLM_MODEL = "sarvam-2b"

        with patch("app.providers.stt.sarvam.get_settings", return_value=mock_empty_settings):
            with pytest.raises(MissingAPIKeyError) as exc_stt:
                SarvamSTTProvider(api_key=None)
            assert "SARVAM_API_KEY" in str(exc_stt.value)

        with patch("app.providers.tts.sarvam.get_settings", return_value=mock_empty_settings):
            with pytest.raises(MissingAPIKeyError) as exc_tts:
                SarvamTTSProvider(api_key=None)
            assert "SARVAM_API_KEY" in str(exc_tts.value)

        with patch("app.providers.llm.sarvam.get_settings", return_value=mock_empty_settings):
            with pytest.raises(MissingAPIKeyError) as exc_llm:
                SarvamLLMProvider(api_key=None)
            assert "SARVAM_API_KEY" in str(exc_llm.value)

    def test_credential_masking_integrity(self):
        """Verify API keys are safely masked in logs and exceptions."""
        test_key = "sk_live_secret_sarvam_api_key_987654321"
        masked = mask_credential(test_key)
        assert test_key not in masked
        assert masked.endswith("4321")
        assert "***" in masked


# =============================================================================
# 2. PROVIDER FAILURE & GRACEFUL DEGRADATION TESTS
# =============================================================================

class TestProviderFailureResilience:
    """Test suite verifying system resilience against external provider errors and timeouts."""

    @pytest.mark.asyncio
    async def test_stt_timeout_handling(self):
        """Verify STT provider timeout raises ProviderTimeoutError without leaking secrets."""
        stt = SarvamSTTProvider(api_key="sk_live_dummy_test_key_12345")
        with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("STT Connection timed out")):
            with pytest.raises(ProviderTimeoutError) as exc_info:
                await stt.transcribe(audio_bytes=VALID_WAV_HEADER, language="en")
            assert exc_info.value.provider_name == "sarvam"
            assert "sk_live" not in str(exc_info.value)

    def test_tts_failure_returns_partial_success(self, client):
        """Verify TTS failure produces a partial_success response containing the text answer."""
        files = {"audio": ("audio.wav", io.BytesIO(VALID_WAV_HEADER), "audio/wav")}
        data = {"language": "en", "top_k": 3, "synthesize_speech": "true"}

        with patch("app.providers.tts.mock.MockTTSProvider.synthesize", side_effect=RuntimeError("TTS upstream failure")):
            res = client.post("/api/voice-ask", files=files, data=data)
            assert res.status_code == 200
            payload = res.json()
            assert payload["status"] in ("partial_success", "success")
            assert payload["answer"] != ""
            assert payload["fallback_status"] is True or payload.get("audio_output", {}).get("available") is False

    def test_qdrant_failure_triggers_bm25_fallback(self):
        """Verify Qdrant failure triggers fallback to BM25 index."""
        service = get_retrieval_service()
        mock_bm25_res = [
            RetrievalResult(
                chunk_id="c_bm25",
                document_id="doc_1",
                text="BM25 fallback content for Goa",
                chunk_type="fixed",
                language="en",
                bm25_score=14.0,
            )
        ]
        with patch.object(service.dense_retriever, "retrieve", side_effect=ProviderConnectionError(provider_name="qdrant", endpoint="http://qdrant:6333")), \
             patch.object(service.bm25_retriever, "retrieve", return_value=(mock_bm25_res, 5.0)):
            results, lat = service.retrieve(query="Goa tourism", top_k=3, language="en", use_cache=False)
            assert len(results) > 0
            assert lat.fallback_used is True


# =============================================================================
# 3. MULTILINGUAL VOICE RAG END-TO-END VERIFICATION
# =============================================================================

class TestMultilingualVoiceRAGIntegration:
    """Test suite verifying end-to-end Voice RAG execution across all 5 supported languages."""

    @pytest.mark.parametrize("lang", ["en", "hi", "ta", "te", "ml"])
    def test_voice_ask_multilingual_flow(self, client, lang):
        """Verify POST /api/voice-ask processes multilingual requests correctly."""
        files = {"audio": ("sample.wav", io.BytesIO(VALID_WAV_HEADER), "audio/wav")}
        data = {"language": lang, "top_k": 3, "synthesize_speech": "true"}

        res = client.post("/api/voice-ask", files=files, data=data)
        assert res.status_code == 200
        payload = res.json()
        assert payload["status"] in ("success", "partial_success")
        assert payload["transcript"] != ""
        assert payload["answer"] != ""
        assert payload["language"] == lang

    @pytest.mark.parametrize(
        "fmt, header, mime, fname",
        [
            ("wav", VALID_WAV_HEADER, "audio/wav", "test.wav"),
            ("mp3", VALID_MP3_HEADER, "audio/mpeg", "test.mp3"),
            ("ogg", VALID_OGG_HEADER, "audio/ogg", "test.ogg"),
            ("webm", VALID_WEBM_HEADER, "audio/webm", "test.webm"),
            ("flac", VALID_FLAC_HEADER, "audio/flac", "test.flac"),
            ("m4a", VALID_M4A_HEADER, "audio/m4a", "test.m4a"),
        ],
    )
    def test_voice_ask_audio_formats(self, client, fmt, header, mime, fname):
        """Verify all supported binary audio formats pass security and execution checks."""
        files = {"audio": (fname, io.BytesIO(header), mime)}
        data = {"language": "en", "top_k": 3, "synthesize_speech": "true"}

        res = client.post("/api/voice-ask", files=files, data=data)
        assert res.status_code == 200
        payload = res.json()
        assert payload["status"] in ("success", "partial_success")


# =============================================================================
# 4. GATED REAL-PROVIDER LIVE API SMOKE TEST (COST-PROTECTED)
# =============================================================================

@pytest.mark.skipif(
    os.getenv("PHASE612_REAL_PROVIDER_TESTS", "false").lower() != "true"
    or not os.getenv("SARVAM_API_KEY"),
    reason="Real provider live API tests require PHASE612_REAL_PROVIDER_TESTS=true and SARVAM_API_KEY.",
)
class TestLiveSarvamProviderSmoke:
    """Cost-protected live smoke test executing against real Sarvam AI endpoints when explicitly enabled."""

    @pytest.mark.asyncio
    async def test_live_sarvam_llm_generation(self):
        """Test live Sarvam LLM generation with grounded prompt."""
        provider = SarvamLLMProvider()
        res = await provider.generate(
            prompt="What is the capital of Goa?\nContext: Panaji is the official capital city of Goa.",
            language="en",
        )
        assert res.text != ""
        assert res.provider == "sarvam"
        assert res.latency_ms > 0
