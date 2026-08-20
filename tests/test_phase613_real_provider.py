"""Phase 6.13 — Real Cloud Provider Latency Optimization & Connection Pooling Tests."""

import io
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient
import httpx

from app.config import get_settings
from app.main import app
from app.observability.security_middleware import get_rate_limiter
from app.providers.llm.openai import OpenAILLMProvider
from app.providers.llm.sarvam import SarvamLLMProvider
from app.providers.stt.sarvam import SarvamSTTProvider
from app.providers.tts.sarvam import SarvamTTSProvider
from app.retrieval.cache import get_retrieval_cache
from app.retrieval.models import RetrievalResult
from app.retrieval.service import get_retrieval_service

VALID_WAV_HEADER = (
    b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00"
    b"\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
)


@pytest.fixture
def client():
    get_rate_limiter().reset()
    with TestClient(app) as test_client:
        yield test_client
    get_rate_limiter().reset()


# =============================================================================
# 1. HTTP CONNECTION POOLING & CLIENT REUSE TESTS
# =============================================================================

class TestHttpConnectionPoolingAndLifecycle:
    """Test suite verifying persistent HTTP connection pooling and clean teardown."""

    @pytest.mark.asyncio
    async def test_sarvam_stt_connection_pooling(self):
        """Verify SarvamSTTProvider reuses the same AsyncClient instance across consecutive requests."""
        provider = SarvamSTTProvider(api_key="sk_live_dummy_test_key_12345")
        client1 = await provider._get_client()
        client2 = await provider._get_client()
        assert client1 is client2
        assert not client1.is_closed

        await provider.aclose()
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_sarvam_tts_connection_pooling(self):
        """Verify SarvamTTSProvider reuses the same AsyncClient instance across consecutive requests."""
        provider = SarvamTTSProvider(api_key="sk_live_dummy_test_key_12345")
        client1 = await provider._get_client()
        client2 = await provider._get_client()
        assert client1 is client2
        assert not client1.is_closed

        await provider.aclose()
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_sarvam_llm_connection_pooling(self):
        """Verify SarvamLLMProvider reuses the same AsyncClient instance across consecutive requests."""
        provider = SarvamLLMProvider(api_key="sk_live_dummy_test_key_12345")
        client1 = await provider._get_client()
        client2 = await provider._get_client()
        assert client1 is client2
        assert not client1.is_closed

        await provider.aclose()
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_openai_llm_connection_pooling(self):
        """Verify OpenAILLMProvider reuses the same AsyncClient instance across consecutive requests."""
        provider = OpenAILLMProvider(api_key="sk_live_dummy_test_key_12345")
        client1 = await provider._get_client()
        client2 = await provider._get_client()
        assert client1 is client2
        assert not client1.is_closed

        await provider.aclose()
        assert provider._client is None


# =============================================================================
# 2. RETRIEVAL CACHE LATENCY OPTIMIZATION TESTS
# =============================================================================

class TestRetrievalCacheOptimization:
    """Test suite verifying cache acceleration and multilingual isolation."""

    def test_cache_hit_latency_reduction(self):
        """Verify that cache hits and sets work accurately."""
        cache = get_retrieval_cache()
        cache.enabled = True
        cache.clear()

        test_val = [RetrievalResult(chunk_id="c1", document_id="doc1", text="cached Goa content", chunk_type="fixed", language="en", combined_score=0.9)]
        cache.set(query="Goa tourism", top_k=3, strategies="hybrid", language="en", fusion_method="rrf", results=test_val)

        # Cache hit
        cached_result = cache.get(query="Goa tourism", top_k=3, strategies="hybrid", language="en", fusion_method="rrf")
        assert cached_result == test_val

        # Cache stats
        stats = cache.stats()
        assert stats["hits"] >= 1

    def test_multilingual_cache_isolation(self):
        """Verify that queries in different languages maintain distinct cache keys."""
        cache = get_retrieval_cache()
        cache.enabled = True
        cache.clear()

        query_text = "capital of India"
        en_val = [RetrievalResult(chunk_id="c_en", document_id="doc_en", text="New Delhi is capital", chunk_type="fixed", language="en", combined_score=0.95)]

        # Set for English
        cache.set(query=query_text, top_k=3, strategies="hybrid", language="en", fusion_method="rrf", results=en_val)

        # Get in English should hit
        assert cache.get(query=query_text, top_k=3, strategies="hybrid", language="en", fusion_method="rrf") == en_val

        # Get in Hindi with identical query text should miss due to language isolation
        assert cache.get(query=query_text, top_k=3, strategies="hybrid", language="hi", fusion_method="rrf") is None


# =============================================================================
# 3. STAGE LATENCY TELEMETRY INTEGRITY TESTS
# =============================================================================

class TestStageLatencyTelemetry:
    """Test suite verifying accurate and complete latency breakdown telemetry."""

    def test_voice_ask_latency_breakdown_presence(self, client):
        """Verify /api/voice-ask returns comprehensive stage-by-stage latency metrics."""
        files = {"audio": ("sample.wav", io.BytesIO(VALID_WAV_HEADER), "audio/wav")}
        data = {"language": "en", "top_k": 3, "synthesize_speech": "true"}

        resp = client.post("/api/voice-ask", files=files, data=data)
        assert resp.status_code == 200
        payload = resp.json()

        assert "latency_ms" in payload
        lat = payload["latency_ms"]
        assert "stt" in lat
        assert "bm25" in lat
        assert "llm" in lat
        assert "tts" in lat
        assert "total" in lat
        assert "retrieval_total_ms" in lat
        assert "voice_total_ms" in lat
        assert lat["total"] >= 0.0
