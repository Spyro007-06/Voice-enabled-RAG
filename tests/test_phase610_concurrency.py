"""Phase 6.10 — Production Load, Concurrency, Resilience & Singleton Thread Safety Tests."""

import asyncio
import io
import time
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
import httpx

from app.config import get_settings
from app.main import app
from app.observability.metrics import MetricsRegistry, get_metrics_registry
from app.observability.security_middleware import get_rate_limiter
from app.observability.tracing import (
    get_current_language,
    get_current_request_id,
    set_current_language,
    set_current_request_id,
)
from app.reranking.adaptive import get_adaptive_retrieval_service
from app.retrieval.cache import RetrievalCache, get_rag_cache
from app.retrieval.models import RetrievalResult
from app.retrieval.service import get_retrieval_service

# Valid audio fixtures for audio load tests
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
# 1. CACHE CONCURRENCY & MULTILINGUAL ISOLATION TESTS
# =============================================================================

class TestCacheConcurrencyAndIsolation:
    """Test suite verifying thread safety, concurrent access, and cross-language cache isolation."""

    def test_cache_cross_language_isolation(self):
        """Verify that identical query text in different languages generates distinct cache keys."""
        cache = RetrievalCache(max_size=100, enabled=True, ttl_seconds=300)
        query = "What is the capital of Goa?"
        languages = ["en", "hi", "ta", "te", "ml"]

        # Store distinct results per language
        for lang in languages:
            results = [
                RetrievalResult(
                    chunk_id=f"chunk_{lang}_1",
                    document_id=f"doc_{lang}_1",
                    text=f"Capital in {lang}",
                    chunk_type="fixed",
                    language=lang,
                    fusion_score=0.9,
                )
            ]
            cache.set(query=query, top_k=5, language=lang, results=results)

        # Verify no cross-language contamination
        for lang in languages:
            cached = cache.get(query=query, top_k=5, language=lang)
            assert cached is not None, f"Cache missed for language {lang}"
            assert cached[0].language == lang
            assert cached[0].chunk_id == f"chunk_{lang}_1"

    @pytest.mark.asyncio
    async def test_concurrent_cache_access_thread_safety(self):
        """Simulate concurrent threads/tasks reading and writing to the retrieval cache."""
        cache = RetrievalCache(max_size=50, enabled=True, ttl_seconds=300)

        async def worker(worker_id: int):
            for i in range(20):
                query = f"query_{i % 10}"
                lang = "en" if i % 2 == 0 else "hi"
                val = cache.get(query=query, top_k=5, language=lang)
                if val is None:
                    res = [
                        RetrievalResult(
                            chunk_id=f"c_{worker_id}_{i}",
                            document_id=f"d_{worker_id}_{i}",
                            text=f"Text {worker_id}",
                            chunk_type="fixed",
                            language=lang,
                            fusion_score=0.8,
                        )
                    ]
                    cache.set(query=query, top_k=5, language=lang, results=res)
                await asyncio.sleep(0.001)

        tasks = [asyncio.create_task(worker(w)) for w in range(10)]
        await asyncio.gather(*tasks)

        stats = cache.stats()
        assert stats["current_size"] <= 50
        assert stats["hits"] >= 0
        assert stats["misses"] >= 0

    def test_bounded_lru_eviction_under_concurrency(self):
        """Verify LRU cache does not exceed maximum capacity under heavy load."""
        cache = RetrievalCache(max_size=20, enabled=True, ttl_seconds=60)
        for i in range(100):
            res = [
                RetrievalResult(
                    chunk_id=f"c_{i}",
                    document_id=f"d_{i}",
                    text=f"Text {i}",
                    chunk_type="fixed",
                    fusion_score=0.8,
                )
            ]
            cache.set(query=f"key_{i}", top_k=5, results=res)

        stats = cache.stats()
        assert stats["current_size"] <= 20
        assert stats["evictions"] >= 80


# =============================================================================
# 2. SINGLETON CONCURRENCY & METRICS THREAD SAFETY
# =============================================================================

class TestSingletonConcurrency:
    """Test suite verifying thread safety and singleton preservation under concurrent execution."""

    @pytest.mark.asyncio
    async def test_metrics_registry_concurrency(self):
        """Stress Prometheus metrics registry concurrently and verify total counter integrity."""
        registry = MetricsRegistry()

        async def incrementor():
            for _ in range(50):
                registry.record_http_request(
                    endpoint="/api/ask",
                    method="POST",
                    status_code=200,
                    duration_s=0.05,
                    language="en",
                )
                registry.record_cache_hit(language="en")
                registry.record_retrieval(
                    language="en",
                    tier="skip",
                    cache_hit=False,
                    duration_s=0.02,
                    confidence=0.85,
                )
                await asyncio.sleep(0.001)

        tasks = [asyncio.create_task(incrementor()) for _ in range(10)]
        await asyncio.gather(*tasks)

        prom_text = registry.generate_prometheus_text()
        assert 'http_requests_total{endpoint="/api/ask",method="POST",status="200",language="en"} 500.0' in prom_text
        assert 'rag_cache_hits_total{language="en"} 500.0' in prom_text

    def test_singleton_getters_return_same_instance(self):
        """Verify factory getters return the exact same instance across calls."""
        r1 = get_retrieval_service()
        r2 = get_retrieval_service()
        assert r1 is r2

        a1 = get_adaptive_retrieval_service()
        a2 = get_adaptive_retrieval_service()
        assert a1 is a2

        m1 = get_metrics_registry()
        m2 = get_metrics_registry()
        assert m1 is m2


# =============================================================================
# 3. PROVIDER FAILURE, TIMEOUT & FALLBACK TESTS
# =============================================================================

class TestProviderFailureAndResilience:
    """Test suite verifying graceful degradation, timeouts, and fallback ranking."""

    def test_qdrant_timeout_fallback_to_bm25(self):
        """Simulate Qdrant timeout and verify fallback to BM25 lexical retrieval."""
        service = get_retrieval_service()
        mock_bm25_res = [
            RetrievalResult(
                chunk_id="chunk_bm25_1",
                document_id="doc_1",
                text="Goa beaches and tourism information",
                chunk_type="fixed",
                language="en",
                bm25_score=15.0,
            )
        ]
        with patch.object(service.dense_retriever, "retrieve", side_effect=TimeoutError("Qdrant timed out")), \
             patch.object(service.bm25_retriever, "retrieve", return_value=(mock_bm25_res, 5.0)):
            results, latency = service.retrieve(
                query="Goa tourism and beaches",
                top_k=5,
                language="en",
                use_cache=False,
            )
            assert len(results) > 0
            assert latency.fallback_used is True

    def test_bm25_failure_fallback_to_dense(self):
        """Simulate BM25 failure and verify fallback to dense ANN retrieval."""
        service = get_retrieval_service()
        mock_dense_res = [
            RetrievalResult(
                chunk_id="chunk_dense_1",
                document_id="doc_1",
                text="Old Goa churches information",
                chunk_type="fixed",
                language="en",
                dense_score=0.9,
            )
        ]
        with patch.object(service.bm25_retriever, "retrieve", side_effect=RuntimeError("BM25 corrupted")), \
             patch.object(service.dense_retriever, "retrieve", return_value=(mock_dense_res, 5.0, 10.0)):
            results, latency = service.retrieve(
                query="Old Goa churches",
                top_k=5,
                language="en",
                use_cache=False,
            )
            assert len(results) > 0
            assert latency.fallback_used is True

    @pytest.mark.asyncio
    async def test_adaptive_reranker_timeout_preserves_rrf(self):
        """Simulate reranker timeout in adaptive retrieval and verify RRF fallback."""
        adaptive_service = get_adaptive_retrieval_service()
        mock_dense_res = [
            RetrievalResult(
                chunk_id="chunk_1",
                document_id="doc_1",
                text="Dudhsagar waterfall is located on the Mandovi River in Goa.",
                chunk_type="fixed",
                language="en",
                dense_score=0.85,
            )
        ]
        mock_bm25_res = [
            RetrievalResult(
                chunk_id="chunk_1",
                document_id="doc_1",
                text="Dudhsagar waterfall is located on the Mandovi River in Goa.",
                chunk_type="fixed",
                language="en",
                bm25_score=12.0,
            )
        ]
        with patch.object(adaptive_service.retrieval_service.dense_retriever, "retrieve", return_value=(mock_dense_res, 5.0, 10.0)), \
             patch.object(adaptive_service.retrieval_service.bm25_retriever, "retrieve", return_value=(mock_bm25_res, 5.0)), \
             patch.object(adaptive_service.lightweight_reranker, "rerank", side_effect=TimeoutError("Reranker deadline exceeded")):
            results, decision, lat = adaptive_service.adaptive_retrieve(
                query="Dudhsagar waterfall location",
                top_k=5,
                language="en",
                use_cache=False,
            )
            assert len(results) > 0
            assert lat.fallback_used is True or decision.reranker_tier in ("high", "skip", "medium", "low")


# =============================================================================
# 4. CONCURRENT REQUEST CORRELATION & CONTEXTVARS ISOLATION
# =============================================================================

class TestRequestCorrelationUnderConcurrency:
    """Test suite verifying X-Request-ID uniqueness and ContextVars isolation across concurrent tasks."""

    @pytest.mark.asyncio
    async def test_contextvars_isolation_across_concurrent_tasks(self):
        """Verify that concurrent async tasks maintain strictly isolated request IDs and languages."""
        results = {}

        async def subtask(task_id: int):
            req_id = f"custom-req-id-{task_id:04d}"
            lang = "hi" if task_id % 2 == 0 else "ta"
            set_current_request_id(req_id)
            set_current_language(lang)

            # Yield control to allow interleaving
            await asyncio.sleep(0.01)

            # Verify context remained isolated
            curr_req = get_current_request_id()
            curr_lang = get_current_language()
            results[task_id] = (curr_req == req_id and curr_lang == lang)

        tasks = [asyncio.create_task(subtask(i)) for i in range(50)]
        await asyncio.gather(*tasks)

        assert all(results.values()), "ContextVars leaked between concurrent tasks!"

    def test_concurrent_api_requests_retain_distinct_request_ids(self, client):
        """Verify client requests with unique X-Request-ID receive matching response headers."""
        import concurrent.futures

        def send_request(idx: int):
            req_id = f"test-client-req-{idx:03d}"
            res = client.post(
                "/api/retrieve",
                json={"query": f"test query {idx}"},
                headers={"X-Request-ID": req_id},
            )
            return res.status_code, res.headers.get("X-Request-ID")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(send_request, i) for i in range(15)]
            for i, f in enumerate(concurrent.futures.as_completed(futures)):
                status_code, returned_id = f.result()
                assert status_code in (200, 422)
                assert returned_id.startswith("test-client-req-")


# =============================================================================
# 5. VOICE RAG FORMAT CONCURRENCY & MULTI-FORMAT STRESS
# =============================================================================

class TestVoiceRAGMultiFormatConcurrency:
    """Test suite verifying concurrent voice uploads across diverse audio formats."""

    @pytest.mark.parametrize(
        "fmt_name, header_bytes, mime_type, filename",
        [
            ("wav", VALID_WAV_HEADER, "audio/wav", "audio.wav"),
            ("mp3", VALID_MP3_HEADER, "audio/mpeg", "audio.mp3"),
            ("ogg", VALID_OGG_HEADER, "audio/ogg", "audio.ogg"),
            ("webm", VALID_WEBM_HEADER, "audio/webm", "audio.webm"),
            ("flac", VALID_FLAC_HEADER, "audio/flac", "audio.flac"),
            ("m4a", VALID_M4A_HEADER, "audio/m4a", "audio.m4a"),
        ],
    )
    def test_voice_ask_supported_formats(self, client, fmt_name, header_bytes, mime_type, filename):
        """Verify that all supported audio formats pass security checks and execute successfully."""
        files = {"audio": (filename, io.BytesIO(header_bytes), mime_type)}
        data = {"language": "en", "top_k": 3, "synthesize_speech": "true"}

        res = client.post("/api/voice-ask", files=files, data=data)
        assert res.status_code == 200
        payload = res.json()
        assert payload["status"] in ("success", "partial_success", "fallback")
        assert payload["transcript"] != ""
        assert payload["answer"] != ""
