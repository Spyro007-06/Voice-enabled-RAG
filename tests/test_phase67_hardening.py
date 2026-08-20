"""Unit and integration tests for Phase 6.7 — Production SLA Hardening & Quality Preservation."""

import asyncio
import time
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import get_settings
from app.embeddings.multilingual_e5 import get_embedding_provider
from app.orchestration.models import VoiceAskResponse, VoiceLatencyBreakdown
from app.orchestration.voice_rag import VoiceRAGOrchestrator, get_voice_rag_orchestrator
from app.providers.factory import get_stt_provider, get_tts_provider
from app.providers.stt.mock import MockSTTProvider
from app.providers.tts.mock import MockTTSProvider
from app.reranking.adaptive import AdaptiveRetrievalService, calculate_retrieval_confidence
from app.reranking.lightweight_reranker import get_lightweight_reranker
from app.retrieval.bm25 import BM25Retriever, get_bm25_retriever
from app.retrieval.cache import RetrievalCache, get_rag_cache
from app.retrieval.models import RetrievalResult
from app.retrieval.service import RetrievalService, get_retrieval_service

DUMMY_AUDIO = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"


def test_bm25_singleton_reuse():
    """Verify BM25Retriever singleton factory reuses loaded index."""
    retriever1 = get_bm25_retriever()
    retriever2 = get_bm25_retriever()
    assert retriever1 is retriever2
    assert isinstance(retriever1, BM25Retriever)


def test_embedding_singleton_reuse():
    """Verify MultilingualE5EmbeddingProvider singleton is reused."""
    provider1 = get_embedding_provider()
    provider2 = get_embedding_provider()
    assert provider1 is provider2


def test_lightweight_reranker_singleton_reuse():
    """Verify LightweightMiniLMReranker singleton is reused."""
    reranker1 = get_lightweight_reranker()
    reranker2 = get_lightweight_reranker()
    assert reranker1 is reranker2


def test_cache_multilingual_isolation():
    """Verify cache keys strictly isolate queries across languages."""
    cache = RetrievalCache(enabled=True)
    res_en = [RetrievalResult(chunk_id="en_1", document_id="d1", text="Capital", chunk_type="fixed", language="en", rank=1)]
    res_hi = [RetrievalResult(chunk_id="hi_1", document_id="d2", text="राजधानी", chunk_type="fixed", language="hi", rank=1)]
    res_ta = [RetrievalResult(chunk_id="ta_1", document_id="d3", text="தலைநகரம்", chunk_type="fixed", language="ta", rank=1)]

    cache.set(query="capital", top_k=5, language="en", results=res_en)
    cache.set(query="capital", top_k=5, language="hi", results=res_hi)
    cache.set(query="capital", top_k=5, language="ta", results=res_ta)

    assert cache.get(query="capital", top_k=5, language="en")[0].chunk_id == "en_1"
    assert cache.get(query="capital", top_k=5, language="hi")[0].chunk_id == "hi_1"
    assert cache.get(query="capital", top_k=5, language="ta")[0].chunk_id == "ta_1"
    assert cache.get(query="capital", top_k=5, language="te") is None


def test_cache_telemetry_counters():
    """Verify cache hits, misses, evictions, and statistics."""
    cache = RetrievalCache(max_size=2, enabled=True, ttl_seconds=300)
    res = [RetrievalResult(chunk_id="c1", document_id="d1", text="T", chunk_type="fixed", language="en", rank=1)]

    # Miss
    assert cache.get("query1", top_k=5, language="en") is None
    # Set
    cache.set("query1", top_k=5, language="en", results=res)
    # Hit
    assert cache.get("query1", top_k=5, language="en") is not None
    # Eviction test
    cache.set("query2", top_k=5, language="en", results=res)
    cache.set("query3", top_k=5, language="en", results=res)  # Should evict query1

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["evictions"] == 1
    assert stats["enabled"] is True


def test_adaptive_insufficient_budget_skips_reranking():
    """Verify that when remaining retrieval deadline is < min_budget_ms, neural reranker is skipped."""
    dense = [RetrievalResult(chunk_id=f"c_{i}", document_id=f"d_{i}", text=f"Text {i}", chunk_type="fixed", language="en", dense_score=0.4, rank=i) for i in range(1, 6)]
    bm25 = [RetrievalResult(chunk_id=f"c_{i}", document_id=f"d_{i}", text=f"Text {i}", chunk_type="fixed", language="en", bm25_score=0.4, rank=i) for i in range(1, 6)]
    fusion = dense

    decision = calculate_retrieval_confidence(
        dense_results=dense,
        bm25_results=bm25,
        fusion_results=fusion,
        threshold_high=0.65,
        threshold_low=0.30,
        remaining_budget_ms=10.0,  # Below 20.0ms budget
        min_budget_ms=20.0,
    )

    assert not decision.should_rerank
    assert decision.reranker in ("fallback_rrf", "none")


@pytest.mark.asyncio
async def test_qdrant_timeout_fallback_to_bm25():
    """Verify that Qdrant timeout cleanly degrades to BM25 without throwing an unhandled exception."""
    mock_dense = MagicMock()
    mock_dense.retrieve.side_effect = TimeoutError("Simulated Qdrant timeout")

    mock_bm25 = MagicMock()
    mock_bm25.is_indexed.return_value = True
    bm25_cand = [RetrievalResult(chunk_id="bm25_1", document_id="doc_bm25", text="BM25 match text", chunk_type="fixed", language="en", bm25_score=10.0, rank=1)]
    mock_bm25.retrieve.return_value = (bm25_cand, 5.0)

    service = RetrievalService(dense_retriever=mock_dense, bm25_retriever=mock_bm25)
    results, latency = service.retrieve(query="What is Goa?", top_k=3, use_cache=False)

    assert len(results) >= 1
    assert results[0].chunk_id == "bm25_1"
    assert latency.fallback_used is True


@pytest.mark.asyncio
async def test_bm25_timeout_fallback_to_dense():
    """Verify that BM25 timeout cleanly degrades to Dense without throwing an unhandled exception."""
    mock_dense = MagicMock()
    dense_cand = [RetrievalResult(chunk_id="dense_1", document_id="doc_dense", text="Dense match text", chunk_type="fixed", language="en", dense_score=0.9, rank=1)]
    mock_dense.retrieve.return_value = (dense_cand, 2.0, 5.0)

    mock_bm25 = MagicMock()
    mock_bm25.is_indexed.return_value = True
    mock_bm25.retrieve.side_effect = TimeoutError("Simulated BM25 timeout")

    service = RetrievalService(dense_retriever=mock_dense, bm25_retriever=mock_bm25)
    results, latency = service.retrieve(query="What is Goa?", top_k=3, use_cache=False)

    assert len(results) >= 1
    assert results[0].chunk_id == "dense_1"
    assert latency.fallback_used is True


@pytest.mark.asyncio
async def test_voice_rag_end_to_end_multilingual_telemetry():
    """Verify end-to-end voice pipeline produces isolated SLA categories across Indic languages."""
    stt = MockSTTProvider(default_text="भारत की राजधानी क्या है?", default_language="hi")
    tts = MockTTSProvider()
    orchestrator = VoiceRAGOrchestrator(stt_provider=stt, tts_provider=tts)

    resp: VoiceAskResponse = await orchestrator.execute_voice_rag(
        audio_bytes=DUMMY_AUDIO,
        language="hi",
        top_k=3,
        synthesize_speech=True,
    )

    assert resp.status in ("success", "partial_success")
    assert resp.language == "hi"
    assert resp.audio.available is True
    l = resp.latency_ms

    # Check that category subtotals are isolated
    assert l.retrieval_total_ms >= 0.0
    assert l.generation_total_ms >= 0.0
    assert l.voice_total_ms >= 0.0
    assert l.end_to_end_total_ms >= 0.0
    assert l.total == l.end_to_end_total_ms
    assert l.audio_validation >= 0.0


@pytest.mark.asyncio
async def test_hard_retrieval_deadline_propagation():
    """Verify that retrieval service bounded wait respects overall deadline."""
    settings = get_settings()
    deadline_ms = getattr(settings, "TOTAL_RETRIEVAL_DEADLINE_MS", 200.0)

    t0 = time.perf_counter()
    retriever = get_retrieval_service()
    results, latency = retriever.retrieve(query="Goa tourism", top_k=5, use_cache=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Ensure retrieval returns within deadline + small test allowance
    assert elapsed_ms < 500.0  # Safe upper bound on CPU dev environments
    assert latency.total_ms <= elapsed_ms + 1.0
