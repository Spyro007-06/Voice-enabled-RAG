"""Unit and regression tests for Phase 6.4 latency, quality, and adaptive retrieval hardening."""

import os
import time
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.reranking.adaptive import (
    AdaptiveDecision,
    AdaptiveRetrievalService,
    ConfidenceSignals,
    calculate_retrieval_confidence,
    calculate_retriever_agreement,
    compute_confidence_signals,
    rerank_candidate_count,
    routing_decision,
    should_rerank,
)
from app.reranking.models import AdaptiveRetrieveRequest, AdaptiveRetrieveResponse, RerankResult
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.cache import RetrievalCache, get_rag_cache
from app.retrieval.filters import get_language_synonyms, matches_filter
from app.retrieval.models import RetrievalResult
from app.retrieval.qdrant_store import QdrantVectorStore, get_qdrant_store, get_shared_qdrant_client


def _make_dummy_result(
    chunk_id: str,
    doc_id: str,
    text: str = "sample text",
    dense_score: float = 0.85,
    bm25_score: float = 18.0,
    fusion_score: float = 0.03,
    language: str = "hi",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=text,
        chunk_type="semantic",
        language=language,
        dense_score=dense_score,
        bm25_score=bm25_score,
        normalized_dense_score=dense_score,
        normalized_bm25_score=bm25_score,
        fusion_score=fusion_score,
        rank=1,
        metadata={"language": language, "query_id": "q1", "passage_index": 0},
    )


class TestPhase64AdaptiveConfidenceAndSignals:
    """Test 8-signal confidence breakdown, bounding, and normalization."""

    def test_compute_confidence_signals_structure(self):
        dense = [_make_dummy_result("c1", "d1", dense_score=0.88), _make_dummy_result("c2", "d2", dense_score=0.80)]
        bm25 = [_make_dummy_result("c1", "d1", bm25_score=24.0), _make_dummy_result("c3", "d3", bm25_score=15.0)]
        fusion = [_make_dummy_result("c1", "d1", fusion_score=0.033), _make_dummy_result("c2", "d2", fusion_score=0.028)]

        signals = compute_confidence_signals(dense, bm25, fusion, language="hi")

        assert isinstance(signals, ConfidenceSignals)
        assert 0.0 <= signals.dense_similarity <= 1.0
        assert 0.0 <= signals.dense_margin <= 1.0
        assert 0.0 <= signals.bm25_strength <= 1.0
        assert 0.0 <= signals.retriever_agreement <= 1.0
        assert 0.0 <= signals.rrf_margin <= 1.0
        assert 0.0 <= signals.language_confidence <= 1.0
        assert 0.0 <= signals.document_diversity <= 1.0
        assert 0.0 <= signals.final_confidence <= 1.0

        # Verify serialization dict
        d = signals.to_dict()
        assert "dense_similarity" in d
        assert "final_confidence" in d
        assert len(d) == 8

    def test_empty_signals_return_zeros(self):
        signals = compute_confidence_signals([], [], [])
        assert signals.final_confidence == 0.0
        assert signals.dense_similarity == 0.0
        assert signals.bm25_strength == 0.0

    def test_high_confidence_routes_to_skip(self):
        signals = ConfidenceSignals(
            dense_similarity=0.95,
            dense_margin=0.85,
            bm25_strength=0.90,
            retriever_agreement=0.80,
            rrf_margin=0.90,
            language_confidence=1.0,
            document_diversity=1.0,
            final_confidence=0.88,
        )
        decision = routing_decision(signals, threshold_high=0.70, threshold_low=0.40)
        assert decision.should_rerank is False
        assert decision.reranker_tier == "high"
        assert decision.candidate_k == 0
        assert decision.reranker == "none"

    def test_medium_confidence_routes_to_k3(self):
        signals = ConfidenceSignals(
            dense_similarity=0.60,
            dense_margin=0.40,
            bm25_strength=0.50,
            retriever_agreement=0.40,
            rrf_margin=0.40,
            language_confidence=1.0,
            document_diversity=1.0,
            final_confidence=0.55,
        )
        decision = routing_decision(signals, threshold_high=0.70, threshold_low=0.40, medium_k=3, low_k=5)
        assert decision.should_rerank is True
        assert decision.reranker_tier == "medium"
        assert decision.candidate_k == 3
        assert decision.reranker == "minilm"

    def test_low_confidence_routes_to_k5(self):
        signals = ConfidenceSignals(
            dense_similarity=0.20,
            dense_margin=0.10,
            bm25_strength=0.10,
            retriever_agreement=0.00,
            rrf_margin=0.10,
            language_confidence=0.5,
            document_diversity=1.0,
            final_confidence=0.25,
        )
        decision = routing_decision(signals, threshold_high=0.70, threshold_low=0.40, medium_k=3, low_k=5)
        assert decision.should_rerank is True
        assert decision.reranker_tier == "low"
        assert decision.candidate_k == 5
        assert decision.reranker == "minilm"

    def test_early_exit_downgrade_from_low_to_k3(self):
        # Low confidence but with high RRF dominance and language match
        signals = ConfidenceSignals(
            dense_similarity=0.35,
            dense_margin=0.20,
            bm25_strength=0.20,
            retriever_agreement=0.30,
            rrf_margin=0.75,
            language_confidence=0.90,
            document_diversity=1.0,
            final_confidence=0.38,  # Below 0.40
        )
        k = rerank_candidate_count(0.38, threshold_high=0.70, threshold_low=0.40, medium_k=3, low_k=5, signals=signals)
        assert k == 3  # Downgraded to 3 via early exit check


class TestPhase64CacheArchitecture:
    """Test thread-safe bounded LRU cache with TTL expiration."""

    def test_cache_disabled_returns_none(self):
        cache = RetrievalCache(max_size=10, enabled=False, ttl_seconds=300)
        cache.set("test query", 5, results=[_make_dummy_result("c1", "d1")])
        assert cache.get("test query", 5) is None
        assert cache.size == 0

    def test_cache_hit_and_eviction(self):
        cache = RetrievalCache(max_size=2, enabled=True, ttl_seconds=300)
        res1 = [_make_dummy_result("c1", "d1")]
        res2 = [_make_dummy_result("c2", "d2")]
        res3 = [_make_dummy_result("c3", "d3")]

        cache.set("q1", 5, results=res1)
        cache.set("q2", 5, results=res2)
        assert cache.get("q1", 5) is not None

        # Adding 3rd should evict q2 (since q1 was accessed most recently)
        cache.set("q3", 5, results=res3)
        assert cache.get("q1", 5) is not None
        assert cache.get("q3", 5) is not None
        assert cache.get("q2", 5) is None

    def test_cache_ttl_expiration(self):
        cache = RetrievalCache(max_size=10, enabled=True, ttl_seconds=1)
        cache.set("q_expire", 5, results=[_make_dummy_result("c1", "d1")])
        assert cache.get("q_expire", 5) is not None
        time.sleep(1.1)
        assert cache.get("q_expire", 5) is None

    def test_embedding_and_adaptive_cache(self):
        cache = RetrievalCache(max_size=10, enabled=True, ttl_seconds=300)
        cache.set_embedding("test embedding", [0.1, 0.2, 0.3])
        assert cache.get_embedding("test embedding") == [0.1, 0.2, 0.3]

        cache.set_adaptive("test adaptive", "hi", ("mock_results", "mock_decision"))
        cached = cache.get_adaptive("test adaptive", "hi")
        assert cached == ("mock_results", "mock_decision")


class TestPhase64MultilingualAndOptimization:
    """Test multilingual normalization, Qdrant client reuse, and BM25 index reuse."""

    def test_language_synonym_normalization(self):
        assert "hi" in get_language_synonyms("hi-IN")
        assert "hin_Deva" in get_language_synonyms("hi")
        assert "ta" in get_language_synonyms("Tamil")
        assert "te" in get_language_synonyms("tel_Telu")
        assert "ml" in get_language_synonyms("malayalam")

    def test_matches_filter_multilingual(self):
        meta = {"language": "hin_Deva"}
        assert matches_filter(meta, chunk_type="semantic", language="hi")
        assert matches_filter(meta, chunk_type="semantic", language="hin_deva")
        assert not matches_filter(meta, chunk_type="semantic", language="ta")

    def test_qdrant_client_reuse(self):
        client1 = get_shared_qdrant_client()
        client2 = get_shared_qdrant_client()
        assert client1 is client2

    def test_bm25_index_reuse(self):
        bm25_1 = BM25Retriever()
        assert hasattr(bm25_1, "is_indexed")


class TestPhase64FallbackAndTelemetry:
    """Test graceful fallback on reranker failure and API endpoints."""

    def test_reranker_failure_falls_back_to_rrf(self):
        mock_retrieval = MagicMock()
        dummy_res = [_make_dummy_result("c1", "d1", dense_score=0.6, bm25_score=10.0, fusion_score=0.03)]
        mock_retrieval.dense_retriever.retrieve.return_value = (dummy_res, 5.0, 5.0)
        mock_retrieval.bm25_retriever.retrieve.return_value = (dummy_res, 5.0)
        mock_retrieval.bm25_retriever.is_indexed.return_value = True
        mock_retrieval.fusion.fuse_results.return_value = dummy_res
        service = AdaptiveRetrievalService(retrieval_service=mock_retrieval)
        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = RuntimeError("CUDA OOM / model failure")
        service._lightweight_reranker = mock_reranker

        results, decision, lat = service.adaptive_retrieve(
            query="भारत की राजधानी क्या है?",
            top_k=3,
            language="hi",
            threshold_high=0.99,  # Force reranking
            threshold_low=0.95,
        )

        assert len(results) > 0
        assert decision.reranker == "fallback_rrf"
        assert decision.should_rerank is True
        assert lat.total > 0

    def test_adaptive_api_endpoint_backward_compatibility(self):
        client = TestClient(app)
        resp = client.post("/api/adaptive-retrieve", json={"query": "What is machine learning?", "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert "query" in data
        assert "results" in data
        assert "retrieval_confidence" in data
        assert "reranking_used" in data
        assert "reranker" in data
        assert "routing_tier" in data
        assert "candidate_k" in data
        assert "latency_ms" in data
        assert "embedding" in data["latency_ms"]
        assert "retrieval" in data["latency_ms"]

    def test_no_credential_leakage_in_api(self):
        client = TestClient(app)
        resp = client.post("/api/adaptive-retrieve", json={"query": "test query", "top_k": 2})
        text = resp.text.lower()
        assert "api_key" not in text
        assert "bearer" not in text
        assert "sk-" not in text
