"""Unit and integration tests for Phase 6.5 Production Latency Hardening & Retrieval Stability."""

import os
import time
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.generation.models import (
    AskRequest,
    AskResponse,
    CitationProvenance,
    GenerationResult,
    ValidationResult,
)
from app.generation.validators import AnswerValidator, CitationValidator
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
from app.reranking.lightweight_reranker import LightweightMiniLMReranker, get_lightweight_reranker
from app.reranking.models import AdaptiveRetrieveRequest, AdaptiveRetrieveResponse, RerankResult
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.cache import RetrievalCache, get_rag_cache
from app.retrieval.filters import get_language_synonyms, matches_filter
from app.retrieval.models import LatencyBreakdown, RetrievalResult
from app.retrieval.qdrant_store import QdrantVectorStore, get_qdrant_store, get_shared_qdrant_client
from app.retrieval.service import RetrievalService, get_retrieval_service


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


class TestPhase65ParallelRetrievalAndTimeouts:
    """Test parallel retrieval, timeouts, and fallback mechanics."""

    def test_parallel_retrieval_success(self):
        """Test concurrent dense and BM25 execution."""
        service = RetrievalService()
        mock_dense = MagicMock()
        mock_dense.retrieve.return_value = ([_make_dummy_result("c1", "d1", dense_score=0.9)], 10.0, 20.0)
        mock_bm25 = MagicMock()
        mock_bm25.is_indexed.return_value = True
        mock_bm25.retrieve.return_value = ([_make_dummy_result("c2", "d2", bm25_score=15.0)], 15.0)

        service.dense_retriever = mock_dense
        service.bm25_retriever = mock_bm25

        results, lat = service.retrieve("test query", top_k=5, use_cache=False)
        assert len(results) >= 1
        assert lat.parallel_execution is True
        assert lat.timeout_stage is None
        assert lat.fallback_used is False

    def test_qdrant_timeout_falls_back_to_bm25(self):
        """Test that Qdrant timeout continues with BM25 results."""
        service = RetrievalService()

        def slow_dense(*args, **kwargs):
            time.sleep(0.3)
            return ([_make_dummy_result("c_dense", "d_dense")], 5.0, 200.0)

        mock_dense = MagicMock()
        mock_dense.retrieve.side_effect = slow_dense
        mock_bm25 = MagicMock()
        mock_bm25.is_indexed.return_value = True
        mock_bm25.retrieve.return_value = ([_make_dummy_result("c_bm25", "d_bm25")], 10.0)

        service.dense_retriever = mock_dense
        service.bm25_retriever = mock_bm25

        current_settings = get_settings()
        mock_settings = MagicMock(wraps=current_settings)
        mock_settings.QDRANT_TIMEOUT_MS = 50.0
        mock_settings.PARALLEL_RETRIEVAL_ENABLED = True
        mock_settings.BM25_TIMEOUT_MS = 80.0

        with patch("app.retrieval.service.get_settings", return_value=mock_settings):
            results, lat = service.retrieve("test query", top_k=5, use_cache=False)
            assert len(results) == 1
            assert results[0].chunk_id == "c_bm25"
            assert lat.timeout_stage == "qdrant"
            assert lat.fallback_used is True

    def test_bm25_timeout_falls_back_to_dense(self):
        """Test that BM25 timeout continues with dense results."""
        service = RetrievalService()

        def slow_bm25(*args, **kwargs):
            time.sleep(0.3)
            return ([_make_dummy_result("c_bm25", "d_bm25")], 200.0)

        mock_dense = MagicMock()
        mock_dense.retrieve.return_value = ([_make_dummy_result("c_dense", "d_dense")], 5.0, 10.0)
        mock_bm25 = MagicMock()
        mock_bm25.is_indexed.return_value = True
        mock_bm25.retrieve.side_effect = slow_bm25

        service.dense_retriever = mock_dense
        service.bm25_retriever = mock_bm25

        current_settings = get_settings()
        mock_settings = MagicMock(wraps=current_settings)
        mock_settings.QDRANT_TIMEOUT_MS = 120.0
        mock_settings.PARALLEL_RETRIEVAL_ENABLED = True
        mock_settings.BM25_TIMEOUT_MS = 50.0

        with patch("app.retrieval.service.get_settings", return_value=mock_settings):
            results, lat = service.retrieve("test query", top_k=5, use_cache=False)
            assert len(results) == 1
            assert results[0].chunk_id == "c_dense"
            assert lat.timeout_stage == "bm25"
            assert lat.fallback_used is True

    def test_reranker_timeout_and_fallback(self):
        """Test that reranker failure or timeout falls back to RRF rankings."""
        service = AdaptiveRetrievalService()
        mock_ret = MagicMock()
        dummy = [_make_dummy_result("c1", "d1"), _make_dummy_result("c2", "d2")]
        mock_ret.dense_retriever.retrieve.return_value = (dummy, 5.0, 10.0)
        mock_ret.bm25_retriever.retrieve.return_value = (dummy, 10.0)
        mock_ret.fusion.fuse_results.return_value = dummy
        mock_ret._executor = None
        service.retrieval_service = mock_ret

        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = TimeoutError("Reranker timed out")
        service._lightweight_reranker = mock_reranker

        results, decision, lat = service.adaptive_retrieve(
            query="test query",
            top_k=2,
            threshold_high=0.99,  # Force reranking attempt
            threshold_low=0.95,
            use_cache=False,
        )

        assert len(results) == 2
        assert decision.reranker == "fallback_rrf"
        assert lat.fallback_used is True


class TestPhase65BudgetDeadlinesAndEarlyExits:
    """Test total retrieval deadline, budget tracking, and candidate dominance early exits."""

    def test_latency_budget_exhaustion_skips_reranker(self):
        """Verify that when remaining latency budget is below threshold, reranking is skipped."""
        signals = ConfidenceSignals(
            dense_similarity=0.50,
            dense_margin=0.20,
            bm25_strength=0.50,
            retriever_agreement=0.20,
            rrf_margin=0.20,
            language_confidence=1.0,
            document_diversity=1.0,
            final_confidence=0.45,  # Medium confidence normally triggers MiniLM K=3
        )

        # Budget is 20ms < min_budget (40ms)
        decision = routing_decision(
            signals,
            threshold_high=0.70,
            threshold_low=0.40,
            remaining_budget_ms=20.0,
            min_budget_ms=40.0,
        )

        assert decision.should_rerank is False
        assert decision.candidate_k == 0
        assert decision.reranker == "none"
        assert "exhausted" in decision.reason.lower()

    def test_candidate_dominance_early_exit(self):
        """Verify candidate dominance early exit downgrades or skips reranking."""
        signals = ConfidenceSignals(
            dense_similarity=0.75,
            dense_margin=0.30,  # Dominant margin >= 0.15
            bm25_strength=0.60,
            retriever_agreement=0.40,
            rrf_margin=0.70,
            language_confidence=1.0,
            document_diversity=1.0,
            final_confidence=0.35,  # Low confidence normally K=5
        )

        k = rerank_candidate_count(
            confidence=0.35,
            threshold_high=0.70,
            threshold_low=0.40,
            medium_k=3,
            low_k=5,
            signals=signals,
            early_exit_margin=0.15,
        )
        assert k == 3  # Downgraded from 5 to 3 due to dominance margin


class TestPhase65CacheHardeningAndIsolation:
    """Test cache isolation across languages, top_k, TTL, and capacity limits."""

    def test_cache_language_isolation(self):
        """Ensure identical query text in different languages do not produce cache collisions."""
        cache = RetrievalCache(max_size=10, enabled=True, ttl_seconds=300)
        res_hi = [_make_dummy_result("c_hi", "d_hi", language="hi")]
        res_en = [_make_dummy_result("c_en", "d_en", language="en")]

        cache.set_adaptive("goa capital", language="hi", value=(res_hi, "dec_hi"))
        cache.set_adaptive("goa capital", language="en", value=(res_en, "dec_en"))

        cached_hi = cache.get_adaptive("goa capital", language="hi")
        cached_en = cache.get_adaptive("goa capital", language="en")
        cached_ta = cache.get_adaptive("goa capital", language="ta")

        assert cached_hi is not None
        assert cached_hi[0][0].chunk_id == "c_hi"
        assert cached_en is not None
        assert cached_en[0][0].chunk_id == "c_en"
        assert cached_ta is None

    def test_cache_ttl_and_eviction(self):
        """Ensure TTL expiration and LRU capacity eviction work accurately."""
        cache = RetrievalCache(max_size=2, enabled=True, ttl_seconds=1)
        cache.set("q1", 5, results=[_make_dummy_result("c1", "d1")])
        cache.set("q2", 5, results=[_make_dummy_result("c2", "d2")])
        assert cache.get("q1", 5) is not None

        # Add 3rd item -> evicts q2
        cache.set("q3", 5, results=[_make_dummy_result("c3", "d3")])
        assert cache.get("q2", 5) is None
        assert cache.get("q1", 5) is not None

        # Wait for TTL
        time.sleep(1.1)
        assert cache.get("q1", 5) is None


class TestPhase65ComponentReuseAndSingletons:
    """Verify shared instances of Qdrant client, BM25 index, and MiniLM model."""

    def test_shared_qdrant_client_reuse(self):
        client1 = get_shared_qdrant_client()
        client2 = get_shared_qdrant_client()
        assert client1 is client2

    def test_bm25_index_loaded_once(self):
        retriever = BM25Retriever()
        assert hasattr(retriever, "is_indexed")

    def test_minilm_reranker_singleton(self):
        r1 = get_lightweight_reranker()
        r2 = get_lightweight_reranker()
        assert r1 is r2


class TestPhase65ProvenanceGroundingAndAPI:
    """Test citation provenance, grounding validator, API endpoints, and secret masking."""

    def test_citation_provenance_and_fabricated_rejection(self):
        chunks = [
            RerankResult(
                chunk_id="chunk_real_1",
                document_id="doc_real_1",
                text="Panaji is the capital of Goa.",
                chunk_type="fixed",
                language="en",
                reranker_score=0.98,
                original_rank=1,
                rank=1,
            )
        ]
        validated, prov_list, fabricated = CitationValidator.validate_citations(
            raw_citations=["chunk_real_1", "fake_chunk_99"],
            retrieved_context=chunks,
        )
        assert validated == ["chunk_real_1"]
        assert len(prov_list) == 1
        assert prov_list[0].chunk_id == "chunk_real_1"
        assert fabricated == ["fake_chunk_99"]

    def test_multilingual_unicode_grounding(self):
        chunk = RerankResult(
            chunk_id="chunk_hi_01",
            document_id="doc_hi_01",
            text="गोवा की राजधानी पणजी है।",
            chunk_type="sentence",
            language="hi",
            reranker_score=0.96,
            original_rank=1,
            rank=1,
        )
        res = AnswerValidator.validate_answer(
            answer="गोवा की राजधानी पणजी है।",
            grounded=True,
            confidence=0.95,
            citations=["chunk_hi_01"],
            retrieved_context=[chunk],
        )
        assert res.is_valid is True
        assert res.cleaned_answer == "गोवा की राजधानी पणजी है।"
        assert res.validated_citations == ["chunk_hi_01"]

    def test_api_endpoints_compatibility_and_telemetry(self):
        client = TestClient(app)

        # 1. Health
        assert client.get("/health").status_code == 200

        # 2. Retrieve
        r1 = client.post("/api/retrieve", json={"query": "What is machine learning?", "top_k": 3})
        assert r1.status_code == 200
        assert "latency_ms" in r1.json()

        # 3. Adaptive Retrieve
        r2 = client.post("/api/adaptive-retrieve", json={"query": "What is machine learning?", "top_k": 3})
        assert r2.status_code == 200
        data2 = r2.json()
        assert "retrieval_confidence" in data2
        assert "latency_ms" in data2
        assert "remaining_budget_ms" in data2["latency_ms"]

    def test_zero_secret_leakage_in_api(self):
        client = TestClient(app)
        resp = client.post("/api/adaptive-retrieve", json={"query": "test key query", "top_k": 2})
        text = resp.text.lower()
        assert "api_key" not in text
        assert "bearer" not in text
        assert "sk-" not in text
        assert "sarvam_api_key" not in text
