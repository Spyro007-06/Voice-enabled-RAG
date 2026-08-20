"""Unit and integration tests for Phase 5.5 Adaptive Retrieval and Latency Optimization."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.chunking.models import Chunk
from app.main import app
from app.reranking.adaptive import (
    AdaptiveDecision,
    AdaptiveRetrievalService,
    calculate_retrieval_confidence,
    calculate_retriever_agreement,
)
from app.reranking.base import Reranker
from app.reranking.lightweight_reranker import FastLexicalSemanticRescorer, LightweightMiniLMReranker
from app.reranking.models import RerankResult
from app.retrieval.models import LatencyBreakdown, RetrievalResult
from app.retrieval.service import RetrievalService


# --- Unit Tests & Fixtures ---

class DummyMockReranker(Reranker):
    """Mock reranker returning deterministic scores and fast latency."""

    def __init__(self, multiplier: float = 1.0):
        self.multiplier = multiplier

    def rerank(self, query, candidates, top_k=None):
        scored = []
        for orig_idx, c in enumerate(candidates, start=1):
            res = RerankResult(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                chunk_type=c.chunk_type,
                language=c.language,
                dense_score=c.dense_score,
                bm25_score=c.bm25_score,
                fusion_score=c.fusion_score,
                reranker_score=round(float(c.dense_score or 0.5) * self.multiplier, 4),
                original_rank=c.rank if c.rank is not None else orig_idx,
                rank=orig_idx,
                metadata=c.metadata,
            )
            scored.append(res)
        scored.sort(key=lambda x: x.reranker_score, reverse=True)
        for i, item in enumerate(scored, start=1):
            item.rank = i
        k = top_k if top_k is not None else len(scored)
        return scored[:k], 5.0


def test_retriever_agreement_calculation():
    """Verify Jaccard retriever agreement calculation across overlapping candidate pools."""
    dense_res = [
        RetrievalResult(chunk_id=f"c_{i}", document_id=f"doc_{i}", text="test", chunk_type="fixed", dense_score=0.9 - (i * 0.05), rank=i)
        for i in range(1, 6)  # c_1, c_2, c_3, c_4, c_5
    ]
    bm25_res_full_overlap = [
        RetrievalResult(chunk_id=f"c_{i}", document_id=f"doc_{i}", text="test", chunk_type="fixed", bm25_score=10.0 - i, rank=i)
        for i in range(1, 6)  # c_1, c_2, c_3, c_4, c_5
    ]
    bm25_res_partial_overlap = [
        RetrievalResult(chunk_id=f"c_{i}", document_id=f"doc_{i}", text="test", chunk_type="fixed", bm25_score=10.0 - i, rank=i)
        for i in [1, 2, 6, 7, 8]  # c_1, c_2 in common (2 / 8 = 0.25)
    ]
    bm25_res_zero_overlap = [
        RetrievalResult(chunk_id=f"c_{i}", document_id=f"doc_{i}", text="test", chunk_type="fixed", bm25_score=10.0 - i, rank=i)
        for i in range(6, 11)  # c_6 .. c_10
    ]

    # Full overlap (5 / 5 = 1.0)
    assert calculate_retriever_agreement(dense_res, bm25_res_full_overlap, k=5) == 1.0

    # Partial overlap (2 / 8 = 0.25)
    assert calculate_retriever_agreement(dense_res, bm25_res_partial_overlap, k=5) == 0.25

    # Zero overlap (0 / 10 = 0.0)
    assert calculate_retriever_agreement(dense_res, bm25_res_zero_overlap, k=5) == 0.0

    # Empty edge case
    assert calculate_retriever_agreement([], [], k=5) == 0.0


def test_confidence_score_and_adaptive_decision():
    """Verify weighted confidence calculation and 3-tier routing (high, medium, low)."""
    # 1. High Confidence Case (High Dense similarity + High Agreement)
    dense_high = [
        RetrievalResult(chunk_id="c_1", document_id="d_1", text="text1", chunk_type="fixed", dense_score=0.88, rank=1),
        RetrievalResult(chunk_id="c_2", document_id="d_2", text="text2", chunk_type="fixed", dense_score=0.72, rank=2),
    ]
    bm25_high = [
        RetrievalResult(chunk_id="c_1", document_id="d_1", text="text1", chunk_type="fixed", bm25_score=15.0, rank=1),
        RetrievalResult(chunk_id="c_2", document_id="d_2", text="text2", chunk_type="fixed", bm25_score=12.0, rank=2),
    ]
    fusion_high = dense_high

    decision_high = calculate_retrieval_confidence(
        dense_results=dense_high,
        bm25_results=bm25_high,
        fusion_results=fusion_high,
        threshold_high=0.70,
        threshold_low=0.40,
    )
    assert decision_high.confidence_score >= 0.70
    assert decision_high.should_rerank is False
    assert decision_high.reranker_tier == "skip"
    assert "High retrieval confidence" in decision_high.reason

    # 2. Medium Confidence Case
    dense_med = [
        RetrievalResult(chunk_id="c_1", document_id="d_1", text="text1", chunk_type="fixed", dense_score=0.74, rank=1),
        RetrievalResult(chunk_id="c_2", document_id="d_2", text="text2", chunk_type="fixed", dense_score=0.68, rank=2),
    ]
    bm25_med = [
        RetrievalResult(chunk_id="c_1", document_id="d_1", text="text1", chunk_type="fixed", bm25_score=10.0, rank=1),
        RetrievalResult(chunk_id="c_9", document_id="d_9", text="text9", chunk_type="fixed", bm25_score=9.0, rank=2),
    ]
    decision_med = calculate_retrieval_confidence(
        dense_results=dense_med,
        bm25_results=bm25_med,
        fusion_results=dense_med,
        threshold_high=0.70,
        threshold_low=0.40,
    )
    assert 0.40 <= decision_med.confidence_score < 0.70
    assert decision_med.should_rerank is True
    assert decision_med.reranker_tier == "lightweight"
    assert decision_med.candidate_k in (3, 5)

    # 3. Low Confidence Case (Low similarity + Zero Agreement + Flat Margin)
    dense_low = [
        RetrievalResult(chunk_id="c_1", document_id="d_1", text="text1", chunk_type="fixed", dense_score=0.52, rank=1),
        RetrievalResult(chunk_id="c_2", document_id="d_2", text="text2", chunk_type="fixed", dense_score=0.52, rank=2),
    ]
    bm25_low = [
        RetrievalResult(chunk_id="c_8", document_id="d_8", text="text8", chunk_type="fixed", bm25_score=4.0, rank=1),
        RetrievalResult(chunk_id="c_9", document_id="d_9", text="text9", chunk_type="fixed", bm25_score=3.0, rank=2),
    ]
    decision_low = calculate_retrieval_confidence(
        dense_results=dense_low,
        bm25_results=bm25_low,
        fusion_results=dense_low,
        threshold_high=0.70,
        threshold_low=0.40,
    )
    assert decision_low.confidence_score < 0.40
    assert decision_low.should_rerank is True
    assert decision_low.reranker_tier == "full"
    assert decision_low.candidate_k in (5, 10)


def test_fast_lexical_semantic_rescorer():
    """Verify sub-millisecond FastLexicalSemanticRescorer."""
    rescorer = FastLexicalSemanticRescorer(dense_weight=0.7, overlap_weight=0.3)
    cands = [
        RetrievalResult(chunk_id="c1", document_id="d1", text="भारत की राजधानी नई दिल्ली है", chunk_type="fixed", dense_score=0.8, rank=1),
        RetrievalResult(chunk_id="c2", document_id="d2", text="क्रिकेट एक लोकप्रिय खेल है", chunk_type="fixed", dense_score=0.75, rank=2),
    ]
    results, lat_ms = rescorer.rerank(query="भारत की राजधानी", candidates=cands)
    assert len(results) == 2
    assert results[0].chunk_id == "c1"  # c1 has exact lexical overlap + higher dense score
    assert lat_ms < 5.0  # Ultra-fast


def test_adaptive_retrieval_service_flow():
    """Verify end-to-end AdaptiveRetrievalService pipeline with mock retrievers."""
    service = AdaptiveRetrievalService(
        heavy_reranker=DummyMockReranker(multiplier=2.0),
        lightweight_reranker=DummyMockReranker(multiplier=1.0),
    )

    # Test with live query
    results, decision, lat = service.adaptive_retrieve(
        query="भारत की राजधानी क्या है?",
        top_k=3,
    )
    assert len(results) <= 3
    assert 0.0 <= decision.confidence_score <= 1.0
    assert decision.reranker_tier in ["skip", "lightweight", "full"]
    assert lat.total > 0
    assert lat.embedding >= 0
    assert lat.retrieval >= 0


# --- API Endpoint Test ---

@pytest.mark.asyncio
async def test_post_adaptive_retrieve_endpoint():
    """Verify live POST /api/adaptive-retrieve endpoint."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        payload = {
            "query": "भारत की राजधानी क्या है?",
            "top_k": 3,
        }
        response = await client.post("/api/adaptive-retrieve", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "भारत की राजधानी क्या है?"
        assert len(data["results"]) == 3
        assert "retrieval_confidence" in data
        assert isinstance(data["reranking_used"], bool)
        assert "latency_ms" in data
        assert "total" in data["latency_ms"]


@pytest.mark.asyncio
async def test_post_adaptive_retrieve_empty_query():
    """Verify 422 error on empty query string."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/adaptive-retrieve", json={"query": "   ", "top_k": 5})
        assert response.status_code == 422
