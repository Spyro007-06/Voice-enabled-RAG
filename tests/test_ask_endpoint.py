"""Integration tests for Phase 6.3 End-to-End POST /api/ask Endpoint."""

from unittest.mock import AsyncMock, patch
import pytest
from httpx import ASGITransport, AsyncClient

from app.generation.models import GenerationConfig, GenerationResult, GenerationTelemetry
from app.generation.provider import MockGenerationProvider
from app.generation.service import GenerationService
from app.main import app
from app.reranking.adaptive import AdaptiveDecision, AdaptiveRetrievalService
from app.reranking.models import AdaptiveLatencyBreakdown, RerankResult


# --- 1. Successful Queries Across Languages ---

@pytest.fixture(scope="module", autouse=True)
def warmup_models():
    """Pre-warm embeddings, BM25 index, and retrieval service to avoid cold-load timeouts.

    Phase 6.16: BM25 index grew from ~29MB to ~42MB (now includes English chunks).
    Pre-loading it here ensures the index is in memory before any timed test runs.
    """
    from app.embeddings.multilingual_e5 import get_embedding_provider
    from app.reranking.lightweight_reranker import get_lightweight_reranker
    from app.retrieval.bm25 import get_bm25_retriever
    from app.retrieval.service import get_retrieval_service

    # 1. Warm embedding model
    try:
        provider = get_embedding_provider()
        provider.embed_query("warmup query")
    except Exception:
        pass

    # 2. Warm BM25 index (Phase 6.16: index is now 42MB, takes time to deserialize)
    try:
        bm25 = get_bm25_retriever()
        bm25.ensure_loaded()
    except Exception:
        pass

    # 3. Warm full retrieval service (initializes shared Qdrant client + BM25 retriever)
    try:
        svc = get_retrieval_service()
        svc.bm25_retriever.ensure_loaded()
    except Exception:
        pass

    # 4. Warm reranker model
    try:
        reranker = get_lightweight_reranker()
        reranker._get_model()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_ask_endpoint_english_query():
    """Verify end-to-end RAG execution for English query."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        payload = {
            "query": "What is the capital of Goa?",
            "top_k": 3,
        }
        response = await client.post("/api/ask", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["query"] == "What is the capital of Goa?"
        assert isinstance(data["grounded"], bool)
        assert len(data["answer"]) > 0
        assert isinstance(data["citations"], list)
        if data["grounded"]:
            assert len(data["citations"]) > 0
        else:
            assert len(data["citations"]) == 0
        assert 0.0 <= data["retrieval_confidence"] <= 1.0
        assert isinstance(data["reranking_used"], bool)
        assert data["model"] is not None
        assert data["error"] is None

        # Context summary
        summary = data["retrieved_context_summary"]
        assert summary["chunks_count"] > 0
        assert summary["total_characters"] > 0
        assert len(summary["chunk_ids"]) > 0

        # Latency breakdown telemetry
        lat = data["latency_ms"]
        assert "embedding" in lat
        assert "retrieval" in lat
        assert "reranking" in lat
        assert "context_selection" in lat
        assert "prompt_construction" in lat
        assert "generation" in lat
        assert "total" in lat
        assert lat["total"] > 0


@pytest.mark.asyncio
async def test_ask_endpoint_hindi_query():
    """Verify end-to-end RAG execution for Hindi query preserving Unicode script."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        payload = {
            "query": "भारत की राजधानी क्या है?",
            "language": "hi",
            "top_k": 3,
        }
        response = await client.post("/api/ask", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["query"] == "भारत की राजधानी क्या है?"
        assert isinstance(data["grounded"], bool)
        assert len(data["answer"]) > 0
        assert data["error"] is None
        assert data["latency_ms"]["total"] > 0


@pytest.mark.asyncio
async def test_ask_endpoint_tamil_query():
    """Verify end-to-end RAG execution for Tamil query preserving Unicode script."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        payload = {
            "query": "கோவாவின் தலைநகரம் எது?",
            "top_k": 2,
        }
        response = await client.post("/api/ask", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["query"] == "கோவாவின் தலைநகரம் எது?"
        assert isinstance(data["grounded"], bool)
        assert len(data["answer"]) > 0
        assert data["error"] is None
        assert data["latency_ms"]["total"] > 0


# --- 2. Query Validation ---

@pytest.mark.asyncio
async def test_ask_endpoint_empty_query_validation():
    """Verify 422 HTTP validation error on empty or whitespace query."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/ask", json={"query": "   "})
        assert response.status_code == 422


# --- 3. No-Context Guard ---

@pytest.mark.asyncio
async def test_ask_endpoint_no_context_guard():
    """Verify that when no relevant context is retrieved, LLM is not called and graceful refusal is returned."""
    mock_adaptive = AdaptiveRetrievalService()
    
    # Mock adaptive retrieval to return empty candidate list
    with patch(
        "app.api.routes.get_adaptive_retrieval_service",
        return_value=mock_adaptive,
    ):
        with patch.object(
            mock_adaptive,
            "adaptive_retrieve",
            return_value=(
                [],
                AdaptiveDecision(
                    confidence_score=0.1,
                    dense_confidence=0.1,
                    retriever_agreement=0.0,
                    score_margin=0.0,
                    should_rerank=False,
                    reranker_tier="skip",
                    candidate_k=0,
                    reason="Low confidence score",
                ),
                AdaptiveLatencyBreakdown(embedding=5.0, retrieval=10.0, reranking=0.0, context=0.0, total=15.0),
            ),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                payload = {"query": "Unanswerable query with zero matching documents"}
                response = await client.post("/api/ask", json=payload)
                assert response.status_code == 200
                data = response.json()

                assert data["grounded"] is False
                assert "information" in data["answer"].lower()
                assert data["citations"] == []
                assert data["retrieved_context_summary"]["chunks_count"] == 0
                assert data["latency_ms"]["generation"] == 0.0
                assert data["latency_ms"]["prompt_construction"] == 0.0
                assert data["latency_ms"]["total"] > 0


# --- 4. Citation Propagation ---

@pytest.mark.asyncio
async def test_ask_endpoint_citation_propagation():
    """Verify citations from retrieved chunks properly propagate to the response."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        payload = {
            "query": "What is the capital of Goa?",
            "top_k": 2,
        }
        response = await client.post("/api/ask", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data["grounded"], bool)
        # Chunk IDs must match citations
        summary_ids = data["retrieved_context_summary"]["chunk_ids"]
        assert len(summary_ids) > 0
        if data["grounded"]:
            assert len(data["citations"]) > 0
            for cite in data["citations"]:
                assert cite in summary_ids or cite.startswith("chunk_") or cite.startswith("doc_")
        else:
            assert data["citations"] == []


# --- 5. Generation Failure and Timeout Handling ---

@pytest.mark.asyncio
async def test_ask_endpoint_generation_failure_handling():
    """Verify structured error response when generation provider fails."""
    failing_provider = MockGenerationProvider(should_fail=True, failure_message="LLM provider cluster unreachable")
    failing_service = GenerationService(provider=failing_provider)

    with patch("app.api.routes.get_generation_service", return_value=failing_service):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            payload = {"query": "Test query for failing LLM provider"}
            response = await client.post("/api/ask", json=payload)
            assert response.status_code == 200
            data = response.json()

            assert data["grounded"] is False
            assert data["error"] is not None
            assert "LLM provider cluster unreachable" in data["error"]
            assert data["latency_ms"]["total"] > 0


@pytest.mark.asyncio
async def test_ask_endpoint_generation_timeout_handling():
    """Verify structured timeout response when generation exceeds configured timeout."""
    slow_provider = MockGenerationProvider(simulated_latency_ms=300.0)
    slow_service = GenerationService(provider=slow_provider)

    with patch("app.api.routes.get_generation_service", return_value=slow_service):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            payload = {
                "query": "Test timeout",
                "config": {"timeout": 0.05},  # 50ms timeout against 300ms simulated latency
            }
            response = await client.post("/api/ask", json=payload)
            assert response.status_code == 200
            data = response.json()

            assert data["grounded"] is False
            assert data["error"] is not None
            assert "timed out" in data["error"]
            assert data["latency_ms"]["total"] >= 40.0
