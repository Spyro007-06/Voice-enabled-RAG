"""Unit and integration tests for Phase 5 Cross-Encoder Reranking and Context Selection."""

from typing import List, Tuple
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.reranking.base import Reranker
from app.reranking.context_selector import ContextSelector
from app.reranking.models import RerankResult
from app.reranking.service import RerankingService
from app.retrieval.models import LatencyBreakdown, RetrievalResult
from app.retrieval.service import RetrievalService


class MockReranker(Reranker):
    """Deterministic mock reranker for unit testing without downloading model weights."""

    def __init__(self):
        self.model_name = "mock/bge-reranker-test"
        self.device = "cpu"

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
    ) -> Tuple[List[RerankResult], float]:
        scored = []
        for orig_idx, c in enumerate(candidates, start=1):
            # Deterministic score based on text length or mock heuristic
            mock_score = 0.95 - (orig_idx * 0.05)
            # Favor candidates matching "capital" in text
            if "राजधानी" in c.text or "capital" in c.text.lower():
                mock_score += 0.2

            res = RerankResult(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                chunk_type=c.chunk_type,
                language=c.language,
                dense_score=c.dense_score,
                bm25_score=c.bm25_score,
                fusion_score=c.fusion_score,
                reranker_score=round(mock_score, 4),
                original_rank=orig_idx,
                rank=1,
                metadata=c.metadata,
            )
            scored.append(res)

        scored.sort(key=lambda x: x.reranker_score, reverse=True)
        for i, item in enumerate(scored, start=1):
            item.rank = i

        return scored, 5.0


@pytest.fixture
def mock_retrieval_service():
    """Mock RetrievalService returning synthetic candidate chunks."""
    class MockRetService(RetrievalService):
        def retrieve(self, query, top_k=20, strategies=None, language=None, fusion_method="rrf", use_cache=True):
            cands = [
                RetrievalResult(
                    chunk_id=f"chunk_doc1_part_{i}",
                    document_id="doc_1",
                    text=f"Doc 1 chunk text part {i} - भारत की राजधानी",
                    chunk_type="fixed",
                    language="hin_Deva",
                    dense_score=0.8 - (i * 0.05),
                    fusion_score=0.03,
                    rank=i,
                    metadata={"doc_num": 1, "is_selected": 1},
                )
                for i in range(1, 4)  # 3 chunks from doc_1
            ]
            cands.append(
                RetrievalResult(
                    chunk_id="chunk_doc2_part_1",
                    document_id="doc_2",
                    text="Doc 2 chunk text - व्यापारिक केंद्र मुंबई",
                    chunk_type="sentence",
                    language="hin_Deva",
                    dense_score=0.6,
                    fusion_score=0.02,
                    rank=4,
                    metadata={"doc_num": 2, "is_selected": 0},
                )
            )
            cands.append(
                RetrievalResult(
                    chunk_id="chunk_doc3_part_1",
                    document_id="doc_3",
                    text="Doc 3 chunk text - Chennai is in Tamil Nadu",
                    chunk_type="sliding_window",
                    language="eng_Latn",
                    dense_score=0.5,
                    fusion_score=0.01,
                    rank=5,
                    metadata={"doc_num": 3, "is_selected": 1},
                )
            )
            lat = LatencyBreakdown(embedding_ms=5.0, dense_retrieval_ms=10.0, total_ms=15.0)
            return cands[:top_k], lat

    return MockRetService()


# --- Unit Tests ---

def test_mock_reranker_scoring():
    """Test reranking reordering and score preservation."""
    reranker = MockReranker()
    cands = [
        RetrievalResult(
            chunk_id="c1",
            document_id="d1",
            text="कुछ अन्य जानकारी",
            chunk_type="fixed",
            fusion_score=0.8,
            rank=1,
            metadata={},
        ),
        RetrievalResult(
            chunk_id="c2",
            document_id="d2",
            text="भारत की राजधानी नई दिल्ली है",
            chunk_type="fixed",
            fusion_score=0.6,
            rank=2,
            metadata={},
        ),
    ]

    reranked, lat = reranker.rerank("राजधानी क्या है?", cands)
    assert len(reranked) == 2
    # c2 contains "राजधानी", so mock reranker scores it higher
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].rank == 1
    assert reranked[0].reranker_score > reranked[1].reranker_score
    assert lat >= 0.0


def test_context_selector_diversity_cap():
    """Test that max chunks per document constraint is strictly enforced."""
    selector = ContextSelector(
        final_top_k=5,
        max_chunks_per_doc=2,
        max_context_chars=6000,
        diversity_enabled=True,
    )
    # 4 chunks from the same document
    reranked_inputs = [
        RerankResult(
            chunk_id=f"c_{i}",
            document_id="doc_repeat",
            text=f"Repeated text content from doc {i}",
            chunk_type="fixed",
            reranker_score=0.9 - (i * 0.05),
            original_rank=i,
            rank=i,
            metadata={},
        )
        for i in range(1, 5)
    ]
    # Add 1 chunk from another document
    reranked_inputs.append(
        RerankResult(
            chunk_id="c_other",
            document_id="doc_other",
            text="Unique text from other doc",
            chunk_type="fixed",
            reranker_score=0.6,
            original_rank=5,
            rank=5,
            metadata={},
        )
    )

    selected, stats, _ = selector.select_context(reranked_inputs, top_k=5)
    doc_repeat_count = sum(1 for c in selected if c.document_id == "doc_repeat")
    assert doc_repeat_count == 2
    assert stats.diversity_filtered_count == 2
    assert any(c.document_id == "doc_other" for c in selected)


def test_context_selector_budget_limit():
    """Test that total character budget is not exceeded."""
    selector = ContextSelector(
        final_top_k=10,
        max_chunks_per_doc=5,
        max_context_chars=100,  # Tight character limit
        diversity_enabled=False,
    )
    reranked_inputs = [
        RerankResult(
            chunk_id=f"c_{i}",
            document_id=f"d_{i}",
            text="A" * 60,  # 60 chars each
            chunk_type="fixed",
            reranker_score=0.9 - (i * 0.05),
            original_rank=i,
            rank=i,
            metadata={},
        )
        for i in range(1, 5)
    ]

    selected, stats, _ = selector.select_context(reranked_inputs, top_k=10)
    # Only 1 chunk fits within 100 chars (60 <= 100, 60+60 = 120 > 100)
    assert len(selected) == 1
    assert stats.total_characters == 60


def test_reranking_service_end_to_end(mock_retrieval_service):
    """Test full RerankingService pipeline flow."""
    reranker = MockReranker()
    selector = ContextSelector(final_top_k=3, max_chunks_per_doc=2, max_context_chars=5000)
    service = RerankingService(
        retrieval_service=mock_retrieval_service,
        reranker=reranker,
        context_selector=selector,
    )

    results, stats, lat = service.rerank_and_select(
        query="भारत की राजधानी",
        top_k=3,
        candidate_k=10,
    )
    assert len(results) == 3
    assert stats.selected_chunks == 3
    assert lat.retrieval_ms > 0
    assert lat.reranking_ms > 0
    assert lat.total_ms >= (lat.retrieval_ms + lat.reranking_ms)


# --- API Endpoint Tests ---

@pytest.mark.asyncio
async def test_post_rerank_endpoint():
    """Verify live POST /api/rerank endpoint."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        payload = {
            "query": "भारत की राजधानी क्या है?",
            "top_k": 3,
            "candidate_k": 10,
            "retrieval_mode": "rrf_hybrid",
        }
        response = await client.post("/api/rerank", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "भारत की राजधानी क्या है?"
        assert len(data["results"]) <= 3
        assert "context_stats" in data
        assert "total_characters" in data["context_stats"]
        assert "latency_ms" in data
        assert "reranking_ms" in data["latency_ms"]


@pytest.mark.asyncio
async def test_post_rerank_empty_query_error():
    """Verify validation error on empty query string."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/rerank", json={"query": "   "})
        assert response.status_code == 422
