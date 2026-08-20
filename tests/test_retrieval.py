"""Unit and integration tests for Phase 4 Hybrid Retrieval, BM25, Fusion, and Evaluation."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.chunking.models import Chunk
from app.main import app
from app.retrieval.bm25 import BM25Retriever, IndicUnicodeTokenizer
from app.retrieval.cache import RetrievalCache
from app.retrieval.dense import DenseRetriever
from app.retrieval.evaluation import RetrievalEvaluator, compute_mrr_at_k, compute_recall_at_k
from app.retrieval.filters import matches_filter
from app.retrieval.fusion import HybridFusion, min_max_normalize
from app.retrieval.models import RetrievalResult
from app.retrieval.qdrant_store import QdrantVectorStore


@pytest.fixture
def mock_chunks():
    """Synthetic test chunks with diverse metadata and strategy types."""
    return [
        Chunk(
            chunk_id="chunk_doc1_fixed_0",
            document_id="doc_1",
            text="नई दिल्ली भारत की राजधानी है।",
            chunk_type="fixed",
            chunk_index=0,
            start_position=0,
            end_position=30,
            token_count=10,
            metadata={"language": "hin_Deva", "query_id": 100, "is_selected": 1},
        ),
        Chunk(
            chunk_id="chunk_doc1_semantic_0",
            document_id="doc_1",
            text="नई दिल्ली में भारत सरकार का मुख्यालय है।",
            chunk_type="semantic",
            chunk_index=0,
            start_position=0,
            end_position=40,
            token_count=12,
            metadata={"language": "hin_Deva", "query_id": 100, "is_selected": 1},
        ),
        Chunk(
            chunk_id="chunk_doc2_fixed_0",
            document_id="doc_2",
            text="मुंबई महाराष्ट्र की राजधानी और व्यापारिक केंद्र है।",
            chunk_type="fixed",
            chunk_index=0,
            start_position=0,
            end_position=50,
            token_count=14,
            metadata={"language": "hin_Deva", "query_id": 200, "is_selected": 0},
        ),
        Chunk(
            chunk_id="chunk_doc3_sentence_0",
            document_id="doc_3",
            text="Chennai is the capital of Tamil Nadu.",
            chunk_type="sentence",
            chunk_index=0,
            start_position=0,
            end_position=37,
            token_count=8,
            metadata={"language": "eng_Latn", "query_id": 300, "is_selected": 1},
        ),
    ]


@pytest.fixture
def mock_retrieval_results():
    """Synthetic dense and BM25 retrieval results for fusion testing."""
    dense_res = [
        RetrievalResult(
            chunk_id="chunk_A",
            document_id="doc_1",
            text="Passage A text",
            chunk_type="fixed",
            language="hin_Deva",
            dense_score=0.85,
            fusion_score=0.85,
            rank=1,
            metadata={"query_id": 100, "is_selected": 1},
        ),
        RetrievalResult(
            chunk_id="chunk_B",
            document_id="doc_2",
            text="Passage B text",
            chunk_type="fixed",
            language="hin_Deva",
            dense_score=0.60,
            fusion_score=0.60,
            rank=2,
            metadata={"query_id": 100, "is_selected": 0},
        ),
    ]
    bm25_res = [
        RetrievalResult(
            chunk_id="chunk_B",
            document_id="doc_2",
            text="Passage B text",
            chunk_type="fixed",
            language="hin_Deva",
            bm25_score=12.5,
            fusion_score=12.5,
            rank=1,
            metadata={"query_id": 100, "is_selected": 0},
        ),
        RetrievalResult(
            chunk_id="chunk_C",
            document_id="doc_3",
            text="Passage C text",
            chunk_type="sentence",
            language="eng_Latn",
            bm25_score=8.0,
            fusion_score=8.0,
            rank=2,
            metadata={"query_id": 200, "is_selected": 1},
        ),
    ]
    return dense_res, bm25_res


# --- Tokenizer & BM25 Tests ---

def test_multilingual_tokenizer():
    """Test Indic and English tokenization."""
    tokenizer = IndicUnicodeTokenizer()
    tokens_hi = tokenizer.tokenize("नई दिल्ली भारत")
    assert "नई" in tokens_hi
    assert "दिल्ली" in tokens_hi

    tokens_en = tokenizer.tokenize("What is a corporation?")
    assert "what" in tokens_en
    assert "corporation" in tokens_en


def test_bm25_build_and_retrieve(mock_chunks):
    """Test BM25 index construction, search, and filtering."""
    retriever = BM25Retriever()
    retriever.build_index_from_chunks(mock_chunks)

    results, lat = retriever.retrieve("नई दिल्ली", top_k=2)
    assert len(results) >= 1
    assert "नई दिल्ली" in results[0].text
    assert lat >= 0.0

    # Strategy filter test
    results_filtered, _ = retriever.retrieve("नई दिल्ली", strategies=["semantic"])
    assert all(r.chunk_type == "semantic" for r in results_filtered)


# --- Score Normalization & Fusion Tests ---

def test_min_max_normalize():
    """Test min-max normalization edge cases."""
    assert min_max_normalize([]) == []
    assert min_max_normalize([5.0]) == [1.0]
    assert min_max_normalize([5.0, 5.0]) == [1.0, 1.0]

    norm = min_max_normalize([10.0, 20.0, 30.0])
    assert norm == [0.0, 0.5, 1.0]


def test_weighted_fusion(mock_retrieval_results):
    """Test weighted score fusion and evidence combination."""
    dense_res, bm25_res = mock_retrieval_results
    fusion = HybridFusion(alpha=0.6)

    fused = fusion.fuse_results(dense_res, bm25_res, top_k=5, method="weighted")
    assert len(fused) == 3  # A, B, C deduplicated
    # chunk_B appears in both sets, its dense and bm25 scores must be preserved
    chunk_b = next(r for r in fused if r.chunk_id == "chunk_B")
    assert chunk_b.dense_score == 0.60
    assert chunk_b.bm25_score == 12.5


def test_rrf_fusion(mock_retrieval_results):
    """Test Reciprocal Rank Fusion (RRF)."""
    dense_res, bm25_res = mock_retrieval_results
    fusion = HybridFusion(rrf_k=60)

    fused = fusion.fuse_results(dense_res, bm25_res, top_k=5, method="rrf")
    assert len(fused) == 3
    assert fused[0].rank == 1
    assert fused[0].fusion_score > 0.0


# --- Cache Tests ---

def test_retrieval_cache(mock_retrieval_results):
    """Test LRU caching and retrieval."""
    dense_res, _ = mock_retrieval_results
    cache = RetrievalCache(max_size=2, enabled=True)

    cache.set("capital india", 5, "all", "hi", "rrf", dense_res)
    hit = cache.get("capital india", 5, "all", "hi", "rrf")
    assert hit is not None
    assert len(hit) == len(dense_res)

    miss = cache.get("different query", 5, "all", "hi", "rrf")
    assert miss is None


# --- Evaluation Metrics Tests ---

def test_evaluation_metrics(mock_retrieval_results):
    """Test Recall@K and MRR@K calculations."""
    dense_res, _ = mock_retrieval_results
    relevant_ids = {"doc_1"}

    r1 = compute_recall_at_k(dense_res, relevant_ids, k=1)
    assert r1 == 1.0

    mrr = compute_mrr_at_k(dense_res, relevant_ids, k=10)
    assert mrr == 1.0

    # Irrelevant query
    r_empty = compute_recall_at_k(dense_res, {"doc_non_existent"}, k=1)
    # Since metadata has is_selected == 1 for chunk_A, fallback checks metadata
    assert r_empty == 1.0


# --- API Endpoint Test ---

@pytest.mark.asyncio
async def test_post_retrieve_endpoint():
    """Verify POST /api/retrieve live behavior."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        payload = {
            "query": "भारत की राजधानी क्या है?",
            "top_k": 3,
            "fusion_method": "rrf",
        }
        response = await client.post("/api/retrieve", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "भारत की राजधानी क्या है?"
        assert data["fusion_method"] == "rrf"
        assert "latency_ms" in data
        assert "embedding_ms" in data["latency_ms"]
        assert "total_ms" in data["latency_ms"]


@pytest.mark.asyncio
async def test_post_retrieve_empty_query_error():
    """Verify validation error on empty query string."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/retrieve", json={"query": "   "})
        assert response.status_code == 422
