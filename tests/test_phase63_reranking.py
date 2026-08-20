"""Tests for Phase 6.3 production reranking optimization and latency architecture."""

import pytest
from unittest.mock import MagicMock, patch

from app.config import get_settings
from app.generation.models import AskRequest
from app.reranking.adaptive import (
    AdaptiveDecision,
    AdaptiveRetrievalService,
    calculate_retrieval_confidence,
    calculate_retriever_agreement,
)
from app.reranking.lightweight_reranker import (
    LightweightMiniLMReranker,
    get_lightweight_reranker,
)
from app.retrieval.models import RetrievalResult


def _make_dummy_result(chunk_id: str, dense_score: float = 0.85, bm25_score: float = 12.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        text=f"Text for {chunk_id}",
        chunk_type="sentence",
        language="hi",
        dense_score=dense_score,
        bm25_score=bm25_score,
        fusion_score=0.016,
        rank=1,
    )


class TestPhase63RerankingOptimization:
    """Test suite for Phase 6.3 production reranker policies, calibration, and lazy loading."""

    def test_settings_production_defaults(self):
        """Verify production reranker configuration defaults."""
        settings = get_settings()
        assert settings.RERANKER_PROVIDER == "minilm"
        assert "mMiniLMv2" in settings.LIGHTWEIGHT_RERANKER_MODEL
        assert settings.HEAVY_RERANKER_ENABLED is False
        assert settings.RERANKER_MAX_LENGTH == 128
        assert settings.RERANKER_DEFAULT_CANDIDATE_K == 5
        assert settings.RERANKER_HIGH_CONFIDENCE_SKIP is True

    def test_confidence_calibration_bounds(self):
        """Verify confidence is always bounded in [0.0, 1.0] across edge score inputs."""
        # 1. Very high dense score
        d1 = [_make_dummy_result("c1", dense_score=0.98), _make_dummy_result("c2", dense_score=0.70)]
        b1 = [_make_dummy_result("c1")]
        f1 = [_make_dummy_result("c1")]
        dec1 = calculate_retrieval_confidence(d1, b1, f1)
        assert 0.0 <= dec1.confidence_score <= 1.0
        assert dec1.confidence_score >= 0.70
        assert dec1.should_rerank is False
        assert dec1.reranker_tier == "skip"

        # 2. Negative / zero dense score
        d2 = [_make_dummy_result("c1", dense_score=0.10)]
        dec2 = calculate_retrieval_confidence(d2, [], d2)
        assert 0.0 <= dec2.confidence_score <= 1.0
        assert dec2.dense_confidence == 0.0

        # 3. Empty results
        dec3 = calculate_retrieval_confidence([], [], [])
        assert dec3.confidence_score == 0.0
        assert dec3.dense_confidence == 0.0
        assert dec3.retriever_agreement == 0.0

    def test_high_confidence_skip_policy(self):
        """Verify high confidence triggers skip fast-path with candidate_k=0 and reranker='none'."""
        dense = [_make_dummy_result(f"c{i}", dense_score=0.90 - i * 0.05) for i in range(5)]
        bm25 = [_make_dummy_result(f"c{i}") for i in range(5)]
        fusion = [_make_dummy_result(f"c{i}") for i in range(5)]

        decision = calculate_retrieval_confidence(dense, bm25, fusion, threshold_high=0.65)
        assert decision.should_rerank is False
        assert decision.reranker_tier == "skip"
        assert decision.reranker == "none"
        assert decision.reranking_used is False
        assert decision.candidate_k == 0

    def test_medium_confidence_minilm_policy(self):
        """Verify medium confidence uses lightweight MiniLM with candidate_k=5."""
        dense = [_make_dummy_result("c1", dense_score=0.78), _make_dummy_result("c2", dense_score=0.70)]
        bm25 = [_make_dummy_result("c99")]  # Low agreement
        fusion = dense

        decision = calculate_retrieval_confidence(
            dense, bm25, fusion, threshold_high=0.85, threshold_low=0.25
        )
        assert decision.should_rerank is True
        assert decision.reranker_tier == "lightweight"
        assert decision.reranker == "minilm"
        assert decision.reranking_used is True
        assert decision.candidate_k in (3, 5)

    def test_lazy_reranker_instantiation(self):
        """Verify BGE heavy reranker is NOT instantiated during AdaptiveRetrievalService initialization."""
        service = AdaptiveRetrievalService()
        # Internal heavy reranker attribute must remain None until property access
        assert service._heavy_reranker is None
        assert service._lightweight_reranker is None

    def test_lightweight_minilm_initialization(self):
        """Verify LightweightMiniLMReranker sets correct params."""
        reranker = LightweightMiniLMReranker(max_length=128, batch_size=16)
        assert reranker.max_length == 128
        assert reranker.batch_size == 16
        assert reranker._model is None  # Lazy loading check

    @pytest.mark.asyncio
    async def test_adaptive_retrieve_skip_path_execution(self):
        """Verify adaptive_retrieve fast path preserves metadata without neural reranking."""
        mock_retrieval = MagicMock()
        dummy_dense = [_make_dummy_result(f"chunk_{i}", dense_score=0.92 - i * 0.05) for i in range(5)]
        mock_retrieval.dense_retriever.retrieve.return_value = (dummy_dense, 5.0, 10.0)
        mock_retrieval.bm25_retriever.retrieve.return_value = (dummy_dense, 5.0)
        mock_retrieval.fusion.fuse_results.return_value = dummy_dense

        service = AdaptiveRetrievalService(retrieval_service=mock_retrieval)
        results, decision, latency = service.adaptive_retrieve("test high confidence query", top_k=3)

        assert decision.should_rerank is False
        assert decision.reranker == "none"
        assert latency.reranking == 0.0
        assert len(results) <= 3
        assert results[0].chunk_id == "chunk_0"

    @pytest.mark.asyncio
    async def test_adaptive_retrieve_minilm_path_execution(self):
        """Verify adaptive_retrieve medium confidence path routes to MiniLM reranker."""
        mock_retrieval = MagicMock()
        mock_minilm = MagicMock()
        dummy_cands = [_make_dummy_result(f"chunk_{i}", dense_score=0.68) for i in range(5)]
        mock_retrieval.dense_retriever.retrieve.return_value = (dummy_cands, 5.0, 10.0)
        mock_retrieval.bm25_retriever.retrieve.return_value = ([], 5.0)
        mock_retrieval.fusion.fuse_results.return_value = dummy_cands

        mock_minilm.rerank.return_value = (
            [
                _make_dummy_result("chunk_1", dense_score=0.68),
                _make_dummy_result("chunk_0", dense_score=0.68),
            ],
            35.5,
        )

        service = AdaptiveRetrievalService(
            retrieval_service=mock_retrieval,
            lightweight_reranker=mock_minilm,
        )
        results, decision, latency = service.adaptive_retrieve(
            "test medium confidence query", top_k=2, threshold_high=0.90, threshold_low=0.20
        )

        assert decision.should_rerank is True
        assert decision.reranker == "minilm"
        mock_minilm.rerank.assert_called_once()
        assert latency.reranking == 35.5
