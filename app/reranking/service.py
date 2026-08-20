"""Reranking service coordinating candidate retrieval, cross-encoder scoring, and context selection."""

import logging
import time
from functools import lru_cache
from typing import List, Optional, Tuple, Union

from app.config import get_settings
from app.reranking.base import Reranker
from app.reranking.bge_reranker import get_reranker
from app.reranking.context_selector import ContextSelector
from app.reranking.models import ContextSelectionStats, RerankLatencyBreakdown, RerankResult
from app.retrieval.service import RetrievalService, get_retrieval_service

logger = logging.getLogger(__name__)


class RerankingService:
    """Orchestrates candidate retrieval, cross-encoder reranking, and context selection."""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        reranker: Optional[Reranker] = None,
        context_selector: Optional[ContextSelector] = None,
    ):
        self.retrieval_service = retrieval_service or get_retrieval_service()
        self.reranker = reranker or get_reranker()
        self.context_selector = context_selector or ContextSelector()

    def rerank_and_select(
        self,
        query: str,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        strategies: Optional[Union[str, List[str]]] = None,
        language: Optional[str] = None,
        retrieval_mode: str = "rrf_hybrid",
    ) -> Tuple[List[RerankResult], ContextSelectionStats, RerankLatencyBreakdown]:
        """Execute candidate retrieval, cross-encoder reranking, and context selection."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        settings = get_settings()
        k_final = top_k if top_k is not None else settings.RERANK_TOP_K
        k_cands = candidate_k if candidate_k is not None else settings.RERANK_CANDIDATE_K

        # 1. Candidate Retrieval
        fusion_arg = "rrf" if retrieval_mode == "rrf_hybrid" else (
            "weighted" if retrieval_mode == "weighted_hybrid" else retrieval_mode
        )
        candidates, ret_latency = self.retrieval_service.retrieve(
            query=query,
            top_k=k_cands,
            strategies=strategies,
            language=language,
            fusion_method=fusion_arg,
            use_cache=False,
        )
        t_retrieval_ms = ret_latency.total_ms

        if not candidates:
            stats = ContextSelectionStats()
            lat = RerankLatencyBreakdown(
                retrieval_ms=round(t_retrieval_ms, 2),
                reranking_ms=0.0,
                context_selection_ms=0.0,
                total_ms=round(t_retrieval_ms, 2),
            )
            return [], stats, lat

        # 2. Deduplicate Candidates by chunk_id
        unique_cands = []
        seen = set()
        for c in candidates:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                unique_cands.append(c)

        # 3. Cross-Encoder Batch Reranking
        reranked_cands, t_rerank_ms = self.reranker.rerank(
            query=query,
            candidates=unique_cands,
        )

        # 4. Context Selection & Diversity Control
        selected_results, context_stats, t_selection_ms = self.context_selector.select_context(
            reranked_candidates=reranked_cands,
            top_k=k_final,
        )

        t_total_ms = t_retrieval_ms + t_rerank_ms + t_selection_ms

        latency_breakdown = RerankLatencyBreakdown(
            retrieval_ms=round(t_retrieval_ms, 2),
            reranking_ms=round(t_rerank_ms, 2),
            context_selection_ms=round(t_selection_ms, 2),
            total_ms=round(t_total_ms, 2),
        )

        return selected_results, context_stats, latency_breakdown


@lru_cache()
def get_reranking_service() -> RerankingService:
    """Return singleton cached RerankingService instance."""
    return RerankingService()
