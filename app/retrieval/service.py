"""Retrieval service coordinating dense vector, BM25 lexical, and hybrid fusion pipelines."""

import concurrent.futures
import logging
import time
from functools import lru_cache
from typing import List, Optional, Tuple, Union

from app.config import get_settings
from app.retrieval.bm25 import BM25Retriever, get_bm25_retriever
from app.retrieval.cache import RetrievalCache
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import HybridFusion
from app.retrieval.models import LatencyBreakdown, RetrievalResult

logger = logging.getLogger(__name__)


class RetrievalService:
    """End-to-end multi-strategy hybrid retrieval coordinator with parallel execution and timeouts."""

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        fusion: Optional[HybridFusion] = None,
        cache: Optional[RetrievalCache] = None,
    ):
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.bm25_retriever = bm25_retriever or get_bm25_retriever()
        self.fusion = fusion or HybridFusion()
        self.cache = cache or RetrievalCache()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        strategies: Optional[Union[str, List[str]]] = None,
        language: Optional[str] = None,
        fusion_method: str = "rrf",
        use_cache: bool = True,
    ) -> Tuple[List[RetrievalResult], LatencyBreakdown]:
        """Execute full hybrid retrieval pipeline with granular stage latency telemetry, timeouts, and parallelization."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        t_total_start = time.perf_counter()
        settings = get_settings()

        # 1. Check Query Cache
        if use_cache:
            cached_results = self.cache.get(
                query=query,
                top_k=top_k,
                strategies=strategies,
                language=language,
                fusion_method=fusion_method,
            )
            if cached_results is not None:
                t_total_ms = (time.perf_counter() - t_total_start) * 1000
                latency = LatencyBreakdown(
                    embedding_ms=0.0,
                    dense_retrieval_ms=0.0,
                    bm25_retrieval_ms=0.0,
                    fusion_ms=0.0,
                    total_ms=round(t_total_ms, 2),
                    timeout_stage=None,
                    fallback_used=False,
                    parallel_execution=False,
                )
                return cached_results, latency

        clean_fusion = fusion_method.lower().strip()
        dense_results: List[RetrievalResult] = []
        bm25_results: List[RetrievalResult] = []
        t_embed_ms = 0.0
        t_dense_ms = 0.0
        t_bm25_ms = 0.0
        timeout_stage: Optional[str] = None
        fallback_used = False
        is_parallel = getattr(settings, "PARALLEL_RETRIEVAL_ENABLED", True)

        try:
            total_deadline_ms = float(getattr(settings, "TOTAL_RETRIEVAL_DEADLINE_MS", 200.0))
        except (TypeError, ValueError):
            total_deadline_ms = 200.0
        try:
            qdrant_timeout_ms = float(getattr(settings, "QDRANT_TIMEOUT_MS", 157.0))
        except (TypeError, ValueError):
            qdrant_timeout_ms = 157.0
        try:
            bm25_timeout_ms = float(getattr(settings, "BM25_TIMEOUT_MS", 136.0))
        except (TypeError, ValueError):
            bm25_timeout_ms = 136.0

        total_deadline_s = total_deadline_ms / 1000.0
        deadline = t_total_start + total_deadline_s

        run_dense = clean_fusion in ("rrf", "weighted", "dense_only")
        run_bm25 = clean_fusion in ("rrf", "weighted", "bm25_only") and self.bm25_retriever.is_indexed()

        if is_parallel and run_dense and run_bm25:
            # Parallel execution of Dense and BM25 branches
            candidate_k = max(top_k * 2, 20)

            def _fetch_dense():
                return self.dense_retriever.retrieve(
                    query=query, top_k=candidate_k, strategies=strategies, language=language
                )

            def _fetch_bm25():
                return self.bm25_retriever.retrieve(
                    query=query, top_k=candidate_k, strategies=strategies, language=language
                )

            dense_future = self._executor.submit(_fetch_dense)
            bm25_future = self._executor.submit(_fetch_bm25)

            remaining_s = max(0.001, deadline - time.perf_counter())
            qdrant_timeout_s = min(qdrant_timeout_ms / 1000.0, remaining_s)
            bm25_timeout_s = min(bm25_timeout_ms / 1000.0, remaining_s)
            wait_timeout_s = min(max(qdrant_timeout_s, bm25_timeout_s), remaining_s)

            done, not_done = concurrent.futures.wait(
                [dense_future, bm25_future],
                timeout=wait_timeout_s,
                return_when=concurrent.futures.ALL_COMPLETED,
            )

            # Process Dense branch
            if dense_future in done:
                try:
                    dense_results, t_embed_ms, t_dense_ms = dense_future.result()
                except Exception as ex:
                    logger.warning("Dense retrieval error (%s); falling back to BM25.", ex)
                    dense_results = []
                    fallback_used = True
            else:
                logger.warning("Dense Qdrant retrieval timed out after %.2f s; falling back to BM25.", qdrant_timeout_s)
                dense_results = []
                timeout_stage = "qdrant"
                fallback_used = True

            # Process BM25 branch
            if bm25_future in done:
                try:
                    bm25_results, t_bm25_ms = bm25_future.result()
                except Exception as ex:
                    logger.warning("BM25 retrieval error (%s); falling back to Dense.", ex)
                    bm25_results = []
                    fallback_used = True
            else:
                logger.warning("BM25 retrieval timed out after %.2f s; falling back to Dense.", bm25_timeout_s)
                bm25_results = []
                timeout_stage = "bm25" if not timeout_stage else f"{timeout_stage},bm25"
                fallback_used = True

        else:
            # Sequential execution
            if run_dense:
                try:
                    dense_results, t_embed_ms, t_dense_ms = self.dense_retriever.retrieve(
                        query=query,
                        top_k=max(top_k * 2, 20),
                        strategies=strategies,
                        language=language,
                    )
                except Exception as ex:
                    logger.warning("Dense retrieval error in sequential mode: %s", ex)
                    dense_results = []
                    fallback_used = True

            if run_bm25:
                try:
                    bm25_results, t_bm25_ms = self.bm25_retriever.retrieve(
                        query=query,
                        top_k=max(top_k * 2, 20),
                        strategies=strategies,
                        language=language,
                    )
                except Exception as ex:
                    logger.warning("BM25 retrieval error in sequential mode: %s", ex)
                    bm25_results = []
                    fallback_used = True

        # 4. Fusion and Deduplication
        t_fusion_start = time.perf_counter()
        final_results = self.fusion.fuse_results(
            dense_results=dense_results,
            bm25_results=bm25_results,
            top_k=top_k,
            method=clean_fusion,
        )
        t_fusion_ms = (time.perf_counter() - t_fusion_start) * 1000

        # 5. Populate Cache
        if use_cache and final_results:
            self.cache.set(
                query=query,
                top_k=top_k,
                strategies=strategies,
                language=language,
                fusion_method=fusion_method,
                results=final_results,
            )

        t_total_ms = (time.perf_counter() - t_total_start) * 1000

        latency = LatencyBreakdown(
            embedding_ms=round(t_embed_ms, 2),
            dense_retrieval_ms=round(t_dense_ms, 2),
            bm25_retrieval_ms=round(t_bm25_ms, 2),
            fusion_ms=round(t_fusion_ms, 2),
            total_ms=round(t_total_ms, 2),
            timeout_stage=timeout_stage,
            fallback_used=fallback_used,
            parallel_execution=is_parallel and run_dense and run_bm25,
        )

        return final_results, latency


@lru_cache()
def get_retrieval_service() -> RetrievalService:
    """Return singleton cached RetrievalService instance."""
    return RetrievalService()
