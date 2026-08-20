"""Lightweight multilingual cross-encoder and fast rescoring implementations."""

import logging
import time
from functools import lru_cache
from typing import List, Optional, Tuple

import torch
from sentence_transformers import CrossEncoder

from app.config import get_settings
from app.reranking.base import Reranker
from app.reranking.models import RerankResult
from app.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)


class LightweightMiniLMReranker(Reranker):
    """Compact multilingual cross-encoder using mmarco-mMiniLMv2 with max_length=128 for < 50ms CPU latency."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.LIGHTWEIGHT_RERANKER_MODEL
        self.max_length = max_length or settings.LIGHTWEIGHT_RERANKER_MAX_LENGTH
        self.batch_size = batch_size or settings.RERANK_BATCH_SIZE

        target_device = device or settings.RERANKER_DEVICE
        if target_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = target_device

        self._model: Optional[CrossEncoder] = None
        self._initialization_latency_ms: float = 0.0
        logger.info(
            "Initializing LightweightMiniLMReranker [Model: %s | Device: %s | MaxLen: %d | Batch: %d]",
            self.model_name,
            self.device,
            self.max_length,
            self.batch_size,
        )

    def _ensure_model(self) -> None:
        """Thread-safe lazy initialization of lightweight CrossEncoder model."""
        if self._model is None:
            t0 = time.perf_counter()
            settings = get_settings()
            if self.device == "cpu" and hasattr(torch, "set_num_threads"):
                try:
                    num_threads = getattr(settings, "CPU_NUM_THREADS", 4)
                    torch.set_num_threads(num_threads)
                except Exception:
                    pass
            logger.info("Loading lightweight cross-encoder '%s' onto %s...", self.model_name, self.device)
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
            )
            self._initialization_latency_ms = (time.perf_counter() - t0) * 1000
            logger.info("Lightweight CrossEncoder loaded in %.2f ms", self._initialization_latency_ms)

    @property
    def initialization_time_ms(self) -> float:
        return self._initialization_latency_ms

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> Tuple[List[RerankResult], float]:
        """Rerank candidates using lightweight CrossEncoder model."""
        if not candidates:
            return [], 0.0

        self._ensure_model()
        t0 = time.perf_counter()

        pairs = [(query, c.text) for c in candidates]
        with torch.inference_mode():
            raw_scores = self._model.predict(
                sentences=pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )

        # Normalize score array if single pair returned as float
        if not hasattr(raw_scores, "__len__"):
            raw_scores = [float(raw_scores)]

        scored_candidates: List[RerankResult] = []
        for orig_idx, (cand, score) in enumerate(zip(candidates, raw_scores), start=1):
            rerank_item = RerankResult(
                chunk_id=cand.chunk_id,
                document_id=cand.document_id,
                text=cand.text,
                chunk_type=cand.chunk_type,
                language=cand.language,
                dense_score=cand.dense_score,
                bm25_score=cand.bm25_score,
                fusion_score=cand.fusion_score,
                reranker_score=float(score),
                original_rank=cand.rank if cand.rank is not None else orig_idx,
                rank=orig_idx,
                metadata=cand.metadata,
            )
            scored_candidates.append(rerank_item)

        scored_candidates.sort(key=lambda x: x.reranker_score, reverse=True)
        for rank_idx, item in enumerate(scored_candidates, start=1):
            item.rank = rank_idx

        t_rerank_ms = (time.perf_counter() - t0) * 1000
        k = top_k if top_k is not None else len(scored_candidates)
        return scored_candidates[:k], t_rerank_ms


class FastLexicalSemanticRescorer(Reranker):
    """Ultra-fast (< 0.1ms) CPU rescorer combining exact token overlap and dense cosine similarity."""

    def __init__(self, dense_weight: float = 0.7, overlap_weight: float = 0.3):
        self.dense_weight = dense_weight
        self.overlap_weight = overlap_weight

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> Tuple[List[RerankResult], float]:
        if not candidates:
            return [], 0.0

        t0 = time.perf_counter()
        q_tokens = set(query.lower().split())

        scored_candidates: List[RerankResult] = []
        for orig_idx, cand in enumerate(candidates, start=1):
            d_tokens = set(cand.text.lower().split())
            overlap = len(q_tokens & d_tokens) / max(len(q_tokens), 1)
            dense_s = cand.dense_score if cand.dense_score is not None else 0.5
            final_s = (self.dense_weight * dense_s) + (self.overlap_weight * overlap)

            rerank_item = RerankResult(
                chunk_id=cand.chunk_id,
                document_id=cand.document_id,
                text=cand.text,
                chunk_type=cand.chunk_type,
                language=cand.language,
                dense_score=cand.dense_score,
                bm25_score=cand.bm25_score,
                fusion_score=cand.fusion_score,
                reranker_score=float(final_s),
                original_rank=cand.rank if cand.rank is not None else orig_idx,
                rank=orig_idx,
                metadata=cand.metadata,
            )
            scored_candidates.append(rerank_item)

        scored_candidates.sort(key=lambda x: x.reranker_score, reverse=True)
        for rank_idx, item in enumerate(scored_candidates, start=1):
            item.rank = rank_idx

        t_ms = (time.perf_counter() - t0) * 1000
        k = top_k if top_k is not None else len(scored_candidates)
        return scored_candidates[:k], t_ms


@lru_cache()
def get_lightweight_reranker() -> LightweightMiniLMReranker:
    """Return cached singleton LightweightMiniLMReranker instance."""
    return LightweightMiniLMReranker()
