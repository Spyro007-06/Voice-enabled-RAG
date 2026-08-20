"""Multilingual BGE Reranker v2 (m3) implementation using SentenceTransformers CrossEncoder."""

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


class BGEReranker(Reranker):
    """Cross-encoder reranker using BAAI/bge-reranker-v2-m3 supporting 100+ languages including Indic."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.RERANKER_MODEL
        self.batch_size = batch_size or settings.RERANK_BATCH_SIZE

        # Device selection & fallback
        req_dev = (device or settings.RERANKER_DEVICE).lower().strip()
        if req_dev == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif req_dev == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

        logger.info(
            "Initializing BGEReranker [Model: %s | Device: %s | Batch size: %d]",
            self.model_name,
            self.device,
            self.batch_size,
        )

        self._model: Optional[CrossEncoder] = None
        self._init_time_ms: float = 0.0

    def _ensure_model(self) -> CrossEncoder:
        """Lazy-load cross-encoder model on first inference call."""
        if self._model is None:
            t0 = time.perf_counter()
            logger.info("Loading cross-encoder reranker '%s' onto %s...", self.model_name, self.device)
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=512,
            )
            self._init_time_ms = (time.perf_counter() - t0) * 1000
            logger.info("CrossEncoder loaded in %.2f ms", self._init_time_ms)
        return self._model

    @property
    def initialization_time_ms(self) -> float:
        return self._init_time_ms

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
    ) -> Tuple[List[RerankResult], float]:
        """Batch cross-encoder inference for query and candidate text pairs."""
        if not candidates or not query or not query.strip():
            return [], 0.0

        model = self._ensure_model()
        t0 = time.perf_counter()

        # Build pair list: (query, passage_text)
        pairs = [(query.strip(), c.text.strip()) for c in candidates]

        # Batch inference
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        t_inference_ms = (time.perf_counter() - t0) * 1000

        # Construct RerankResult models
        scored_results: List[RerankResult] = []
        for orig_rank, (cand, score) in enumerate(zip(candidates, scores), start=1):
            rerank_score = float(score)
            res = RerankResult(
                chunk_id=cand.chunk_id,
                document_id=cand.document_id,
                text=cand.text,
                chunk_type=cand.chunk_type,
                language=cand.language,
                dense_score=cand.dense_score,
                bm25_score=cand.bm25_score,
                fusion_score=cand.fusion_score,
                reranker_score=round(rerank_score, 6),
                original_rank=orig_rank,
                rank=1,
                metadata=cand.metadata,
            )
            scored_results.append(res)

        # Sort descending by reranker_score
        scored_results.sort(key=lambda x: x.reranker_score, reverse=True)
        for i, item in enumerate(scored_results, start=1):
            item.rank = i

        return scored_results, t_inference_ms


@lru_cache()
def get_reranker() -> Reranker:
    """Return singleton cached BGEReranker instance."""
    return BGEReranker()
