"""Score normalization, rank fusion algorithms (RRF & Weighted), and deduplication."""

from collections import defaultdict
from typing import Dict, List, Optional
from app.config import get_settings
from app.retrieval.models import RetrievalResult


def min_max_normalize(scores: List[float]) -> List[float]:
    """Normalize a list of raw scores to [0.0, 1.0] range using Min-Max scaling with zero-division safety."""
    if not scores:
        return []
    min_val = min(scores)
    max_val = max(scores)
    spread = max_val - min_val

    if spread <= 1e-9:
        # All scores identical or single score: return 1.0 for each
        return [1.0] * len(scores)

    return [(s - min_val) / spread for s in scores]


class HybridFusion:
    """Combines dense and lexical retrieval results via Weighted Score Fusion or Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        alpha: Optional[float] = None,
        rrf_k: Optional[int] = None,
    ):
        settings = get_settings()
        self.alpha = alpha if alpha is not None else settings.HYBRID_ALPHA
        self.rrf_k = rrf_k if rrf_k is not None else settings.RRF_K

    def fuse_results(
        self,
        dense_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        top_k: int = 10,
        method: str = "rrf",
    ) -> List[RetrievalResult]:
        """Fuse and deduplicate dense and BM25 results into a unified ranked list.
        
        Methods:
        - 'rrf': Reciprocal Rank Fusion based on rank positions.
        - 'weighted': Min-max normalized linear score combination.
        - 'dense_only': Rank exclusively by dense score.
        - 'bm25_only': Rank exclusively by BM25 score.
        """
        clean_method = method.lower().strip()

        if clean_method == "dense_only":
            return dense_results[:top_k]
        elif clean_method == "bm25_only":
            return bm25_results[:top_k]
        elif clean_method == "weighted":
            return self._weighted_score_fusion(dense_results, bm25_results, top_k)
        else:  # Default to RRF
            return self._reciprocal_rank_fusion(dense_results, bm25_results, top_k)

    def _weighted_score_fusion(
        self,
        dense_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Weighted score fusion with min-max normalization."""
        # 1. Normalize dense scores
        dense_raw = [r.dense_score if r.dense_score is not None else 0.0 for r in dense_results]
        dense_norm = min_max_normalize(dense_raw)
        for r, n in zip(dense_results, dense_norm):
            r.normalized_dense_score = n

        # 2. Normalize BM25 scores
        bm25_raw = [r.bm25_score if r.bm25_score is not None else 0.0 for r in bm25_results]
        bm25_norm = min_max_normalize(bm25_raw)
        for r, n in zip(bm25_results, bm25_norm):
            r.normalized_bm25_score = n

        # 3. Combine evidence by chunk_id
        combined_records: Dict[str, Dict] = {}

        for r in dense_results:
            combined_records[r.chunk_id] = {
                "base_result": r,
                "dense_score": r.dense_score,
                "norm_dense": r.normalized_dense_score or 0.0,
                "bm25_score": None,
                "norm_bm25": 0.0,
            }

        for r in bm25_results:
            if r.chunk_id in combined_records:
                combined_records[r.chunk_id]["bm25_score"] = r.bm25_score
                combined_records[r.chunk_id]["norm_bm25"] = r.normalized_bm25_score or 0.0
            else:
                combined_records[r.chunk_id] = {
                    "base_result": r,
                    "dense_score": None,
                    "norm_dense": 0.0,
                    "bm25_score": r.bm25_score,
                    "norm_bm25": r.normalized_bm25_score or 0.0,
                }

        # 4. Compute weighted fusion score: alpha * norm_dense + (1 - alpha) * norm_bm25
        ranked_items = []
        for chunk_id, data in combined_records.items():
            fusion_score = (self.alpha * data["norm_dense"]) + ((1.0 - self.alpha) * data["norm_bm25"])
            base = data["base_result"]

            final_result = RetrievalResult(
                chunk_id=base.chunk_id,
                document_id=base.document_id,
                text=base.text,
                chunk_type=base.chunk_type,
                language=base.language,
                dense_score=data["dense_score"],
                bm25_score=data["bm25_score"],
                normalized_dense_score=data["norm_dense"],
                normalized_bm25_score=data["norm_bm25"],
                fusion_score=round(fusion_score, 6),
                rank=1,
                metadata=base.metadata,
            )
            ranked_items.append(final_result)

        # Sort descending by fusion score
        ranked_items.sort(key=lambda x: x.fusion_score, reverse=True)
        for i, item in enumerate(ranked_items, start=1):
            item.rank = i

        return ranked_items[:top_k]

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Reciprocal Rank Fusion (RRF): RRF(d) = sum(1 / (k + rank(d)))."""
        rrf_scores: Dict[str, float] = defaultdict(float)
        item_evidence: Dict[str, Dict] = {}

        # 1. Accumulate dense ranks
        for rank_idx, r in enumerate(dense_results, start=1):
            score_contribution = 1.0 / (self.rrf_k + rank_idx)
            rrf_scores[r.chunk_id] += score_contribution
            if r.chunk_id not in item_evidence:
                item_evidence[r.chunk_id] = {
                    "base_result": r,
                    "dense_score": r.dense_score,
                    "bm25_score": None,
                }

        # 2. Accumulate BM25 ranks
        for rank_idx, r in enumerate(bm25_results, start=1):
            score_contribution = 1.0 / (self.rrf_k + rank_idx)
            rrf_scores[r.chunk_id] += score_contribution
            if r.chunk_id in item_evidence:
                item_evidence[r.chunk_id]["bm25_score"] = r.bm25_score
            else:
                item_evidence[r.chunk_id] = {
                    "base_result": r,
                    "dense_score": None,
                    "bm25_score": r.bm25_score,
                }

        # 3. Assemble and rank
        ranked_items = []
        for chunk_id, total_rrf in rrf_scores.items():
            data = item_evidence[chunk_id]
            base = data["base_result"]

            final_result = RetrievalResult(
                chunk_id=base.chunk_id,
                document_id=base.document_id,
                text=base.text,
                chunk_type=base.chunk_type,
                language=base.language,
                dense_score=data["dense_score"],
                bm25_score=data["bm25_score"],
                normalized_dense_score=None,
                normalized_bm25_score=None,
                fusion_score=round(total_rrf, 6),
                rank=1,
                metadata=base.metadata,
            )
            ranked_items.append(final_result)

        # Sort descending by RRF score
        ranked_items.sort(key=lambda x: x.fusion_score, reverse=True)
        for i, item in enumerate(ranked_items, start=1):
            item.rank = i

        return ranked_items[:top_k]
