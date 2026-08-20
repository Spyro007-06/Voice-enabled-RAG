"""Retrieval evaluation metrics: Recall@1, Recall@5, Recall@10, and MRR@10."""

from typing import Dict, List, Optional, Set
from app.retrieval.models import RetrievalResult


def compute_recall_at_k(retrieved_results: List[RetrievalResult], relevant_ids: Set[str], k: int) -> float:
    """Calculate Recall@K: whether at least one ground-truth relevant document is present in top K results."""
    if not relevant_ids:
        return 0.0

    top_k_candidates = retrieved_results[:k]
    for res in top_k_candidates:
        # Check by document_id or query_id/passage_index combination
        doc_id = res.document_id
        meta = res.metadata or {}
        is_selected = meta.get("is_selected", 0)

        if doc_id in relevant_ids or is_selected == 1:
            return 1.0

    return 0.0


def compute_mrr_at_k(retrieved_results: List[RetrievalResult], relevant_ids: Set[str], k: int = 10) -> float:
    """Calculate Reciprocal Rank (RR@K) for a single query."""
    if not relevant_ids:
        return 0.0

    top_k_candidates = retrieved_results[:k]
    for rank_idx, res in enumerate(top_k_candidates, start=1):
        doc_id = res.document_id
        meta = res.metadata or {}
        is_selected = meta.get("is_selected", 0)

        if doc_id in relevant_ids or is_selected == 1:
            return 1.0 / rank_idx

    return 0.0


class RetrievalEvaluator:
    """Computes aggregated retrieval quality metrics across query evaluation sets."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset internal accumulator metrics."""
        self.queries_evaluated = 0
        self.recall_at_1_scores: List[float] = []
        self.recall_at_5_scores: List[float] = []
        self.recall_at_10_scores: List[float] = []
        self.mrr_at_10_scores: List[float] = []
        self.latencies_ms: List[float] = []

    def evaluate_query(
        self,
        retrieved_results: List[RetrievalResult],
        relevant_doc_ids: Set[str],
        latency_ms: Optional[float] = None,
    ) -> Dict[str, float]:
        """Evaluate a single query retrieval and accumulate scores."""
        r1 = compute_recall_at_k(retrieved_results, relevant_doc_ids, k=1)
        r5 = compute_recall_at_k(retrieved_results, relevant_doc_ids, k=5)
        r10 = compute_recall_at_k(retrieved_results, relevant_doc_ids, k=10)
        mrr10 = compute_mrr_at_k(retrieved_results, relevant_doc_ids, k=10)

        self.queries_evaluated += 1
        self.recall_at_1_scores.append(r1)
        self.recall_at_5_scores.append(r5)
        self.recall_at_10_scores.append(r10)
        self.mrr_at_10_scores.append(mrr10)
        if latency_ms is not None:
            self.latencies_ms.append(latency_ms)

        return {
            "recall@1": r1,
            "recall@5": r5,
            "recall@10": r10,
            "mrr@10": mrr10,
        }

    def get_summary(self) -> Dict[str, float]:
        """Compute mean metrics across all evaluated queries."""
        if not self.queries_evaluated:
            return {
                "queries_evaluated": 0,
                "recall@1": 0.0,
                "recall@5": 0.0,
                "recall@10": 0.0,
                "mrr@10": 0.0,
            }

        return {
            "queries_evaluated": self.queries_evaluated,
            "recall@1": round(sum(self.recall_at_1_scores) / self.queries_evaluated, 4),
            "recall@5": round(sum(self.recall_at_5_scores) / self.queries_evaluated, 4),
            "recall@10": round(sum(self.recall_at_10_scores) / self.queries_evaluated, 4),
            "mrr@10": round(sum(self.mrr_at_10_scores) / self.queries_evaluated, 4),
            "avg_latency_ms": round(sum(self.latencies_ms) / len(self.latencies_ms), 2) if self.latencies_ms else 0.0,
        }
