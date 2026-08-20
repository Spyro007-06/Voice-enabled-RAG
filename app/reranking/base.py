"""Abstract base class for cross-encoder reranking models."""

from abc import ABC, abstractmethod
from typing import List, Tuple

from app.retrieval.models import RetrievalResult
from app.reranking.models import RerankResult


class Reranker(ABC):
    """Abstract interface for candidate rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
    ) -> Tuple[List[RerankResult], float]:
        """Rerank retrieval candidates for a query, returning ranked results and inference latency in ms."""
        pass
