"""Dense vector retrieval implementation using Multilingual E5 and Qdrant."""

import logging
import time
from typing import List, Optional, Tuple, Union

from app.config import get_settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.multilingual_e5 import get_embedding_provider
from app.retrieval.filters import build_qdrant_filter
from app.retrieval.models import RetrievalResult
from app.retrieval.qdrant_store import QdrantVectorStore, get_qdrant_store

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Dense retriever performing semantic vector search in Qdrant using Multilingual E5."""

    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        collection_name: Optional[str] = None,
    ):
        settings = get_settings()
        self.vector_store = vector_store or get_qdrant_store()
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        strategies: Optional[Union[str, List[str]]] = None,
        language: Optional[str] = None,
    ) -> Tuple[List[RetrievalResult], float, float]:
        """Execute dense retrieval returning results, embedding latency (ms), and search latency (ms)."""
        settings = get_settings()
        k = top_k if top_k is not None else settings.DENSE_TOP_K

        # 1. Query Embedding
        t_embed_start = time.perf_counter()
        query_vector = self.embedding_provider.embed_query(query)
        t_embed_ms = (time.perf_counter() - t_embed_start) * 1000

        # 2. Qdrant Vector Search
        t_search_start = time.perf_counter()
        qdrant_filter = build_qdrant_filter(strategies=strategies, language=language)

        scored_points = self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
            query_filter=qdrant_filter,
        )
        t_search_ms = (time.perf_counter() - t_search_start) * 1000

        # 3. Format as RetrievalResult
        results: List[RetrievalResult] = []
        for rank_idx, point in enumerate(scored_points, start=1):
            payload = point.payload or {}
            raw_score = float(point.score)

            result = RetrievalResult(
                chunk_id=payload.get("chunk_id", str(point.id)),
                document_id=payload.get("document_id", ""),
                text=payload.get("text", ""),
                chunk_type=payload.get("chunk_type", "dense"),
                language=payload.get("language"),
                dense_score=raw_score,
                bm25_score=None,
                normalized_dense_score=raw_score,
                normalized_bm25_score=None,
                fusion_score=raw_score,
                rank=rank_idx,
                metadata=payload.get("metadata", {}),
            )
            results.append(result)

        return results, t_embed_ms, t_search_ms
