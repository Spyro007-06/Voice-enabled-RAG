"""Qdrant Vector Database Provider supporting local, in-memory, and Qdrant Cloud."""

import logging
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient

from app.chunking.models import Chunk
from app.config import get_settings
from app.providers.exceptions import (
    MissingAPIKeyError,
    ProviderConnectionError,
    mask_credential,
)
from app.providers.vector.base import VectorStoreProvider
from app.retrieval.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class QdrantVectorProvider(VectorStoreProvider):
    """Qdrant vector store provider wrapping QdrantVectorStore."""

    def __init__(
        self,
        provider_type: str = "qdrant_local",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        location: Optional[str] = None,
        in_memory: bool = False,
    ):
        settings = get_settings()
        self._provider_type = provider_type.lower().strip()

        if self._provider_type == "qdrant_cloud":
            target_url = url or settings.VECTOR_DB_URL
            target_key = api_key or settings.VECTOR_DB_API_KEY

            if not target_url or not target_url.strip():
                raise MissingAPIKeyError(provider_name="qdrant_cloud", key_name="VECTOR_DB_URL")

            logger.info(
                "Initializing QdrantCloudVectorProvider at %s (Key: %s)",
                target_url,
                mask_credential(target_key),
            )
            try:
                client = QdrantClient(url=target_url, api_key=target_key)
            except Exception as exc:
                raise ProviderConnectionError(provider_name="qdrant_cloud", endpoint=target_url) from exc

            self.store = QdrantVectorStore(client=client)
        elif self._provider_type == "memory" or in_memory:
            logger.info("Initializing in-memory QdrantVectorProvider")
            self.store = QdrantVectorStore(in_memory=True)
        else:
            # Local persistent disk client
            target_path = location or settings.QDRANT_PATH
            logger.info("Initializing local QdrantVectorProvider at %s", target_path)
            self.store = QdrantVectorStore(location=target_path, in_memory=False)

    @property
    def provider_name(self) -> str:
        return self._provider_type

    def upsert(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        return self.store.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            collection_name=collection_name,
            **kwargs,
        )

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Any]:
        return self.store.search(
            query_vector=query_vector,
            top_k=top_k,
            collection_name=collection_name,
            filter_dict=filters,
            **kwargs,
        )

    def count(self, collection_name: Optional[str] = None) -> int:
        target_col = collection_name or get_settings().QDRANT_COLLECTION_NAME
        try:
            if not self.store.collection_exists(target_col):
                return 0
            info = self.store.client.get_collection(collection_name=target_col)
            return info.points_count or 0
        except Exception:
            return 0
