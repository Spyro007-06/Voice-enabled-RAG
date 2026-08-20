"""Multilingual E5 embedding provider implementation using SentenceTransformers."""

import logging
from functools import lru_cache
from typing import List, Optional, Sequence

import torch
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class MultilingualE5EmbeddingProvider(EmbeddingProvider):
    """Multilingual E5 embedding provider (e.g. intfloat/multilingual-e5-small).
    
    Supports 100+ languages including 14 Indic languages from MSMARCO-XI.
    E5 requires specific task prefixes:
    - Passages: 'passage: <text>'
    - Queries:  'query: <text>'
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        settings = get_settings()
        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

        # Device selection & fallback
        requested_device = (device or settings.EMBEDDING_DEVICE).lower().strip()
        if requested_device == "cuda" and torch.cuda.is_available():
            self._device = "cuda"
        elif requested_device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = "cpu"

        logger.info(
            "Initializing MultilingualE5EmbeddingProvider [Model: %s | Device: %s | Batch size: %d]",
            self._model_name,
            self._device,
            self._batch_size,
        )

        self._model: Optional[SentenceTransformer] = None
        self._dimension: Optional[int] = None

    def _ensure_model(self) -> SentenceTransformer:
        """Lazy load model on first inference call to keep startup/health endpoints fast."""
        if self._model is None:
            settings = get_settings()
            if self._device == "cpu" and hasattr(torch, "set_num_threads"):
                try:
                    num_threads = getattr(settings, "CPU_NUM_THREADS", 4)
                    torch.set_num_threads(num_threads)
                except Exception:
                    pass
            logger.info("Loading embedding model '%s' onto device '%s'...", self._model_name, self._device)
            self._model = SentenceTransformer(self._model_name, device=self._device)
            if hasattr(self._model, "get_embedding_dimension"):
                self._dimension = self._model.get_embedding_dimension()
            else:
                self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info("Model loaded successfully. Vector dimension: %d", self._dimension)
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Return the vector embedding dimension."""
        if self._dimension is None:
            self._ensure_model()
        return self._dimension or 384

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode a batch of document/passage texts with the 'passage: ' prefix."""
        if not texts:
            return []

        model = self._ensure_model()
        # E5 expects 'passage: ' prefix for corpus documents
        prefixed_texts = [f"passage: {t.strip()}" for t in texts]

        with torch.inference_mode():
            embeddings = model.encode(
                prefixed_texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Encode a search query with the 'query: ' prefix."""
        if not text or not text.strip():
            # Return zero vector of appropriate dimension for empty text
            return [0.0] * self.embedding_dimension

        model = self._ensure_model()
        # E5 expects 'query: ' prefix for queries
        prefixed_query = f"query: {text.strip()}"

        with torch.inference_mode():
            embedding = model.encode(
                prefixed_query,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return embedding.tolist()


@lru_cache()
def get_embedding_provider() -> EmbeddingProvider:
    """Return singleton cached instance of MultilingualE5EmbeddingProvider."""
    return MultilingualE5EmbeddingProvider()


# Alias for backward compatibility
MultilingualE5Embedder = MultilingualE5EmbeddingProvider

