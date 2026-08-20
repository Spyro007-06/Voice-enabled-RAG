"""Abstract interface for embedding providers."""

from abc import ABC, abstractmethod
from typing import List, Sequence


class EmbeddingProvider(ABC):
    """Abstract base class defining the standard interface for vector embedding models."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Compute normalized vector embeddings for a sequence of document/passage texts.
        
        Must use batch encoding and appropriate document/passage prefixes.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Compute normalized vector embedding for a single search query text.
        
        Must use appropriate query prefix as expected by the model architecture.
        """
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the vector dimension of generated embeddings."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Return the compute device (e.g. 'cpu', 'cuda')."""
        pass
