"""Base abstraction for all service providers in HH Goa 2026 Voice RAG."""

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """Abstract base class for all providers (STT, TTS, LLM, Vector)."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique identifier for this provider."""
        pass
