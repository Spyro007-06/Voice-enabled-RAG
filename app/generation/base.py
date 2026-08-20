"""Abstract base class for LLM generation providers."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

from app.generation.models import GenerationConfig, GenerationResult


class GenerationProvider(ABC):
    """Abstract interface for provider-independent LLM text generation."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique identifier for this generation provider."""
        pass

    @abstractmethod
    async def generate(
        self,
        query: str,
        context: Union[str, List[Any]],
        generation_config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate a structured response for a query and context given configuration.

        Args:
            query: The user query or question.
            context: Retrieved context (raw text or list of chunk objects/strings).
            generation_config: Optional generation parameters (temperature, max_tokens, etc.).

        Returns:
            GenerationResult: Structured generation output including answer, citations,
                model name, latency, finish_reason, and error information.
        """
        pass
