"""Abstract Base Interface for LLM Generation Providers."""

from app.generation.base import GenerationProvider

# Re-export GenerationProvider as LLMProvider for naming consistency across the provider architecture
LLMProvider = GenerationProvider

__all__ = ["GenerationProvider", "LLMProvider"]
