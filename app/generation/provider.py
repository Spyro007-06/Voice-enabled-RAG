"""Deterministic Mock LLM Generation Provider for testing and architectural validation."""

import asyncio
import time
from typing import Any, List, Optional, Union

from app.generation.base import GenerationProvider
from app.generation.models import GenerationConfig, GenerationResult, GenerationTelemetry


class MockGenerationProvider(GenerationProvider):
    """Deterministic mock generation provider that produces predictable responses without calling an external LLM."""

    def __init__(
        self,
        default_model: str = "mock-model",
        simulated_latency_ms: float = 0.0,
        should_fail: bool = False,
        failure_message: str = "Simulated mock generation failure",
    ):
        """Initialize mock generation provider.

        Args:
            default_model: Default model identifier string.
            simulated_latency_ms: Optional artificial delay in milliseconds.
            should_fail: Whether the provider should simulate a failure.
            failure_message: Exception message to raise when should_fail is True.
        """
        self._default_model = default_model
        self.simulated_latency_ms = simulated_latency_ms
        self.should_fail = should_fail
        self.failure_message = failure_message

    @property
    def provider_name(self) -> str:
        """Return the provider identifier."""
        return "mock"

    async def generate(
        self,
        query: str,
        context: Union[str, List[Any]],
        generation_config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Deterministically produce a structured mock generation result.

        Args:
            query: Input question or search string.
            context: Retrieved context (raw text or list of chunks/objects).
            generation_config: Optional configuration override.

        Returns:
            GenerationResult: Deterministic structured result with telemetry and citations.
        """
        start_time = time.perf_counter()

        if self.simulated_latency_ms > 0:
            await asyncio.sleep(self.simulated_latency_ms / 1000.0)

        if self.should_fail:
            raise RuntimeError(self.failure_message)

        config = generation_config or GenerationConfig(model_name=self._default_model)
        model_name = config.model_name or self._default_model

        # Extract citations and normalize context items
        citations: List[str] = []
        context_texts: List[str] = []

        if isinstance(context, str):
            if context.strip():
                context_texts.append(context.strip())
        elif isinstance(context, list):
            for idx, item in enumerate(context):
                if isinstance(item, str):
                    if item.strip():
                        context_texts.append(item.strip())
                elif hasattr(item, "text"):
                    # Object like RerankResult or Chunk
                    if getattr(item, "text", ""):
                        context_texts.append(str(item.text).strip())
                    chunk_id = getattr(item, "chunk_id", None) or getattr(item, "document_id", None)
                    if chunk_id:
                        citations.append(str(chunk_id))
                    else:
                        citations.append(f"doc_{idx+1}")
                elif isinstance(item, dict):
                    text_val = item.get("text") or item.get("content") or ""
                    if text_val:
                        context_texts.append(str(text_val).strip())
                    chunk_id = item.get("chunk_id") or item.get("document_id") or item.get("id")
                    if chunk_id:
                        citations.append(str(chunk_id))
                    else:
                        citations.append(f"doc_{idx+1}")

        total_context_chars = sum(len(t) for t in context_texts)
        has_context = len(context_texts) > 0

        if not has_context:
            answer = f"[MOCK] No context available to answer: '{query}'"
            grounded = False
        else:
            # Check content-word coverage to simulate truthful LLM adherence
            from app.guardrails.relevance import compute_content_overlap
            combined_context = " ".join(context_texts)
            overlap_ratio = compute_content_overlap(query, combined_context)

            if overlap_ratio < 0.35:
                # Retrieved context lacks sufficient coverage for the question
                answer = "I don't have enough information in the retrieved context to answer that."
                grounded = False
            else:
                # Deterministic mock response referencing query and snippet of context
                first_snippet = context_texts[0][:150].strip()
                answer = (
                    f"[MOCK] Answer to '{query}' based on {len(context_texts)} source(s): "
                    f"{first_snippet}"
                )
                grounded = True

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Deterministic token estimation (~4 chars per token heuristic for mock telemetry)
        prompt_tokens = max(1, (len(query) + total_context_chars) // 4)
        completion_tokens = max(1, len(answer) // 4)

        telemetry = GenerationTelemetry(
            latency_ms=round(elapsed_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            context_chunks_count=len(context_texts),
            context_characters=total_context_chars,
        )

        return GenerationResult(
            answer=answer,
            grounded=grounded,
            citations=citations,
            model=model_name,
            latency_ms=round(elapsed_ms, 2),
            finish_reason="stop",
            error=None,
            telemetry=telemetry,
        )


def get_generation_provider(provider_name: str = "mock", **kwargs: Any) -> GenerationProvider:
    """Factory to instantiate generation providers by name.

    Args:
        provider_name: Provider name ('mock', 'sarvam', 'openai', etc.).
        **kwargs: Additional parameters passed to provider constructor.

    Returns:
        GenerationProvider: Instantiated provider.
    """
    normalized_name = provider_name.lower().strip()
    if normalized_name == "mock":
        return MockGenerationProvider(**kwargs)
    try:
        from app.providers.factory import get_llm_provider
        return get_llm_provider(provider_name=provider_name, **kwargs)
    except Exception as exc:
        raise ValueError(f"Unsupported generation provider: '{provider_name}'") from exc
