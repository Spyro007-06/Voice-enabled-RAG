"""Generation service orchestrating LLM request validation, context preparation, provider execution, and telemetry."""

import asyncio
import logging
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

from app.config import get_settings
from app.generation.base import GenerationProvider
from app.generation.models import (
    GenerationConfig,
    GenerationRequest,
    GenerationResult,
    GenerationTelemetry,
)
from app.generation.provider import get_generation_provider
from app.generation.validators import AnswerValidator, CitationValidator

logger = logging.getLogger(__name__)


class GenerationService:
    """Orchestrates LLM generation across interchangeable providers with validation and telemetry."""

    def __init__(
        self,
        provider: Optional[GenerationProvider] = None,
    ):
        """Initialize GenerationService.

        Args:
            provider: Optional custom GenerationProvider. If None, initialized from application settings.
        """
        settings = get_settings()
        self.provider = provider or get_generation_provider(provider_name=settings.LLM_PROVIDER)

    def _prepare_config(self, config_override: Optional[GenerationConfig] = None) -> GenerationConfig:
        """Merge configuration overrides with application settings defaults."""
        settings = get_settings()
        default_model = (
            settings.GEMINI_MODEL
            if settings.LLM_PROVIDER == "gemini"
            else settings.LLM_MODEL_NAME
        )
        if config_override is None:
            return GenerationConfig(
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                top_p=settings.LLM_TOP_P,
                timeout=settings.LLM_TIMEOUT,
                streaming=settings.LLM_STREAMING,
                model_name=default_model,
                system_prompt=settings.LLM_SYSTEM_PROMPT,
            )

        if not config_override.model_name:
            config_override.model_name = default_model
        return config_override

    def _validate_and_extract_context(
        self,
        context: Union[str, List[Any]],
    ) -> Tuple[Union[str, List[Any]], int, int, List[str]]:
        """Validate context and extract metadata, character length, chunk counts, and candidate citation IDs.

        Preserves multilingual Unicode content without lossy ASCII normalization.

        Returns:
            Tuple of (normalized_context, chunk_count, total_characters, citation_ids)
        """
        citations: List[str] = []
        total_chars = 0
        chunk_count = 0

        if context is None:
            return "", 0, 0, []

        if isinstance(context, str):
            clean_str = context.strip()
            total_chars = len(clean_str)
            chunk_count = 1 if clean_str else 0
            return clean_str, chunk_count, total_chars, []

        if isinstance(context, list):
            chunk_count = len(context)
            for idx, item in enumerate(context):
                if isinstance(item, str):
                    total_chars += len(item)
                elif hasattr(item, "text"):
                    text_val = str(getattr(item, "text", ""))
                    total_chars += len(text_val)
                    chunk_id = getattr(item, "chunk_id", None) or getattr(item, "document_id", None)
                    if chunk_id:
                        citations.append(str(chunk_id))
                    else:
                        citations.append(f"chunk_{idx+1}")
                elif isinstance(item, dict):
                    text_val = str(item.get("text") or item.get("content") or "")
                    total_chars += len(text_val)
                    chunk_id = item.get("chunk_id") or item.get("document_id") or item.get("id")
                    if chunk_id:
                        citations.append(str(chunk_id))
                    else:
                        citations.append(f"chunk_{idx+1}")
                else:
                    total_chars += len(str(item))

            return context, chunk_count, total_chars, citations

        # Fallback for unexpected context types
        str_val = str(context)
        return str_val, 1 if str_val else 0, len(str_val), []

    async def generate(
        self,
        request: Optional[Union[GenerationRequest, str]] = None,
        query: Optional[str] = None,
        context: Optional[Union[str, List[Any]]] = None,
        config: Optional[GenerationConfig] = None,
        language: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        """Execute LLM generation pipeline with validation, provider invocation, timeout handling, and telemetry.

        Args:
            request: Optional GenerationRequest instance or query string.
            query: Optional search query string (if request object is not passed).
            context: Context string or list of chunks/results.
            config: Generation configuration override.
            language: Optional language code (e.g. 'hi', 'ta', 'te', 'ml', 'en').
            metadata: Optional metadata dictionary.

        Returns:
            GenerationResult: Structured generation result.
        """
        start_time = time.perf_counter()

        # 1. Normalize Request parameters
        if isinstance(request, GenerationRequest):
            req_query = request.query
            req_context = request.context
            req_config = request.config
            req_lang = request.language
            req_meta = request.metadata
        else:
            req_query = query if query is not None else (request if isinstance(request, str) else "")
            req_context = context if context is not None else ""
            req_config = config
            req_lang = language
            req_meta = metadata or {}

        # 2. Validate Query
        if not req_query or not req_query.strip():
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GenerationResult(
                answer="",
                grounded=False,
                citations=[],
                model="unknown",
                latency_ms=round(elapsed_ms, 2),
                finish_reason="error",
                error="Query cannot be empty or whitespace.",
                telemetry=GenerationTelemetry(
                    latency_ms=round(elapsed_ms, 2),
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    context_chunks_count=0,
                    context_characters=0,
                ),
            )

        # 3. Validate and Prepare Context
        clean_context, chunk_count, total_chars, extracted_citations = self._validate_and_extract_context(req_context)

        # 4. Prepare Configuration
        gen_config = self._prepare_config(req_config)
        timeout_seconds = gen_config.timeout

        # 5. Invoke Provider with Latency Measurement and Timeout Protection
        try:
            result = await asyncio.wait_for(
                self.provider.generate(
                    query=req_query,
                    context=clean_context,
                    generation_config=gen_config,
                ),
                timeout=timeout_seconds,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # 6. Validate Answer, Calibrate Confidence, and Filter Fabricated Citations
            val_result = AnswerValidator.validate_answer(
                answer=result.answer,
                grounded=result.grounded,
                confidence=getattr(result, "confidence", 1.0),
                citations=result.citations if result.citations else extracted_citations,
                retrieved_context=req_context,
            )

            telemetry = result.telemetry or GenerationTelemetry(
                latency_ms=round(elapsed_ms, 2),
                prompt_tokens=max(1, (len(req_query) + total_chars) // 4),
                completion_tokens=max(1, len(val_result.cleaned_answer) // 4) if val_result.cleaned_answer else 0,
                total_tokens=max(1, (len(req_query) + total_chars + len(val_result.cleaned_answer)) // 4),
                context_chunks_count=chunk_count,
                context_characters=total_chars,
            )
            telemetry.latency_ms = round(elapsed_ms, 2)
            telemetry.context_chunks_count = chunk_count
            telemetry.context_characters = total_chars

            return GenerationResult(
                answer=val_result.cleaned_answer,
                grounded=val_result.is_valid and result.grounded and bool(clean_context),
                confidence=val_result.confidence,
                citations=val_result.validated_citations,
                citation_provenance=val_result.citation_provenance,
                model=result.model or gen_config.model_name,
                latency_ms=round(elapsed_ms, 2),
                finish_reason=result.finish_reason or "stop",
                error=result.error,
                telemetry=telemetry,
            )

        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = f"Generation timed out after {timeout_seconds:.1f}s"
            logger.warning(f"GenerationService: {err_msg} for query '{req_query[:50]}'")
            return GenerationResult(
                answer="",
                grounded=False,
                citations=[],
                model=gen_config.model_name,
                latency_ms=round(elapsed_ms, 2),
                finish_reason="timeout",
                error=err_msg,
                telemetry=GenerationTelemetry(
                    latency_ms=round(elapsed_ms, 2),
                    context_chunks_count=chunk_count,
                    context_characters=total_chars,
                ),
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = f"Provider execution failed: {str(exc)}"
            logger.error(f"GenerationService error: {err_msg}", exc_info=True)
            return GenerationResult(
                answer="",
                grounded=False,
                citations=[],
                model=gen_config.model_name,
                latency_ms=round(elapsed_ms, 2),
                finish_reason="error",
                error=err_msg,
                telemetry=GenerationTelemetry(
                    latency_ms=round(elapsed_ms, 2),
                    context_chunks_count=chunk_count,
                    context_characters=total_chars,
                ),
            )


@lru_cache()
def get_generation_service() -> GenerationService:
    """Return cached singleton instance of GenerationService."""
    return GenerationService()
