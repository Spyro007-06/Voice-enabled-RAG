"""Google Gemini LLM Generation Provider for Grounded Multilingual RAG."""

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
import httpx

from app.config import get_settings
from app.generation.base import GenerationProvider
from app.generation.models import GenerationConfig, GenerationResult, GenerationTelemetry
from app.providers.exceptions import (
    InvalidAPIKeyError,
    MalformedResponseError,
    MissingAPIKeyError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    mask_credential,
)

logger = logging.getLogger(__name__)

# Grounded system instruction adhering to strict retrieval-only rules
DEFAULT_GEMINI_SYSTEM_INSTRUCTION = (
    "You are a multilingual retrieval-grounded question answering system.\n\n"
    "Your task is to answer the user's question using ONLY the retrieved passages supplied in the context.\n"
    "The retrieved passages come from the MSMARCO-XI multilingual corpus.\n\n"
    "Rules:\n"
    "1. Do not use outside knowledge.\n"
    "2. Do not invent facts.\n"
    "3. Do not infer unsupported information.\n"
    "4. Do not fabricate citations.\n"
    "5. If the retrieved context does not contain enough information to answer the question, clearly state "
    "that the information is not available in the retrieved context.\n"
    "6. Answer in the requested language whenever the retrieved context supports answering in that language.\n"
    "7. Keep the answer concise and directly related to the question.\n"
    "8. Preserve factual meaning from the retrieved passages.\n"
    "9. Never claim that information exists in the corpus unless it is present in the supplied context."
)


class GeminiLLMProvider(GenerationProvider):
    """Google Gemini LLM generation provider using official REST API with async HTTP/SSE."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """Initialize Google Gemini LLM generation provider.

        Args:
            api_key: Google Gemini API key (defaults to settings.GEMINI_API_KEY).
            base_url: Google Gemini REST base URL (defaults to settings.GEMINI_BASE_URL).
            model: Gemini model identifier (defaults to settings.GEMINI_MODEL, e.g., 'gemini-2.5-flash').
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.api_key = api_key or settings.GEMINI_API_KEY or settings.LLM_API_KEY
        self.base_url = (base_url or getattr(settings, "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")).rstrip("/")
        self.model = model or getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash")
        self.timeout = timeout if timeout is not None else settings.LLM_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None

        if not self.api_key or not self.api_key.strip():
            raise MissingAPIKeyError(provider_name="gemini", key_name="GEMINI_API_KEY")

        logger.info(
            "Initialized GeminiLLMProvider [Model: %s | Key: %s]",
            self.model,
            mask_credential(self.api_key),
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or initialize persistent, connection-pooled AsyncClient."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._client is not None and not getattr(self._client, "is_closed", False):
            self._loop = current_loop
            return self._client

        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=30.0)
        timeout = httpx.Timeout(timeout=self.timeout, connect=3.0, read=self.timeout)
        self._client = httpx.AsyncClient(limits=limits, timeout=timeout)
        self._loop = current_loop
        return self._client

    async def aclose(self) -> None:
        """Close persistent HTTP client session."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @property
    def provider_name(self) -> str:
        """Return unique provider identifier."""
        return "gemini"

    def _format_context_and_citations(
        self, context: Union[str, List[Any]]
    ) -> tuple[str, List[str], int, int]:
        """Extract structured source passages, citations, character counts, and chunk counts."""
        context_blocks: List[str] = []
        citations: List[str] = []

        if isinstance(context, str):
            if context.strip():
                context_blocks.append(context.strip())
        elif isinstance(context, list):
            for idx, item in enumerate(context, start=1):
                if isinstance(item, str):
                    if item.strip():
                        context_blocks.append(f"SOURCE {idx}\nPassage:\n{item.strip()}")
                elif hasattr(item, "text"):
                    txt = str(getattr(item, "text", "")).strip()
                    cid = getattr(item, "chunk_id", None) or getattr(item, "document_id", None)
                    lang = getattr(item, "language", None) or "unknown"
                    score = getattr(item, "reranker_score", None) or getattr(item, "fusion_score", None) or getattr(item, "score", None)
                    strat = getattr(item, "chunk_type", None) or "semantic"

                    header = f"SOURCE {idx}\nLanguage: {lang}"
                    if score is not None:
                        header += f"\nRelevance: {round(float(score), 2)}"
                    header += f"\nChunk Strategy: {strat}"
                    context_blocks.append(f"{header}\n\nPassage:\n{txt}")

                    if cid:
                        citations.append(str(cid))
                elif isinstance(item, dict):
                    txt = str(item.get("text") or item.get("content") or "").strip()
                    cid = item.get("chunk_id") or item.get("document_id") or item.get("id")
                    lang = item.get("language") or "unknown"
                    score = item.get("score")
                    strat = item.get("chunk_type") or item.get("chunking_strategy") or "semantic"

                    header = f"SOURCE {idx}\nLanguage: {lang}"
                    if score is not None:
                        header += f"\nRelevance: {round(float(score), 2)}"
                    header += f"\nChunk Strategy: {strat}"
                    context_blocks.append(f"{header}\n\nPassage:\n{txt}")

                    if cid:
                        citations.append(str(cid))

        combined_context = "\n\n---\n\n".join(context_blocks)
        total_chars = sum(len(b) for b in context_blocks)
        chunk_count = len(context_blocks)
        return combined_context, citations, chunk_count, total_chars

    async def generate(
        self,
        query: str,
        context: Union[str, List[Any]],
        generation_config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate a grounded answer for query and context using Google Gemini.

        Args:
            query: User query string.
            context: Retrieved context chunks/passages.
            generation_config: Optional GenerationConfig settings.

        Returns:
            GenerationResult: Structured generation output.
        """
        settings = get_settings()
        config = generation_config or GenerationConfig(
            model_name=self.model,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        target_model = (
            config.model_name
            if config.model_name and config.model_name != "mock-model"
            else self.model
        )

        t_start = time.perf_counter()
        combined_context, citations, chunk_count, total_chars = self._format_context_and_citations(context)

        system_instruction = config.system_prompt or DEFAULT_GEMINI_SYSTEM_INSTRUCTION

        user_content = (
            f"Retrieved Context:\n{combined_context}\n\n"
            f"User Question:\n{query}"
        ) if combined_context else f"User Question:\n{query}\n\n[NO RETRIEVED CONTEXT AVAILABLE]"

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_content}],
                }
            ],
            "systemInstruction": {
                "parts": [{"text": system_instruction}],
            },
            "generationConfig": {
                "temperature": config.temperature,
                "maxOutputTokens": config.max_tokens or 2048,
                "topP": config.top_p if config.top_p is not None else 0.9,
            },
        }

        candidate_models = [target_model]
        for fallback in ["gemini-2.0-flash", "gemini-1.5-flash"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        resp = None
        for cand_model in candidate_models:
            url = f"{self.base_url}/models/{cand_model}:generateContent"
            params = {"key": self.api_key}
            headers = {"Content-Type": "application/json"}

            try:
                client = await self._get_client()
                resp = await client.post(url, params=params, headers=headers, json=payload)
                if resp.status_code == 404 and cand_model != candidate_models[-1]:
                    logger.warning("Gemini model %s returned 404, falling back to next candidate", cand_model)
                    continue
                break
            except httpx.TimeoutException as exc:
                raise ProviderTimeoutError(provider_name="gemini", timeout_seconds=self.timeout) from exc
            except httpx.RequestError as exc:
                raise ProviderConnectionError(provider_name="gemini", endpoint=url) from exc

        lat_ms = (time.perf_counter() - t_start) * 1000.0

        if resp is None:
            raise ProviderConnectionError(provider_name="gemini", endpoint=url)
        if resp.status_code in (401, 403):
            raise InvalidAPIKeyError(provider_name="gemini", key_name="GEMINI_API_KEY")
        if resp.status_code == 429:
            raise ProviderRateLimitError(provider_name="gemini")
        if resp.status_code >= 400:
            raise MalformedResponseError(
                provider_name="gemini",
                message=f"Gemini API returned HTTP {resp.status_code}: {resp.text[:250]}",
            )

        try:
            res_json = resp.json()
            candidates = res_json.get("candidates", [])
            if not candidates:
                # Handle prompt safety block or empty response
                prompt_feedback = res_json.get("promptFeedback", {})
                block_reason = prompt_feedback.get("blockReason", "empty_candidates")
                answer = "I couldn't find enough relevant information in the indexed multilingual corpus to answer this reliably."
                finish_reason = f"blocked_{block_reason}"
                prompt_tokens = 0
                completion_tokens = 0
            else:
                first_cand = candidates[0]
                content_obj = first_cand.get("content", {})
                parts = content_obj.get("parts", [])
                answer = "".join(part.get("text", "") for part in parts).strip()
                finish_reason = first_cand.get("finishReason", "STOP")

                usage = res_json.get("usageMetadata", {})
                prompt_tokens = usage.get("promptTokenCount", max(1, (len(query) + total_chars) // 4))
                completion_tokens = usage.get("candidatesTokenCount", max(1, len(answer) // 4))

        except Exception as exc:
            raise MalformedResponseError(
                provider_name="gemini",
                message="Failed to parse Google Gemini response payload.",
            ) from exc

        telemetry = GenerationTelemetry(
            latency_ms=round(lat_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            context_chunks_count=chunk_count,
            context_characters=total_chars,
        )

        return GenerationResult(
            answer=answer,
            grounded=bool(combined_context and answer),
            citations=citations,
            model=target_model,
            latency_ms=round(lat_ms, 2),
            finish_reason=finish_reason,
            error=None,
            telemetry=telemetry,
        )

    async def generate_stream(
        self,
        query: str,
        context: Optional[List[Any]] = None,
        config: Optional[Any] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response tokens via Google Gemini streamGenerateContent SSE endpoint."""
        from app.generation.models import GenerationConfig

        if config is None:
            config = GenerationConfig()

        raw_model = getattr(config, "model_name", None)
        target_model = (
            raw_model
            if raw_model and raw_model != "mock-model"
            else self.model
        )
        combined_context, _, _, _ = self._format_context_and_citations(context or [])

        system_instruction = getattr(config, "system_prompt", None) or DEFAULT_GEMINI_SYSTEM_INSTRUCTION

        user_content = (
            f"Retrieved Context:\n{combined_context}\n\n"
            f"User Question:\n{query}"
        ) if combined_context else f"User Question:\n{query}\n\n[NO RETRIEVED CONTEXT AVAILABLE]"

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_content}],
                }
            ],
            "systemInstruction": {
                "parts": [{"text": system_instruction}],
            },
            "generationConfig": {
                "temperature": getattr(config, "temperature", 0.7),
                "maxOutputTokens": getattr(config, "max_tokens", 2048) or 2048,
                "topP": getattr(config, "top_p", 0.9) or 0.9,
            },
        }

        candidate_models = [target_model]
        for fallback in ["gemini-2.0-flash", "gemini-1.5-flash"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        client = await self._get_client()
        stream_opened = False

        for cand_model in candidate_models:
            url = f"{self.base_url}/models/{cand_model}:streamGenerateContent"
            params = {"alt": "sse", "key": self.api_key}
            headers = {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }

            try:
                async with client.stream("POST", url, params=params, headers=headers, json=payload) as resp:
                    if resp.status_code == 404 and cand_model != candidate_models[-1]:
                        logger.warning("Gemini streaming model %s returned 404, falling back to next candidate", cand_model)
                        continue
                    if resp.status_code in (401, 403):
                        raise InvalidAPIKeyError(provider_name="gemini", key_name="GEMINI_API_KEY")
                    if resp.status_code == 429:
                        raise ProviderRateLimitError(provider_name="gemini")
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        raise MalformedResponseError(
                            provider_name="gemini",
                            message=f"Gemini streaming returned HTTP {resp.status_code}: {body[:250]}",
                        )

                    stream_opened = True
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                candidates = chunk.get("candidates", [])
                                if candidates:
                                    parts = candidates[0].get("content", {}).get("parts", [])
                                    for part in parts:
                                        delta = part.get("text", "")
                                        if delta:
                                            yield delta
                            except (json.JSONDecodeError, IndexError, KeyError):
                                continue
                    if stream_opened:
                        return
            except (InvalidAPIKeyError, ProviderRateLimitError, MalformedResponseError):
                raise
            except httpx.TimeoutException as exc:
                raise ProviderTimeoutError(provider_name="gemini", timeout_seconds=self.timeout) from exc
            except httpx.RequestError as exc:
                raise ProviderConnectionError(provider_name="gemini", endpoint=url) from exc
