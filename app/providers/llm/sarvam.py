import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, List, Optional, Union
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


class SarvamLLMProvider(GenerationProvider):
    """Sarvam AI LLM generation provider (sarvam-2b model)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.SARVAM_API_KEY
        self.base_url = (base_url or settings.SARVAM_BASE_URL).rstrip("/")
        self.model = model or settings.SARVAM_LLM_MODEL
        self.timeout = timeout if timeout is not None else settings.LLM_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None

        # Production credential validation at provider initialization time
        if not self.api_key or not self.api_key.strip():
            raise MissingAPIKeyError(provider_name="sarvam", key_name="SARVAM_API_KEY")

        logger.info(
            "Initialized SarvamLLMProvider [Model: %s | Key: %s]",
            self.model,
            mask_credential(self.api_key),
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or initialize persistent, connection-pooled AsyncClient."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        client_closed = self._client is None or self._client.is_closed
        if not client_closed and getattr(self, "_loop", None) != current_loop:
            client_closed = True
            self._client = None

        if client_closed:
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
        return "sarvam"

    async def generate(
        self,
        query: str,
        context: Union[str, List[Any]],
        generation_config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        settings = get_settings()
        config = generation_config or GenerationConfig(
            model_name=self.model,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )

        t_start = time.perf_counter()

        # Format context and extract citation metadata
        context_texts: List[str] = []
        citations: List[str] = []
        if isinstance(context, str):
            if context.strip():
                context_texts.append(context.strip())
        elif isinstance(context, list):
            for idx, item in enumerate(context):
                if isinstance(item, str):
                    if item.strip():
                        context_texts.append(item.strip())
                elif hasattr(item, "text"):
                    if getattr(item, "text", ""):
                        context_texts.append(str(item.text).strip())
                    cid = getattr(item, "chunk_id", None) or getattr(item, "document_id", None)
                    if cid:
                        citations.append(str(cid))
                elif isinstance(item, dict):
                    t_val = item.get("text") or item.get("content") or ""
                    if t_val:
                        context_texts.append(str(t_val).strip())
                    cid = item.get("chunk_id") or item.get("document_id") or item.get("id")
                    if cid:
                        citations.append(str(cid))

        combined_context = "\n\n".join(context_texts)
        total_context_chars = sum(len(t) for t in context_texts)

        system_instruction = (
            config.system_prompt
            or "You are a helpful and truthful assistant. Answer based only on the provided context."
        )

        user_content = f"Context:\n{combined_context}\n\nQuestion:\n{query}" if combined_context else query

        payload = {
            "model": config.model_name or self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
        }

        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            client = await self._get_client()
            resp = await client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(provider_name="sarvam", timeout_seconds=self.timeout) from exc
        except httpx.RequestError as exc:
            raise ProviderConnectionError(provider_name="sarvam", endpoint=url) from exc

        lat_ms = (time.perf_counter() - t_start) * 1000.0

        if resp.status_code in (401, 403):
            raise InvalidAPIKeyError(provider_name="sarvam", key_name="SARVAM_API_KEY")
        if resp.status_code == 429:
            raise ProviderRateLimitError(provider_name="sarvam")
        if resp.status_code >= 400:
            raise MalformedResponseError(
                provider_name="sarvam",
                message=f"Sarvam LLM returned HTTP {resp.status_code}: {resp.text[:200]}",
            )

        try:
            res_json = resp.json()
            choices = res_json.get("choices", [])
            msg_obj = choices[0].get("message", {})
            raw_content = msg_obj.get("content")
            raw_reasoning = msg_obj.get("reasoning_content")
            answer = (raw_content or raw_reasoning or "").strip()
            finish_reason = choices[0].get("finish_reason", "stop")

            usage = res_json.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", max(1, (len(query) + total_context_chars) // 4))
            completion_tokens = usage.get("completion_tokens", max(1, len(answer) // 4))
        except Exception as exc:
            raise MalformedResponseError(
                provider_name="sarvam",
                message="Failed to parse Sarvam LLM response payload.",
            ) from exc

        telemetry = GenerationTelemetry(
            latency_ms=round(lat_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            context_chunks_count=len(context_texts),
            context_characters=total_context_chars,
        )

        return GenerationResult(
            answer=answer,
            grounded=bool(combined_context),
            citations=citations,
            model=config.model_name or self.model,
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
        """Stream LLM response as token chunks via Sarvam SSE (stream=true).

        Yields plain-text token delta strings as they arrive from the API.
        Raises ProviderTimeoutError, ProviderConnectionError, InvalidAPIKeyError,
        ProviderRateLimitError, or MalformedResponseError on failure.

        STREAMING STATUS: LLM streaming supported via OpenAI-compatible SSE.
        STREAMING STATUS: TTS DOES NOT SUPPORT STREAMING (single base64 payload).
        """
        from app.generation.models import GenerationConfig  # avoid circular at module level

        if config is None:
            config = GenerationConfig()

        # --- Build context text (identical logic to generate()) ---
        context_texts: List[str] = []
        if context:
            for item in context:
                if hasattr(item, "text") and item.text:
                    context_texts.append(str(item.text).strip())
                elif isinstance(item, dict):
                    t_val = item.get("text") or item.get("content") or ""
                    if t_val:
                        context_texts.append(str(t_val).strip())

        combined_context = "\n\n".join(context_texts)

        system_instruction = (
            config.system_prompt
            or "You are a helpful and truthful assistant. Answer based only on the provided context."
        )
        user_content = f"Context:\n{combined_context}\n\nQuestion:\n{query}" if combined_context else query

        payload = {
            "model": (config.model_name if hasattr(config, "model_name") else None) or self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "stream": True,
        }

        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        try:
            client = await self._get_client()
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code in (401, 403):
                    raise InvalidAPIKeyError(provider_name="sarvam", key_name="SARVAM_API_KEY")
                if resp.status_code == 429:
                    raise ProviderRateLimitError(provider_name="sarvam")
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise MalformedResponseError(
                        provider_name="sarvam",
                        message=f"Sarvam LLM stream returned HTTP {resp.status_code}: {body[:200]}",
                    )

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
                            delta = (
                                chunk.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, IndexError, KeyError):
                            continue  # skip malformed chunks silently

        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(provider_name="sarvam", timeout_seconds=self.timeout) from exc
        except httpx.RequestError as exc:
            raise ProviderConnectionError(provider_name="sarvam", endpoint=url) from exc
