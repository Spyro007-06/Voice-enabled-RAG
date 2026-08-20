import asyncio
import base64
import logging
import time
from typing import Optional
import httpx

from app.config import get_settings
from app.providers.exceptions import (
    InvalidAPIKeyError,
    MalformedResponseError,
    MissingAPIKeyError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    mask_credential,
)
from app.providers.tts.base import TTSProvider, TTSResult
from app.retrieval.language import to_sarvam_code

logger = logging.getLogger(__name__)


class SarvamTTSProvider(TTSProvider):
    """Sarvam AI Text-to-Speech provider (Bulbul v1 model)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        default_speaker: str = "anushka",
        timeout: float = 15.0,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.SARVAM_API_KEY
        self.base_url = (base_url or settings.SARVAM_BASE_URL).rstrip("/")
        self.model = model or settings.SARVAM_TTS_MODEL
        self.default_speaker = default_speaker
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

        # Production credential check at provider initialization time
        if not self.api_key or not self.api_key.strip():
            raise MissingAPIKeyError(provider_name="sarvam", key_name="SARVAM_API_KEY")

        logger.info(
            "Initialized SarvamTTSProvider [Model: %s | Key: %s]",
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
            timeout = httpx.Timeout(timeout=self.timeout, connect=5.0, read=self.timeout)
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

    async def synthesize(
        self,
        text: str,
        language: str = "hi",
        speaker: Optional[str] = None,
        pace: Optional[float] = None,
    ) -> TTSResult:
        if not text or not text.strip():
            return TTSResult(
                audio_bytes=b"",
                audio_format="wav",
                sample_rate=24000,
                duration_s=0.0,
                latency_ms=0.0,
                provider=self.provider_name,
            )

        t_start = time.perf_counter()
        url = f"{self.base_url}/text-to-speech"
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json",
        }

        # Format language code (e.g. 'hi' -> 'hi-IN')
        lang_code = to_sarvam_code(language, default="hi-IN") if language else "hi-IN"

        payload = {
            "inputs": [text.strip()],
            "target_language_code": lang_code,
            "speaker": speaker or self.default_speaker,
            "model": self.model,
            "enable_preprocessing": True,
        }

        if pace is not None:
            payload["pace"] = pace

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
                message=f"Sarvam TTS returned HTTP {resp.status_code}: {resp.text[:200]}",
            )

        try:
            res_json = resp.json()
            audios = res_json.get("audios", [])
            if not audios:
                raise ValueError("No audio returned in Sarvam TTS payload.")
            audio_bytes = base64.b64decode(audios[0])
        except Exception as exc:
            raise MalformedResponseError(
                provider_name="sarvam",
                message="Failed to parse/decode Sarvam TTS base64 audio output.",
            ) from exc

        duration_est = max(0.1, len(audio_bytes) / 48000.0)  # 24kHz 16-bit mono PCM

        return TTSResult(
            audio_bytes=audio_bytes,
            audio_format="wav",
            sample_rate=24000,
            duration_s=round(duration_est, 2),
            latency_ms=round(lat_ms, 2),
            provider=self.provider_name,
        )
