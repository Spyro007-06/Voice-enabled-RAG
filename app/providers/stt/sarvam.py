import asyncio
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
from app.providers.stt.base import STTProvider, STTResult
from app.retrieval.language import to_sarvam_code
from app.speech.validation import verify_audio_magic_bytes

logger = logging.getLogger(__name__)


class SarvamSTTProvider(STTProvider):
    """Sarvam AI Speech-to-Text provider (Saarika v2 model)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 15.0,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.SARVAM_API_KEY
        self.base_url = (base_url or settings.SARVAM_BASE_URL).rstrip("/")
        self.model = model or settings.SARVAM_STT_MODEL
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

        # Production credential check at provider initialization time
        if not self.api_key or not self.api_key.strip():
            raise MissingAPIKeyError(provider_name="sarvam", key_name="SARVAM_API_KEY")

        logger.info(
            "Initialized SarvamSTTProvider [Model: %s | Key: %s]",
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

    async def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> STTResult:
        if not audio_bytes:
            return STTResult(
                text="",
                language=language,
                confidence=0.0,
                duration_s=0.0,
                latency_ms=0.0,
                provider=self.provider_name,
            )

        t_start = time.perf_counter()
        url = f"{self.base_url}/speech-to-text"
        headers = {
            "api-subscription-key": self.api_key,
        }

        # Language code normalization for Sarvam (e.g. 'en' -> 'en-IN', 'hi' -> 'hi-IN', default 'unknown' for auto-detect)
        lang_code = to_sarvam_code(language, default="unknown") if language else "unknown"

        data = {
            "model": self.model,
            "language_code": lang_code,
        }
        if prompt:
            data["prompt"] = prompt

        # Determine audio extension and MIME type from binary signatures
        detected_fmt = verify_audio_magic_bytes(audio_bytes) or "wav"
        mime_map = {
            "wav": ("audio.wav", "audio/wav"),
            "mp3": ("audio.mp3", "audio/mpeg"),
            "ogg": ("audio.ogg", "audio/ogg"),
            "webm": ("audio.webm", "audio/webm"),
            "m4a": ("audio.m4a", "audio/mp4"),
            "mp4": ("audio.mp4", "audio/mp4"),
            "flac": ("audio.flac", "audio/flac"),
            "aac": ("audio.aac", "audio/aac"),
        }
        file_name, mime_type = mime_map.get(detected_fmt, ("audio.wav", "audio/wav"))

        files = {
            "file": (file_name, audio_bytes, mime_type),
        }

        try:
            client = await self._get_client()
            resp = await client.post(url, headers=headers, data=data, files=files)
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
                message=f"Sarvam STT returned HTTP {resp.status_code}: {resp.text[:200]}",
            )

        try:
            res_json = resp.json()
            transcript = res_json.get("transcript", "").strip()
            confidence = float(res_json.get("confidence", 0.95))
            duration_s = float(res_json.get("duration", 0.0))
        except Exception as exc:
            raise MalformedResponseError(
                provider_name="sarvam",
                message="Failed to parse Sarvam STT JSON response payload.",
            ) from exc

        return STTResult(
            text=transcript,
            language=res_json.get("language_code", lang_code),
            confidence=confidence,
            duration_s=duration_s,
            latency_ms=round(lat_ms, 2),
            provider=self.provider_name,
        )

