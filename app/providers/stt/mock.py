"""Mock Speech-to-Text provider for tests and development environments."""

import time
from typing import Optional

from app.providers.stt.base import STTProvider, STTResult


class MockSTTProvider(STTProvider):
    """Deterministic mock STT provider."""

    def __init__(
        self,
        default_text: str = "गोवा की राजधानी क्या है?",
        default_language: str = "hi",
        confidence: float = 0.98,
        simulated_latency_ms: float = 0.0,
    ):
        self.default_text = default_text
        self.default_language = default_language
        self.confidence = confidence
        self.simulated_latency_ms = simulated_latency_ms

    @property
    def provider_name(self) -> str:
        return "mock"

    async def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> STTResult:
        t_start = time.perf_counter()
        if not audio_bytes:
            return STTResult(
                text="",
                language=language or self.default_language,
                confidence=0.0,
                duration_s=0.0,
                latency_ms=0.0,
                provider=self.provider_name,
            )

        duration_est = max(0.1, len(audio_bytes) / 32000.0)  # rough 16kHz 16-bit mono estimation
        lat_ms = (time.perf_counter() - t_start) * 1000.0 + self.simulated_latency_ms

        return STTResult(
            text=self.default_text,
            language=language or self.default_language,
            confidence=self.confidence,
            duration_s=round(duration_est, 2),
            latency_ms=round(lat_ms, 2),
            provider=self.provider_name,
        )
