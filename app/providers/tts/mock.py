"""Mock Text-to-Speech provider for tests and development environments."""

import time
from typing import Optional

from app.providers.tts.base import TTSProvider, TTSResult


class MockTTSProvider(TTSProvider):
    """Deterministic mock TTS provider that produces synthetic audio frames without external calls."""

    def __init__(
        self,
        default_sample_rate: int = 24000,
        simulated_latency_ms: float = 0.0,
    ):
        self.default_sample_rate = default_sample_rate
        self.simulated_latency_ms = simulated_latency_ms

    @property
    def provider_name(self) -> str:
        return "mock"

    async def synthesize(
        self,
        text: str,
        language: str = "hi",
        speaker: Optional[str] = None,
        pace: Optional[float] = None,
    ) -> TTSResult:
        t_start = time.perf_counter()
        clean_text = text.strip() if text else ""

        # Generate deterministic synthetic WAV-like header + payload (~100ms audio per 5 chars)
        duration_est = max(0.1, len(clean_text) * 0.05)
        num_samples = int(duration_est * self.default_sample_rate)
        # Synthetic PCM byte payload (silent / flat audio)
        dummy_audio = b"\x00\x00" * num_samples

        lat_ms = (time.perf_counter() - t_start) * 1000.0 + self.simulated_latency_ms

        return TTSResult(
            audio_bytes=dummy_audio,
            audio_format="wav",
            sample_rate=self.default_sample_rate,
            duration_s=round(duration_est, 2),
            latency_ms=round(lat_ms, 2),
            provider=self.provider_name,
        )
