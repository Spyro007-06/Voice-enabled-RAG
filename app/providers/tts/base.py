"""Abstract Base Interface for Text-to-Speech (TTS) Providers."""

from abc import abstractmethod
from typing import Optional
from pydantic import BaseModel, Field

from app.providers.base import BaseProvider


class TTSResult(BaseModel):
    """Structured result returned by Text-to-Speech synthesis."""

    audio_bytes: bytes = Field(..., description="Synthesized binary audio payload.")
    audio_format: str = Field("wav", description="Audio container format ('wav', 'mp3', etc.).")
    sample_rate: int = Field(24000, description="Sampling rate in Hz.")
    duration_s: float = Field(0.0, ge=0.0, description="Estimated duration in seconds.")
    latency_ms: float = Field(0.0, ge=0.0, description="Synthesis latency in milliseconds.")
    provider: str = Field(..., description="Name of the TTS provider.")


class TTSProvider(BaseProvider):
    """Abstract interface for Text-to-Speech synthesis providers."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        language: str = "hi",
        speaker: Optional[str] = None,
        pace: Optional[float] = None,
    ) -> TTSResult:
        """Synthesize text into speech audio bytes.

        Args:
            text: Input text transcript to synthesize.
            language: BCP-47 / ISO language code (e.g. 'hi-IN', 'ta-IN', 'en-IN').
            speaker: Optional speaker identifier voice profile.
            pace: Speech rate multiplier (e.g. 1.0 = normal).

        Returns:
            TTSResult: Structured audio synthesis result.
        """
        pass
