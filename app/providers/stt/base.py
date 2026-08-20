"""Abstract Base Interface for Speech-to-Text (STT) Providers."""

from abc import abstractmethod
from typing import Optional
from pydantic import BaseModel, Field

from app.providers.base import BaseProvider


class STTResult(BaseModel):
    """Structured result returned by Speech-to-Text transcription."""

    text: str = Field(..., description="Transcribed text transcript.")
    language: Optional[str] = Field(None, description="Detected or specified BCP-47 language code.")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence score for transcription.")
    duration_s: float = Field(0.0, ge=0.0, description="Duration of the audio in seconds.")
    latency_ms: float = Field(0.0, ge=0.0, description="Latency of the STT call in milliseconds.")
    provider: str = Field(..., description="Name of the STT provider.")


class STTProvider(BaseProvider):
    """Abstract interface for Speech-to-Text providers."""

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> STTResult:
        """Transcribe raw audio bytes into text.

        Args:
            audio_bytes: Raw audio binary data (WAV, MP3, etc.).
            language: Optional BCP-47 / ISO language code (e.g. 'hi-IN', 'en-IN', 'ta-IN').
            prompt: Optional context prompt or hotwords for boosting domain recognition.

        Returns:
            STTResult: Structured transcription result.
        """
        pass
