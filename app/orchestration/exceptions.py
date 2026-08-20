"""Domain-specific exceptions for Voice RAG Orchestration."""

from typing import Optional


class VoiceOrchestrationError(Exception):
    """Base exception for Voice RAG orchestration pipeline errors."""

    def __init__(self, message: str, stage: str = "orchestration", status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.status_code = status_code


class AudioValidationError(VoiceOrchestrationError):
    """Exception raised when uploaded audio fails format, size, or content validation."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message=message, stage="audio_validation", status_code=status_code)


class AudioPayloadTooLargeError(VoiceOrchestrationError):
    """Exception raised when uploaded audio payload exceeds configured size limit."""

    def __init__(self, message: str = "Audio payload exceeds maximum permitted size."):
        super().__init__(message=message, stage="audio_validation", status_code=413)


class UnsupportedAudioFormatError(VoiceOrchestrationError):
    """Exception raised when uploaded audio MIME type or extension is unsupported."""

    def __init__(self, message: str = "Unsupported audio content type."):
        super().__init__(message=message, stage="audio_validation", status_code=415)


class STTError(VoiceOrchestrationError):
    """Exception raised when Speech-to-Text transcription fails or times out."""

    def __init__(self, message: str, provider: Optional[str] = None):
        prov_info = f" ({provider})" if provider else ""
        super().__init__(message=f"STT transcription failed{prov_info}: {message}", stage="stt", status_code=502)
        self.provider = provider


class TTSError(VoiceOrchestrationError):
    """Exception raised when Text-to-Speech synthesis fails or times out."""

    def __init__(self, message: str, provider: Optional[str] = None):
        prov_info = f" ({provider})" if provider else ""
        super().__init__(message=f"TTS synthesis failed{prov_info}: {message}", stage="tts", status_code=502)
        self.provider = provider


class GroundingRefusalError(VoiceOrchestrationError):
    """Exception raised when grounding validation fails and answer cannot be safely delivered."""

    def __init__(self, message: str = "Generated response failed grounding validation and was rejected."):
        super().__init__(message=message, stage="grounding", status_code=422)
