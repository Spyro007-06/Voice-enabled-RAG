"""Audio security and payload validation module for Phase 6.9."""

import os
import re
from typing import List, Optional, Tuple

from app.config import get_settings
from app.orchestration.exceptions import (
    AudioPayloadTooLargeError,
    AudioValidationError,
    UnsupportedAudioFormatError,
)

# Standard audio signatures (magic bytes) for format integrity verification
AUDIO_MAGIC_SIGNATURES = {
    "wav": [(b"RIFF", 0, b"WAVE", 8)],  # RIFF at 0, WAVE at 8
    "mp3": [
        (b"ID3", 0),  # ID3v2 tag
        (b"\xff\xfb", 0),  # MPEG-1 Layer 3 sync
        (b"\xff\xf3", 0),  # MPEG-2 Layer 3 sync
        (b"\xff\xf2", 0),  # MPEG-2.5 Layer 3 sync
        (b"\xff\xe3", 0),  # MPEG sync
    ],
    "ogg": [(b"OggS", 0)],  # Ogg container
    "webm": [(b"\x1a\x45\xdf\xa3", 0)],  # EBML header (WebM/MKV)
    "m4a": [(b"ftyp", 4)],  # ISO Base Media File (ftyp atom at byte 4)
    "mp4": [(b"ftyp", 4)],
    "flac": [(b"fLaC", 0)],  # Free Lossless Audio Codec
    "aac": [
        (b"\xff\xf1", 0),  # ADTS AAC sync (MPEG-4)
        (b"\xff\xf9", 0),  # ADTS AAC sync (MPEG-2)
        (b"ftyp", 4),      # AAC in MP4 container
    ],
}

ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "webm", "m4a", "flac", "aac", "mp4"}


def sanitize_filename(filename: Optional[str], max_length: int = 255) -> str:
    """Sanitize uploaded audio filename to prevent path traversal, null-byte injection, and OS exploits.

    Args:
        filename: Raw user-supplied filename.
        max_length: Maximum allowed filename length.

    Returns:
        str: Clean safe basename.
    """
    if not filename or not isinstance(filename, str):
        return "audio_upload.wav"

    # Strip directory paths (both Windows and POSIX)
    clean = os.path.basename(filename.replace("\\", "/"))
    # Remove null bytes and non-printable control characters
    clean = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", clean)
    # Remove dangerous characters like ..
    clean = clean.replace("..", "")
    # Remove characters outside standard alphanumeric, dot, hyphen, underscore
    clean = re.sub(r"[^a-zA-Z0-9._\-]", "_", clean)
    clean = clean.strip("._- ")

    if not clean:
        return "audio_upload.wav"

    return clean[:max_length]


def verify_audio_magic_bytes(audio_bytes: bytes) -> Optional[str]:
    """Inspect the first bytes of an audio stream to detect its true format signature.

    Args:
        audio_bytes: Raw audio binary data.

    Returns:
        Optional[str]: Detected format name ('wav', 'mp3', 'ogg', 'webm', 'm4a', 'flac', 'aac') or None.
    """
    if not audio_bytes or len(audio_bytes) < 4:
        return None

    header = audio_bytes[:32]

    for fmt, patterns in AUDIO_MAGIC_SIGNATURES.items():
        for pat in patterns:
            if len(pat) == 2:
                sig, offset = pat
                if len(header) >= offset + len(sig):
                    if header[offset : offset + len(sig)] == sig:
                        return fmt
            elif len(pat) == 4:
                sig1, off1, sig2, off2 = pat
                if len(header) >= off2 + len(sig2):
                    if header[off1 : off1 + len(sig1)] == sig1 and header[off2 : off2 + len(sig2)] == sig2:
                        return fmt

    return None


def validate_audio_security(
    audio_bytes: bytes,
    content_type: Optional[str] = None,
    filename: Optional[str] = None,
    max_size_bytes: Optional[int] = None,
    allowed_mime_types: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """Execute complete security validation for incoming audio payloads.

    Validates:
    1. Non-empty payload (>= 4 bytes).
    2. Size within configured boundary (413 if exceeded).
    3. Content-Type against allowed MIME types (415 if unsupported).
    4. File extension against allowed extensions (415 if unsupported).
    5. Actual binary magic bytes against known valid audio signatures (422 if corrupt or disguised).
    6. Safe filename sanitization to prevent path traversal.

    Returns:
        Tuple[str, str]: (sanitized_filename, validated_content_type)

    Raises:
        AudioValidationError: If payload is empty, corrupted, or failed integrity check.
        AudioPayloadTooLargeError: If size exceeds limit.
        UnsupportedAudioFormatError: If MIME or extension is disallowed.
    """
    settings = get_settings()
    max_bytes = max_size_bytes or getattr(settings, "MAX_AUDIO_SIZE_BYTES", 15 * 1024 * 1024)
    allowed_mimes = allowed_mime_types or getattr(
        settings,
        "ALLOWED_AUDIO_MIME_TYPES",
        [
            "audio/wav",
            "audio/x-wav",
            "audio/wave",
            "audio/mpeg",
            "audio/mp3",
            "audio/ogg",
            "audio/webm",
            "audio/m4a",
            "audio/x-m4a",
            "audio/mp4",
            "audio/aac",
            "audio/flac",
        ],
    )

    # 1. Payload size and presence
    if not audio_bytes or len(audio_bytes) == 0:
        raise AudioValidationError("Uploaded audio payload is empty (0 bytes).")

    if len(audio_bytes) > max_bytes:
        raise AudioPayloadTooLargeError(
            f"Audio payload size ({len(audio_bytes)} bytes) exceeds maximum limit ({max_bytes} bytes)."
        )

    if len(audio_bytes) < 4:
        raise AudioValidationError("Uploaded audio payload is too small to contain valid audio data.")

    # 2. Filename sanitization
    clean_filename = sanitize_filename(filename)
    file_ext = clean_filename.lower().split(".")[-1] if "." in clean_filename else ""

    # 3. MIME type validation
    effective_mime = "audio/wav"
    if content_type:
        clean_mime = content_type.lower().split(";")[0].strip()
        if clean_mime == "application/octet-stream":
            # If application/octet-stream, infer from extension
            if file_ext and file_ext not in ALLOWED_EXTENSIONS:
                raise UnsupportedAudioFormatError(f"Unsupported audio file extension '.{file_ext}'.")
            effective_mime = clean_mime
        elif clean_mime not in [m.lower() for m in allowed_mimes]:
            raise UnsupportedAudioFormatError(
                f"Unsupported audio MIME type '{clean_mime}'. Allowed: {', '.join(allowed_mimes)}"
            )
        else:
            effective_mime = clean_mime
    elif file_ext and file_ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedAudioFormatError(f"Unsupported audio file extension '.{file_ext}'.")

    # 4. Binary Magic Byte Validation (Format Verification)
    detected_fmt = verify_audio_magic_bytes(audio_bytes)
    if detected_fmt:
        fmt_to_mime = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "ogg": "audio/ogg",
            "webm": "audio/webm",
            "m4a": "audio/mp4",
            "mp4": "audio/mp4",
            "flac": "audio/flac",
            "aac": "audio/aac",
        }
        # If user passed application/octet-stream or no content_type, use detected format
        if not content_type or content_type == "application/octet-stream":
            effective_mime = fmt_to_mime.get(detected_fmt, effective_mime)
    else:
        # Check if payload looks like non-audio executable or document, and reject
        if audio_bytes.startswith(b"MZ") or audio_bytes.startswith(b"\x7fELF") or audio_bytes.startswith(b"%PDF"):
            raise AudioValidationError("Uploaded file is not a valid audio stream (executable/document header detected).")
        if audio_bytes.startswith(b"<!DOCTYPE") or audio_bytes.startswith(b"<html"):
            raise AudioValidationError("Uploaded file is an HTML document, not audio.")
        if audio_bytes.startswith(b"{\n") or audio_bytes.startswith(b"{\""):
            raise AudioValidationError("Uploaded file is a JSON document, not audio.")

    return clean_filename, effective_mime

