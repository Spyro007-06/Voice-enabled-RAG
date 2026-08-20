"""Security utilities for credential redaction, path masking, header sanitization, and output scrubbing."""

import re
from typing import Any, Dict, List, Optional, Union

# Common credential patterns to scrub from outputs, errors, and logs
SENSITIVE_PATTERNS = [
    re.compile(r"(api[_-]?key\s*[:=]\s*['\"]?)([a-zA-Z0-9_\-\.]{8,})(['\"]?)", re.IGNORECASE),
    re.compile(r"(bearer\s+)([a-zA-Z0-9_\-\.]{8,})", re.IGNORECASE),
    re.compile(r"(authorization\s*[:=]\s*['\"]?)([a-zA-Z0-9_\-\.\s]{8,})(['\"]?)", re.IGNORECASE),
    re.compile(r"(sarvam[_-]?api[_-]?key\s*[:=]\s*['\"]?)([a-zA-Z0-9_\-\.]{8,})(['\"]?)", re.IGNORECASE),
    re.compile(r"(openai[_-]?api[_-]?key\s*[:=]\s*['\"]?)([a-zA-Z0-9_\-\.]{8,})(['\"]?)", re.IGNORECASE),
    re.compile(r"(qdrant[_-]?api[_-]?key\s*[:=]\s*['\"]?)([a-zA-Z0-9_\-\.]{8,})(['\"]?)", re.IGNORECASE),
    re.compile(r"(sk-[a-zA-Z0-9]{20,})", re.IGNORECASE),
]

# Internal filesystem path patterns (Windows & POSIX)
PATH_PATTERNS = [
    re.compile(r"[a-zA-Z]:\\[^\s\"\'<>]+", re.IGNORECASE),  # Windows absolute paths
    re.compile(r"(?:/home/|/var/|/tmp/|/app/|/etc/|/root/|/Users/)[^\s\"\'<>]+", re.IGNORECASE),  # POSIX absolute paths
]

# Header / Request-ID sanitization
SAFE_HEADER_REGEX = re.compile(r"[^a-zA-Z0-9\-_.]")


def sanitize_output_text(text: str) -> str:
    """Scrub sensitive credentials, API keys, and internal filesystem paths from text responses.

    Args:
        text: Raw response or message text.

    Returns:
        str: Sanitized text safe for client-facing exposure.
    """
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 1. Scrub credentials
    for pattern in SENSITIVE_PATTERNS:
        text = pattern.sub(r"\1***REDACTED***", text)

    # 2. Scrub internal filesystem paths
    for pattern in PATH_PATTERNS:
        text = pattern.sub("[INTERNAL_PATH]", text)

    return text


def sanitize_header_value(value: Optional[str], max_len: int = 64) -> str:
    """Sanitize user-provided header values (e.g. X-Request-ID) to prevent CRLF/header injection.

    Args:
        value: Raw header value string.
        max_len: Maximum permitted character length.

    Returns:
        str: Sanitized header string or empty string if invalid.
    """
    if not value or not isinstance(value, str):
        return ""
    clean = SAFE_HEADER_REGEX.sub("", value.strip())
    return clean[:max_len]


def sanitize_error_detail(detail: Any) -> Any:
    """Sanitize error detail message to prevent leaking internal stack traces or paths."""
    if isinstance(detail, str):
        return sanitize_output_text(detail)
    if isinstance(detail, list):
        return [sanitize_error_detail(item) for item in detail]
    if isinstance(detail, dict):
        return {k: sanitize_error_detail(v) for k, v in detail.items()}
    return detail
