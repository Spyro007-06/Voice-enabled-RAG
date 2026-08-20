"""Structured JSON logging with automated credential and audio redaction for Phase 6.8."""

import json
import logging
import re
import time
from typing import Any, Dict, Optional

from app.observability.tracing import get_current_language, get_current_request_id

# Regex patterns matching secret tokens, API keys, bearer auth headers
SENSITIVE_PATTERNS = [
    re.compile(r"(api[_-]?key\s*[:=]\s*['\"]?)([a-zA-Z0-9_\-\.]{8,})(['\"]?)", re.IGNORECASE),
    re.compile(r"(bearer\s+)([a-zA-Z0-9_\-\.]{8,})", re.IGNORECASE),
    re.compile(r"(authorization\s*[:=]\s*['\"]?)([a-zA-Z0-9_\-\.\s]{8,})(['\"]?)", re.IGNORECASE),
    re.compile(r"(sarvam[_-]?api[_-]?key\s*[:=]\s*['\"]?)([a-zA-Z0-9_\-\.]{8,})(['\"]?)", re.IGNORECASE),
    re.compile(r"(openai[_-]?api[_-]?key\s*[:=]\s*['\"]?)([a-zA-Z0-9_\-\.]{8,})(['\"]?)", re.IGNORECASE),
    re.compile(r"(qdrant[_-]?api[_-]?key\s*[:=]\s*['\"]?)([a-zA-Z0-9_\-\.]{8,})(['\"]?)", re.IGNORECASE),
    re.compile(r"(sk-[a-zA-Z0-9]{20,})", re.IGNORECASE),
]

# Pattern matching long base64 audio strings (80+ chars)
AUDIO_BASE64_PATTERN = re.compile(r"([A-Za-z0-9+/=]{80,})")


def sanitize_log_message(msg: str) -> str:
    """Mask credentials, API keys, tokens, and raw base64 payloads from log messages."""
    if not isinstance(msg, str):
        msg = str(msg)

    # Redact sensitive key patterns
    for pattern in SENSITIVE_PATTERNS:
        msg = pattern.sub(r"\1***REDACTED***", msg)

    # Redact raw binary byte representations
    if "b'RIFF" in msg or 'b"RIFF' in msg:
        msg = re.sub(r"b['\"]RIFF.*?['\"]", "<AUDIO_PAYLOAD_REDACTED>", msg)

    # Redact long base64 audio payloads (>=80 chars)
    msg = AUDIO_BASE64_PATTERN.sub("<BASE64_AUDIO_REDACTED>", msg)

    return msg


class StructuredJSONFormatter(logging.Formatter):
    """Logging formatter outputting RFC 3339 timestamped structured JSON with context correlation."""

    def format(self, record: logging.LogRecord) -> str:
        # Monotonic/UTC timestamp
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created))

        # Retrieve request context
        req_id = getattr(record, "request_id", None) or get_current_request_id()
        lang = getattr(record, "language", None) or get_current_language()

        # Format message safely
        raw_message = record.getMessage()
        sanitized_msg = sanitize_log_message(raw_message)

        log_data: Dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": sanitized_msg,
            "request_id": req_id,
            "language": lang,
        }

        # Include custom extra fields if provided
        if hasattr(record, "endpoint"):
            log_data["endpoint"] = record.endpoint
        if hasattr(record, "status_code"):
            log_data["status"] = record.status_code
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = round(float(record.latency_ms), 2)
        if hasattr(record, "event"):
            log_data["event"] = record.event

        # Include exception trace if present (sanitized)
        if record.exc_info:
            if isinstance(record.exc_info, tuple) and len(record.exc_info) == 3:
                log_data["exception"] = sanitize_log_message(self.formatException(record.exc_info))
            else:
                log_data["exception"] = sanitize_log_message(str(record.exc_info))

        return json.dumps(log_data, ensure_ascii=False)


def setup_observability_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure root / application logger to use StructuredJSONFormatter."""
    root_logger = logging.getLogger()
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Update existing StreamHandlers or add one
    has_stream_handler = False
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(StructuredJSONFormatter())
            has_stream_handler = True

    if not has_stream_handler:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredJSONFormatter())
        root_logger.addHandler(handler)

    return logging.getLogger("hhgoa_voice_rag")
