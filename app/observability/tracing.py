"""Context-aware request tracing and correlation utilities for Phase 6.8."""

import contextvars
import time
import uuid
from typing import Optional

# Async-safe context variables for request correlation
_REQUEST_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")
_LANGUAGE_CTX: contextvars.ContextVar[str] = contextvars.ContextVar("language", default="en")
_START_TIME_CTX: contextvars.ContextVar[float] = contextvars.ContextVar("start_time", default=0.0)


def generate_request_id() -> str:
    """Generate a unique UUID4 request correlation ID."""
    return str(uuid.uuid4())


def get_current_request_id() -> str:
    """Retrieve current request ID from async context, or generate one if unset."""
    req_id = _REQUEST_ID_CTX.get()
    if not req_id:
        req_id = generate_request_id()
        _REQUEST_ID_CTX.set(req_id)
    return req_id


def set_current_request_id(request_id: str) -> None:
    """Set the current request ID in async context."""
    _REQUEST_ID_CTX.set(request_id)


def get_current_language() -> str:
    """Retrieve current request language code from async context."""
    return _LANGUAGE_CTX.get()


def set_current_language(language: str) -> None:
    """Set the current request language code in async context."""
    _LANGUAGE_CTX.set(language or "en")


def set_request_start_time(start_time: Optional[float] = None) -> None:
    """Set the request monotonic start timestamp."""
    _START_TIME_CTX.set(start_time if start_time is not None else time.perf_counter())


def get_request_elapsed_ms() -> float:
    """Calculate elapsed time in ms since request started."""
    start_t = _START_TIME_CTX.get()
    if start_t <= 0.0:
        return 0.0
    return max(0.0, (time.perf_counter() - start_t) * 1000.0)
