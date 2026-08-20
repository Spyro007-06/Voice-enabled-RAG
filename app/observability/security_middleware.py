"""Production Security, Rate Limiting, and Request Validation Middleware for FastAPI."""

import logging
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple
from threading import Lock

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import get_settings
from app.observability.security import sanitize_header_value
from app.observability.tracing import get_current_request_id

logger = logging.getLogger("hhgoa_voice_rag.security")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware attaching standard HTTP security headers to all HTTP responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        settings = get_settings()

        if getattr(settings, "SECURITY_HEADERS_ENABLED", True):
            # Prevent MIME sniffing
            response.headers["X-Content-Type-Options"] = "nosniff"
            # Prevent clickjacking / frame embedding
            response.headers["X-Frame-Options"] = "DENY"
            # Strict referrer policy
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            # Restrict resource loading & frame ancestors
            response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'"
            # Restrict browser permissions (allow self for microphone voice queries)
            response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=(self)"

            # Cache control for dynamic/sensitive API endpoints
            if request.url.path.startswith("/api/"):
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
                response.headers["Pragma"] = "no-cache"

        return response


class InMemoryRateLimiter:
    """Thread-safe sliding-window rate limiter keyed by client identifier."""

    def __init__(self, requests_per_minute: int = 120):
        self.requests_per_minute = requests_per_minute
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._lock = Lock()

    def is_allowed(self, client_id: str, limit: Optional[int] = None) -> Tuple[bool, int]:
        """Check if request from client_id is allowed under rate limits.

        Returns:
            Tuple[bool, int]: (is_allowed, retry_after_seconds)
        """
        now = time.time()
        window = 60.0
        effective_limit = limit or self.requests_per_minute

        with self._lock:
            # Purge timestamps older than 60s
            timestamps = [t for t in self._history[client_id] if now - t < window]
            self._history[client_id] = timestamps

            if len(timestamps) >= effective_limit:
                oldest = timestamps[0]
                retry_after = max(1, int(window - (now - oldest)))
                return False, retry_after

            self._history[client_id].append(now)
            return True, 0

    def reset(self) -> None:
        """Clear rate limit history (useful for testing)."""
        with self._lock:
            self._history.clear()


# Global rate limiter instance
_global_rate_limiter = InMemoryRateLimiter()


def get_rate_limiter() -> InMemoryRateLimiter:
    """Return singleton rate limiter instance."""
    return _global_rate_limiter


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing per-client rate limiting on sensitive and expensive API endpoints."""

    def __init__(self, app, limiter: Optional[InMemoryRateLimiter] = None):
        super().__init__(app)
        self.limiter = limiter or get_rate_limiter()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        settings = get_settings()

        if not getattr(settings, "RATE_LIMIT_ENABLED", True):
            return await call_next(request)

        path = request.url.path
        # Exclude telemetry, docs, static frontend, and health check endpoints
        if path in ("/health", "/metrics", "/", "/docs", "/redoc", "/openapi.json") or path.startswith("/app"):
            return await call_next(request)

        # Resolve client identifier (IP address or X-Forwarded-For)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_id = forwarded.split(",")[0].strip()
        elif request.client and request.client.host:
            client_id = request.client.host
        else:
            client_id = "unknown_client"

        rpm = getattr(settings, "RATE_LIMIT_REQUESTS_PER_MINUTE", 120)
        allowed, retry_after = self.limiter.is_allowed(client_id, limit=rpm)

        if not allowed:
            req_id = get_current_request_id() or ""
            logger.warning(
                "Rate limit exceeded for client '%s' on %s %s",
                client_id,
                request.method,
                path,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "detail": "Rate limit exceeded. Please retry after the cooldown period.",
                    "path": path,
                    "request_id": req_id,
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)
