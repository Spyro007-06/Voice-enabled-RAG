"""HTTP Request correlation and observability middleware for FastAPI."""

import logging
import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.observability.metrics import get_metrics_registry
from app.observability.tracing import (
    get_current_language,
    get_current_request_id,
    set_current_language,
    set_current_request_id,
    set_request_start_time,
)

logger = logging.getLogger("hhgoa_voice_rag.middleware")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Middleware that injects request correlation IDs, tracks HTTP latencies, and emits telemetry."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        t0 = time.perf_counter()
        set_request_start_time(t0)

        # 1. Resolve or Generate Request ID
        from app.observability.security import sanitize_header_value
        raw_req_id = request.headers.get("X-Request-ID") or request.headers.get("x-request-id")
        req_id = sanitize_header_value(raw_req_id)
        if not req_id:
            from app.observability.tracing import generate_request_id
            req_id = generate_request_id()
        set_current_request_id(req_id)

        # 2. Resolve Language Hint
        lang = request.headers.get("Accept-Language", "en").split(",")[0].split("-")[0].strip().lower()
        if lang in ("hi", "ta", "te", "ml", "en"):
            set_current_language(lang)
        else:
            set_current_language("en")

        status_code = 500
        path = request.url.path
        method = request.method
        registry = get_metrics_registry()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            duration_s = time.perf_counter() - t0
            current_lang = get_current_language()
            registry.record_http_request(
                endpoint=path,
                method=method,
                status_code=500,
                duration_s=duration_s,
                language=current_lang,
            )
            logger.error(
                "HTTP request error: %s %s -> %s (%.2fms)",
                method,
                path,
                exc,
                duration_s * 1000.0,
                extra={
                    "request_id": req_id,
                    "endpoint": path,
                    "status_code": 500,
                    "latency_ms": duration_s * 1000.0,
                    "language": current_lang,
                    "event": "request_error",
                },
                exc_info=True,
            )
            raise exc

        duration_s = time.perf_counter() - t0
        current_lang = get_current_language()

        # 3. Attach Request Correlation ID Header
        response.headers["X-Request-ID"] = req_id

        # 4. Record Metrics
        registry.record_http_request(
            endpoint=path,
            method=method,
            status_code=status_code,
            duration_s=duration_s,
            language=current_lang,
        )

        # 5. Log structured completion (debug/info)
        logger.info(
            "%s %s %d (%.2fms)",
            method,
            path,
            status_code,
            duration_s * 1000.0,
            extra={
                "request_id": req_id,
                "endpoint": path,
                "status_code": status_code,
                "latency_ms": duration_s * 1000.0,
                "language": current_lang,
                "event": "request_completed",
            },
        )

        return response
