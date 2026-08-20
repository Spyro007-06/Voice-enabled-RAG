"""Observability package for Phase 6.8: metrics, structured logging, middleware, tracing."""

from app.observability.logging import StructuredJSONFormatter, setup_observability_logging
from app.observability.metrics import MetricsRegistry, get_metrics_registry
from app.observability.middleware import ObservabilityMiddleware
from app.observability.tracing import (
    generate_request_id,
    get_current_language,
    get_current_request_id,
    get_request_elapsed_ms,
    set_current_language,
    set_current_request_id,
)

__all__ = [
    "MetricsRegistry",
    "get_metrics_registry",
    "StructuredJSONFormatter",
    "setup_observability_logging",
    "ObservabilityMiddleware",
    "generate_request_id",
    "get_current_request_id",
    "set_current_request_id",
    "get_current_language",
    "set_current_language",
    "get_request_elapsed_ms",
]
