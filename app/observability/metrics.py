"""Thread-safe Prometheus-compatible metrics registry and collectors for Phase 6.8."""

import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class MetricValue:
    """Base class for metric data."""
    pass


class Counter:
    """Thread-safe Prometheus Counter metric."""

    def __init__(self, name: str, description: str, label_names: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._lock = threading.Lock()
        self._values: Dict[Tuple[str, ...], float] = {}

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment counter by specified amount (default 1.0)."""
        if amount < 0:
            raise ValueError("Counter increments must be non-negative.")
        key = self._format_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value for labels."""
        key = self._format_key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def _format_key(self, labels: Optional[Dict[str, str]]) -> Tuple[str, ...]:
        if not self.label_names:
            return ()
        labels = labels or {}
        return tuple(str(labels.get(name, "")) for name in self.label_names)

    def export(self) -> str:
        """Export counter in standard Prometheus format."""
        with self._lock:
            lines = [
                f"# HELP {self.name} {self.description}",
                f"# TYPE {self.name} counter",
            ]
            if not self._values and not self.label_names:
                lines.append(f"{self.name} 0.0")
            for label_tuple, val in sorted(self._values.items()):
                if self.label_names:
                    label_pairs = [f'{name}="{val}"' for name, val in zip(self.label_names, label_tuple)]
                    label_str = "{" + ",".join(label_pairs) + "}"
                    lines.append(f"{self.name}{label_str} {val}")
                else:
                    lines.append(f"{self.name} {val}")
            return "\n".join(lines)


class Gauge:
    """Thread-safe Prometheus Gauge metric."""

    def __init__(self, name: str, description: str, label_names: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._lock = threading.Lock()
        self._values: Dict[Tuple[str, ...], float] = {}

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge to specified value."""
        key = self._format_key(labels)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment gauge value."""
        key = self._format_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement gauge value."""
        key = self._format_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) - amount

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        key = self._format_key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def _format_key(self, labels: Optional[Dict[str, str]]) -> Tuple[str, ...]:
        if not self.label_names:
            return ()
        labels = labels or {}
        return tuple(str(labels.get(name, "")) for name in self.label_names)

    def export(self) -> str:
        """Export gauge in standard Prometheus format."""
        with self._lock:
            lines = [
                f"# HELP {self.name} {self.description}",
                f"# TYPE {self.name} gauge",
            ]
            if not self._values and not self.label_names:
                lines.append(f"{self.name} 0.0")
            for label_tuple, val in sorted(self._values.items()):
                if self.label_names:
                    label_pairs = [f'{name}="{val}"' for name, val in zip(self.label_names, label_tuple)]
                    label_str = "{" + ",".join(label_pairs) + "}"
                    lines.append(f"{self.name}{label_str} {val}")
                else:
                    lines.append(f"{self.name} {val}")
            return "\n".join(lines)


class Histogram:
    """Thread-safe Prometheus Histogram metric with configurable bucket boundaries."""

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self.buckets = tuple(sorted(buckets or self.DEFAULT_BUCKETS)) + (float("inf"),)
        self._lock = threading.Lock()
        # Structure: key -> {"sum": float, "count": int, "buckets": {bound: int}}
        self._data: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record an observed value into the histogram."""
        key = self._format_key(labels)
        with self._lock:
            if key not in self._data:
                self._data[key] = {
                    "sum": 0.0,
                    "count": 0,
                    "buckets": {b: 0 for b in self.buckets},
                }
            entry = self._data[key]
            entry["sum"] += float(value)
            entry["count"] += 1
            for b in self.buckets:
                if value <= b:
                    entry["buckets"][b] += 1

    def _format_key(self, labels: Optional[Dict[str, str]]) -> Tuple[str, ...]:
        if not self.label_names:
            return ()
        labels = labels or {}
        return tuple(str(labels.get(name, "")) for name in self.label_names)

    def export(self) -> str:
        """Export histogram series, buckets, sum, and count in standard Prometheus format."""
        with self._lock:
            lines = [
                f"# HELP {self.name} {self.description}",
                f"# TYPE {self.name} histogram",
            ]
            for label_tuple, entry in sorted(self._data.items()):
                base_labels = []
                if self.label_names:
                    base_labels = [f'{name}="{val}"' for name, val in zip(self.label_names, label_tuple)]

                # Cumulative buckets
                for b in self.buckets:
                    b_str = "+Inf" if b == float("inf") else str(b)
                    bucket_labels = base_labels + [f'le="{b_str}"']
                    label_str = "{" + ",".join(bucket_labels) + "}"
                    lines.append(f"{self.name}_bucket{label_str} {entry['buckets'][b]}")

                # Sum and count
                label_str = "{" + ",".join(base_labels) + "}" if base_labels else ""
                lines.append(f"{self.name}_sum{label_str} {entry['sum']}")
                lines.append(f"{self.name}_count{label_str} {entry['count']}")

            return "\n".join(lines)


class MetricsRegistry:
    """Centralized thread-safe Prometheus metrics registry for the Voice RAG backend."""

    def __init__(self):
        self._lock = threading.Lock()

        # ---------------------------------------------------------------------
        # 1. HTTP Server Metrics
        # ---------------------------------------------------------------------
        self.http_requests_total = Counter(
            name="http_requests_total",
            description="Total HTTP requests processed by endpoint, method, status, and language.",
            label_names=["endpoint", "method", "status", "language"],
        )
        self.http_request_duration_seconds = Histogram(
            name="http_request_duration_seconds",
            description="HTTP request processing latency in seconds.",
            label_names=["endpoint", "method", "status"],
            buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
        )

        # ---------------------------------------------------------------------
        # 2. RAG Retrieval Metrics
        # ---------------------------------------------------------------------
        self.rag_retrieval_requests_total = Counter(
            name="rag_retrieval_requests_total",
            description="Total retrieval executions by language, tier, and cache state.",
            label_names=["language", "tier", "cache_hit"],
        )
        self.rag_retrieval_duration_seconds = Histogram(
            name="rag_retrieval_duration_seconds",
            description="Retrieval and neural reranking latency in seconds.",
            label_names=["language", "tier"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3),
        )
        self.rag_dense_duration_seconds = Histogram(
            name="rag_dense_duration_seconds",
            description="Dense embedding and Qdrant ANN search latency in seconds.",
            label_names=["language"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2),
        )
        self.rag_bm25_duration_seconds = Histogram(
            name="rag_bm25_duration_seconds",
            description="BM25 lexical retrieval latency in seconds.",
            label_names=["language"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
        )
        self.rag_rerank_duration_seconds = Histogram(
            name="rag_rerank_duration_seconds",
            description="Neural Cross-Encoder reranking latency in seconds.",
            label_names=["language"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.15),
        )

        # ---------------------------------------------------------------------
        # 3. Generation & Guardrails Metrics
        # ---------------------------------------------------------------------
        self.rag_generation_duration_seconds = Histogram(
            name="rag_generation_duration_seconds",
            description="Prompt building, LLM generation, and grounding validation latency in seconds.",
            label_names=["language", "status"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
        )
        self.rag_guardrail_refusals_total = Counter(
            name="rag_guardrail_refusals_total",
            description="Total requests refused or filtered by guardrails.",
            label_names=["language", "stage"],
        )
        self.rag_unsupported_answers_total = Counter(
            name="rag_unsupported_answers_total",
            description="Total ungrounded answers detected and replaced with fallback.",
            label_names=["language"],
        )

        # ---------------------------------------------------------------------
        # 4. Voice I/O Metrics
        # ---------------------------------------------------------------------
        self.rag_voice_duration_seconds = Histogram(
            name="rag_voice_duration_seconds",
            description="Voice I/O latency (STT + TTS) in seconds.",
            label_names=["component", "language"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
        )
        self.rag_end_to_end_duration_seconds = Histogram(
            name="rag_end_to_end_duration_seconds",
            description="Complete end-to-end Voice RAG pipeline latency in seconds.",
            label_names=["language", "status"],
            buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5),
        )

        # ---------------------------------------------------------------------
        # 5. Cache Metrics
        # ---------------------------------------------------------------------
        self.rag_cache_hits_total = Counter(
            name="rag_cache_hits_total",
            description="Total retrieval cache hits by language.",
            label_names=["language"],
        )
        self.rag_cache_misses_total = Counter(
            name="rag_cache_misses_total",
            description="Total retrieval cache misses by language.",
            label_names=["language"],
        )
        self.rag_cache_evictions_total = Counter(
            name="rag_cache_evictions_total",
            description="Total cache entries evicted due to capacity bounds.",
        )
        self.rag_cache_expirations_total = Counter(
            name="rag_cache_expirations_total",
            description="Total cache entries expired due to TTL expiry.",
        )
        self.rag_cache_entries = Gauge(
            name="rag_cache_entries",
            description="Current count of active in-memory cached entries.",
        )

        # ---------------------------------------------------------------------
        # 6. Provider Health, Timeouts & Fallbacks
        # ---------------------------------------------------------------------
        self.rag_provider_requests_total = Counter(
            name="rag_provider_requests_total",
            description="Total calls executed against external providers.",
            label_names=["provider", "component", "status"],
        )
        self.rag_provider_failures_total = Counter(
            name="rag_provider_failures_total",
            description="Total provider failures encountered.",
            label_names=["provider", "component", "error_type"],
        )
        self.rag_provider_timeouts_total = Counter(
            name="rag_provider_timeouts_total",
            description="Total component timeouts encountered.",
            label_names=["provider", "component"],
        )
        self.rag_fallbacks_total = Counter(
            name="rag_fallbacks_total",
            description="Total graceful fallbacks triggered across subsystems.",
            label_names=["component", "fallback_type"],
        )
        self.rag_rerank_skips_total = Counter(
            name="rag_rerank_skips_total",
            description="Total times adaptive routing safely skipped neural reranking.",
            label_names=["language"],
        )
        self.rag_provider_health = Gauge(
            name="rag_provider_health",
            description="Lightweight health indicator of providers (1=healthy, 0=unhealthy).",
            label_names=["provider", "component"],
        )
        self.rag_last_retrieval_confidence = Gauge(
            name="rag_last_retrieval_confidence",
            description="Most recent retrieval confidence score by language.",
            label_names=["language"],
        )

        # Initialize default provider health states (1.0 = healthy)
        self._provider_health_map: Dict[str, Dict[str, Any]] = {
            "sarvam_stt": {"status": "healthy", "healthy": True, "last_check": time.time()},
            "sarvam_tts": {"status": "healthy", "healthy": True, "last_check": time.time()},
            "sarvam_llm": {"status": "healthy", "healthy": True, "last_check": time.time()},
            "qdrant": {"status": "healthy", "healthy": True, "last_check": time.time()},
        }
        self.rag_provider_health.set(1.0, {"provider": "sarvam", "component": "stt"})
        self.rag_provider_health.set(1.0, {"provider": "sarvam", "component": "tts"})
        self.rag_provider_health.set(1.0, {"provider": "sarvam", "component": "llm"})
        self.rag_provider_health.set(1.0, {"provider": "qdrant", "component": "vector"})

    # -------------------------------------------------------------------------
    # Convenience Recording Methods
    # -------------------------------------------------------------------------
    def record_http_request(
        self, endpoint: str, method: str, status_code: int, duration_s: float, language: str = "en"
    ) -> None:
        """Record an incoming HTTP request and its latency."""
        norm_lang = (language or "en").lower()[:5]
        norm_ep = endpoint if endpoint in ("/api/ask", "/api/voice-ask", "/api/retrieve", "/health", "/metrics", "/") else "other"
        status_str = str(status_code)
        self.http_requests_total.inc(labels={"endpoint": norm_ep, "method": method, "status": status_str, "language": norm_lang})
        self.http_request_duration_seconds.observe(duration_s, labels={"endpoint": norm_ep, "method": method, "status": status_str})

    def record_retrieval(
        self,
        language: str,
        tier: str,
        cache_hit: bool,
        duration_s: float,
        dense_s: float = 0.0,
        bm25_s: float = 0.0,
        rerank_s: float = 0.0,
        confidence: float = 0.0,
    ) -> None:
        """Record retrieval telemetry metrics."""
        norm_lang = (language or "en").lower()[:5]
        tier_str = str(tier or "none")
        self.rag_retrieval_requests_total.inc(labels={"language": norm_lang, "tier": tier_str, "cache_hit": str(cache_hit).lower()})
        self.rag_retrieval_duration_seconds.observe(duration_s, labels={"language": norm_lang, "tier": tier_str})
        if dense_s > 0:
            self.rag_dense_duration_seconds.observe(dense_s, labels={"language": norm_lang})
        if bm25_s > 0:
            self.rag_bm25_duration_seconds.observe(bm25_s, labels={"language": norm_lang})
        if rerank_s > 0:
            self.rag_rerank_duration_seconds.observe(rerank_s, labels={"language": norm_lang})
        self.rag_last_retrieval_confidence.set(confidence, labels={"language": norm_lang})

    def record_generation(self, language: str, duration_s: float, status: str = "success") -> None:
        """Record generation latency."""
        norm_lang = (language or "en").lower()[:5]
        self.rag_generation_duration_seconds.observe(duration_s, labels={"language": norm_lang, "status": status})

    def record_voice(self, component: str, language: str, duration_s: float) -> None:
        """Record voice I/O latency (stt / tts)."""
        norm_lang = (language or "en").lower()[:5]
        self.rag_voice_duration_seconds.observe(duration_s, labels={"component": component, "language": norm_lang})

    def record_end_to_end(self, duration_s: float, language: str = "en", status: str = "success") -> None:
        """Record end-to-end Voice RAG pipeline latency."""
        norm_lang = (language or "en").lower()[:5]
        self.rag_end_to_end_duration_seconds.observe(duration_s, labels={"language": norm_lang, "status": status})

    def record_cache_hit(self, language: str = "en") -> None:
        """Record cache hit."""
        self.rag_cache_hits_total.inc(labels={"language": (language or "en").lower()[:5]})

    def record_cache_miss(self, language: str = "en") -> None:
        """Record cache miss."""
        self.rag_cache_misses_total.inc(labels={"language": (language or "en").lower()[:5]})

    def record_cache_eviction(self) -> None:
        """Record cache eviction."""
        self.rag_cache_evictions_total.inc()

    def record_cache_expiration(self) -> None:
        """Record cache expiration."""
        self.rag_cache_expirations_total.inc()

    def record_provider_failure(self, provider: str, component: str, error_type: str = "exception") -> None:
        """Record provider failure and update health gauge."""
        self.rag_provider_failures_total.inc(labels={"provider": provider, "component": component, "error_type": error_type})
        self.rag_provider_health.set(0.0, labels={"provider": provider, "component": component})
        key = f"{provider}_{component}"
        if key in self._provider_health_map:
            self._provider_health_map[key] = {"status": "degraded", "healthy": False, "last_check": time.time()}

    def record_provider_timeout(self, provider: str, component: str) -> None:
        """Record provider timeout."""
        self.rag_provider_timeouts_total.inc(labels={"provider": provider, "component": component})
        self.record_provider_failure(provider, component, error_type="timeout")

    def record_fallback(self, component: str, fallback_type: str) -> None:
        """Record subsystem fallback usage."""
        self.rag_fallbacks_total.inc(labels={"component": component, "fallback_type": fallback_type})

    def record_guardrail_refusal(self, language: str, stage: str = "pre_generation") -> None:
        """Record guardrail refusal."""
        norm_lang = (language or "en").lower()[:5]
        self.rag_guardrail_refusals_total.inc(labels={"language": norm_lang, "stage": stage})

    def record_unsupported_answer(self, language: str) -> None:
        """Record ungrounded answer detection."""
        norm_lang = (language or "en").lower()[:5]
        self.rag_unsupported_answers_total.inc(labels={"language": norm_lang})

    def record_rerank_skip(self, language: str) -> None:
        """Record reranker skip."""
        norm_lang = (language or "en").lower()[:5]
        self.rag_rerank_skips_total.inc(labels={"language": norm_lang})

    def update_provider_health(self, provider: str, component: str, is_healthy: bool) -> None:
        """Update provider health state."""
        self.rag_provider_health.set(1.0 if is_healthy else 0.0, labels={"provider": provider, "component": component})
        key = f"{provider}_{component}"
        self._provider_health_map[key] = {
            "status": "healthy" if is_healthy else "degraded",
            "healthy": is_healthy,
            "last_check": time.time(),
        }

    def get_provider_health_summary(self) -> Dict[str, Any]:
        """Return lightweight provider health summary for /health endpoint."""
        return {k: dict(v) for k, v in self._provider_health_map.items()}

    def generate_prometheus_text(self) -> str:
        """Export all registered metrics in standard Prometheus text format."""
        sections = [
            self.http_requests_total.export(),
            self.http_request_duration_seconds.export(),
            self.rag_retrieval_requests_total.export(),
            self.rag_retrieval_duration_seconds.export(),
            self.rag_dense_duration_seconds.export(),
            self.rag_bm25_duration_seconds.export(),
            self.rag_rerank_duration_seconds.export(),
            self.rag_generation_duration_seconds.export(),
            self.rag_voice_duration_seconds.export(),
            self.rag_end_to_end_duration_seconds.export(),
            self.rag_cache_hits_total.export(),
            self.rag_cache_misses_total.export(),
            self.rag_cache_evictions_total.export(),
            self.rag_cache_expirations_total.export(),
            self.rag_cache_entries.export(),
            self.rag_provider_requests_total.export(),
            self.rag_provider_failures_total.export(),
            self.rag_provider_timeouts_total.export(),
            self.rag_fallbacks_total.export(),
            self.rag_guardrail_refusals_total.export(),
            self.rag_unsupported_answers_total.export(),
            self.rag_rerank_skips_total.export(),
            self.rag_provider_health.export(),
            self.rag_last_retrieval_confidence.export(),
        ]
        return "\n\n".join(s for s in sections if s) + "\n"


_REGISTRY_INSTANCE: Optional[MetricsRegistry] = None


def get_metrics_registry() -> MetricsRegistry:
    """Return singleton MetricsRegistry instance."""
    global _REGISTRY_INSTANCE
    if _REGISTRY_INSTANCE is None:
        _REGISTRY_INSTANCE = MetricsRegistry()
    return _REGISTRY_INSTANCE
