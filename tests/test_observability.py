"""Tests for Phase 6.8 — Production Observability, Metrics, Structured Logging, and Tracing."""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.observability.logging import StructuredJSONFormatter, sanitize_log_message
from app.observability.metrics import Counter, Gauge, Histogram, MetricsRegistry, get_metrics_registry
from app.observability.tracing import (
    generate_request_id,
    get_current_language,
    get_current_request_id,
    set_current_language,
    set_current_request_id,
)

DUMMY_AUDIO = (
    b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00"
    b"\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
)


def test_counter_metrics_increment():
    """Verify Counter increments and thread-safe export."""
    counter = Counter(name="test_counter_total", description="Test counter", label_names=["status", "lang"])
    assert counter.get({"status": "200", "lang": "en"}) == 0.0

    counter.inc(labels={"status": "200", "lang": "en"})
    counter.inc(amount=2.5, labels={"status": "200", "lang": "en"})
    counter.inc(labels={"status": "500", "lang": "hi"})

    assert counter.get({"status": "200", "lang": "en"}) == 3.5
    assert counter.get({"status": "500", "lang": "hi"}) == 1.0

    exported = counter.export()
    assert "# HELP test_counter_total" in exported
    assert "# TYPE test_counter_total counter" in exported
    assert 'test_counter_total{status="200",lang="en"} 3.5' in exported


def test_gauge_metrics_operations():
    """Verify Gauge set, inc, dec, and export."""
    gauge = Gauge(name="test_active_sessions", description="Active sessions", label_names=["region"])
    gauge.set(10.0, labels={"region": "in"})
    gauge.inc(2.0, labels={"region": "in"})
    gauge.dec(1.0, labels={"region": "in"})

    assert gauge.get({"region": "in"}) == 11.0
    exported = gauge.export()
    assert "# HELP test_active_sessions" in exported
    assert "# TYPE test_active_sessions gauge" in exported
    assert 'test_active_sessions{region="in"} 11.0' in exported


def test_histogram_metrics_observation():
    """Verify Histogram observes values and creates cumulative buckets."""
    hist = Histogram(
        name="test_latency_seconds",
        description="Latency in seconds",
        label_names=["endpoint"],
        buckets=(0.05, 0.1, 0.5, 1.0),
    )
    hist.observe(0.04, labels={"endpoint": "/api/ask"})
    hist.observe(0.08, labels={"endpoint": "/api/ask"})
    hist.observe(0.75, labels={"endpoint": "/api/ask"})

    exported = hist.export()
    assert "# HELP test_latency_seconds" in exported
    assert "# TYPE test_latency_seconds histogram" in exported
    assert 'test_latency_seconds_bucket{endpoint="/api/ask",le="0.05"} 1' in exported
    assert 'test_latency_seconds_bucket{endpoint="/api/ask",le="0.1"} 2' in exported
    assert 'test_latency_seconds_bucket{endpoint="/api/ask",le="1.0"} 3' in exported
    assert 'test_latency_seconds_count{endpoint="/api/ask"} 3' in exported


def test_structured_json_logging_sanitization():
    """Verify StructuredJSONFormatter sanitizes API keys and sensitive tokens."""
    formatter = StructuredJSONFormatter()
    logger = logging.getLogger("test_security_logger")
    record = logger.makeRecord(
        name="test_logger",
        level=logging.INFO,
        fn="test.py",
        lno=10,
        msg="Connecting with api_key: 'sk-1234567890abcdef123456' and bearer abcdefgh12345678",
        args=(),
        exc_info=None,
    )
    record.request_id = "test-req-123"
    record.language = "hi"

    formatted = formatter.format(record)
    parsed = json.loads(formatted)

    assert parsed["level"] == "INFO"
    assert parsed["request_id"] == "test-req-123"
    assert parsed["language"] == "hi"
    assert "sk-1234567890abcdef123456" not in parsed["message"]
    assert "***REDACTED***" in parsed["message"]


def test_sanitize_log_message_audio_payloads():
    """Verify sanitize_log_message redacts long base64 and binary audio headers."""
    fake_audio_msg = "Received audio: b'RIFF\\x24\\x00\\x00\\x00WAVEfmt...' base64: " + ("A" * 150)
    cleaned = sanitize_log_message(fake_audio_msg)
    assert "<AUDIO_PAYLOAD_REDACTED>" in cleaned or "RIFF" not in cleaned
    assert "<BASE64_AUDIO_REDACTED>" in cleaned


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_prometheus_format():
    """Verify GET /metrics returns 200 with Prometheus text formatting."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
        text = response.text
        assert "# HELP http_requests_total" in text
        assert "# TYPE http_requests_total counter" in text
        assert "rag_retrieval_requests_total" in text
        assert "rag_cache_hits_total" in text


@pytest.mark.asyncio
async def test_request_id_generation_and_propagation():
    """Verify X-Request-ID header is generated and returned by middleware."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Request without X-Request-ID
        res1 = await client.get("/health")
        assert res1.status_code == 200
        assert "x-request-id" in res1.headers
        generated_id = res1.headers["x-request-id"]
        assert len(generated_id) > 10

        # Request with client-supplied X-Request-ID
        custom_id = "client-trace-id-998877"
        res2 = await client.get("/health", headers={"X-Request-ID": custom_id})
        assert res2.status_code == 200
        assert res2.headers.get("x-request-id") == custom_id


@pytest.mark.asyncio
async def test_voice_ask_telemetry_and_correlation():
    """Verify /api/voice-ask response includes matching request_id and sets X-Request-ID header."""
    custom_id = "voice-test-req-5544"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"audio": ("test.wav", DUMMY_AUDIO, "audio/wav")}
        data = {"language": "hi", "synthesize_speech": "true"}
        headers = {"X-Request-ID": custom_id}

        response = await client.post("/api/voice-ask", files=files, data=data, headers=headers)
        assert response.status_code == 200
        assert response.headers.get("x-request-id") == custom_id
        body = response.json()
        assert body["request_id"] == custom_id
        assert body["language"] == "hi"
        assert body["latency_ms"]["total"] > 0.0


def test_provider_health_summary_tracking():
    """Verify MetricsRegistry tracks provider health states and fallbacks."""
    registry = MetricsRegistry()
    summary_initial = registry.get_provider_health_summary()
    assert summary_initial["sarvam_stt"]["status"] == "healthy"

    # Simulate provider failure
    registry.record_provider_failure("sarvam", "stt", error_type="timeout")
    summary_after = registry.get_provider_health_summary()
    assert summary_after["sarvam_stt"]["status"] == "degraded"
    assert summary_after["sarvam_stt"]["healthy"] is False

    # Restore health
    registry.update_provider_health("sarvam", "stt", is_healthy=True)
    summary_restored = registry.get_provider_health_summary()
    assert summary_restored["sarvam_stt"]["status"] == "healthy"
    assert summary_restored["sarvam_stt"]["healthy"] is True
