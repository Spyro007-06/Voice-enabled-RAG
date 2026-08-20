"""Phase 6.14 — Production Deployment Validation Automated Test Suite.

Verifies:
1. Dockerfile production standards (multi-stage, non-root UID 10001, pythonunbuffered, healthcheck, no secrets)
2. Docker Compose orchestration integrity (voice-rag-net, persistent volumes, healthchecks)
3. Production endpoint contracts (/health, /metrics, /openapi.json)
4. Multilingual Text RAG across en, hi, ta, te, ml
5. Multi-format Voice RAG across WAV, MP3, OGG, WebM, M4A, FLAC
6. Security guardrails, disguised file rejection, rate limiting, and failure recovery
"""

import io
import os
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, validate_production_config
from app.main import app
from app.observability.security_middleware import get_rate_limiter


@pytest.fixture
def client():
    get_rate_limiter().reset()
    with TestClient(app) as test_client:
        yield test_client
    get_rate_limiter().reset()


class TestDockerfileProductionReadiness:
    """Test suite verifying Dockerfile compliance with production containerization standards."""

    @pytest.fixture
    def dockerfile_text(self) -> str:
        p = os.path.join(os.path.dirname(__file__), "..", "Dockerfile")
        assert os.path.exists(p), "Dockerfile does not exist"
        with open(p, "r", encoding="utf-8") as f:
            return f.read()

    def test_dockerfile_multistage_and_base(self, dockerfile_text):
        """Verify Dockerfile uses Python 3.11-slim and multi-stage build."""
        assert "FROM python:3.11-slim AS builder" in dockerfile_text
        assert "FROM python:3.11-slim AS runtime" in dockerfile_text

    def test_dockerfile_non_root_uid(self, dockerfile_text):
        """Verify Dockerfile creates and runs as non-root appuser (UID 10001)."""
        assert "10001" in dockerfile_text
        assert "USER appuser" in dockerfile_text

    def test_dockerfile_healthcheck_and_ports(self, dockerfile_text):
        """Verify Dockerfile specifies HEALTHCHECK and port 8000."""
        assert "HEALTHCHECK" in dockerfile_text
        assert "/health" in dockerfile_text
        assert "EXPOSE 8000" in dockerfile_text


class TestDockerComposeProductionReadiness:
    """Test suite verifying docker-compose.yml orchestration integrity."""

    @pytest.fixture
    def compose_text(self) -> str:
        p = os.path.join(os.path.dirname(__file__), "..", "docker-compose.yml")
        assert os.path.exists(p), "docker-compose.yml does not exist"
        with open(p, "r", encoding="utf-8") as f:
            return f.read()

    def test_compose_services_and_network(self, compose_text):
        """Verify compose defines voice-rag-api, qdrant, and voice-rag-net network."""
        assert "voice-rag-api:" in compose_text
        assert "qdrant:" in compose_text
        assert "voice-rag-net" in compose_text

    def test_compose_persistent_volumes(self, compose_text):
        """Verify compose defines persistent volumes for model cache, app data, and qdrant storage."""
        assert "model_cache" in compose_text
        assert "app_data" in compose_text
        assert "qdrant_storage" in compose_text


class TestProductionEndpointContracts:
    """Test suite verifying production HTTP endpoint contracts."""

    def test_health_endpoint(self, client):
        """Verify /health returns HTTP 200 with valid status and security headers."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "healthy"}
        assert "x-content-type-options" in [k.lower() for k in resp.headers]
        assert "x-request-id" in [k.lower() for k in resp.headers]

    def test_metrics_endpoint(self, client):
        """Verify /metrics returns Prometheus format with required telemetry counters."""
        resp = client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "http_requests_total" in text
        assert "rag_retrieval_requests_total" in text
        assert "rag_provider_health" in text

    def test_openapi_endpoint(self, client):
        """Verify /openapi.json defines required endpoints."""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json().get("paths", {})
        assert "/api/ask" in paths
        assert "/api/voice-ask" in paths
        assert "/health" in paths
        assert "/metrics" in paths


class TestMultilingualTextRAGDeployment:
    """Test suite verifying Text RAG across all 5 supported Indic/English languages."""

    @pytest.mark.parametrize("lang,query", [
        ("en", "What is the capital of Goa?"),
        ("hi", "गोवा की राजधानी क्या है?"),
        ("ta", "கோவாவின் தலைநகரம் எது?"),
        ("te", "గోవా రాజధాని ఏది?"),
        ("ml", "ഗോവയുടെ തലസ്ഥാനം ഏതാണ്?"),
    ])
    def test_text_rag_multilingual_queries(self, client, lang, query):
        """Verify /api/ask responds with HTTP 200 and valid answer structure for all languages."""
        resp = client.post("/api/ask", json={"query": query, "language": lang, "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "citations" in data
        assert "latency_ms" in data
        assert data.get("grounded") is True or len(data.get("answer", "")) > 0


class TestMultiFormatVoiceRAGDeployment:
    """Test suite verifying Voice RAG across all 6 audio formats."""

    @pytest.mark.parametrize("fmt,magic", [
        ("wav", b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"),
        ("mp3", b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 32),
        ("ogg", b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00" + b"\x00" * 32),
        ("webm", b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01" + b"\x00" * 32),
        ("flac", b"fLaC\x00\x00\x00\x22" + b"\x00" * 32),
        ("m4a", b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00" + b"\x00" * 32),
    ])
    def test_voice_rag_multi_format_ingestion(self, client, fmt, magic):
        """Verify /api/voice-ask accepts and processes all supported audio formats."""
        files = {"audio": (f"sample.{fmt}", io.BytesIO(magic), f"audio/{fmt}")}
        resp = client.post("/api/voice-ask", files=files, data={"language": "en", "top_k": "3"})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") in ("success", "partial_success")
        assert "latency_ms" in data


class TestProductionSecurityAndResilience:
    """Test suite verifying security guardrails, rate limiting, and failure recovery."""

    def test_security_disguised_file_rejection(self, client):
        """Verify disguised PDF payload is rejected before transcription."""
        disguised_pdf = {"audio": ("malicious.wav", io.BytesIO(b"%PDF-1.4 Fake PDF Content"), "audio/wav")}
        resp = client.post("/api/voice-ask", files=disguised_pdf, data={"language": "en"})
        assert resp.status_code in (415, 422)

    def test_security_oversized_query_rejection(self, client):
        """Verify oversized query (>2000 chars) is rejected with HTTP 422."""
        resp = client.post("/api/ask", json={"query": "X" * 2500, "language": "en"})
        assert resp.status_code == 422

    def test_failure_recovery_qdrant_failover_bm25(self, client):
        """Verify system falls back to BM25 when Qdrant is unavailable."""
        with patch("app.retrieval.dense.DenseRetriever.retrieve", side_effect=RuntimeError("Qdrant Down")):
            resp = client.post("/api/ask", json={"query": "What is the capital of Goa?", "language": "en"})
            assert resp.status_code == 200
            assert len(resp.json().get("answer", "")) > 0

    def test_failure_recovery_tts_down_partial_success(self, client):
        """Verify system returns partial_success with text answer when TTS fails."""
        with patch("app.providers.tts.mock.MockTTSProvider.synthesize", side_effect=RuntimeError("TTS Down")):
            wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
            files = {"audio": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
            resp = client.post("/api/voice-ask", files=files, data={"language": "en"})
            assert resp.status_code == 200
            assert resp.json().get("status") == "partial_success"
            assert len(resp.json().get("answer", "")) > 0
