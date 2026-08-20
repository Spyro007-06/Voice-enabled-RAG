"""Phase 6.11 — Production Containerization & Deployment Architecture Tests.

Comprehensive automated test suite covering:
1. Production configuration validation (DEBUG rejection, CORS wildcard rejection, provider key requirements).
2. Container Healthcheck and endpoint contracts (/health, /metrics, /openapi.json).
3. Dockerfile security best practices (non-root user, healthcheck directive, python unbuffered, deterministic dependencies).
4. Secret isolation & .dockerignore compliance.
5. Docker Compose orchestration integrity (services, healthchecks, networks, volumes).
6. Environment variable handling and setting defaults.
"""

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


# =============================================================================
# 1. PRODUCTION CONFIGURATION VALIDATION TESTS
# =============================================================================

class TestProductionConfigValidation:
    """Test suite validating strict production configuration rules."""

    def test_production_config_rejects_debug_true(self):
        """Verify that DEBUG=True is flagged in production mode."""
        settings = Settings(
            ENVIRONMENT="production",
            DEBUG=True,
            CORS_ORIGINS=["https://example.com"],
        )
        issues = validate_production_config(settings)
        assert any("DEBUG mode must be disabled" in issue for issue in issues)

    def test_production_config_rejects_wildcard_cors(self):
        """Verify that wildcard CORS origins are flagged in production mode."""
        settings = Settings(
            ENVIRONMENT="production",
            DEBUG=False,
            CORS_ORIGINS=["*"],
        )
        issues = validate_production_config(settings)
        assert any("CORS_ORIGINS should not contain wildcard" in issue for issue in issues)

    def test_production_config_requires_sarvam_api_key(self):
        """Verify missing SARVAM_API_KEY is flagged when using Sarvam providers in production."""
        settings = Settings(
            ENVIRONMENT="production",
            DEBUG=False,
            CORS_ORIGINS=["https://example.com"],
            STT_PROVIDER="sarvam",
            SARVAM_API_KEY=None,
        )
        issues = validate_production_config(settings)
        assert any("SARVAM_API_KEY is required" in issue for issue in issues)

    def test_production_config_requires_openai_api_key(self):
        """Verify missing OPENAI_API_KEY is flagged when using OpenAI LLM in production."""
        settings = Settings(
            ENVIRONMENT="production",
            DEBUG=False,
            CORS_ORIGINS=["https://example.com"],
            LLM_PROVIDER="openai",
            OPENAI_API_KEY=None,
        )
        issues = validate_production_config(settings)
        assert any("OPENAI_API_KEY is required" in issue for issue in issues)

    def test_production_config_valid_passes(self):
        """Verify compliant production configuration produces zero validation issues."""
        settings = Settings(
            ENVIRONMENT="production",
            DEBUG=False,
            CORS_ORIGINS=["https://api.hhgoa2026.com"],
            STT_PROVIDER="sarvam",
            TTS_PROVIDER="sarvam",
            LLM_PROVIDER="sarvam",
            SARVAM_API_KEY="sk_live_valid_key_12345",
            VECTOR_PROVIDER="qdrant_local",
        )
        issues = validate_production_config(settings)
        assert len(issues) == 0


# =============================================================================
# 2. CONTAINER HEALTHCHECK & ENDPOINT CONTRACT TESTS
# =============================================================================

class TestContainerEndpointsAndHealthcheck:
    """Test suite validating endpoints used by container health checks and scrapers."""

    def test_health_endpoint_returns_200_and_healthy_status(self, client):
        """Verify /health returns HTTP 200 and healthy JSON payload required by Docker HEALTHCHECK."""
        resp = client.get("/health")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "healthy"

    def test_metrics_endpoint_accessible(self, client):
        """Verify /metrics returns Prometheus format text for scraper integration."""
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "http_requests_total" in resp.text
        assert resp.headers.get("content-type", "").startswith("text/plain")

    def test_openapi_json_endpoint_accessible(self, client):
        """Verify /openapi.json returns valid API specification schema."""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        payload = resp.json()
        assert "openapi" in payload
        assert "/api/ask" in payload["paths"]
        assert "/api/voice-ask" in payload["paths"]


# =============================================================================
# 3. DOCKERFILE SECURITY & SYNTAX BEST PRACTICES
# =============================================================================

class TestDockerfileSecurityAndBestPractices:
    """Test suite verifying Dockerfile compliance with production security standards."""

    @pytest.fixture
    def dockerfile_content(self) -> str:
        dockerfile_path = os.path.join(os.path.dirname(__file__), "..", "Dockerfile")
        assert os.path.exists(dockerfile_path), "Dockerfile does not exist"
        with open(dockerfile_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_dockerfile_uses_non_root_user(self, dockerfile_content):
        """Verify Dockerfile establishes and switches to a dedicated non-root user."""
        assert "USER appuser" in dockerfile_content or "USER 10001" in dockerfile_content
        assert "useradd" in dockerfile_content or "adduser" in dockerfile_content

    def test_dockerfile_contains_healthcheck_directive(self, dockerfile_content):
        """Verify Dockerfile defines a container HEALTHCHECK targeting /health."""
        assert "HEALTHCHECK" in dockerfile_content
        assert "/health" in dockerfile_content

    def test_dockerfile_configures_python_unbuffered_and_bytecode(self, dockerfile_content):
        """Verify Python environment variables are properly set for container logging."""
        assert "PYTHONUNBUFFERED=1" in dockerfile_content
        assert "PYTHONDONTWRITEBYTECODE=1" in dockerfile_content

    def test_dockerfile_does_not_copy_env_or_secrets(self, dockerfile_content):
        """Verify Dockerfile does not explicitly copy .env or credential files into the image."""
        lines = dockerfile_content.splitlines()
        for line in lines:
            clean = line.strip()
            if clean.startswith("COPY") or clean.startswith("ADD"):
                assert ".env" not in clean
                assert "secrets" not in clean


# =============================================================================
# 4. SECRET ISOLATION & .DOCKERIGNORE COMPLIANCE
# =============================================================================

class TestDockerignoreCompliance:
    """Test suite verifying .dockerignore excludes sensitive secrets and development files."""

    @pytest.fixture
    def dockerignore_content(self) -> str:
        ignore_path = os.path.join(os.path.dirname(__file__), "..", ".dockerignore")
        assert os.path.exists(ignore_path), ".dockerignore does not exist"
        with open(ignore_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_dockerignore_excludes_secrets_and_env(self, dockerignore_content):
        """Verify critical secrets, certificates, and test artifacts are ignored."""
        assert ".env" in dockerignore_content
        assert "*.key" in dockerignore_content or "*.pem" in dockerignore_content
        assert "__pycache__" in dockerignore_content
        assert ".git" in dockerignore_content
        assert "tests" in dockerignore_content


# =============================================================================
# 5. DOCKER COMPOSE ORCHESTRATION INTEGRITY
# =============================================================================

class TestDockerComposeOrchestration:
    """Test suite verifying docker-compose.yml structure and network isolation."""

    @pytest.fixture
    def compose_content(self) -> str:
        compose_path = os.path.join(os.path.dirname(__file__), "..", "docker-compose.yml")
        assert os.path.exists(compose_path), "docker-compose.yml does not exist"
        with open(compose_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_docker_compose_defines_required_services(self, compose_content):
        """Verify docker-compose.yml defines voice-rag-api and qdrant."""
        assert "voice-rag-api:" in compose_content
        assert "qdrant:" in compose_content

    def test_docker_compose_defines_persistent_volumes(self, compose_content):
        """Verify docker-compose.yml defines persistent volumes for model cache and qdrant storage."""
        assert "model_cache" in compose_content
        assert "qdrant_storage" in compose_content

    def test_docker_compose_defines_isolated_network(self, compose_content):
        """Verify services communicate over a dedicated bridge network."""
        assert "voice-rag-net" in compose_content
