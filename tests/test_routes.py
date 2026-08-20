"""Tests for Phase 1 base endpoints and configurations."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_root_endpoint():
    """Verify GET / returns expected online status, project name, and version."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/")
        assert response.status_code == 200
        assert response.json() == {
            "status": "online",
            "project": "HH Goa 2026 Voice RAG",
            "version": "1.0.0",
        }


@pytest.mark.asyncio
async def test_health_endpoint():
    """Verify GET /health returns healthy status."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "healthy",
        }


@pytest.mark.asyncio
async def test_openapi_docs():
    """Verify GET /docs and /openapi.json are accessible."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        docs_response = await client.get("/docs")
        assert docs_response.status_code == 200

        openapi_response = await client.get("/openapi.json")
        assert openapi_response.status_code == 200
        data = openapi_response.json()
        assert data["info"]["title"] == "HH Goa 2026 Voice RAG"
