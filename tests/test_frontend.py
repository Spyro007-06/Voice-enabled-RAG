"""Tests for Phase 6.25 Dataset-Centric Multilingual RAG Application Redesign."""

import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_frontend_index_serving():
    """Verify GET /app/ returns the dataset-centric Multilingual Passage Intelligence interface."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/app/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        text = response.text

        # Core branding & title
        assert "HH Goa 2026" in text
        assert "Multilingual Voice RAG" in text
        assert "Multilingual Passage Intelligence" in text
        assert "MSMARCO-XI" in text

        # Conversation UI elements
        assert 'id="chatMessages"' in text
        assert 'id="welcomeState"' in text
        assert 'id="welcomeMicBtn"' in text
        assert 'id="queryInput"' in text
        assert 'id="micBtn"' in text
        assert 'id="sendBtn"' in text
        assert 'id="recordingBar"' in text
        assert 'id="charCounter"' in text

        # Composer validation (no number spinners, proper multiline textarea)
        assert "<textarea" in text
        assert 'maxlength="2000"' in text
        assert "type=\"number\"" not in text

        # Drawers (Modals)
        assert 'id="menuDrawer"' in text
        assert 'id="datasetDrawer"' in text
        assert 'id="pipelineDrawer"' in text
        assert 'id="statusDrawer"' in text
        assert 'id="aboutDrawer"' in text
        assert 'id="drawerBackdrop"' in text

        # Multilingual selector with dual native + English names
        assert 'id="langSelect"' in text
        assert "English" in text
        assert "हिन्दी — Hindi" in text
        assert "தமிழ் — Tamil" in text
        assert "తెలుగు — Telugu" in text
        assert "മലയാളം — Malayalam" in text

        # Dataset information exposed in UI
        assert "48,206" in text
        assert "28,541" in text
        assert "9,858" in text  # English
        assert "9,355" in text  # Hindi
        assert "9,910" in text  # Tamil
        assert "9,168" in text  # Telugu
        assert "9,915" in text  # Malayalam

        # Pipeline steps in UI
        assert "Language Resolution" in text
        assert "Saarika v2.5" in text
        assert "Google Gemini" in text
        assert "Bulbul v2 TTS" in text


@pytest.mark.asyncio
async def test_frontend_all_static_assets_serving():
    """Verify all static CSS, JS, and SVG assets are properly delivered without 404s."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # CSS files
        for css_file in ["main.css", "components.css", "animations.css", "responsive.css"]:
            res = await client.get(f"/app/css/{css_file}")
            assert res.status_code == 200, f"Failed loading /app/css/{css_file}"
            assert "text/css" in res.headers.get("content-type", "")

        # JS modules
        for js_file in ["app.js", "api.js", "state.js", "chat.js", "voice.js", "ui.js", "health.js"]:
            res = await client.get(f"/app/js/{js_file}")
            assert res.status_code == 200, f"Failed loading /app/js/{js_file}"
            assert any(t in res.headers.get("content-type", "") for t in ["javascript", "text/plain", "application/x-javascript"])

        # SVG Logo
        svg_res = await client.get("/app/assets/logo.svg")
        assert svg_res.status_code == 200
        assert "image/svg+xml" in svg_res.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_root_endpoint_unaffected():
    """Ensure root JSON endpoint remains unchanged for existing backend consumers."""
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
