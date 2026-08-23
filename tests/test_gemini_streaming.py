"""Tests for Google Gemini Streaming and SSE Generator (app.providers.llm.gemini)."""

import json
import pytest
import httpx
from httpx import ASGITransport, AsyncClient

from app.config import get_settings
from app.main import app
from app.providers.llm.gemini import GeminiLLMProvider


@pytest.mark.asyncio
async def test_gemini_generate_stream_tokens(monkeypatch):
    """Test that GeminiLLMProvider.generate_stream parses SSE chunk lines and yields text deltas."""
    # Simulated SSE chunks from Google Gemini streamGenerateContent?alt=sse
    stream_lines = [
        b'data: {"candidates": [{"content": {"parts": [{"text": "Artificial "}]}}]}\r\n\r\n',
        b'data: {"candidates": [{"content": {"parts": [{"text": "intelligence "}]}}]}\r\n\r\n',
        b'data: {"candidates": [{"content": {"parts": [{"text": "is the simulation of human intelligence."}]}}]}\r\n\r\n',
        b'data: [DONE]\r\n\r\n',
    ]

    class MockStreamResponse:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def aiter_lines(self):
            for raw in stream_lines:
                line = raw.decode("utf-8").strip()
                if line:
                    yield line

    class MockAsyncClient:
        is_closed = False

        def stream(self, method, url, *args, **kwargs):
            return MockStreamResponse()

        async def aclose(self):
            self.is_closed = True

    provider = GeminiLLMProvider(api_key="mock_key", model="gemini-2.5-flash")
    provider._client = MockAsyncClient()

    collected_tokens = []
    async for token in provider.generate_stream(
        query="What is AI?",
        context=[{"chunk_id": "c1", "text": "AI simulates human intelligence."}],
    ):
        collected_tokens.append(token)

    assert len(collected_tokens) == 3
    assert collected_tokens[0] == "Artificial "
    assert collected_tokens[1] == "intelligence "
    assert "".join(collected_tokens) == "Artificial intelligence is the simulation of human intelligence."


@pytest.mark.asyncio
async def test_ask_stream_endpoint_sse(monkeypatch):
    """Test /api/ask-stream SSE events emit stage and done payloads cleanly."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/ask-stream", params={"query": "What is a computer?", "language": "en"})
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        content = response.text
        assert "event: stage" in content
        assert "retrieving" in content
        assert "event: done" in content
