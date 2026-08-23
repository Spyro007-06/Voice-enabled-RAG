"""Tests for Google Gemini LLM Provider (app.providers.llm.gemini)."""

import json
import pytest
import httpx

from app.config import get_settings
from app.generation.models import GenerationConfig, GenerationResult
from app.providers.exceptions import (
    InvalidAPIKeyError,
    MalformedResponseError,
    MissingAPIKeyError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.providers.factory import get_llm_provider
from app.providers.llm.gemini import GeminiLLMProvider


def test_gemini_provider_initialization_missing_key(monkeypatch):
    """Test that GeminiLLMProvider raises MissingAPIKeyError when API key is missing."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("LLM_API_KEY", "")
    get_settings.cache_clear()

    with pytest.raises(MissingAPIKeyError) as exc_info:
        GeminiLLMProvider(api_key=None)

    assert exc_info.value.provider_name == "gemini"
    assert "GEMINI_API_KEY" in str(exc_info.value)


def test_gemini_provider_initialization_success():
    """Test successful initialization of GeminiLLMProvider with configured parameters."""
    provider = GeminiLLMProvider(
        api_key="mock_gemini_key_12345",
        model="gemini-2.5-flash",
        timeout=45.0,
    )
    assert provider.provider_name == "gemini"
    assert provider.model == "gemini-2.5-flash"
    assert provider.timeout == 45.0


def test_factory_get_llm_provider_gemini():
    """Test factory resolution of Gemini LLM provider."""
    provider = get_llm_provider(
        provider_name="gemini",
        api_key="mock_factory_key",
        model="gemini-1.5-flash",
    )
    assert isinstance(provider, GeminiLLMProvider)
    assert provider.provider_name == "gemini"


@pytest.mark.asyncio
async def test_gemini_generate_success(monkeypatch):
    """Test successful generation from Gemini REST API response."""
    mock_response_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "A computer is an electronic device that manipulates information or data."
                        }
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 35,
            "candidatesTokenCount": 15,
            "totalTokenCount": 50,
        },
    }

    async def mock_post(url, *args, **kwargs):
        req = httpx.Request("POST", url)
        return httpx.Response(200, json=mock_response_payload, request=req)

    provider = GeminiLLMProvider(api_key="mock_key", model="gemini-2.5-flash")

    # Mock the internal client's post method
    client = await provider._get_client()
    monkeypatch.setattr(client, "post", mock_post)

    context = [
        {"chunk_id": "chunk_1", "text": "Computers process data according to programmed instructions."},
        {"chunk_id": "chunk_2", "text": "A computer consists of hardware and software components."},
    ]

    result = await provider.generate(
        query="What is a computer?",
        context=context,
        generation_config=GenerationConfig(temperature=0.3, max_tokens=100),
    )

    assert isinstance(result, GenerationResult)
    assert "electronic device" in result.answer
    assert result.grounded is True
    assert result.model == "gemini-2.5-flash"
    assert result.finish_reason == "STOP"
    assert "chunk_1" in result.citations
    assert "chunk_2" in result.citations
    assert result.telemetry.prompt_tokens == 35
    assert result.telemetry.completion_tokens == 15


@pytest.mark.asyncio
async def test_gemini_generate_invalid_api_key(monkeypatch):
    """Test InvalidAPIKeyError on 401/403 HTTP error from Gemini."""
    async def mock_post_403(url, *args, **kwargs):
        req = httpx.Request("POST", url)
        return httpx.Response(403, json={"error": {"message": "API key not valid."}}, request=req)

    provider = GeminiLLMProvider(api_key="invalid_key")
    client = await provider._get_client()
    monkeypatch.setattr(client, "post", mock_post_403)

    with pytest.raises(InvalidAPIKeyError) as exc_info:
        await provider.generate(query="Test", context="Context")

    assert exc_info.value.provider_name == "gemini"


@pytest.mark.asyncio
async def test_gemini_generate_rate_limit(monkeypatch):
    """Test ProviderRateLimitError on 429 HTTP response."""
    async def mock_post_429(url, *args, **kwargs):
        req = httpx.Request("POST", url)
        return httpx.Response(429, json={"error": {"message": "Resource exhausted"}}, request=req)

    provider = GeminiLLMProvider(api_key="mock_key")
    client = await provider._get_client()
    monkeypatch.setattr(client, "post", mock_post_429)

    with pytest.raises(ProviderRateLimitError) as exc_info:
        await provider.generate(query="Test", context="Context")

    assert exc_info.value.provider_name == "gemini"


@pytest.mark.asyncio
async def test_gemini_generate_timeout(monkeypatch):
    """Test ProviderTimeoutError on httpx TimeoutException."""
    async def mock_post_timeout(url, *args, **kwargs):
        raise httpx.TimeoutException("Read timed out")

    provider = GeminiLLMProvider(api_key="mock_key")
    client = await provider._get_client()
    monkeypatch.setattr(client, "post", mock_post_timeout)

    with pytest.raises(ProviderTimeoutError) as exc_info:
        await provider.generate(query="Test", context="Context")

    assert exc_info.value.provider_name == "gemini"


@pytest.mark.asyncio
async def test_gemini_generate_connection_error(monkeypatch):
    """Test ProviderConnectionError on network connection failure."""
    async def mock_post_conn(url, *args, **kwargs):
        raise httpx.RequestError("Failed to connect")

    provider = GeminiLLMProvider(api_key="mock_key")
    client = await provider._get_client()
    monkeypatch.setattr(client, "post", mock_post_conn)

    with pytest.raises(ProviderConnectionError) as exc_info:
        await provider.generate(query="Test", context="Context")

    assert exc_info.value.provider_name == "gemini"
