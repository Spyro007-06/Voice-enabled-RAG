"""Unit tests for Phase 6.1 LLM Generation Architecture."""

import asyncio
import pytest
from pydantic import ValidationError

from app.generation.base import GenerationProvider
from app.generation.models import (
    GenerationConfig,
    GenerationRequest,
    GenerationResult,
    GenerationTelemetry,
)
from app.generation.provider import (
    MockGenerationProvider,
    get_generation_provider,
)
from app.generation.service import GenerationService, get_generation_service
from app.reranking.models import RerankResult


# --- 1. GenerationConfig Tests ---

def test_generation_config_defaults():
    """Verify default generation config values."""
    config = GenerationConfig()
    assert config.temperature == 0.7
    assert config.max_tokens == 1024
    assert config.top_p == 0.9
    assert config.timeout == 30.0
    assert config.streaming is False
    assert config.model_name == "mock-model"
    assert config.system_prompt is None
    assert config.stop_sequences is None


def test_generation_config_custom():
    """Verify custom generation config parameters."""
    config = GenerationConfig(
        temperature=0.2,
        max_tokens=512,
        top_p=0.95,
        timeout=15.0,
        streaming=True,
        model_name="custom-llm-v1",
        system_prompt="You are a specialized assistant.",
        stop_sequences=["\n\n", "END"],
    )
    assert config.temperature == 0.2
    assert config.max_tokens == 512
    assert config.top_p == 0.95
    assert config.timeout == 15.0
    assert config.streaming is True
    assert config.model_name == "custom-llm-v1"
    assert config.stop_sequences == ["\n\n", "END"]


def test_generation_config_validation_bounds():
    """Verify bounds validation for generation configuration."""
    with pytest.raises(ValidationError):
        GenerationConfig(temperature=-0.1)

    with pytest.raises(ValidationError):
        GenerationConfig(temperature=2.5)

    with pytest.raises(ValidationError):
        GenerationConfig(max_tokens=0)

    with pytest.raises(ValidationError):
        GenerationConfig(top_p=-0.1)

    with pytest.raises(ValidationError):
        GenerationConfig(top_p=1.5)

    with pytest.raises(ValidationError):
        GenerationConfig(timeout=0)


# --- 2. GenerationResult & Telemetry Tests ---

def test_generation_result_and_telemetry_models():
    """Verify serialization and validation of GenerationResult and GenerationTelemetry."""
    telemetry = GenerationTelemetry(
        latency_ms=45.5,
        prompt_tokens=120,
        completion_tokens=40,
        total_tokens=160,
        context_chunks_count=3,
        context_characters=850,
    )
    result = GenerationResult(
        answer="The capital of Goa is Panaji.",
        grounded=True,
        citations=["chunk_1", "chunk_2"],
        model="mock-model",
        latency_ms=45.5,
        finish_reason="stop",
        error=None,
        telemetry=telemetry,
    )
    data = result.model_dump()
    assert data["answer"] == "The capital of Goa is Panaji."
    assert data["grounded"] is True
    assert len(data["citations"]) == 2
    assert data["model"] == "mock-model"
    assert data["latency_ms"] == 45.5
    assert data["finish_reason"] == "stop"
    assert data["error"] is None
    assert data["telemetry"]["context_characters"] == 850


# --- 3. GenerationProvider Interface Tests ---

def test_abstract_generation_provider_cannot_be_instantiated():
    """Verify that GenerationProvider cannot be instantiated directly."""
    with pytest.raises(TypeError):
        GenerationProvider()


# --- 4. MockGenerationProvider Tests ---

@pytest.mark.asyncio
async def test_mock_provider_basic_generation():
    """Verify MockGenerationProvider generates predictable deterministic response."""
    provider = MockGenerationProvider(default_model="mock-v1")
    assert provider.provider_name == "mock"

    query = "What is the capital of Goa?"
    context = "Panaji is the state capital of Goa, located on the banks of the Mandovi river."

    result = await provider.generate(query=query, context=context)
    assert isinstance(result, GenerationResult)
    assert result.grounded is True
    assert "[MOCK] Answer to 'What is the capital of Goa?'" in result.answer
    assert "Panaji" in result.answer
    assert result.model == "mock-v1"
    assert result.finish_reason == "stop"
    assert result.error is None
    assert result.telemetry is not None
    assert result.telemetry.context_characters == len(context)


@pytest.mark.asyncio
async def test_mock_provider_empty_context():
    """Verify MockGenerationProvider handles empty context deterministically."""
    provider = MockGenerationProvider()
    result = await provider.generate(query="Explain quantum computing", context="")
    assert result.grounded is False
    assert "[MOCK] No context available" in result.answer
    assert result.telemetry.context_chunks_count == 0


@pytest.mark.asyncio
async def test_mock_provider_structured_chunks():
    """Verify MockGenerationProvider extracts citations from structured chunk objects."""
    provider = MockGenerationProvider()
    chunks = [
        RerankResult(
            chunk_id="doc_101_c1",
            document_id="doc_101",
            text="Calangute is one of the most popular beaches in North Goa.",
            chunk_type="fixed",
            reranker_score=0.92,
            original_rank=1,
            rank=1,
        ),
        RerankResult(
            chunk_id="doc_102_c2",
            document_id="doc_102",
            text="Baga beach is known for nightlife and water sports.",
            chunk_type="sentence",
            reranker_score=0.88,
            original_rank=2,
            rank=2,
        ),
    ]

    result = await provider.generate(query="Popular beaches in Goa", context=chunks)
    assert result.grounded is True
    assert "doc_101_c1" in result.citations
    assert "doc_102_c2" in result.citations
    assert result.telemetry.context_chunks_count == 2


@pytest.mark.asyncio
async def test_mock_provider_simulated_delay_and_failure():
    """Verify simulated delay and failure injection in mock provider."""
    provider_delay = MockGenerationProvider(simulated_latency_ms=50.0)
    result = await provider_delay.generate("test query", "test context")
    assert result.latency_ms >= 40.0

    provider_fail = MockGenerationProvider(should_fail=True, failure_message="Injected provider crash")
    with pytest.raises(RuntimeError, match="Injected provider crash"):
        await provider_fail.generate("test query", "test context")


def test_get_generation_provider_factory():
    """Verify provider factory instantiates mock provider and rejects unknown providers."""
    mock_p = get_generation_provider("mock", default_model="test-mock")
    assert isinstance(mock_p, MockGenerationProvider)

    with pytest.raises(ValueError, match="Unsupported generation provider"):
        get_generation_provider("unknown_llm_provider_xyz")


# --- 5. GenerationService Tests ---

@pytest.mark.asyncio
async def test_generation_service_with_request_model():
    """Verify GenerationService generates response using GenerationRequest."""
    service = GenerationService(provider=MockGenerationProvider())
    req = GenerationRequest(
        query="Tell me about Dudhsagar falls",
        context="Dudhsagar Falls is a four-tiered waterfall located on the Mandovi River in Goa.",
        config=GenerationConfig(temperature=0.5, model_name="mock-model-v2"),
        language="en",
    )
    result = await service.generate(req)
    assert isinstance(result, GenerationResult)
    assert result.grounded is True
    assert result.model == "mock-model-v2"
    assert result.error is None
    assert result.finish_reason == "stop"
    assert result.latency_ms >= 0.0
    assert result.telemetry is not None
    assert result.telemetry.context_characters > 0


@pytest.mark.asyncio
async def test_generation_service_empty_query_validation():
    """Verify GenerationService gracefully validates empty or whitespace query."""
    service = GenerationService()
    result = await service.generate(query="   ", context="Some context")
    assert result.error == "Query cannot be empty or whitespace."
    assert result.finish_reason == "error"
    assert result.grounded is False
    assert result.answer == ""
    assert result.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_generation_service_with_rerank_results():
    """Verify GenerationService extracts citations and chunks correctly from RerankResult list."""
    service = GenerationService(provider=MockGenerationProvider())
    chunks = [
        RerankResult(
            chunk_id="chunk_alpha",
            document_id="doc_alpha",
            text="Old Goa contains churches recognized by UNESCO as World Heritage Sites.",
            chunk_type="semantic",
            reranker_score=0.95,
            original_rank=1,
            rank=1,
        )
    ]
    result = await service.generate(query="Churches in Goa", context=chunks)
    assert result.grounded is True
    assert "chunk_alpha" in result.citations
    assert result.telemetry.context_chunks_count == 1
    assert "Old Goa" in result.answer


@pytest.mark.asyncio
async def test_generation_service_timeout_handling():
    """Verify GenerationService handles provider timeouts gracefully."""
    # Slow provider simulating 200ms delay
    slow_provider = MockGenerationProvider(simulated_latency_ms=250.0)
    service = GenerationService(provider=slow_provider)

    # Config with 50ms timeout (0.05s)
    short_config = GenerationConfig(timeout=0.05)
    result = await service.generate(
        query="Slow query test",
        context="Some context",
        config=short_config,
    )
    assert result.finish_reason == "timeout"
    assert "timed out" in result.error
    assert result.grounded is False
    assert result.answer == ""
    assert result.latency_ms >= 40.0


@pytest.mark.asyncio
async def test_generation_service_provider_error_handling():
    """Verify GenerationService captures unexpected provider exceptions gracefully."""
    failing_provider = MockGenerationProvider(should_fail=True, failure_message="Connection refused to LLM backend")
    service = GenerationService(provider=failing_provider)

    result = await service.generate(
        query="Test failing provider",
        context="Sample context",
    )
    assert result.finish_reason == "error"
    assert "Connection refused" in result.error
    assert result.grounded is False
    assert result.answer == ""
    assert result.latency_ms >= 0.0


# --- 6. Multilingual Unicode Preservation Tests ---

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "language,query,context_text",
    [
        (
            "hi",
            "गोवा की राजधानी क्या है?",
            "पणजी भारतीय राज्य गोवा की राजधानी है। यह मांडवी नदी के मुहाने पर स्थित है।",
        ),
        (
            "ta",
            "கோவாவின் தலைநகரம் எது?",
            "பனாஜி என்பது இந்தியாவின் கோவா மாநிலத்தின் தலைநகரம் ஆகும். இது மாண்டோவி ஆற்றின் கரையில் அமைந்துள்ளது.",
        ),
        (
            "te",
            "గోవా రాజధాని ఏమిటి?",
            "పనాజీ భారతదేశంలోని గోవా రాష్ట్ర రాజధాని. ఇది మాండోవి నది ఒడ్డున ఉంది.",
        ),
        (
            "ml",
            "ഗോവയുടെ തലസ്ഥാനം ഏതാണ്?",
            "പനാജി ഇന്ത്യൻ സംസ്ഥാനമായ ഗോവയുടെ തലസ്ഥാനമാണ്. ഇത് മാണ്ഡോവി നദിയുടെ തീരത്താണ് സ്ഥിതി ചെയ്യുന്നത്.",
        ),
    ],
)
async def test_multilingual_unicode_preservation(language: str, query: str, context_text: str):
    """Verify query and context preserve full multilingual Unicode characters across Indian languages."""
    service = GenerationService(provider=MockGenerationProvider())
    req = GenerationRequest(
        query=query,
        context=context_text,
        language=language,
    )
    result = await service.generate(req)
    assert result.grounded is True
    assert query in result.answer
    # First part of context text should be intact and preserved in unicode
    snippet = context_text[:30]
    assert snippet in result.answer
    assert result.telemetry.context_characters == len(context_text)


def test_get_generation_service_singleton():
    """Verify get_generation_service returns singleton instance."""
    s1 = get_generation_service()
    s2 = get_generation_service()
    assert s1 is s2
