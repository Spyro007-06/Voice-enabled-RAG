"""Tests for Gemini Grounding, Anti-Hallucination, and Refusal Guardrails."""

import pytest
from unittest.mock import AsyncMock, patch

from app.generation.models import GenerationConfig, GenerationResult
from app.generation.service import GenerationService
from app.guardrails.service import GuardrailService
from app.orchestration.voice_rag import VoiceRAGOrchestrator
from app.providers.llm.gemini import GeminiLLMProvider
from app.providers.stt.mock import MockSTTProvider
from app.providers.tts.mock import MockTTSProvider
from app.reranking.adaptive import AdaptiveDecision, AdaptiveRetrievalService, TierString
from app.reranking.models import AdaptiveLatencyBreakdown, RerankResult


@pytest.mark.asyncio
async def test_empty_retrieval_never_calls_gemini():
    """Verify that when retrieval returns 0 passages, Gemini is NEVER called and refusal is returned."""
    mock_gemini = AsyncMock(spec=GeminiLLMProvider)
    mock_gemini.provider_name = "gemini"

    gen_service = GenerationService(provider=mock_gemini)
    guard_service = GuardrailService()

    mock_adaptive = AsyncMock(spec=AdaptiveRetrievalService)
    mock_adaptive.adaptive_retrieve.return_value = (
        [],  # Empty results
        AdaptiveDecision(
            confidence_score=0.0,
            dense_confidence=0.0,
            retriever_agreement=0.0,
            score_margin=0.0,
            should_rerank=False,
            reranker_tier=TierString("high"),
            candidate_k=0,
            reranker="none",
            reranking_used=False,
            reason="Empty retrieval context.",
        ),
        AdaptiveLatencyBreakdown(),
    )

    orchestrator = VoiceRAGOrchestrator(
        stt_provider=MockSTTProvider(default_text="What is the capital of Atlantis?"),
        tts_provider=MockTTSProvider(),
        adaptive_retrieval_service=mock_adaptive,
        generation_service=gen_service,
        guardrail_service=guard_service,
    )

    # Simulate voice query for out-of-corpus question (e.g., Atlantis)
    fake_audio = b"RIFF" + b"\x00" * 100
    response = await orchestrator.execute_voice_rag(
        audio_bytes=fake_audio,
        language="en",
        filename="test.wav",
    )

    # Assertions: Gemini was NEVER called
    mock_gemini.generate.assert_not_called()

    # Response indicates safe refusal
    assert response.grounded is False
    assert response.citations == []
    assert "don't have enough information" in response.answer.lower() or "insufficient" in response.answer.lower()


@pytest.mark.asyncio
async def test_low_relevance_never_calls_gemini():
    """Verify that when retrieved passages are unrelated, Gemini is NEVER called."""
    mock_gemini = AsyncMock(spec=GeminiLLMProvider)
    mock_gemini.provider_name = "gemini"

    gen_service = GenerationService(provider=mock_gemini)
    guard_service = GuardrailService(min_relevance_threshold=0.50)

    # Passages completely unrelated to query
    unrelated_results = [
        RerankResult(
            chunk_id="chunk_99",
            document_id="doc_99",
            text="The recipe for strawberry cheesecake requires fresh strawberries and cream cheese.",
            dense_score=0.05,
            reranker_score=0.08,
            fusion_score=0.06,
            rank=1,
            original_rank=1,
            chunk_type="sentence",
            language="en",
        )
    ]

    mock_adaptive = AsyncMock(spec=AdaptiveRetrievalService)
    mock_adaptive.adaptive_retrieve.return_value = (
        unrelated_results,
        AdaptiveDecision(
            confidence_score=0.08,
            dense_confidence=0.05,
            retriever_agreement=0.0,
            score_margin=0.0,
            should_rerank=False,
            reranker_tier=TierString("high"),
            candidate_k=1,
            reranker="minilm",
            reranking_used=True,
            reason="Low relevance score.",
        ),
        AdaptiveLatencyBreakdown(),
    )

    orchestrator = VoiceRAGOrchestrator(
        stt_provider=MockSTTProvider(default_text="What is quantum computing?"),
        tts_provider=MockTTSProvider(),
        adaptive_retrieval_service=mock_adaptive,
        generation_service=gen_service,
        guardrail_service=guard_service,
    )

    fake_audio = b"RIFF" + b"\x00" * 100
    response = await orchestrator.execute_voice_rag(
        audio_bytes=fake_audio,
        language="en",
        filename="test.wav",
    )

    mock_gemini.generate.assert_not_called()
    assert response.grounded is False
    assert response.citations == []


@pytest.mark.asyncio
async def test_gemini_receives_structured_retrieved_context():
    """Verify that Gemini receives retrieved passages in structured format."""
    mock_gemini = AsyncMock(spec=GeminiLLMProvider)
    mock_gemini.provider_name = "gemini"
    mock_gemini.generate.return_value = GenerationResult(
        answer="Machine learning is a subset of artificial intelligence.",
        grounded=True,
        citations=["chunk_42"],
        model="gemini-2.5-flash",
        latency_ms=120.0,
    )

    gen_service = GenerationService(provider=mock_gemini)
    guard_service = GuardrailService()

    relevant_results = [
        RerankResult(
            chunk_id="chunk_42",
            document_id="doc_10",
            text="Machine learning involves training algorithms on data to make predictions.",
            dense_score=0.88,
            reranker_score=0.92,
            fusion_score=0.90,
            rank=1,
            original_rank=1,
            language="en",
            chunk_type="semantic",
        )
    ]

    mock_adaptive = AsyncMock(spec=AdaptiveRetrievalService)
    mock_adaptive.adaptive_retrieve.return_value = (
        relevant_results,
        AdaptiveDecision(
            confidence_score=0.92,
            dense_confidence=0.88,
            retriever_agreement=0.85,
            score_margin=0.2,
            should_rerank=True,
            reranker_tier=TierString("high"),
            candidate_k=1,
            reranker="minilm",
            reranking_used=True,
            reason="High relevance.",
        ),
        AdaptiveLatencyBreakdown(),
    )

    orchestrator = VoiceRAGOrchestrator(
        stt_provider=MockSTTProvider(default_text="What is machine learning?"),
        tts_provider=MockTTSProvider(),
        adaptive_retrieval_service=mock_adaptive,
        generation_service=gen_service,
        guardrail_service=guard_service,
    )

    fake_audio = b"RIFF" + b"\x00" * 100
    response = await orchestrator.execute_voice_rag(
        audio_bytes=fake_audio,
        language="en",
        filename="test.wav",
    )

    # Verify Gemini WAS called once with the retrieved context
    mock_gemini.generate.assert_called_once()
    call_kwargs = mock_gemini.generate.call_args.kwargs
    assert "context" in call_kwargs
    assert len(call_kwargs["context"]) == 1
    assert call_kwargs["context"][0].chunk_id == "chunk_42"

    assert response.grounded is True
    assert "chunk_42" in response.citations
