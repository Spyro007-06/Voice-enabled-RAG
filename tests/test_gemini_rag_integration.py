"""End-to-End RAG Integration Tests with Google Gemini LLM Provider."""

import pytest
from unittest.mock import AsyncMock, patch
from httpx import ASGITransport, AsyncClient

from app.config import get_settings
from app.generation.models import GenerationConfig, GenerationResult
from app.generation.service import GenerationService
from app.guardrails.service import GuardrailService
from app.main import app
from app.orchestration.voice_rag import VoiceRAGOrchestrator
from app.providers.llm.gemini import GeminiLLMProvider
from app.providers.stt.mock import MockSTTProvider
from app.providers.tts.mock import MockTTSProvider
from app.reranking.adaptive import AdaptiveDecision, AdaptiveRetrievalService, TierString
from app.reranking.models import AdaptiveLatencyBreakdown, RerankResult


@pytest.mark.parametrize(
    "lang,query,expected_keyword",
    [
        ("en", "What is a computer?", "electronic"),
        ("hi", "कंप्यूटर क्या है?", "इलेक्ट्रॉनिक"),
        ("ta", "கணினி என்றால் என்ன?", "மின்னணு"),
        ("te", "కంప్యూటర్ అంటే ఏమిటి?", "ఎలక్ట్రానిక్"),
        ("ml", "കമ്പ്യൂട്ടർ എന്താണ്?", "ഇലക്ട്രോണിക്"),
    ],
)
@pytest.mark.asyncio
async def test_multilingual_voice_rag_gemini(lang, query, expected_keyword):
    """Test full voice query -> STT -> RAG -> Gemini -> TTS pipeline across all 5 Indic languages."""
    mock_gemini = AsyncMock(spec=GeminiLLMProvider)
    mock_gemini.provider_name = "gemini"
    mock_gemini.generate.return_value = GenerationResult(
        answer=f"Answer in {lang} containing {expected_keyword}.",
        grounded=True,
        citations=[f"chunk_{lang}_1"],
        model="gemini-2.5-flash",
        latency_ms=95.0,
    )

    gen_service = GenerationService(provider=mock_gemini)
    guard_service = GuardrailService()

    retrieved_results = [
        RerankResult(
            chunk_id=f"chunk_{lang}_1",
            document_id=f"doc_{lang}_1",
            text=f"Passage content describing {expected_keyword} for {query}",
            dense_score=0.90,
            reranker_score=0.95,
            fusion_score=0.92,
            rank=1,
            original_rank=1,
            chunk_type="semantic",
            language=lang,
        )
    ]

    mock_adaptive = AsyncMock(spec=AdaptiveRetrievalService)
    mock_adaptive.adaptive_retrieve.return_value = (
        retrieved_results,
        AdaptiveDecision(
            confidence_score=0.95,
            dense_confidence=0.90,
            retriever_agreement=0.88,
            score_margin=0.25,
            should_rerank=True,
            reranker_tier=TierString("high"),
            candidate_k=1,
            reranker="minilm",
            reranking_used=True,
            reason="High relevance.",
        ),
        AdaptiveLatencyBreakdown(),
    )

    stt_provider = MockSTTProvider(default_text=query, default_language=lang)
    tts_provider = MockTTSProvider()

    orchestrator = VoiceRAGOrchestrator(
        stt_provider=stt_provider,
        tts_provider=tts_provider,
        adaptive_retrieval_service=mock_adaptive,
        generation_service=gen_service,
        guardrail_service=guard_service,
    )

    fake_audio = b"RIFF" + b"\x00" * 100
    response = await orchestrator.execute_voice_rag(
        audio_bytes=fake_audio,
        language=lang,
        filename="test.wav",
        synthesize_speech=True,
    )

    assert response.status in ("success", "partial_success")
    assert response.grounded is True
    assert expected_keyword in response.answer
    assert f"chunk_{lang}_1" in response.citations
    assert response.audio.available is True
    assert response.audio.audio_base64 is not None


@pytest.mark.asyncio
async def test_tts_failure_preserves_gemini_grounded_text():
    """Verify that if TTS synthesis fails, the grounded Gemini text answer is still preserved."""
    mock_gemini = AsyncMock(spec=GeminiLLMProvider)
    mock_gemini.provider_name = "gemini"
    mock_gemini.generate.return_value = GenerationResult(
        answer="The internet is a global network of interconnected computers.",
        grounded=True,
        citations=["chunk_net_1"],
        model="gemini-2.5-flash",
        latency_ms=110.0,
    )

    gen_service = GenerationService(provider=mock_gemini)
    guard_service = GuardrailService()

    retrieved_results = [
        RerankResult(
            chunk_id="chunk_net_1",
            document_id="doc_net_1",
            text="The internet connects billions of devices worldwide.",
            dense_score=0.85,
            reranker_score=0.91,
            fusion_score=0.88,
            rank=1,
            original_rank=1,
            chunk_type="semantic",
            language="en",
        )
    ]

    mock_adaptive = AsyncMock(spec=AdaptiveRetrievalService)
    mock_adaptive.adaptive_retrieve.return_value = (
        retrieved_results,
        AdaptiveDecision(
            confidence_score=0.91,
            dense_confidence=0.85,
            retriever_agreement=0.80,
            score_margin=0.20,
            should_rerank=True,
            reranker_tier=TierString("high"),
            candidate_k=1,
            reranker="minilm",
            reranking_used=True,
            reason="High relevance.",
        ),
        AdaptiveLatencyBreakdown(),
    )

    # TTS provider that throws an error
    failing_tts = MockTTSProvider()
    failing_tts.synthesize = AsyncMock(side_effect=Exception("TTS Server Down"))

    orchestrator = VoiceRAGOrchestrator(
        stt_provider=MockSTTProvider(default_text="What is the internet?"),
        tts_provider=failing_tts,
        adaptive_retrieval_service=mock_adaptive,
        generation_service=gen_service,
        guardrail_service=guard_service,
    )

    fake_audio = b"RIFF" + b"\x00" * 100
    response = await orchestrator.execute_voice_rag(
        audio_bytes=fake_audio,
        language="en",
        filename="test.wav",
        synthesize_speech=True,
    )

    # Response still preserves text answer and citations, status is partial_success
    assert response.status == "partial_success"
    assert response.grounded is True
    assert "global network" in response.answer
    assert response.audio.available is False
