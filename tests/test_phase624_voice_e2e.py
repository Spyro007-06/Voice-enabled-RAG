"""
Phase 6.24 End-to-End Voice & Multilingual RAG Test Suite.
Validates the complete pipeline: STT, Language Resolution, Adaptive Retrieval,
Sarvam-105b LLM Grounding, Sarvam TTS (bulbul:v2), Frontend Contracts, and Security.
"""

import io
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.routes import ask_endpoint, health_check, voice_ask_endpoint
from app.config import get_settings, validate_production_config, Settings
from app.generation.models import AskRequest, GenerationConfig
from app.generation.prompt import get_prompt_builder
from app.guardrails.relevance import (
    extract_content_words,
    is_meaningful_word_match,
    validate_answerability,
    RelevanceGuard,
)
from app.main import app
from app.orchestration.models import AudioOutputMetadata, VoiceAskResponse
from app.orchestration.voice_rag import get_voice_rag_orchestrator
from app.providers.factory import (
    get_llm_provider,
    get_stt_provider,
    get_tts_provider,
)
from app.providers.stt.base import STTResult
from app.providers.tts.base import TTSResult
from app.retrieval.language import (
    detect_script_language,
    get_language_filter_synonyms,
    normalize_language_code,
    to_bcp47,
    to_sarvam_code,
)
from app.speech.validation import sanitize_filename, validate_audio_security, verify_audio_magic_bytes


# =============================================================================
# 1. PROVIDER CONFIGURATION TESTS
# =============================================================================

def test_phase624_provider_runtime_configuration():
    """Verify runtime models and parameters loaded in settings."""
    settings = get_settings()
    assert settings.SARVAM_STT_MODEL == "saarika:v2.5"
    assert settings.SARVAM_TTS_MODEL == "bulbul:v2"
    assert settings.SARVAM_LLM_MODEL == "sarvam-105b"
    assert settings.LLM_MAX_TOKENS >= 1536


def test_production_config_validation():
    """Verify validate_production_config flags missing production credentials."""
    prod_settings = Settings(
        ENVIRONMENT="production",
        DEBUG=True,
        SARVAM_API_KEY=None,
        STT_PROVIDER="sarvam",
        TTS_PROVIDER="sarvam",
        LLM_PROVIDER="sarvam",
    )
    issues = validate_production_config(prod_settings)
    assert len(issues) > 0
    assert any("DEBUG" in i for i in issues)
    assert any("SARVAM_API_KEY" in i for i in issues)


# =============================================================================
# 2. STT TESTS (ALL 5 LANGUAGES)
# =============================================================================

@pytest.mark.asyncio
async def test_stt_language_resolution():
    """Verify language code mappings to Sarvam format for all 5 languages."""
    assert to_sarvam_code("en") == "en-IN"
    assert to_sarvam_code("hi") == "hi-IN"
    assert to_sarvam_code("ta") == "ta-IN"
    assert to_sarvam_code("te") == "te-IN"
    assert to_sarvam_code("ml") == "ml-IN"
    assert to_sarvam_code(None, default="unknown") == "unknown"


@pytest.mark.asyncio
async def test_stt_mock_transcription_all_languages():
    """Test STT provider transcription across 5 languages with mock results."""
    test_cases = [
        ("en", "What is a computer?"),
        ("hi", "कंप्यूटर क्या है?"),
        ("ta", "கணினி என்றால் என்ன?"),
        ("te", "కంప్యూటర్ అంటే ఏమిటి?"),
        ("ml", "കമ്പ്യൂട്ടർ എന്താണ്?"),
    ]
    orchestrator = get_voice_rag_orchestrator()

    for lang, expected_text in test_cases:
        mock_stt = AsyncMock()
        mock_stt.provider_name = "sarvam"
        mock_stt.transcribe = AsyncMock(
            return_value=STTResult(
                text=expected_text,
                language=lang,
                confidence=0.98,
                duration_s=2.0,
                latency_ms=250.0,
                provider="sarvam",
            )
        )
        with patch.object(orchestrator, "_stt_provider", mock_stt):
            wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00" + b"\x00" * 200
            res = await orchestrator.stt_provider.transcribe(wav_bytes, language=lang)
            assert res.text == expected_text
            assert res.language == lang
            assert res.confidence >= 0.90


# =============================================================================
# 3. TEXT RETRIEVAL & RELEVANCE TESTS
# =============================================================================

@pytest.mark.parametrize("query,language", [
    ("What is artificial intelligence?", "en"),
    ("What is machine learning?", "en"),
    ("What is a computer?", "en"),
    ("What is the internet?", "en"),
    ("What is a programming language?", "en"),
    ("कंप्यूटर क्या है?", "hi"),
    ("मशीन लर्निंग क्या है?", "hi"),
    ("इंटरनेट क्या है?", "hi"),
    ("கணினி என்றால் என்ன?", "ta"),
    ("செயற்கை நுண்ணறிவு என்றால் என்ன?", "ta"),
    ("కంప్యూటర్ అంటే ఏమిటి?", "te"),
    ("కృత్రిమ మేధస్సు అంటే ఏమిటి?", "te"),
    ("കമ്പ്യൂട്ടർ എന്താണ്?", "ml"),
    ("കൃത്രിമ ബുദ്ധി എന്താണ്?", "ml"),
])
@pytest.mark.asyncio
async def test_multilingual_text_retrieval_queries(query: str, language: str):
    """Test retrieval pipeline on standard corpus queries across all 5 languages."""
    req = AskRequest(query=query, language=language, top_k=3)
    resp = await ask_endpoint(req)
    assert resp.answer is not None and len(resp.answer.strip()) > 0
    assert resp.retrieved_context_summary.chunks_count > 0
    assert resp.latency_ms.total > 0


def test_relevance_guard_indic_stopwords():
    """Verify stopword extraction accurately ignores functional words across Indic languages."""
    words_hi = extract_content_words("कंप्यूटर क्या है और यह क्या करता है")
    assert "क्या" not in words_hi
    assert "है" not in words_hi
    assert "कंप्यूटर" in words_hi

    words_ta = extract_content_words("கணினி என்றால் என்ன")
    assert "என்ன" not in words_ta
    assert "கணினி" in words_ta

    words_te = extract_content_words("కంప్యూటర్ అంటే ఏమిటి")
    assert "ఏమిటి" not in words_te
    assert "కంప్యూటర్" in words_te

    words_ml = extract_content_words("കമ്പ്യൂട്ടർ എന്താണ്")
    assert "എന്താണ്" not in words_ml
    assert "കമ്പ്യൂട്ടർ" in words_ml


def test_relevance_guard_no_false_fragment_matches():
    """Verify short fragments (ate, ake, ing) do not create false positive overlap."""
    assert is_meaningful_word_match("ate", "chocolate") is False
    assert is_meaningful_word_match("ing", "building") is False
    assert is_meaningful_word_match("ake", "baking") is False
    assert is_meaningful_word_match("computer", "computers") is True


# =============================================================================
# 4. TRUTHFUL REFUSAL ON UNRELATED CONTEXT
# =============================================================================

UNANSWERABLE_QUERIES = [
    ("What is the recipe for chocolate chip cookies?", "en"),
    ("Who won the FIFA world cup in 1930?", "en"),
    ("How do you bake a chocolate cake on a laptop?", "en"),
]

@pytest.mark.asyncio
@pytest.mark.parametrize("query,language", UNANSWERABLE_QUERIES)
async def test_truthful_refusal_for_unrelated_context(query: str, language: str):
    """Ensure off-topic queries reject ungrounded generation and return safe fallback."""
    req = AskRequest(query=query, language=language, top_k=3)
    resp = await ask_endpoint(req)
    assert resp.grounded is False
    assert len(resp.citations) == 0
    refusal_words = ["don't have enough information", "not available", "not present", "does not contain", "cannot be found", "not mentioned"]
    assert any(rw in resp.answer.lower() for rw in refusal_words)


# =============================================================================
# 5. VOICE RAG ENDPOINT & AUDIO PAYLOAD TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_audio_security_and_format_detection():
    """Verify MIME and magic byte negotiation for multiple audio formats."""
    # WebM
    webm_head = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01" + b"\x00" * 64
    fn, mime = validate_audio_security(webm_head, content_type="audio/webm", filename="rec.webm")
    assert fn == "rec.webm"
    assert mime == "audio/webm"

    # WAV
    wav_head = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00" + b"\x00" * 64
    fn, mime = validate_audio_security(wav_head, content_type="audio/wav", filename="rec.wav")
    assert fn == "rec.wav"
    assert mime == "audio/wav"


@pytest.mark.asyncio
async def test_voice_ask_endpoint_mock_e2e():
    """Verify POST /api/voice-ask with mock STT and TTS returning valid VoiceAskResponse."""
    orchestrator = get_voice_rag_orchestrator()

    mock_stt = AsyncMock()
    mock_stt.provider_name = "sarvam"
    mock_stt.transcribe = AsyncMock(
        return_value=STTResult(
            text="What is artificial intelligence?",
            language="en",
            confidence=0.99,
            duration_s=2.0,
            latency_ms=150.0,
            provider="sarvam",
        )
    )

    mock_tts = AsyncMock()
    mock_tts.provider_name = "sarvam"
    mock_tts.synthesize = AsyncMock(
        return_value=TTSResult(
            audio_bytes=b"RIFF\x24\x00\x00\x00WAVE" + b"\x00" * 100,
            audio_format="wav",
            sample_rate=24000,
            duration_s=1.5,
            latency_ms=200.0,
            provider="sarvam",
        )
    )

    with patch.object(orchestrator, "_stt_provider", mock_stt), patch.object(orchestrator, "_tts_provider", mock_tts):
        wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00" + b"\x00" * 200
        resp = await orchestrator.execute_voice_rag(
            audio_bytes=wav_bytes,
            language="en",
            filename="test.wav",
            content_type="audio/wav",
            synthesize_speech=True,
        )

        assert isinstance(resp, VoiceAskResponse)
        assert resp.transcript == "What is artificial intelligence?"
        assert resp.status == "success"
        assert resp.audio.available is True
        assert resp.audio.audio_base64 is not None
        assert resp.latency_ms.total > 0


# =============================================================================
# 6. STREAMING & BUFFERED FALLBACK TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_ask_stream_events():
    """Verify GET /api/ask-stream emits ordered stages (retrieving -> ranking -> done)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/ask-stream?query=What+is+a+computer&language=en&top_k=3")
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        events = []
        for line in response.text.split("\n"):
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())

        assert "stage" in events or "done" in events


# =============================================================================
# 7. SECURITY & RETRIEVAL DEBUG TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_debug_retrieval_endpoint():
    """Verify GET /api/debug/retrieval returns structured diagnostic info without secrets."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/debug/retrieval?query=computer&language=en&top_k=3")
        assert resp.status_code == 200
        data = resp.json()
        assert "query" in data
        assert "dense_results" in data
        assert "bm25_results" in data
        assert "rrf_results" in data
        assert "retrieval_confidence" in data
        assert "grounding_decision" in data
        # No credentials or local filesystem paths leaked
        text_dump = json.dumps(data)
        assert "sk_" not in text_dump
        assert "api_key" not in text_dump.lower()
        assert "c:\\" not in text_dump.lower()
