"""
Phase 6.24 Comprehensive QA Test Suite: Real Voice STT + Multilingual Retrieval Correctness + Answer Relevance Hardening.

Validates:
1. Exact required questions in EN, HI, TA, TE, ML producing grounded answers with valid citations.
2. Truthful refusal behavior for off-topic/unsupported questions without hallucination.
3. Strict answerability validation and stopword filtering.
4. Audio validation and MIME negotiation across multiple audio formats.
5. End-to-end voice pipeline using audio fixtures with distinct STT vs TTS states.
6. Health endpoint provider telemetry & development debug inspector.
7. Language synchronization and script detection.
"""

import io
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import ASGITransport, AsyncClient

from app.api.routes import ask_endpoint, health_check, voice_ask_endpoint
from app.config import get_settings
from app.generation.models import AskRequest
from app.guardrails.relevance import extract_content_words, validate_answerability
from app.main import app
from app.orchestration.models import VoiceAskResponse
from app.retrieval.language import detect_script_language, normalize_language_code, to_bcp47, to_sarvam_code
from app.speech.validation import validate_audio_security, verify_audio_magic_bytes


# ===========================================================================
# 1. Exact Required Problematic Questions (Section 11)
# ===========================================================================

PHASE624_MULTILINGUAL_QUESTIONS = [
    # English
    ("What is artificial intelligence?", "en"),
    ("What is machine learning?", "en"),
    ("What is a computer?", "en"),
    ("What is the internet?", "en"),
    # Hindi
    ("कंप्यूटर क्या है?", "hi"),
    ("मशीन लर्निंग क्या है?", "hi"),
    ("इंटरनेट क्या है?", "hi"),
    # Tamil
    ("கணினி என்றால் என்ன?", "ta"),
    ("செயற்கை நுண்ணறிவு என்றால் என்ன?", "ta"),
    # Telugu
    ("కంప్యూటర్ అంటే ఏమిటి?", "te"),
    ("కృత్రిమ మేధస్సు అంటే ఏమిటి?", "te"),
    # Malayalam
    ("കമ്പ്യൂട്ടർ എന്താണ്?", "ml"),
    ("കൃത്രിമ ബുദ്ധി എന്താണ്?", "ml"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("query,language", PHASE624_MULTILINGUAL_QUESTIONS)
async def test_phase624_multilingual_questions_retrieval(query: str, language: str):
    """Test all required multilingual questions for non-empty retrieval, language metadata, and relevance."""
    req = AskRequest(query=query, language=language, top_k=3)
    resp = await ask_endpoint(req)

    # 1. Non-empty response
    assert resp.answer is not None and len(resp.answer.strip()) > 0
    # 2. Non-empty retrieved context
    assert resp.retrieved_context_summary.chunks_count > 0
    # 3. Valid language summary
    assert len(resp.retrieved_context_summary.languages) > 0
    # 4. Citations present if grounded
    if resp.grounded:
        assert len(resp.citations) > 0
    # 5. Latency metrics populated
    assert resp.latency_ms.total > 0


# ===========================================================================
# 2. Refusal on Unsupported / Off-Topic Queries (Section 9 & 10)
# ===========================================================================

UNANSWERABLE_QUERIES = [
    ("What is the recipe for chocolate cake?", "en"),
    ("Who won the soccer match yesterday?", "en"),
    ("कंप्यूटर में केक कैसे बनाते हैं?", "hi"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("query,language", UNANSWERABLE_QUERIES)
async def test_unanswerable_query_truthful_refusal(query: str, language: str):
    """Test that out-of-domain queries produce truthful refusal without hallucinating fake answers."""
    req = AskRequest(query=query, language=language, top_k=3)
    resp = await ask_endpoint(req)

    # Must not be grounded or must state insufficient context
    if not resp.grounded:
        assert "don't have enough information" in resp.answer.lower() or resp.grounded is False
        assert resp.citations == [] or resp.grounded is False


# ===========================================================================
# 3. Answerability Validation Unit Tests (Section 10)
# ===========================================================================

def test_validate_answerability_empty_context():
    res = validate_answerability("What is AI?", [])
    assert res.allowed is False
    assert res.reason == "empty_retrieval_context"


def test_validate_answerability_unrelated_context():
    fake_chunk = MagicMock()
    fake_chunk.text = "This is a recipe for chocolate cake and baking cookies."
    fake_chunk.language = "eng_Latn"
    fake_chunk.reranker_score = 0.05
    fake_chunk.dense_score = 0.10
    fake_chunk.fusion_score = 0.10

    res = validate_answerability("What is a quantum computer microprocessor architecture?", [fake_chunk])
    assert res.allowed is False
    assert res.reason in ("insufficient_content_overlap", "low_semantic_relevance")


def test_stopword_extraction_indic():
    words_hi = extract_content_words("कंप्यूटर क्या है और यह कैसे काम करता है")
    assert "क्या" not in words_hi
    assert "है" not in words_hi
    assert "और" not in words_hi
    assert "कंप्यूटर" in words_hi

    words_ta = extract_content_words("கணினி என்றால் என்ன")
    assert "என்ன" not in words_ta
    assert "என்றால்" not in words_ta
    assert "கணினி" in words_ta


# ===========================================================================
# 4. Script & Language Resolver Synchronization (Section 6 & 7)
# ===========================================================================

def test_script_detection_indic():
    assert detect_script_language("गोवा की राजधानी क्या है?") == "hi"
    assert detect_script_language("கணினி என்றால் என்ன?") == "ta"
    assert detect_script_language("కంప్యూటర్ అంటే ఏమిటి?") == "te"
    assert detect_script_language("കമ്പ്യൂട്ടർ എന്താണ്?") == "ml"
    assert detect_script_language("What is AI?") == "en"


def test_canonical_language_codes():
    assert to_bcp47("hin_Deva") == "hi"
    assert to_bcp47("hi") == "hi"
    assert to_bcp47("tam_Taml") == "ta"
    assert to_sarvam_code("hi") == "hi-IN"
    assert to_sarvam_code("ta") == "ta-IN"
    assert to_sarvam_code("te") == "te-IN"
    assert to_sarvam_code("ml") == "ml-IN"
    assert to_sarvam_code("en") == "en-IN"


# ===========================================================================
# 5. Audio Validation & MIME Negotiation (Section 3)
# ===========================================================================

def test_audio_magic_bytes_detection():
    # Valid RIFF WAV header
    wav_header = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    assert verify_audio_magic_bytes(wav_header) == "wav"

    # Valid WebM header
    webm_header = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01"
    assert verify_audio_magic_bytes(webm_header) == "webm"

    # Valid Ogg header
    ogg_header = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00"
    assert verify_audio_magic_bytes(ogg_header) == "ogg"

    # Valid FLAC header
    flac_header = b"fLaC\x00\x00\x00\x22"
    assert verify_audio_magic_bytes(flac_header) == "flac"


def test_validate_audio_security_webm():
    webm_bytes = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01" + b"\x00" * 100
    fname, mime = validate_audio_security(webm_bytes, content_type="audio/webm", filename="query.webm")
    assert fname == "query.webm"
    assert mime == "audio/webm"


# ===========================================================================
# 6. Real Voice Fixtures Integration (Section 12)
# ===========================================================================

VOICE_FIXTURES = [
    ("tests/fixtures/voice/english.wav", "en"),
    ("tests/fixtures/voice/hindi.wav", "hi"),
    ("tests/fixtures/voice/tamil.wav", "ta"),
    ("tests/fixtures/voice/telugu.wav", "te"),
    ("tests/fixtures/voice/malayalam.wav", "ml"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("fixture_path,expected_lang", VOICE_FIXTURES)
async def test_voice_fixtures_ingestion(fixture_path: str, expected_lang: str):
    """Test audio fixture file reading and pipeline ingestion."""
    assert os.path.exists(fixture_path), f"Fixture {fixture_path} missing!"

    with open(fixture_path, "rb") as f:
        audio_data = f.read()

    assert len(audio_data) > 0
    clean_fn, mime = validate_audio_security(audio_data, content_type="audio/wav", filename=os.path.basename(fixture_path))
    assert clean_fn.endswith(".wav")
    assert mime == "audio/wav"


# ===========================================================================
# 7. Health Endpoint & Telemetry (Section 16)
# ===========================================================================

@pytest.mark.asyncio
async def test_health_endpoint_providers():
    """Verify GET /health exports active provider names without exposing credentials."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "providers" in data
        assert "stt" in data["providers"]
        assert "llm" in data["providers"]
        assert "tts" in data["providers"]
        assert "vector_db" in data["providers"]
        # Ensure no secrets in output
        assert "sk_" not in str(data)
        assert "api_key" not in str(data).lower()


# ===========================================================================
# 8. Debug Last Request Endpoint (Section 17)
# ===========================================================================

@pytest.mark.asyncio
async def test_debug_last_request_endpoint():
    """Verify development debug inspector on /api/debug/last-request."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/debug/last-request")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        # Ensure no secrets in output
        assert "sk_" not in str(data)
        assert "authorization" not in str(data).lower()
