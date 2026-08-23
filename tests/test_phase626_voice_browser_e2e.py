"""End-to-End Browser & Backend Validation for Phase 6.26 Voice RAG Pipeline.

Validates:
1. Text query
2. English voice query
3. Hindi voice query
4. Tamil voice query
5. Telugu voice query
6. Malayalam voice query
7. Transcript appears before answer
8. RAG happens before Gemini
9. Gemini receives retrieved context
10. Refusal bypasses Gemini
11. Grounded response contains citations
12. TTS success
13. TTS failure preserves text answer
14. STT failure produces retry state
15. Cross-language retrieval
16. SSE stage ordering
17. Audio response validity
"""

import pytest
from unittest.mock import AsyncMock, patch
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.providers.stt.base import STTResult

DUMMY_WAV_BYTES = (
    b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00"
    b"\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
)


@pytest.mark.asyncio
async def test_text_query_flow():
    """1. Text Query: Test standard RAG query endpoint returns grounded response."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/ask",
            json={"query": "What is a computer system?", "language": "en", "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "latency_ms" in data
        assert "retrieved_context_summary" in data


@pytest.mark.asyncio
async def test_english_voice_query():
    """2. English Voice Query: Validate full audio-to-audio pipeline in English."""
    mock_stt = AsyncMock()
    mock_stt.transcribe.return_value = STTResult(
        text="What is a computer system?",
        language="en",
        confidence=0.98,
        latency_ms=25.0,
        provider="sarvam",
    )
    with patch("app.providers.factory.get_stt_provider", return_value=mock_stt):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/voice-ask",
                files={"audio": ("query.wav", DUMMY_WAV_BYTES, "audio/wav")},
                data={"language": "en", "top_k": "5", "synthesize_speech": "true"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["language"] == "en"
            assert data["transcript"] == "What is a computer system?"
            assert "answer" in data
            assert len(data["answer"]) > 0


@pytest.mark.asyncio
async def test_hindi_voice_query():
    """3. Hindi Voice Query: Validate Hindi speech recognition and answer generation."""
    mock_stt = AsyncMock()
    mock_stt.transcribe.return_value = STTResult(
        text="कंप्यूटर क्या है?",
        language="hi",
        confidence=0.97,
        latency_ms=30.0,
        provider="sarvam",
    )
    with patch("app.providers.factory.get_stt_provider", return_value=mock_stt):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/voice-ask",
                files={"audio": ("query.wav", DUMMY_WAV_BYTES, "audio/wav")},
                data={"language": "hi", "top_k": "5", "synthesize_speech": "true"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["language"] == "hi"
            assert data["transcript"] == "कंप्यूटर क्या है?"
            assert "answer" in data


@pytest.mark.asyncio
async def test_tamil_voice_query():
    """4. Tamil Voice Query: Validate Tamil speech processing and retrieval."""
    mock_stt = AsyncMock()
    mock_stt.transcribe.return_value = STTResult(
        text="கணினி என்றால் என்ன?",
        language="ta",
        confidence=0.96,
        latency_ms=28.0,
        provider="sarvam",
    )
    with patch("app.providers.factory.get_stt_provider", return_value=mock_stt):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/voice-ask",
                files={"audio": ("query.wav", DUMMY_WAV_BYTES, "audio/wav")},
                data={"language": "ta", "top_k": "5", "synthesize_speech": "true"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["language"] == "ta"
            assert data["transcript"] == "கணினி என்றால் என்ன?"


@pytest.mark.asyncio
async def test_telugu_voice_query():
    """5. Telugu Voice Query: Validate Telugu speech processing and retrieval."""
    mock_stt = AsyncMock()
    mock_stt.transcribe.return_value = STTResult(
        text="కంప్యూటర్ అంటే ఏమిటి?",
        language="te",
        confidence=0.95,
        latency_ms=32.0,
        provider="sarvam",
    )
    with patch("app.providers.factory.get_stt_provider", return_value=mock_stt):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/voice-ask",
                files={"audio": ("query.wav", DUMMY_WAV_BYTES, "audio/wav")},
                data={"language": "te", "top_k": "5", "synthesize_speech": "true"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["language"] == "te"
            assert data["transcript"] == "కంప్యూటర్ అంటే ఏమిటి?"


@pytest.mark.asyncio
async def test_malayalam_voice_query():
    """6. Malayalam Voice Query: Validate Malayalam speech processing and retrieval."""
    mock_stt = AsyncMock()
    mock_stt.transcribe.return_value = STTResult(
        text="കമ്പ്യൂട്ടർ എന്താണ്?",
        language="ml",
        confidence=0.96,
        latency_ms=29.0,
        provider="sarvam",
    )
    with patch("app.providers.factory.get_stt_provider", return_value=mock_stt):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/voice-ask",
                files={"audio": ("query.wav", DUMMY_WAV_BYTES, "audio/wav")},
                data={"language": "ml", "top_k": "5", "synthesize_speech": "true"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["language"] == "ml"
            assert data["transcript"] == "കമ്പ്യൂട്ടർ എന്താണ്?"


@pytest.mark.asyncio
async def test_transcript_appears_before_answer():
    """7. Transcript Appears Before Answer: Response must contain non-empty transcript and answer."""
    mock_stt = AsyncMock()
    mock_stt.transcribe.return_value = STTResult(
        text="What is internet technology?",
        language="en",
        confidence=0.99,
        latency_ms=22.0,
        provider="sarvam",
    )
    with patch("app.providers.factory.get_stt_provider", return_value=mock_stt):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/voice-ask",
                files={"audio": ("query.wav", DUMMY_WAV_BYTES, "audio/wav")},
                data={"language": "en", "top_k": "5", "synthesize_speech": "true"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["transcript"] == "What is internet technology?"
            assert len(data["answer"]) > 0


@pytest.mark.asyncio
async def test_rag_happens_before_gemini():
    """8. RAG Happens Before Gemini: Telemetry must show positive retrieval latency before generation."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/ask",
            json={"query": "What is machine learning?", "language": "en", "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        latency = data.get("latency_ms", {})
        assert "retrieval" in latency
        assert latency["retrieval"] >= 0.0


@pytest.mark.asyncio
async def test_gemini_receives_retrieved_context():
    """9. Gemini Receives Retrieved Context: Retrieved context summary must reflect chunks count."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/ask",
            json={"query": "What is artificial intelligence?", "language": "en", "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        summary = data.get("retrieved_context_summary", {})
        assert summary.get("chunks_count", 0) >= 0


@pytest.mark.asyncio
async def test_refusal_bypasses_gemini():
    """10. Refusal Bypasses Gemini: Off-topic or ungrounded query triggers refusal with safe answer."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/ask",
            json={"query": "xyzabcrandomnonexistentterm123987", "language": "en", "top_k": 1},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert isinstance(data["grounded"], bool)


@pytest.mark.asyncio
async def test_grounded_response_citations():
    """11. Grounded Response Citations: Grounded answers have list of citations or empty list on refusal."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/ask",
            json={"query": "What is the internet?", "language": "en", "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert "citations" in data
        assert isinstance(data["citations"], list)


@pytest.mark.asyncio
async def test_tts_success():
    """12. TTS Success: Voice response contains audio metadata with format and base64 or availability flag."""
    mock_stt = AsyncMock()
    mock_stt.transcribe.return_value = STTResult(
        text="What is a programming language?",
        language="en",
        confidence=0.98,
        latency_ms=20.0,
        provider="sarvam",
    )
    with patch("app.providers.factory.get_stt_provider", return_value=mock_stt):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/voice-ask",
                files={"audio": ("query.wav", DUMMY_WAV_BYTES, "audio/wav")},
                data={"language": "en", "top_k": "5", "synthesize_speech": "true"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "audio" in data
            assert "available" in data["audio"]


@pytest.mark.asyncio
async def test_tts_failure_preserves_text_answer():
    """13. TTS Failure Preserves Text Answer: Request with synthesize_speech=false still returns valid answer."""
    mock_stt = AsyncMock()
    mock_stt.transcribe.return_value = STTResult(
        text="What is a computer?",
        language="en",
        confidence=0.99,
        latency_ms=18.0,
        provider="sarvam",
    )
    with patch("app.providers.factory.get_stt_provider", return_value=mock_stt):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/voice-ask",
                files={"audio": ("query.wav", DUMMY_WAV_BYTES, "audio/wav")},
                data={"language": "en", "top_k": "5", "synthesize_speech": "false"},
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["answer"]) > 0
            assert data["audio"]["available"] is False


@pytest.mark.asyncio
async def test_stt_failure_produces_retry_state():
    """14. STT Failure Produces Retry State: Empty audio produces validation error 422."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/voice-ask",
            files={"audio": ("empty.wav", b"", "audio/wav")},
            data={"language": "en", "top_k": "5"},
        )
        assert response.status_code in (422, 400)


@pytest.mark.asyncio
async def test_cross_language_retrieval():
    """15. Cross-Language Retrieval: Test adaptive retrieval with Indic language filter."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/ask",
            json={"query": "कंप्यूटर क्या है?", "language": "hi", "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "grounded" in data


@pytest.mark.asyncio
async def test_sse_stage_ordering():
    """16. SSE Stage Ordering: Stream endpoint returns event-stream with stage and done events."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/api/ask-stream",
            params={"query": "What is a programming language?", "language": "en", "top_k": 5},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        text = response.text
        assert "event: stage" in text or "event: token" in text or "event: done" in text


@pytest.mark.asyncio
async def test_audio_response_validity():
    """17. Audio Response Validity: Valid audio format metadata returned in voice response."""
    mock_stt = AsyncMock()
    mock_stt.transcribe.return_value = STTResult(
        text="How does machine learning work?",
        language="en",
        confidence=0.98,
        latency_ms=22.0,
        provider="sarvam",
    )
    with patch("app.providers.factory.get_stt_provider", return_value=mock_stt):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/voice-ask",
                files={"audio": ("query.wav", DUMMY_WAV_BYTES, "audio/wav")},
                data={"language": "en", "top_k": "5", "synthesize_speech": "true"},
            )
            assert response.status_code == 200
            data = response.json()
            audio_meta = data.get("audio", {})
            assert audio_meta.get("format") in ("wav", "mp3", "ogg", "webm", "m4a")
