"""
Phase 6.26 Retrieval Relevance & Grounding Quality Unit and Integration Tests.
Covers:
- Multilingual functional stopword handling (EN, HI, TA, TE, ML)
- Content-word token extraction and morphological stem matching
- False substring rejection (no spurious hits on "is", "the", "for", "ate", "ake")
- Multi-signal answerability validation (reranker, lexical, dense, agreement)
- Unrelated and off-topic query refusal
- Zero citations on ungrounded/refused queries
- Cross-language relevance consistency
- API response compatibility
- Phase 6.23 regression tests
"""

import pytest
from unittest.mock import MagicMock
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.guardrails.relevance import (
    MULTILINGUAL_STOPWORDS,
    extract_content_words,
    is_meaningful_word_match,
    compute_content_overlap,
    validate_answerability,
    RelevanceGuard,
)
from app.guardrails.models import RelevanceAssessment, GuardrailAction
from app.guardrails.service import GuardrailService
from app.reranking.models import RerankResult
from app.retrieval.language import to_bcp47, to_sarvam_code, detect_script_language


def make_rerank_result(
    chunk_id: str,
    text: str,
    reranker_score: float,
    dense_score: float = 0.5,
    fusion_score: float = 0.02,
    language: str = "eng_Latn",
    document_id: str = "doc_01",
    chunk_type: str = "semantic",
    rank: int = 1,
    original_rank: int = 1,
) -> RerankResult:
    """Helper to instantiate valid RerankResult models for testing."""
    return RerankResult(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        chunk_type=chunk_type,
        language=language,
        dense_score=dense_score,
        bm25_score=dense_score * 10,
        fusion_score=fusion_score,
        reranker_score=reranker_score,
        original_rank=original_rank,
        rank=rank,
    )


# ===========================================================================
# 1. Multilingual Stopword and Content-Word Extraction Tests
# ===========================================================================

def test_multilingual_stopwords_coverage():
    """Verify stopword sets exist and contain functional words across 5 languages."""
    # English
    assert "what" in MULTILINGUAL_STOPWORDS
    assert "is" in MULTILINGUAL_STOPWORDS
    assert "the" in MULTILINGUAL_STOPWORDS
    assert "computer" not in MULTILINGUAL_STOPWORDS

    # Hindi
    assert "क्या" in MULTILINGUAL_STOPWORDS
    assert "है" in MULTILINGUAL_STOPWORDS
    assert "कंप्यूटर" not in MULTILINGUAL_STOPWORDS

    # Tamil
    assert "என்ன" in MULTILINGUAL_STOPWORDS
    assert "என்றால்" in MULTILINGUAL_STOPWORDS
    assert "கணினி" not in MULTILINGUAL_STOPWORDS

    # Telugu
    assert "ఏమిటి" in MULTILINGUAL_STOPWORDS
    assert "అంటే" in MULTILINGUAL_STOPWORDS
    assert "కంప్యూటర్" not in MULTILINGUAL_STOPWORDS

    # Malayalam
    assert "എന്താണ്" in MULTILINGUAL_STOPWORDS
    assert "കമ്പ്യൂട്ടർ" not in MULTILINGUAL_STOPWORDS


def test_content_word_extraction_multilingual():
    """Verify extraction of meaningful terms while removing punctuation and stopwords."""
    # English
    en_words = extract_content_words("What is artificial intelligence and machine learning?")
    assert en_words == {"artificial", "intelligence", "machine", "learning"}

    # Hindi
    hi_words = extract_content_words("कंप्यूटर क्या है और यह कैसे काम करता है?")
    assert "कंप्यूटर" in hi_words
    assert "क्या" not in hi_words
    assert "है" not in hi_words

    # Tamil
    ta_words = extract_content_words("கணினி என்றால் என்ன?")
    assert "கணினி" in ta_words
    assert "என்ன" not in ta_words

    # Telugu
    te_words = extract_content_words("కంప్యూటర్ అంటే ఏమిటి?")
    assert "కంప్యూటర్" in te_words
    assert "ఏమిటి" not in te_words

    # Malayalam
    ml_words = extract_content_words("കമ്പ്യൂട്ടർ എന്താണ്?")
    assert "കമ്പ്യൂട്ടർ" in ml_words
    assert "എന്താണ്" not in ml_words


def test_false_substring_rejection():
    """Verify short words do not match arbitrary substrings inside unrelated words."""
    # "ai" should not match "train" or "chair"
    assert is_meaningful_word_match("ai", "train") is False
    assert is_meaningful_word_match("ai", "chair") is False

    # Short functional fragments must not match
    assert is_meaningful_word_match("is", "this") is False
    assert is_meaningful_word_match("the", "theater") is False
    assert is_meaningful_word_match("for", "forest") is False
    assert is_meaningful_word_match("ate", "chocolate") is False
    assert is_meaningful_word_match("ake", "cake") is False

    # True inflectional matches (>= 4 chars with >= 60% stem overlap)
    assert is_meaningful_word_match("computer", "computers") is True
    assert is_meaningful_word_match("learning", "learn") is True


def test_content_overlap_computation():
    """Verify overlap ratio computation behaves correctly."""
    query = "What is a computer?"
    matching_ctx = "A computer is an electronic device for storing and processing data."
    unrelated_ctx = "Baking chocolate chip cookies requires flour, sugar, and butter."

    assert compute_content_overlap(query, matching_ctx) == 1.0
    assert compute_content_overlap(query, unrelated_ctx) == 0.0


# ===========================================================================
# 2. Multi-Signal Answerability Validation Tests
# ===========================================================================

def test_validate_answerability_strong_evidence():
    """Verify strong relevant evidence is accepted with high confidence."""
    chunk = make_rerank_result(
        chunk_id="chunk_01",
        text="A computer is a programmable electronic device that accepts raw data as input and processes it.",
        reranker_score=0.88,
        dense_score=0.89,
        fusion_score=0.03,
        language="eng_Latn",
    )
    res = validate_answerability("What is a computer?", [chunk], language="en")
    assert res.allowed is True
    assert res.decision in ("STRONG", "LIMITED")
    assert res.confidence >= 0.60
    assert res.signals["lexical"] == 1.0


def test_validate_answerability_off_topic_refusal():
    """Verify off-topic question with zero overlap and low score is rejected."""
    chunk = make_rerank_result(
        chunk_id="chunk_02",
        text="The recipe calls for 2 cups of sugar, 1 cup of flour, and 3 eggs baked at 350 degrees.",
        reranker_score=0.05,
        dense_score=0.15,
        fusion_score=0.001,
        language="eng_Latn",
    )
    res = validate_answerability("What is the weather in Goa today?", [chunk], language="en")
    assert res.allowed is False
    assert res.decision in ("OFF_TOPIC", "INSUFFICIENT")
    assert res.reason in ("insufficient_content_overlap", "low_semantic_relevance", "off_topic_query")


def test_validate_answerability_empty_context():
    """Verify empty context list returns empty_retrieval_context."""
    res = validate_answerability("What is AI?", [])
    assert res.allowed is False
    assert res.reason == "empty_retrieval_context"
    assert res.confidence == 0.0


def test_validate_answerability_mock_safety():
    """Verify MagicMock objects with unset score attributes do not falsely evaluate to 1.0."""
    mock_chunk = MagicMock()
    mock_chunk.text = "This is a recipe for chocolate cake and baking cookies."
    mock_chunk.language = "eng_Latn"
    mock_chunk.reranker_score = 0.05
    mock_chunk.dense_score = 0.10
    mock_chunk.fusion_score = 0.01

    res = validate_answerability("What is a quantum microprocessor architecture?", [mock_chunk])
    assert res.allowed is False
    assert res.confidence < 0.30


# ===========================================================================
# 3. Guardrail Service & Grounding Decisions
# ===========================================================================

def test_guardrail_pre_generation_relevance():
    """Verify GuardrailService pre-generation blocks off-topic queries."""
    service = GuardrailService()
    unrelated_chunk = make_rerank_result(
        chunk_id="c_unrelated",
        text="The football team won the regional tournament with a 3-1 victory in the final match.",
        reranker_score=0.04,
        dense_score=0.12,
        fusion_score=0.001,
        language="eng_Latn",
    )

    res = service.validate_pre_generation(
        query="What is artificial intelligence?",
        retrieved_context=[unrelated_chunk],
        retrieval_confidence=0.1,
    )
    assert res.allowed is False
    assert res.action == GuardrailAction.REFUSE_GENERATION.value
    assert res.grounded is False


def test_guardrail_post_generation_refusal_handling():
    """Verify post-generation honors honest refusal text without flagging hallucination."""
    service = GuardrailService()
    refusal_answer = "I don't have enough information in the retrieved context to answer that."

    res = service.validate_post_generation(
        query="What is the weather in Goa?",
        answer=refusal_answer,
        retrieved_context=[],
    )
    # Refusal response is allowed and correctly marked ungrounded
    assert res.grounded is False
    assert res.action == GuardrailAction.ALLOW.value
    assert res.reason == "honest_refusal"


# ===========================================================================
# 4. Phase 6.23 Failure Mode Regression
# ===========================================================================

def test_phase623_regression_basic_programming_for_ai():
    """Ensure 'What is AI?' with passages about BASIC programming is strictly refused."""
    basic_chunk = make_rerank_result(
        chunk_id="basic_doc",
        text="BASIC is an acronym for Beginner's All-purpose Symbolic Instruction Code, created in 1964 at Dartmouth College.",
        reranker_score=0.08,
        dense_score=0.22,
        fusion_score=0.002,
        language="eng_Latn",
    )
    res = validate_answerability("What is artificial intelligence?", [basic_chunk])
    assert res.allowed is False
    assert res.decision in ("OFF_TOPIC", "INSUFFICIENT")


# ===========================================================================
# 5. Language Resolver & BCP47 Normalization
# ===========================================================================

def test_language_resolver_canonical_codes():
    """Verify BCP-47 and Sarvam language resolution across all 5 Indic languages."""
    assert to_bcp47("hin_Deva") == "hi"
    assert to_bcp47("hin_deva") == "hi"
    assert to_bcp47("hi") == "hi"
    assert to_bcp47("tam_Taml") == "ta"
    assert to_bcp47("tel_Telu") == "te"
    assert to_bcp47("mal_Mlym") == "ml"
    assert to_bcp47("eng_Latn") == "en"

    assert to_sarvam_code("hi") == "hi-IN"
    assert to_sarvam_code("ta") == "ta-IN"
    assert to_sarvam_code("te") == "te-IN"
    assert to_sarvam_code("ml") == "ml-IN"
    assert to_sarvam_code("en") == "en-IN"


# ===========================================================================
# 6. API Response Contracts Compatibility
# ===========================================================================

@pytest.mark.asyncio
async def test_api_ask_response_relevance_compatibility():
    """Verify /api/ask preserves all required fields on both successful and refused responses."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # 1. In-domain question
        res = await client.post(
            "/api/ask",
            json={"query": "What is a computer?", "language": "en"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "answer" in data
        assert "grounded" in data
        assert "confidence" in data
        assert "citations" in data
        assert "citation_provenance" in data
        assert "language" in data
        assert "latency_ms" in data
        assert "relevance" in data
        assert isinstance(data["grounded"], bool)

        # 2. Refusal question
        off_res = await client.post(
            "/api/ask",
            json={"query": "What is my bank balance?", "language": "en"},
        )
        assert off_res.status_code == 200
        off_data = off_res.json()
        assert off_data["grounded"] is False
        assert off_data["citations"] == []
        assert "relevance" in off_data
