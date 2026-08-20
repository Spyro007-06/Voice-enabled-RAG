"""Unit tests for Phase 6.4 Structured Output and Provenance-Aware Citations."""

import pytest
from pydantic import ValidationError

from app.generation.models import (
    CitationProvenance,
    GenerationResult,
    ValidationResult,
)
from app.generation.validators import AnswerValidator, CitationValidator
from app.reranking.models import RerankResult


# --- 1. Citation Validation Tests ---

def test_citation_validator_valid_citations():
    """Verify CitationValidator accepts valid chunk IDs and extracts rich provenance."""
    chunks = [
        RerankResult(
            chunk_id="chunk_delhi_01",
            document_id="doc_india_01",
            text="New Delhi is the capital of India.",
            chunk_type="fixed",
            language="en",
            reranker_score=0.98,
            original_rank=1,
            rank=1,
        ),
        RerankResult(
            chunk_id="chunk_goa_02",
            document_id="doc_india_02",
            text="Panaji is the capital of Goa.",
            chunk_type="sentence",
            language="en",
            reranker_score=0.88,
            original_rank=2,
            rank=2,
        ),
    ]

    raw_citations = ["chunk_delhi_01", "doc_india_02"]
    validated, prov_list, fabricated = CitationValidator.validate_citations(
        raw_citations=raw_citations,
        retrieved_context=chunks,
    )

    assert len(validated) == 2
    assert "chunk_delhi_01" in validated
    assert "chunk_goa_02" in validated  # mapped doc_india_02 to its chunk_id
    assert len(fabricated) == 0
    assert len(prov_list) == 2
    assert prov_list[0].chunk_id == "chunk_delhi_01"
    assert prov_list[0].rank == 1
    assert prov_list[0].score == 0.98
    assert "New Delhi" in prov_list[0].snippet


def test_citation_validator_rejects_fabricated_citations():
    """Verify CitationValidator rejects and removes unknown/hallucinated chunk IDs."""
    chunks = [
        RerankResult(
            chunk_id="chunk_real_100",
            document_id="doc_real_100",
            text="Real evidence about Indian monuments.",
            chunk_type="semantic",
            reranker_score=0.91,
            original_rank=1,
            rank=1,
        )
    ]

    raw_citations = [
        "chunk_real_100",
        "hallucinated_doc_999",
        "fake_citation_xyz",
    ]

    validated, prov_list, fabricated = CitationValidator.validate_citations(
        raw_citations=raw_citations,
        retrieved_context=chunks,
    )

    assert validated == ["chunk_real_100"]
    assert len(prov_list) == 1
    assert len(fabricated) == 2
    assert "hallucinated_doc_999" in fabricated
    assert "fake_citation_xyz" in fabricated


def test_citation_validator_empty_context():
    """Verify CitationValidator handles empty context safely."""
    validated, prov_list, fabricated = CitationValidator.validate_citations(
        raw_citations=["chunk_1"],
        retrieved_context=[],
    )
    assert validated == []
    assert prov_list == []
    assert fabricated == ["chunk_1"]


# --- 2. Answer Quality and Confidence Validation Tests ---

def test_answer_validator_empty_answer():
    """Verify AnswerValidator flags empty or whitespace-only answer."""
    res = AnswerValidator.validate_answer(
        answer="   ",
        grounded=True,
        confidence=1.0,
    )
    assert res.is_valid is False
    assert res.cleaned_answer == ""
    assert res.confidence == 0.0
    assert "Answer cannot be empty" in res.issues[0]


def test_answer_validator_confidence_bounds():
    """Verify confidence score is clamped to [0.0, 1.0] range."""
    # Test confidence > 1.0
    res_high = AnswerValidator.validate_answer(
        answer="Valid answer text.",
        confidence=1.5,
    )
    assert res_high.confidence == 1.0
    assert any("outside [0.0, 1.0]" in issue for issue in res_high.issues)

    # Test confidence < 0.0
    res_low = AnswerValidator.validate_answer(
        answer="Valid answer text.",
        confidence=-0.5,
    )
    assert res_low.confidence == 0.0
    assert any("outside [0.0, 1.0]" in issue for issue in res_low.issues)


def test_answer_validator_excessive_length():
    """Verify excessively long answer is truncated safely."""
    long_answer = "Goa is a coastal state. " * 500
    res = AnswerValidator.validate_answer(
        answer=long_answer,
        max_length=200,
    )
    assert len(res.cleaned_answer) <= 205
    assert res.cleaned_answer.endswith("...")
    assert any("exceeded maximum length limit" in issue for issue in res.issues)


def test_answer_validator_refusal_detection():
    """Verify insufficient information refusal marks grounded as False."""
    res = AnswerValidator.validate_answer(
        answer="The available context does not contain sufficient information to answer this question.",
        grounded=True,
        confidence=0.9,
    )
    assert res.cleaned_answer.startswith("The available context does not contain")


# --- 3. Multilingual Unicode Preservation in Answer & Citations ---

@pytest.mark.parametrize(
    "language,answer,snippet",
    [
        (
            "hi",
            "भारत की राजधानी नई दिल्ली है।",
            "नई दिल्ली भारत की राजधानी है।",
        ),
        (
            "ta",
            "இந்தியாவின் தலைநகரம் புது தில்லி ஆகும்.",
            "புது தில்லி இந்தியாவின் தலைநகரம் ஆகும்.",
        ),
        (
            "te",
            "భారతదేశ రాజధాని న్యూఢిల్లీ.",
            "న్యూఢిల్లీ భారతదేశ రాజధాని.",
        ),
        (
            "ml",
            "ഇന്ത്യയുടെ തലസ്ഥാനം ന്യൂഡൽഹിയാണ്.",
            "ന്യൂഡൽഹിയാണ് ഇന്ത്യയുടെ തലസ്ഥാനം.",
        ),
    ],
)
def test_answer_validator_multilingual_unicode(language: str, answer: str, snippet: str):
    """Verify full multilingual Unicode preservation without corruption across Indian languages."""
    chunk = RerankResult(
        chunk_id=f"chunk_{language}",
        document_id=f"doc_{language}",
        text=snippet,
        chunk_type="sentence",
        language=language,
        reranker_score=0.95,
        original_rank=1,
        rank=1,
    )

    res = AnswerValidator.validate_answer(
        answer=answer,
        grounded=True,
        confidence=0.95,
        citations=[f"chunk_{language}"],
        retrieved_context=[chunk],
    )

    assert res.is_valid is True
    assert res.cleaned_answer == answer
    assert res.validated_citations == [f"chunk_{language}"]
    assert res.citation_provenance[0].snippet == snippet


# --- 4. GenerationResult Model Validation ---

def test_generation_result_with_provenance():
    """Verify GenerationResult serializes confidence and citation provenance accurately."""
    prov = CitationProvenance(
        chunk_id="chunk_01",
        document_id="doc_01",
        rank=1,
        chunk_type="fixed",
        language="hi",
        score=0.95,
        snippet="Snippet text",
    )
    result = GenerationResult(
        answer="Valid grounded answer.",
        grounded=True,
        confidence=0.92,
        citations=["chunk_01"],
        citation_provenance=[prov],
        model="mock-model",
    )

    data = result.model_dump()
    assert data["confidence"] == 0.92
    assert len(data["citation_provenance"]) == 1
    assert data["citation_provenance"][0]["chunk_id"] == "chunk_01"
    assert data["citation_provenance"][0]["score"] == 0.95


def test_generation_result_confidence_validation():
    """Verify GenerationResult enforces [0.0, 1.0] bounds on confidence."""
    with pytest.raises(ValidationError):
        GenerationResult(answer="Test", confidence=1.2, model="test")

    with pytest.raises(ValidationError):
        GenerationResult(answer="Test", confidence=-0.1, model="test")
