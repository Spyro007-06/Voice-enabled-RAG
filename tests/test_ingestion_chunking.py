"""Unit tests for Phase 2 ingestion and chunking components."""

import pytest
from app.chunking.base import ChunkingStrategy
from app.chunking.fixed import FixedChunkingStrategy
from app.chunking.hierarchical import HierarchicalChunkingStrategy
from app.chunking.models import Chunk, generate_chunk_id
from app.chunking.registry import ChunkingRegistry
from app.chunking.semantic import (
    LightweightTfIdfRepresentationProvider,
    SemanticChunkingStrategy,
    compute_cosine_similarity,
)
from app.chunking.sentence import SentenceChunkingStrategy, split_into_sentences_with_offsets
from app.chunking.sliding_window import SlidingWindowChunkingStrategy
from app.chunking.validator import ChunkValidationError, ChunkValidator
from app.ingestion.dataset_loader import DatasetLoader
from app.ingestion.models import Document
from app.ingestion.normalizer import DocumentNormalizer, normalize_text


# --- Fixtures ---

@pytest.fixture
def sample_msmarco_record():
    """Synthetic MSMARCO-XI record with multiple passages and metadata."""
    return {
        "query_id": 999123,
        "query": "भारत की राजधानी क्या है?",
        "Answer": "भारत की राजधानी नई दिल्ली है।",
        "query_type": "DESCRIPTION",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "Eng_Query": "what is the capital of india?",
        "Eng_Answer": "The capital of India is New Delhi.",
        "meta": {
            "model_name": "indic-trans-v2",
            "temperature": 0.0,
            "max_tokens": 512,
        },
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "New Delhi is the capital of India. It serves as the center of the Government of India.",
                "Mumbai is the financial capital of India. It is located in Maharashtra.",
            ],
            "Translated_passages": [
                "नई दिल्ली भारत की राजधानी है। यह भारत सरकार के केंद्र के रूप में कार्य करती है।",
                "मुंबई भारत की वित्तीय राजधानी है। यह महाराष्ट्र में स्थित है।",
            ],
        },
    }


@pytest.fixture
def sample_document():
    """Single normalized Document fixture."""
    return Document(
        document_id="msmarco_999123_0",
        text="नई दिल्ली भारत की राजधानी है। यह भारत सरकार के केंद्र के रूप में कार्य करती है। राष्ट्रपति भवन यहाँ स्थित है।",
        title=None,
        language="hin_Deva",
        source="ai4bharat/MSMARCO-XI",
        metadata={
            "query_id": 999123,
            "query": "भारत की राजधानी क्या है?",
            "is_selected": 1,
            "passage_index": 0,
        },
    )


# --- Normalization Tests ---

def test_normalize_text():
    """Verify safe whitespace and linebreak normalization."""
    raw = "  यह   एक   परीक्षण    वाक्य है।\r\n\n\n\nदूसरा   पैराग्राफ।  "
    normalized = normalize_text(raw)
    assert normalized == "यह एक परीक्षण वाक्य है。\n\nदूसरा पैराग्राफ।" or "यह एक परीक्षण वाक्य है।\n\nदूसरा पैराग्राफ।" in normalized
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_document_normalizer(sample_msmarco_record):
    """Verify normalizer extracts passages into Documents with preserved metadata."""
    normalizer = DocumentNormalizer()
    docs = normalizer.normalize_record(sample_msmarco_record)

    assert len(docs) == 2
    assert docs[0].document_id == "msmarco_999123_0"
    assert docs[0].language == "hin_Deva"
    assert docs[0].metadata["query_id"] == 999123
    assert docs[0].metadata["is_selected"] == 1
    assert "नई दिल्ली" in docs[0].text

    assert docs[1].document_id == "msmarco_999123_1"
    assert docs[1].metadata["is_selected"] == 0
    assert "मुंबई" in docs[1].text


def test_normalizer_empty_and_corrupt_records():
    """Verify handling of empty or missing passages."""
    normalizer = DocumentNormalizer()
    assert normalizer.normalize_record({}) == []
    assert normalizer.normalize_record({"query_id": None}) == []
    assert normalizer.normalize_record({"query_id": 123, "passages": {}}) == []


# --- Deterministic ID & Model Tests ---

def test_deterministic_chunk_id():
    """Ensure chunk ID is 100% deterministic across multiple calls."""
    id1 = generate_chunk_id("doc_100", "fixed", 0)
    id2 = generate_chunk_id("doc_100", "fixed", 0)
    id3 = generate_chunk_id("doc_100", "fixed", 1)
    id4 = generate_chunk_id("doc_100", "sentence", 0)

    assert id1 == id2
    assert id1 != id3
    assert id1 != id4


# --- Chunking Strategies Tests ---

def test_fixed_chunking(sample_document):
    """Test fixed size chunking strategy."""
    strategy = FixedChunkingStrategy(chunk_size=50, chunk_overlap=10)
    chunks = strategy.chunk(sample_document)

    assert len(chunks) >= 2
    for i, c in enumerate(chunks):
        assert c.chunk_type == "fixed"
        assert c.chunk_index == i
        assert c.document_id == sample_document.document_id
        assert len(c.text) <= 50
        assert c.start_position >= 0
        assert c.end_position >= c.start_position


def test_sentence_chunking(sample_document):
    """Test sentence boundary chunking."""
    strategy = SentenceChunkingStrategy(max_chunk_size=60)
    chunks = strategy.chunk(sample_document)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.chunk_type == "sentence"
        assert c.metadata.get("query_id") == 999123
        assert len(c.text) > 0


def test_sliding_window_chunking(sample_document):
    """Test sliding window overlapping sentence chunking."""
    strategy = SlidingWindowChunkingStrategy(window_size=2, window_overlap=1)
    chunks = strategy.chunk(sample_document)

    assert len(chunks) >= 1
    assert chunks[0].chunk_type == "sliding_window"
    assert chunks[0].metadata["strategy"] == "sliding_window"


def test_semantic_chunking(sample_document):
    """Test semantic chunking boundary detection."""
    strategy = SemanticChunkingStrategy(similarity_threshold=0.3, max_chunk_size=100)
    chunks = strategy.chunk(sample_document)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.chunk_type == "semantic"
        assert c.document_id == sample_document.document_id


def test_semantic_representation_and_similarity():
    """Test TF-IDF representation and cosine similarity computation."""
    provider = LightweightTfIdfRepresentationProvider()
    sents = [
        "भारत की राजधानी नई दिल्ली है।",
        "नई दिल्ली में राष्ट्रपति भवन है।",
        "सौर मंडल में आठ ग्रह हैं।",
    ]
    vectors = provider.encode_sentences(sents)
    assert len(vectors) == 3
    
    sim_related = compute_cosine_similarity(vectors[0], vectors[1])
    sim_unrelated = compute_cosine_similarity(vectors[0], vectors[2])
    assert sim_related >= sim_unrelated


def test_hierarchical_chunking(sample_document):
    """Test hierarchical chunking preserves document hierarchy context."""
    strategy = HierarchicalChunkingStrategy(max_chunk_size=80)
    chunks = strategy.chunk(sample_document)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.chunk_type == "hierarchical"
        assert c.metadata["query_id"] == 999123
        assert c.metadata["is_selected"] == 1
        assert c.metadata["hierarchy_level"] == "passage_leaf"


# --- Registry Tests ---

def test_chunking_registry():
    """Test registry discovery and configuration."""
    available = ChunkingRegistry.list_available_strategies()
    assert "fixed" in available
    assert "sentence" in available
    assert "sliding_window" in available
    assert "semantic" in available
    assert "hierarchical" in available

    strategies = ChunkingRegistry.get_configured_strategies("fixed,sentence")
    assert len(strategies) == 2
    assert strategies[0].strategy_name == "fixed"
    assert strategies[1].strategy_name == "sentence"

    with pytest.raises(ValueError):
        ChunkingRegistry.get_strategy("non_existent_strategy")


# --- Validator Tests ---

def test_chunk_validator(sample_document):
    """Test ChunkValidator correctness and anomaly detection."""
    strategy = SentenceChunkingStrategy(max_chunk_size=100)
    chunks = strategy.chunk(sample_document)

    validator = ChunkValidator(min_char_length=1)
    is_valid, errors = validator.validate_chunks(chunks)
    assert is_valid
    assert len(errors) == 0

    # Test error detection on corrupted chunk
    corrupted_chunk = chunks[0].model_copy(update={"chunk_id": "invalid_random_id"})
    is_corrupt_valid, corrupt_errors = validator.validate_chunks([corrupted_chunk])
    assert not is_corrupt_valid
    assert any("non-deterministic" in err for err in corrupt_errors)

    # Test empty text detection
    empty_chunk = chunks[0].model_copy(update={"text": "   "})
    _, empty_errors = validator.validate_chunks([empty_chunk])
    assert any("empty text" in err for err in empty_errors)
