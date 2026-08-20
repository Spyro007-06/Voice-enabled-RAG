"""Unit and integration tests for Phase 3 Multilingual Embeddings and Qdrant Vector Store."""

import pytest
from app.chunking.models import Chunk
from app.embeddings.multilingual_e5 import MultilingualE5EmbeddingProvider
from app.retrieval.qdrant_store import QdrantVectorStore, chunk_id_to_uuid


@pytest.fixture(scope="module")
def embedding_provider():
    """Module-level singleton embedding provider for fast test execution."""
    return MultilingualE5EmbeddingProvider(device="cpu", batch_size=16)


@pytest.fixture
def in_memory_qdrant():
    """In-memory Qdrant instance for isolated testing."""
    return QdrantVectorStore(in_memory=True)


@pytest.fixture
def sample_test_chunks():
    """Synthetic chunks with varied metadata and languages for testing."""
    return [
        Chunk(
            chunk_id="msmarco_101_fixed_0000_a1b2c3d4",
            document_id="msmarco_101_0",
            text="नई दिल्ली भारत की राजधानी है।",
            chunk_type="fixed",
            chunk_index=0,
            start_position=0,
            end_position=30,
            token_count=10,
            metadata={"language": "hin_Deva", "query_id": 101, "is_selected": 1},
        ),
        Chunk(
            chunk_id="msmarco_101_sentence_0000_e5f6g7h8",
            document_id="msmarco_101_0",
            text="New Delhi is the capital of India.",
            chunk_type="sentence",
            chunk_index=0,
            start_position=0,
            end_position=34,
            token_count=7,
            metadata={"language": "eng_Latn", "query_id": 101, "is_selected": 1},
        ),
    ]


# --- Embedding Provider Tests ---

def test_embedding_provider_initialization(embedding_provider):
    """Verify provider initializes with expected model and dimension."""
    assert embedding_provider.model_name == "intfloat/multilingual-e5-small"
    assert embedding_provider.embedding_dimension == 384
    assert embedding_provider.device in ("cpu", "cuda")


def test_query_and_document_embedding(embedding_provider):
    """Verify query and document embeddings produce normalized 384-d vectors."""
    query_vec = embedding_provider.embed_query("भारत की राजधानी क्या है?")
    assert len(query_vec) == 384
    assert isinstance(query_vec[0], float)

    docs = [
        "नई दिल्ली भारत की राजधानी है।",
        "The Supreme Court of India is located in New Delhi.",
    ]
    doc_vecs = embedding_provider.embed_documents(docs)
    assert len(doc_vecs) == 2
    assert len(doc_vecs[0]) == 384
    assert len(doc_vecs[1]) == 384


def test_multilingual_embedding_support(embedding_provider):
    """Validate query and passage embedding across representative Indic languages."""
    multilingual_samples = {
        "English": "What is the economic capital of India?",
        "Hindi": "भारत की वित्तीय राजधानी क्या है?",
        "Tamil": "இந்தியாவின் பொருளாதார தலைநகரம் எது?",
        "Telugu": "భారతదేశ ఆర్థిక రాజధాని ఏది?",
        "Malayalam": "ഇന്ത്യയുടെ സാമ്പത്തിക തലസ്ഥാനം ഏതാണ്?",
    }

    for lang_name, text in multilingual_samples.items():
        q_vec = embedding_provider.embed_query(text)
        assert len(q_vec) == 384, f"Failed for {lang_name} query"

        p_vecs = embedding_provider.embed_documents([text])
        assert len(p_vecs) == 1 and len(p_vecs[0]) == 384, f"Failed for {lang_name} passage"


# --- Deterministic Point ID & Qdrant Tests ---

def test_deterministic_chunk_to_uuid():
    """Verify chunk ID to UUID mapping is 100% deterministic."""
    uuid1 = chunk_id_to_uuid("msmarco_101_fixed_0000_a1b2c3d4")
    uuid2 = chunk_id_to_uuid("msmarco_101_fixed_0000_a1b2c3d4")
    uuid3 = chunk_id_to_uuid("msmarco_102_fixed_0000_a1b2c3d4")

    assert uuid1 == uuid2
    assert uuid1 != uuid3


def test_qdrant_collection_lifecycle_and_upsert(in_memory_qdrant, embedding_provider, sample_test_chunks):
    """Test Qdrant collection creation, batch upsert, duplicate idempotency, and search."""
    col_name = "test_collection"
    dim = embedding_provider.embedding_dimension

    # 1. Create collection
    created = in_memory_qdrant.create_collection_if_not_exists(collection_name=col_name, vector_size=dim)
    assert created is True
    assert in_memory_qdrant.collection_exists(col_name) is True

    # 2. Embed and Upsert
    texts = [c.text for c in sample_test_chunks]
    vectors = embedding_provider.embed_documents(texts)
    upserted = in_memory_qdrant.upsert_chunks(
        collection_name=col_name,
        chunks=sample_test_chunks,
        vectors=vectors,
        batch_size=2,
    )
    assert upserted == 2

    # 3. Verify Collection Stats
    stats = in_memory_qdrant.get_collection_stats(col_name)
    assert stats["exists"] is True
    assert stats["points_count"] == 2

    # 4. Duplicate Upsert Idempotency Test
    # Upserting the exact same chunks again should update points in-place, NOT increase point count
    upserted_again = in_memory_qdrant.upsert_chunks(
        collection_name=col_name,
        chunks=sample_test_chunks,
        vectors=vectors,
    )
    assert upserted_again == 2
    stats_after = in_memory_qdrant.get_collection_stats(col_name)
    assert stats_after["points_count"] == 2  # No duplicate points created!

    # 5. Search Capability Test
    query_vec = embedding_provider.embed_query("New Delhi capital")
    results = in_memory_qdrant.search(collection_name=col_name, query_vector=query_vec, limit=2)
    assert len(results) == 2
    assert results[0].payload["chunk_id"] in [c.chunk_id for c in sample_test_chunks]
    assert "query_id" in results[0].payload
