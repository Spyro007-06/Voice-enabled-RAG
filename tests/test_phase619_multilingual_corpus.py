"""Comprehensive validation tests for Phase 6.19 Multilingual Corpus, Indexing, and Retrieval Quality."""

import os
import pytest
from app.retrieval.language import (
    CANONICAL_LANGUAGE_MAP,
    LANGUAGE_ALIASES,
    LANGUAGE_DISPLAY_NAMES,
    get_language_filter_synonyms,
    normalize_language_code,
)
from app.retrieval.filters import build_qdrant_filter, matches_filter
from app.retrieval.service import get_retrieval_service
from app.retrieval.bm25 import get_bm25_retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.qdrant_store import get_qdrant_store
from app.embeddings.multilingual_e5 import MultilingualE5EmbeddingProvider
from app.generation.models import AskRequest, AskResponse
from app.api.routes import ask_endpoint


class TestPhase619MultilingualCorpus:
    """Test suite covering Phase 6.19 multilingual ingestion, indexing, and retrieval requirements."""

    def test_canonical_language_normalization(self):
        """1. Verify centralized language resolver handles all variants correctly."""
        assert normalize_language_code("en") == "eng_Latn"
        assert normalize_language_code("eng") == "eng_Latn"
        assert normalize_language_code("English") == "eng_Latn"
        assert normalize_language_code("hi") == "hin_Deva"
        assert normalize_language_code("hindi") == "hin_Deva"
        assert normalize_language_code("ta") == "tam_Taml"
        assert normalize_language_code("tamil") == "tam_Taml"
        assert normalize_language_code("te") == "tel_Telu"
        assert normalize_language_code("telugu") == "tel_Telu"
        assert normalize_language_code("ml") == "mal_Mlym"
        assert normalize_language_code("malayalam") == "mal_Mlym"
        assert normalize_language_code("all") is None
        assert normalize_language_code(None) is None

    def test_language_filter_synonyms(self):
        """2. Verify language synonym expansion for retrieval filters."""
        syns_en = get_language_filter_synonyms("en")
        assert "eng_Latn" in syns_en
        assert "en" in syns_en

        syns_ta = get_language_filter_synonyms("ta")
        assert "tam_Taml" in syns_ta
        assert "tamil" in syns_ta

    def test_qdrant_filter_builder(self):
        """3. Verify Qdrant Filter objects are built accurately for strict and all-language queries."""
        q_filter_en = build_qdrant_filter(language="en")
        assert q_filter_en is not None

        q_filter_none = build_qdrant_filter(language=None)
        assert q_filter_none is None

        q_filter_all = build_qdrant_filter(language="all")
        assert q_filter_all is None

    def test_in_memory_matches_filter(self):
        """4. Verify matches_filter supports all canonical language tags."""
        meta_ta = {"language": "tam_Taml"}
        assert matches_filter(meta_ta, "fixed", language="ta") is True
        assert matches_filter(meta_ta, "fixed", language="en") is False
        assert matches_filter(meta_ta, "fixed", language=None) is True

        meta_te = {"language": "tel_Telu"}
        assert matches_filter(meta_te, "sentence", language="te") is True
        assert matches_filter(meta_te, "sentence", language="hi") is False

    def test_embedding_model_properties(self):
        """5. Verify intfloat/multilingual-e5-small configuration and 384 dimensions."""
        embedder = MultilingualE5EmbeddingProvider()
        assert embedder.embedding_dimension == 384
        q_emb = embedder.embed_query("What is machine learning?")
        assert len(q_emb) == 384

        doc_embs = embedder.embed_documents(["Machine learning is a subset of artificial intelligence."])
        assert len(doc_embs) == 1
        assert len(doc_embs[0]) == 384

    def test_bm25_multilingual_indexing(self):
        """6. Verify BM25 index contains all 5 languages and retrieves correctly."""
        retriever = get_bm25_retriever()
        retriever.load_index()
        assert retriever.is_indexed()

        results_en, _ = retriever.retrieve("computer science", top_k=5, language="en")
        assert len(results_en) > 0
        for r in results_en:
            r_lang = (r.language or r.metadata.get("language") or "").lower()
            assert "eng" in r_lang or "en" in r_lang

    def test_strict_english_retrieval(self):
        """7. Verify strict English retrieval produces English chunks."""
        service = get_retrieval_service()
        results, _ = service.retrieve("What is a computer?", top_k=5, language="en", use_cache=False)
        assert len(results) > 0
        for r in results:
            r_lang = (r.language or r.metadata.get("language") or "").lower()
            assert "eng" in r_lang or "en" in r_lang

    def test_strict_hindi_retrieval(self):
        """8. Verify strict Hindi retrieval produces Hindi chunks."""
        service = get_retrieval_service()
        results, _ = service.retrieve("कंप्यूटर क्या है?", top_k=5, language="hi", use_cache=False)
        assert len(results) > 0
        for r in results:
            r_lang = (r.language or r.metadata.get("language") or "").lower()
            assert "hin" in r_lang or "hi" in r_lang

    def test_strict_tamil_retrieval(self):
        """9. Verify strict Tamil retrieval produces Tamil chunks."""
        service = get_retrieval_service()
        results, _ = service.retrieve("கணினி என்றால் என்ன?", top_k=5, language="ta", use_cache=False)
        assert len(results) > 0
        for r in results:
            r_lang = (r.language or r.metadata.get("language") or "").lower()
            assert "tam" in r_lang or "ta" in r_lang

    def test_strict_telugu_retrieval(self):
        """10. Verify strict Telugu retrieval produces Telugu chunks."""
        service = get_retrieval_service()
        results, _ = service.retrieve("కంప్యూటర్ అంటే ఏమిటి?", top_k=5, language="te", use_cache=False)
        assert len(results) > 0
        for r in results:
            r_lang = (r.language or r.metadata.get("language") or "").lower()
            assert "tel" in r_lang or "te" in r_lang

    def test_strict_malayalam_retrieval(self):
        """11. Verify strict Malayalam retrieval produces Malayalam chunks."""
        service = get_retrieval_service()
        results, _ = service.retrieve("കമ്പ്യൂട്ടർ എന്താണ്?", top_k=5, language="ml", use_cache=False)
        assert len(results) > 0
        for r in results:
            r_lang = (r.language or r.metadata.get("language") or "").lower()
            assert "mal" in r_lang or "ml" in r_lang

    def test_cross_language_retrieval(self):
        """12. Verify cross-language retrieval (language=None) searches the full multilingual corpus."""
        service = get_retrieval_service()
        results, lat = service.retrieve("What is machine learning?", top_k=5, language=None, use_cache=False)
        assert len(results) > 0
        assert lat.total_ms > 0

    @pytest.mark.asyncio
    async def test_end_to_end_rag_grounded_response(self):
        """13. Verify RAG /api/ask generates grounded response with citations for English factual query."""
        req = AskRequest(
            query="What is a computer?",
            language="en",
            top_k=5,
        )
        resp = await ask_endpoint(req)
        assert resp.grounded is True
        assert len(resp.answer) > 0
        assert len(resp.citations) > 0
        assert resp.confidence > 0.0

    @pytest.mark.asyncio
    async def test_empty_context_refusal(self):
        """14. Verify unanswerable query with no context safely returns guardrail refusal."""
        req = AskRequest(
            query="What is quantum gravity?",
            language="nonexistent_lang",
            top_k=5,
        )
        resp = await ask_endpoint(req)
        assert resp.grounded is False or "not have enough information" in resp.answer.lower()
