"""Multilingual BM25 lexical retrieval engine with offline index persistence."""

import logging
import os
import pickle
import re
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

from rank_bm25 import BM25Plus

from app.chunking.models import Chunk
from app.config import get_settings
from app.retrieval.filters import matches_filter
from app.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)


class MultilingualTokenizer(ABC):
    """Abstract interface for multilingual tokenizers."""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Convert text into tokens for indexing and lexical search."""
        pass


class IndicUnicodeTokenizer(MultilingualTokenizer):
    """Robust Unicode tokenizer supporting English, Latin, and Indic scripts.
    
    Generates word tokens and sub-word character tri-grams to handle morphological
    variations in agglutinative Indic languages.
    """

    def tokenize(self, text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []
        
        cleaned = text.lower().strip()
        # Split by whitespace and strip punctuation marks without breaking Indic combining marks
        raw_words = cleaned.split()
        words = [w.strip(".,!?:;\"'()[]{}—–-।|/\\`~@#$%^&*+=<>") for w in raw_words]
        words = [w for w in words if w]
        
        tokens = list(words)
        
        # Add character tri-grams for sub-word matching in Indic and long words
        for word in words:
            if len(word) >= 3:
                tokens.extend([word[i : i + 3] for i in range(len(word) - 2)])
        
        return tokens


class BM25Retriever:
    """In-memory BM25 lexical retriever initialized from persisted offline index."""

    def __init__(
        self,
        index_path: Optional[str] = None,
        tokenizer: Optional[MultilingualTokenizer] = None,
    ):
        settings = get_settings()
        self.index_path = index_path or settings.BM25_INDEX_PATH
        self.tokenizer = tokenizer or IndicUnicodeTokenizer()

        self.bm25_model: Optional[BM25Plus] = None
        self.corpus_chunks: List[Dict[str, Any]] = []
        self._is_loaded = False

    def is_indexed(self) -> bool:
        """Check if BM25 model is loaded in memory or on disk."""
        return self._is_loaded or (self.index_path and os.path.exists(self.index_path))

    def build_index_from_chunks(self, chunks: List[Chunk]) -> int:
        """Build in-memory BM25 model from a list of Chunk models."""
        if not chunks:
            logger.warning("No chunks provided to build BM25 index.")
            return 0

        logger.info("Building BM25 index across %d chunks...", len(chunks))
        self.corpus_chunks = []
        tokenized_corpus: List[List[str]] = []

        for chunk in chunks:
            tokens = self.tokenizer.tokenize(chunk.text)
            tokenized_corpus.append(tokens)
            self.corpus_chunks.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "chunk_type": chunk.chunk_type,
                "chunk_index": chunk.chunk_index,
                "start_position": chunk.start_position,
                "end_position": chunk.end_position,
                "token_count": chunk.token_count,
                "language": chunk.metadata.get("language"),
                "source": chunk.metadata.get("source"),
                "query_id": chunk.metadata.get("query_id"),
                "passage_index": chunk.metadata.get("passage_index"),
                "is_selected": chunk.metadata.get("is_selected", 0),
                "metadata": chunk.metadata,
            })

        self.bm25_model = BM25Plus(tokenized_corpus)
        self._is_loaded = True
        logger.info("BM25 index built successfully with %d documents.", len(self.corpus_chunks))
        return len(self.corpus_chunks)

    def save_index(self, path: Optional[str] = None) -> str:
        """Persist BM25 index and corpus metadata to disk."""
        save_path = path or self.index_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        payload = {
            "bm25_model": self.bm25_model,
            "corpus_chunks": self.corpus_chunks,
        }
        with open(save_path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Saved BM25 index to %s (%d records)", save_path, len(self.corpus_chunks))
        return save_path

    def load_index(self, path: Optional[str] = None) -> bool:
        """Load persisted BM25 index and metadata into memory."""
        load_path = path or self.index_path
        if not os.path.exists(load_path):
            logger.warning("BM25 index file not found at %s", load_path)
            return False

        t0 = time.perf_counter()
        with open(load_path, "rb") as f:
            payload = pickle.load(f)
            self.bm25_model = payload.get("bm25_model")
            self.corpus_chunks = payload.get("corpus_chunks", [])
            self._is_loaded = True

        logger.info(
            "Loaded BM25 index from %s (%d records) in %.2f ms",
            load_path,
            len(self.corpus_chunks),
            (time.perf_counter() - t0) * 1000,
        )
        return True

    def ensure_loaded(self) -> None:
        """Ensure index is loaded before performing search."""
        if not self._is_loaded:
            if not self.load_index():
                raise RuntimeError(
                    f"BM25 index is not initialized or found at {self.index_path}. "
                    "Please run `python scripts/build_bm25.py` first."
                )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        strategies: Optional[Union[str, List[str]]] = None,
        language: Optional[str] = None,
    ) -> Tuple[List[RetrievalResult], float]:
        """Execute BM25 lexical search and return results with latency (ms)."""
        self.ensure_loaded()
        settings = get_settings()
        k = top_k if top_k is not None else settings.BM25_TOP_K

        t_search_start = time.perf_counter()
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens or self.bm25_model is None:
            return [], (time.perf_counter() - t_search_start) * 1000

        # Calculate scores for all corpus documents
        doc_scores = self.bm25_model.get_scores(query_tokens)
        if not hasattr(doc_scores, "__len__") or len(doc_scores) == 0:
            return [], (time.perf_counter() - t_search_start) * 1000

        import numpy as np
        scores_arr = np.asarray(doc_scores, dtype=np.float32)
        pos_indices = np.where(scores_arr > 0.0)[0]

        if len(pos_indices) == 0:
            return [], (time.perf_counter() - t_search_start) * 1000

        # Sort positive scores descending
        pos_scores = scores_arr[pos_indices]
        candidate_count = min(len(pos_indices), max(k * 15, 150))
        if candidate_count < len(pos_indices):
            top_pos_idx = np.argpartition(-pos_scores, candidate_count - 1)[:candidate_count]
            sorted_top = top_pos_idx[np.argsort(-pos_scores[top_pos_idx])]
            sorted_doc_indices = pos_indices[sorted_top]
        else:
            sorted_top = np.argsort(-pos_scores)
            sorted_doc_indices = pos_indices[sorted_top]

        # Filter candidates by strategy/language
        scored_candidates: List[Tuple[int, float]] = []
        for idx in sorted_doc_indices:
            item = self.corpus_chunks[idx]
            if matches_filter(item["metadata"], item["chunk_type"], strategies=strategies, language=language):
                scored_candidates.append((int(idx), float(scores_arr[idx])))
                if len(scored_candidates) >= k:
                    break

        # Fallback if filters were tight and we need more items from the remaining positives
        if len(scored_candidates) < k and len(sorted_doc_indices) < len(pos_indices):
            remaining_indices = set(pos_indices) - set(sorted_doc_indices)
            rem_arr = np.array(list(remaining_indices), dtype=np.int64)
            rem_sorted = rem_arr[np.argsort(-scores_arr[rem_arr])]
            for idx in rem_sorted:
                item = self.corpus_chunks[idx]
                if matches_filter(item["metadata"], item["chunk_type"], strategies=strategies, language=language):
                    scored_candidates.append((int(idx), float(scores_arr[idx])))
                    if len(scored_candidates) >= k:
                        break

        top_candidates = scored_candidates[:k]
        t_search_ms = (time.perf_counter() - t_search_start) * 1000

        # Build RetrievalResult objects
        results: List[RetrievalResult] = []
        for rank_idx, (doc_idx, raw_score) in enumerate(top_candidates, start=1):
            item = self.corpus_chunks[doc_idx]
            result = RetrievalResult(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                text=item["text"],
                chunk_type=item["chunk_type"],
                language=item["language"],
                dense_score=None,
                bm25_score=raw_score,
                normalized_dense_score=None,
                normalized_bm25_score=raw_score,
                fusion_score=raw_score,
                rank=rank_idx,
                metadata=item["metadata"],
            )
            results.append(result)

        return results, t_search_ms


@lru_cache()
def get_bm25_retriever() -> BM25Retriever:
    """Return singleton cached BM25Retriever instance with pre-loaded index."""
    retriever = BM25Retriever()
    if retriever.is_indexed():
        try:
            retriever.ensure_loaded()
        except Exception as exc:
            logger.warning("BM25 pre-warm load notice: %s", exc)
    return retriever
