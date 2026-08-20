"""Hierarchical and metadata-aware chunking strategy."""

from typing import List, Optional

from app.chunking.base import ChunkingStrategy
from app.chunking.models import Chunk
from app.chunking.sentence import split_into_sentences_with_offsets
from app.config import get_settings
from app.ingestion.models import Document


class HierarchicalChunkingStrategy(ChunkingStrategy):
    """Hierarchical and metadata-aware chunking strategy.
    
    Preserves document-level hierarchy and attaches contextual metadata (such as associated query,
    relevance flag, language, and passage rank) directly to child chunks without inventing artificial sections.
    """

    strategy_name: str = "hierarchical"

    def __init__(self, max_chunk_size: Optional[int] = None):
        settings = get_settings()
        self.max_chunk_size = max_chunk_size if max_chunk_size is not None else settings.SENTENCE_MAX_CHUNK_SIZE

    def chunk(self, document: Document) -> List[Chunk]:
        """Produce hierarchical chunks enriched with document-level and query-level metadata."""
        text = document.text
        if not text:
            return []

        sentences = split_into_sentences_with_offsets(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_start: Optional[int] = None
        current_end: Optional[int] = None
        current_len = 0
        chunk_index = 0

        # Extract hierarchical context from metadata
        meta = document.metadata or {}
        hierarchy_context = {
            "query_id": meta.get("query_id"),
            "query_type": meta.get("query_type"),
            "is_selected": meta.get("is_selected", 0),
            "passage_index": meta.get("passage_index", 0),
            "source_lang": meta.get("source_lang"),
            "target_lang": meta.get("target_lang"),
            "strategy": "hierarchical",
            "hierarchy_level": "passage_leaf",
        }

        for sent_text, sent_start, sent_end in sentences:
            sent_len = len(sent_text)

            if current_sentences and (current_len + sent_len + 1 > self.max_chunk_size):
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    self.create_chunk(
                        document=document,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        start_position=current_start if current_start is not None else 0,
                        end_position=current_end if current_end is not None else len(chunk_text),
                        extra_metadata=dict(hierarchy_context),
                    )
                )
                chunk_index += 1
                current_sentences = []
                current_start = None
                current_end = None
                current_len = 0

            if current_start is None:
                current_start = sent_start
            current_end = sent_end
            current_sentences.append(sent_text)
            current_len += sent_len + 1

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                self.create_chunk(
                    document=document,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_position=current_start if current_start is not None else 0,
                    end_position=current_end if current_end is not None else len(text),
                    extra_metadata=dict(hierarchy_context),
                )
            )

        return chunks
