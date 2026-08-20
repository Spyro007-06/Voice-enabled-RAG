"""Sentence-aware chunking strategy respecting linguistic sentence boundaries."""

import re
from typing import List, Optional, Tuple

from app.chunking.base import ChunkingStrategy
from app.chunking.models import Chunk
from app.config import get_settings
from app.ingestion.models import Document

# Sentence delimiter regex supporting English and Indic sentence terminators:
# . ! ? newline Indic Danda (।) Double Danda (॥) Arabic Question Mark (؟)
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?।॥؟\n])\s+")


def split_into_sentences_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Split text into sentences while tracking exact start and end character positions."""
    if not text:
        return []

    sentences_with_offsets: List[Tuple[str, int, int]] = []
    matches = list(SENTENCE_SPLIT_REGEX.finditer(text))

    if not matches:
        return [(text.strip(), 0, len(text))]

    last_end = 0
    for match in matches:
        split_start = match.start()
        split_end = match.end()
        raw_sentence = text[last_end:split_start]
        stripped = raw_sentence.strip()
        if stripped:
            sentence_start = last_end + raw_sentence.find(stripped)
            sentence_end = sentence_start + len(stripped)
            sentences_with_offsets.append((stripped, sentence_start, sentence_end))
        last_end = split_end

    if last_end < len(text):
        raw_sentence = text[last_end:]
        stripped = raw_sentence.strip()
        if stripped:
            sentence_start = last_end + raw_sentence.find(stripped)
            sentence_end = sentence_start + len(stripped)
            sentences_with_offsets.append((stripped, sentence_start, sentence_end))

    return sentences_with_offsets


class SentenceChunkingStrategy(ChunkingStrategy):
    """Chunks documents along linguistic sentence boundaries up to a maximum character size."""

    strategy_name: str = "sentence"

    def __init__(self, max_chunk_size: Optional[int] = None):
        settings = get_settings()
        self.max_chunk_size = max_chunk_size if max_chunk_size is not None else settings.SENTENCE_MAX_CHUNK_SIZE

    def chunk(self, document: Document) -> List[Chunk]:
        """Group full sentences into chunks without exceeding max_chunk_size where possible."""
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
        current_length = 0
        chunk_index = 0

        for sent_text, sent_start, sent_end in sentences:
            sent_len = len(sent_text)

            # If adding this sentence exceeds max_chunk_size and we already have accumulated sentences
            if current_sentences and (current_length + sent_len + 1 > self.max_chunk_size):
                combined_text = " ".join(current_sentences)
                chunks.append(
                    self.create_chunk(
                        document=document,
                        text=combined_text,
                        chunk_index=chunk_index,
                        start_position=current_start if current_start is not None else 0,
                        end_position=current_end if current_end is not None else len(combined_text),
                        extra_metadata={"sentence_count": len(current_sentences), "strategy": "sentence"},
                    )
                )
                chunk_index += 1
                current_sentences = []
                current_start = None
                current_end = None
                current_length = 0

            # If a single sentence by itself exceeds max_chunk_size, add it as its own chunk
            if sent_len > self.max_chunk_size and not current_sentences:
                chunks.append(
                    self.create_chunk(
                        document=document,
                        text=sent_text,
                        chunk_index=chunk_index,
                        start_position=sent_start,
                        end_position=sent_end,
                        extra_metadata={"sentence_count": 1, "oversized": True, "strategy": "sentence"},
                    )
                )
                chunk_index += 1
                continue

            if current_start is None:
                current_start = sent_start
            current_end = sent_end
            current_sentences.append(sent_text)
            current_length += sent_len + 1

        # Flush remaining accumulated sentences
        if current_sentences:
            combined_text = " ".join(current_sentences)
            chunks.append(
                self.create_chunk(
                    document=document,
                    text=combined_text,
                    chunk_index=chunk_index,
                    start_position=current_start if current_start is not None else 0,
                    end_position=current_end if current_end is not None else len(text),
                    extra_metadata={"sentence_count": len(current_sentences), "strategy": "sentence"},
                )
            )

        return chunks
