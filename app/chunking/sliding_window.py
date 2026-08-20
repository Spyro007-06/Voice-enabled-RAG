"""Sliding-window chunking strategy based on overlapping sentence windows."""

from typing import List, Optional

from app.chunking.base import ChunkingStrategy
from app.chunking.models import Chunk
from app.chunking.sentence import split_into_sentences_with_offsets
from app.config import get_settings
from app.ingestion.models import Document


class SlidingWindowChunkingStrategy(ChunkingStrategy):
    """Chunks documents using overlapping sentence-level sliding windows.
    
    - window_size: Number of sentences in each window (default from SLIDING_WINDOW_SIZE).
    - window_overlap: Number of sentences shared with subsequent window (default from SLIDING_WINDOW_OVERLAP).
    """

    strategy_name: str = "sliding_window"

    def __init__(
        self,
        window_size: Optional[int] = None,
        window_overlap: Optional[int] = None,
    ):
        settings = get_settings()
        self.window_size = window_size if window_size is not None else settings.SLIDING_WINDOW_SIZE
        self.window_overlap = window_overlap if window_overlap is not None else settings.SLIDING_WINDOW_OVERLAP

        # Enforce valid window overlap
        if self.window_overlap >= self.window_size:
            self.window_overlap = max(0, self.window_size - 1)

    def chunk(self, document: Document) -> List[Chunk]:
        """Generate chunks using overlapping sentence windows."""
        text = document.text
        if not text:
            return []

        sentences = split_into_sentences_with_offsets(text)
        if not sentences:
            return []

        num_sentences = len(sentences)
        # If total sentences is within single window size, emit 1 chunk without duplication
        if num_sentences <= self.window_size:
            return [
                self.create_chunk(
                    document=document,
                    text=text,
                    chunk_index=0,
                    start_position=sentences[0][1],
                    end_position=sentences[-1][2],
                    extra_metadata={
                        "window_size": num_sentences,
                        "sentence_start_idx": 0,
                        "sentence_end_idx": num_sentences - 1,
                        "strategy": "sliding_window",
                    },
                )
            ]

        chunks: List[Chunk] = []
        step = max(1, self.window_size - self.window_overlap)
        start_idx = 0
        chunk_index = 0

        while start_idx < num_sentences:
            end_idx = min(num_sentences, start_idx + self.window_size)
            window_slice = sentences[start_idx:end_idx]

            window_text = " ".join(s[0] for s in window_slice)
            window_start_pos = window_slice[0][1]
            window_end_pos = window_slice[-1][2]

            chunks.append(
                self.create_chunk(
                    document=document,
                    text=window_text,
                    chunk_index=chunk_index,
                    start_position=window_start_pos,
                    end_position=window_end_pos,
                    extra_metadata={
                        "window_size": len(window_slice),
                        "sentence_start_idx": start_idx,
                        "sentence_end_idx": end_idx - 1,
                        "strategy": "sliding_window",
                    },
                )
            )
            chunk_index += 1

            if end_idx >= num_sentences:
                break

            start_idx += step

        return chunks
