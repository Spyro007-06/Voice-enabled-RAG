"""Fixed-size chunking strategy with configurable size and overlap."""

from typing import List, Optional

from app.chunking.base import ChunkingStrategy
from app.chunking.models import Chunk
from app.config import get_settings
from app.ingestion.models import Document


class FixedChunkingStrategy(ChunkingStrategy):
    """Chunks documents into fixed character-length segments with configurable overlap.
    
    Unit: Characters (with word-boundary snapping when feasible).
    - chunk_size: Maximum number of characters per chunk (default from FIXED_CHUNK_SIZE).
    - chunk_overlap: Number of overlapping characters between adjacent chunks (default from FIXED_CHUNK_OVERLAP).
    """

    strategy_name: str = "fixed"

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size if chunk_size is not None else settings.FIXED_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.FIXED_CHUNK_OVERLAP
        
        # Enforce valid overlap constraint
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = max(0, self.chunk_size // 4)

    def chunk(self, document: Document) -> List[Chunk]:
        """Split document into fixed-size character chunks with overlap."""
        text = document.text
        if not text:
            return []

        doc_len = len(text)
        if doc_len <= self.chunk_size:
            return [
                self.create_chunk(
                    document=document,
                    text=text,
                    chunk_index=0,
                    start_position=0,
                    end_position=doc_len,
                    extra_metadata={"unit": "character", "strategy": "fixed"},
                )
            ]

        chunks: List[Chunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        start = 0
        chunk_index = 0

        while start < doc_len:
            end = min(doc_len, start + self.chunk_size)
            
            # Snap to word boundary if not at the very end of document
            if end < doc_len:
                space_idx = text.rfind(" ", start + (self.chunk_size // 2), end)
                if space_idx > start:
                    end = space_idx

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = self.create_chunk(
                    document=document,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_position=start,
                    end_position=end,
                    extra_metadata={"unit": "character", "strategy": "fixed"},
                )
                chunks.append(chunk)
                chunk_index += 1

            if end >= doc_len:
                break

            start += step
            # Prevent infinite loop if step didn't advance past start
            if start <= chunks[-1].start_position and len(chunks) > 0:
                start = end

        return chunks
