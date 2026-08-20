"""Base interface and common utilities for chunking strategies."""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.chunking.models import Chunk, generate_chunk_id
from app.ingestion.models import Document


class ChunkingStrategy(ABC):
    """Abstract base class defining the common interface for all chunking strategies."""

    strategy_name: str = "base"

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Split a Document into a list of Chunk objects."""
        pass

    def estimate_token_count(self, text: str) -> int:
        """Estimate token count for a text string.
        
        Uses a robust whitespace/punctuation token heuristic suitable for multi-lingual
        and Indic text (approx ~1.3 tokens per word or whitespace segment).
        """
        if not text:
            return 0
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return max(1, len(tokens))

    def create_chunk(
        self,
        document: Document,
        text: str,
        chunk_index: int,
        start_position: int,
        end_position: int,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Chunk:
        """Helper method to construct a valid Chunk with deterministic ID and preserved metadata."""
        chunk_id = generate_chunk_id(
            document_id=document.document_id,
            chunk_type=self.strategy_name,
            chunk_index=chunk_index,
        )
        
        # Merge document metadata with chunk-level metadata
        merged_metadata = dict(document.metadata)
        if document.language:
            merged_metadata["language"] = document.language
        if document.source:
            merged_metadata["source"] = document.source
        if extra_metadata:
            merged_metadata.update(extra_metadata)

        return Chunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            text=text,
            chunk_type=self.strategy_name,
            chunk_index=chunk_index,
            start_position=start_position,
            end_position=end_position,
            token_count=self.estimate_token_count(text),
            metadata=merged_metadata,
        )
