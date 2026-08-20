"""Chunk data models and deterministic ID generation."""

import hashlib
from typing import Any, Dict
from pydantic import BaseModel, Field


def generate_chunk_id(document_id: str, chunk_type: str, chunk_index: int) -> str:
    """Generate a deterministic chunk ID based on document_id, strategy, and index."""
    # Clean string prefix plus deterministic sha256 digest suffix for collision resistance
    raw_key = f"{document_id}:{chunk_type}:{chunk_index}"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:8]
    return f"{document_id}_{chunk_type}_{chunk_index:04d}_{digest}"


class Chunk(BaseModel):
    """Represents an atomic text chunk generated from a Document."""

    chunk_id: str = Field(
        ...,
        description="Deterministic unique ID for this chunk.",
    )
    document_id: str = Field(
        ...,
        description="ID of the parent Document from which this chunk was derived.",
    )
    text: str = Field(
        ...,
        description="Extracted chunk text content.",
    )
    chunk_type: str = Field(
        ...,
        description="Chunking strategy type (e.g. 'fixed', 'sentence', 'sliding_window', 'semantic', 'hierarchical').",
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Zero-based index of this chunk within the parent document.",
    )
    start_position: int = Field(
        ...,
        ge=0,
        description="Character start position in the parent document text.",
    )
    end_position: int = Field(
        ...,
        ge=0,
        description="Character end position in the parent document text.",
    )
    token_count: int = Field(
        ...,
        ge=0,
        description="Estimated token count of the chunk text.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Preserved document and chunk-specific metadata.",
    )
