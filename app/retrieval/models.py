"""Data models for retrieval requests, responses, scoring, and latency breakdown."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """Normalized retrieval result model returned from dense, BM25, or hybrid search."""

    chunk_id: str = Field(..., description="Unique deterministic identifier of the chunk.")
    document_id: str = Field(..., description="Parent document identifier.")
    text: str = Field(..., description="Chunk text content.")
    chunk_type: str = Field(..., description="Chunking strategy used (e.g. 'fixed', 'sentence', 'semantic').")
    language: Optional[str] = Field(default=None, description="Language code if available.")
    dense_score: Optional[float] = Field(default=None, description="Raw dense vector similarity score.")
    bm25_score: Optional[float] = Field(default=None, description="Raw BM25 lexical score.")
    normalized_dense_score: Optional[float] = Field(default=None, description="Min-max normalized dense score.")
    normalized_bm25_score: Optional[float] = Field(default=None, description="Min-max normalized BM25 score.")
    fusion_score: Optional[float] = Field(default=0.0, description="Final combined score after hybrid rank or score fusion.")
    rank: Optional[int] = Field(default=1, ge=1, description="1-based rank position in final retrieved list.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Preserved provenance and dataset metadata.")

    @property
    def score(self) -> float:
        """Compatibility property returning fusion_score or highest available component score."""
        return self.fusion_score or self.dense_score or self.bm25_score or 0.0


class LatencyBreakdown(BaseModel):
    """Detailed millisecond execution time breakdown per retrieval stage."""

    embedding_ms: float = Field(default=0.0, description="Time taken to embed search query.")
    dense_retrieval_ms: float = Field(default=0.0, description="Time taken for Qdrant vector search.")
    bm25_retrieval_ms: float = Field(default=0.0, description="Time taken for BM25 lexical scoring.")
    fusion_ms: float = Field(default=0.0, description="Time taken for score normalization, fusion & deduplication.")
    total_ms: float = Field(default=0.0, description="Total end-to-end retrieval latency in milliseconds.")
    timeout_stage: Optional[str] = Field(default=None, description="Stage that timed out if any.")
    fallback_used: bool = Field(default=False, description="Whether fallback retrieval was invoked due to timeout or error.")
    parallel_execution: bool = Field(default=False, description="Whether dense and BM25 ran in parallel.")
    remaining_budget_ms: Optional[float] = Field(default=None, description="Remaining latency budget in milliseconds.")


class RetrievalRequest(BaseModel):
    """Incoming request model for retrieval operations."""

    query: str = Field(..., min_length=1, description="Search query string.")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of top results to return.")
    language: Optional[str] = Field(default=None, description="Optional language filter (e.g. 'hin_Deva', 'ta', 'all').")
    strategies: Optional[List[str]] = Field(default=None, description="Optional chunk strategy filter (e.g. ['semantic']).")
    fusion_method: str = Field(
        default="rrf",
        description="Rank fusion algorithm: 'rrf' (Reciprocal Rank Fusion) or 'weighted' (Score Fusion).",
    )


class RetrievalResponse(BaseModel):
    """Response model containing retrieved chunks, metadata, and performance metrics."""

    query: str = Field(..., description="Original search query string.")
    fusion_method: str = Field(..., description="Fusion method applied.")
    total_results: int = Field(..., description="Number of returned results.")
    results: List[RetrievalResult] = Field(default_factory=list, description="Ranked list of retrieved chunks.")
    latency_ms: LatencyBreakdown = Field(..., description="Detailed millisecond latency metrics.")
