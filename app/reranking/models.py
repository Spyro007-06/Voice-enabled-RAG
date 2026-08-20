"""Data models for reranking requests, responses, candidate scoring, context statistics, and adaptive retrieval."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RerankResult(BaseModel):
    """Rich reranked candidate model preserving original retrieval scores and metadata."""

    chunk_id: str = Field(..., description="Unique deterministic identifier of the chunk.")
    document_id: str = Field(..., description="Parent document identifier.")
    text: str = Field(..., description="Chunk text content.")
    chunk_type: str = Field(..., description="Chunking strategy (e.g. 'fixed', 'semantic').")
    language: Optional[str] = Field(default=None, description="Language code.")
    dense_score: Optional[float] = Field(default=None, description="Dense retrieval score.")
    bm25_score: Optional[float] = Field(default=None, description="BM25 lexical retrieval score.")
    fusion_score: Optional[float] = Field(default=None, description="Pre-reranking hybrid fusion score.")
    reranker_score: float = Field(..., description="Cross-encoder relevance score.")
    original_rank: int = Field(..., ge=1, description="Rank from initial retrieval stage.")
    rank: int = Field(..., ge=1, description="Final rank after reranking and context selection.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Preserved provenance and dataset metadata.")


class RerankLatencyBreakdown(BaseModel):
    """Detailed millisecond execution time breakdown per reranking stage."""

    retrieval_ms: float = Field(default=0.0, description="Time taken for candidate retrieval (dense/BM25/hybrid).")
    reranking_ms: float = Field(default=0.0, description="Time taken for cross-encoder inference.")
    context_selection_ms: float = Field(default=0.0, description="Time taken for deduplication, diversity, and budgeting.")
    total_ms: float = Field(default=0.0, description="Total pipeline latency in milliseconds.")


class ContextSelectionStats(BaseModel):
    """Telemetry describing context window usage and diversity filtering."""

    total_candidates: int = Field(default=0, description="Number of candidate chunks evaluated.")
    selected_chunks: int = Field(default=0, description="Number of chunks selected in final context.")
    total_characters: int = Field(default=0, description="Total character count in selected context.")
    avg_chunk_characters: float = Field(default=0.0, description="Average character length per selected chunk.")
    max_context_chars: int = Field(default=6000, description="Configured character budget limit.")
    diversity_filtered_count: int = Field(default=0, description="Number of chunks dropped due to document cap or MMR.")


class RerankRequest(BaseModel):
    """Incoming request model for cross-encoder reranking."""

    query: str = Field(..., min_length=1, description="Search query string.")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of top reranked chunks to return.")
    candidate_k: int = Field(default=20, ge=1, le=100, description="Number of initial candidates to retrieve.")
    language: Optional[str] = Field(default=None, description="Optional language filter.")
    strategies: Optional[List[str]] = Field(default=None, description="Optional chunk strategy filter.")
    retrieval_mode: str = Field(
        default="rrf_hybrid",
        description="Candidate retrieval mode: 'rrf_hybrid', 'weighted_hybrid', 'dense_only', 'bm25_only'.",
    )


class RerankResponse(BaseModel):
    """Response model containing reranked results, context telemetry, and latency metrics."""

    query: str = Field(..., description="Original search query string.")
    total_results: int = Field(..., description="Number of returned results.")
    results: List[RerankResult] = Field(default_factory=list, description="Ranked list of reranked results.")
    context_stats: ContextSelectionStats = Field(..., description="Context size and diversity telemetry.")
    latency_ms: RerankLatencyBreakdown = Field(..., description="Stage latency breakdown.")


# --- Phase 5.5 Adaptive Retrieval Models ---

class AdaptiveLatencyBreakdown(BaseModel):
    """Latency breakdown for adaptive retrieval execution."""

    embedding: float = Field(default=0.0, description="Time spent computing query embeddings (ms).")
    retrieval: float = Field(default=0.0, description="Time spent on dense/BM25 retrieval (ms).")
    qdrant: float = Field(default=0.0, description="Time spent on Qdrant vector search (ms).")
    bm25: float = Field(default=0.0, description="Time spent on BM25 search (ms).")
    fusion: float = Field(default=0.0, description="Time spent on RRF fusion (ms).")
    reranking: float = Field(default=0.0, description="Time spent on cross-encoder reranking (ms).")
    context: float = Field(default=0.0, description="Time spent on context selection and diversity budgeting (ms).")
    total: float = Field(default=0.0, description="Total pipeline latency in milliseconds.")
    cache_hit: bool = Field(default=False, description="Whether request was served from cache.")
    timeout_stage: Optional[str] = Field(default=None, description="Stage that timed out if any.")
    fallback_used: bool = Field(default=False, description="Whether fallback ranking was invoked due to budget exhaustion, timeout or error.")
    parallel_execution: bool = Field(default=False, description="Whether dense and BM25 ran concurrently.")
    remaining_budget_ms: Optional[float] = Field(default=None, description="Remaining latency budget in milliseconds.")


class AdaptiveRetrieveRequest(BaseModel):
    """Incoming request for latency-optimized adaptive retrieval."""

    query: str = Field(..., min_length=1, description="Search query string.")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of final context chunks to return.")
    language: Optional[str] = Field(default=None, description="Optional language filter.")
    strategies: Optional[List[str]] = Field(default=None, description="Optional chunk strategy filter.")


class AdaptiveRetrieveResponse(BaseModel):
    """Response model for adaptive retrieval specifying confidence and reranker decision."""

    query: str = Field(..., description="Original query string.")
    results: List[RerankResult] = Field(default_factory=list, description="Final selected context chunks.")
    retrieval_confidence: float = Field(..., description="Calculated confidence score between 0.0 and 1.0.")
    reranking_used: bool = Field(..., description="Whether neural reranking was executed.")
    reranker: str = Field(..., description="Name of reranking engine used or 'none (skipped)'.")
    routing_tier: Optional[str] = Field(default=None, description="Adaptive routing tier ('high', 'medium', 'low').")
    candidate_k: Optional[int] = Field(default=None, description="Number of candidates routed to reranker.")
    confidence_breakdown: Optional[Dict[str, float]] = Field(default=None, description="Deterministic confidence signal breakdown.")
    latency_ms: AdaptiveLatencyBreakdown = Field(..., description="Detailed latency telemetry.")
