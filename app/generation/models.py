"""Pydantic data models for LLM generation requests, results, configuration, and telemetry."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class GenerationConfig(BaseModel):
    """Configuration parameters for LLM generation."""

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature between 0.0 and 2.0.",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=8192,
        description="Maximum number of tokens to generate.",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability between 0.0 and 1.0.",
    )
    timeout: float = Field(
        default=30.0,
        gt=0.0,
        description="Generation timeout in seconds.",
    )
    streaming: bool = Field(
        default=False,
        description="Whether streaming is enabled for generation.",
    )
    model_name: str = Field(
        default="mock-model",
        description="Identifier or name of the generation model.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt overriding default assistant instructions.",
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None,
        description="Optional list of stop sequences to halt generation.",
    )


class GenerationTelemetry(BaseModel):
    """Execution telemetry and token consumption metrics for generation."""

    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total generation latency in milliseconds.",
    )
    prompt_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of prompt tokens evaluated.",
    )
    completion_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of completion tokens generated.",
    )
    total_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total tokens consumed (prompt + completion).",
    )
    context_chunks_count: int = Field(
        default=0,
        ge=0,
        description="Number of context chunks provided in the prompt.",
    )
    context_characters: int = Field(
        default=0,
        ge=0,
        description="Total character count of the context provided.",
    )


class CitationProvenance(BaseModel):
    """Detailed provenance tracking for an individual validated citation."""

    chunk_id: str = Field(..., description="Unique deterministic identifier of the cited chunk.")
    document_id: str = Field(..., description="Parent document identifier.")
    rank: int = Field(default=1, ge=1, description="Rank position of the cited chunk.")
    chunk_type: Optional[str] = Field(default=None, description="Chunking strategy type.")
    language: Optional[str] = Field(default=None, description="Language code.")
    score: Optional[float] = Field(default=None, description="Relevance or reranker score.")
    snippet: Optional[str] = Field(default=None, description="Brief snippet of the cited evidence text.")


class ValidationResult(BaseModel):
    """Result from generation output and citation validation."""

    is_valid: bool = Field(default=True, description="Whether output passed all validation rules.")
    issues: List[str] = Field(default_factory=list, description="Validation issues, warnings, or rejection reasons.")
    cleaned_answer: str = Field(default="", description="Sanitized and validated answer text.")
    validated_citations: List[str] = Field(default_factory=list, description="Validated citation chunk IDs.")
    citation_provenance: List[CitationProvenance] = Field(default_factory=list, description="Preserved provenance for valid citations.")
    fabricated_citations: List[str] = Field(default_factory=list, description="Citations rejected as fabricated or unreferenced.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Calibrated confidence score.")


class GenerationResult(BaseModel):
    """Structured generation output from an LLM provider."""

    answer: str = Field(
        default="",
        description="Generated answer or completion text.",
    )
    grounded: bool = Field(
        default=True,
        description="Whether the answer is grounded in the retrieved context.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the generated answer between 0.0 and 1.0.",
    )
    citations: List[str] = Field(
        default_factory=list,
        description="List of validated citation identifiers supporting the answer.",
    )
    citation_provenance: List[CitationProvenance] = Field(
        default_factory=list,
        description="Rich provenance metadata for validated citations.",
    )
    model: str = Field(
        default="mock-model",
        description="Name or ID of the model used to produce the generation.",
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="End-to-end generation latency in milliseconds.",
    )
    finish_reason: Optional[str] = Field(
        default="stop",
        description="Reason generation completed ('stop', 'length', 'error', 'timeout', etc.).",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error description if generation failed.",
    )
    telemetry: Optional[GenerationTelemetry] = Field(
        default=None,
        description="Detailed performance and token telemetry metrics.",
    )


class GenerationRequest(BaseModel):
    """Request payload for text generation from query and retrieved context."""

    query: str = Field(
        ...,
        min_length=1,
        description="Search query or user question string.",
    )
    context: Union[str, List[Any]] = Field(
        default="",
        description="Retrieved context as a string or list of chunks/objects.",
    )
    config: Optional[GenerationConfig] = Field(
        default=None,
        description="Optional per-request generation configuration overrides.",
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional language code (e.g., 'hi', 'ta', 'te', 'ml', 'en').",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary provenance or request metadata.",
    )


# --- Phase 6.3 & 6.4 End-to-End RAG Ask Models ---

class AskLatencyBreakdown(BaseModel):
    """Detailed millisecond execution time breakdown across all stages from query to generation."""

    embedding: float = Field(default=0.0, description="Time spent on query vector embedding (ms).")
    retrieval: float = Field(default=0.0, description="Time spent on dense/BM25 candidate retrieval (ms).")
    reranking: float = Field(default=0.0, description="Time spent on cross-encoder reranking (ms).")
    context_selection: float = Field(default=0.0, description="Time spent on context selection and diversity budgeting (ms).")
    guardrails: float = Field(default=0.0, description="Time spent on pre- and post-generation guardrail evaluation (ms).")
    prompt_construction: float = Field(default=0.0, description="Time spent on grounded prompt construction and sanitization (ms).")
    generation: float = Field(default=0.0, description="Time spent in LLM provider text generation (ms).")
    total: float = Field(default=0.0, description="Total end-to-end processing latency in milliseconds.")


class RetrievedContextSummary(BaseModel):
    """Summary telemetry of retrieved context chunks selected for generation."""

    chunks_count: int = Field(default=0, ge=0, description="Number of context chunks selected.")
    total_characters: int = Field(default=0, ge=0, description="Total character length of retrieved context.")
    languages: List[str] = Field(default_factory=list, description="Unique language codes present in selected context.")
    chunk_ids: List[str] = Field(default_factory=list, description="Identifiers of selected context chunks.")


class AskRequest(BaseModel):
    """Incoming request model for end-to-end adaptive RAG generation."""

    query: str = Field(..., min_length=1, description="Search query or user question string.")
    language: Optional[str] = Field(default=None, description="Optional target language code.")
    top_k: Optional[int] = Field(default=None, ge=1, le=50, description="Optional number of context chunks to select.")
    strategies: Optional[List[str]] = Field(default=None, description="Optional chunk strategy filters.")
    config: Optional[GenerationConfig] = Field(default=None, description="Optional generation configuration parameters.")


class AskResponse(BaseModel):
    """End-to-end response model combining adaptive retrieval, grounded prompt, generation, and telemetry."""

    query: str = Field(..., description="Original user query.")
    answer: str = Field(..., description="Generated answer or refusal response.")
    grounded: bool = Field(..., description="Whether answer is grounded in retrieved context.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Generation confidence score between 0.0 and 1.0.")
    citations: List[str] = Field(default_factory=list, description="Validated source chunk identifiers supporting the answer.")
    citation_provenance: List[CitationProvenance] = Field(default_factory=list, description="Provenance details for cited chunks.")
    retrieval: Dict[str, Any] = Field(default_factory=dict, description="Structured retrieval telemetry and metadata.")
    guardrail_decision: Optional[Dict[str, Any]] = Field(default=None, description="Structured decision from generation guardrails.")
    retrieval_confidence: float = Field(default=0.0, description="Adaptive retrieval confidence score between 0.0 and 1.0.")
    reranking_used: bool = Field(default=False, description="Whether cross-encoder reranking was executed.")
    model: str = Field(..., description="Generation model identifier.")
    latency_ms: AskLatencyBreakdown = Field(..., description="Granular latency telemetry across all stages.")
    retrieved_context_summary: RetrievedContextSummary = Field(..., description="Telemetry summary of retrieved context.")
    language: Optional[str] = Field(default=None, description="Target or detected language code.")
    relevance: Optional[Dict[str, Any]] = Field(default=None, description="Multi-signal retrieval relevance assessment.")
    error: Optional[str] = Field(default=None, description="Error message if generation or pipeline failed.")



