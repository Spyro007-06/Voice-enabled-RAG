"""Data models for RAG generation guardrails, grounding verification, and safety decisions."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GuardrailAction(str, Enum):
    """Actions recommended by guardrail evaluations."""

    ALLOW = "allow"
    REFUSE_GENERATION = "refuse_generation"
    FLAG_HALLUCINATION = "flag_hallucination"
    REPLACE_WITH_FALLBACK = "replace_with_fallback"
    BLOCK_INPUT = "block_input"
    REVISE_CITATIONS = "revise_citations"


class RelevanceAssessment(BaseModel):
    """Evaluation of query-evidence semantic relatedness and off-topic detection."""

    is_relevant: bool = Field(default=True, description="Whether query is sufficiently related to retrieved context.")
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Estimated relevance score between 0.0 and 1.0.")
    overlap_ratio: float = Field(default=1.0, ge=0.0, le=1.0, description="Token/character lexical or semantic overlap ratio.")
    reason: str = Field(default="Query is relevant to retrieved context.", description="Reason for relevance decision.")
    decision: str = Field(default="STRONG", description="Relevance category: STRONG, LIMITED, INSUFFICIENT, OFF_TOPIC.")
    signals: Optional[Dict[str, float]] = Field(default=None, description="Multi-signal relevance breakdown.")


class GroundingAssessment(BaseModel):
    """Post-generation evaluation of factual alignment between answer and evidence."""

    is_grounded: bool = Field(default=True, description="Whether claims in answer are backed by retrieved context.")
    grounding_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Grounding confidence score between 0.0 and 1.0.")
    unsupported_claims: List[str] = Field(default_factory=list, description="Claims detected without context support.")
    is_refusal: bool = Field(default=False, description="Whether the answer is an explicit refusal/insufficient context statement.")
    reason: str = Field(default="Answer is grounded in provided evidence.", description="Summary of grounding evaluation.")


class GuardrailResult(BaseModel):
    """Structured decision output from guardrail evaluation pipeline."""

    allowed: bool = Field(..., description="Whether generation is permitted to proceed or response is safe to return.")
    reason: str = Field(default="passed", description="Machine-readable code or summary of guardrail decision.")
    grounded: bool = Field(default=True, description="Whether answer is grounded in retrieved context.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Calibrated confidence score between 0.0 and 1.0.")
    action: str = Field(default=GuardrailAction.ALLOW.value, description="Recommended pipeline action.")
    issues: List[str] = Field(default_factory=list, description="List of detected warnings, safety issues, or violations.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Execution time for guardrail checks in milliseconds.")
    safe_fallback_text: Optional[str] = Field(default=None, description="Pre-configured safe message if generation is refused.")
    relevance: Optional[Dict[str, Any]] = Field(default=None, description="Detailed multi-signal relevance assessment telemetry.")
