"""Data models and schemas for Voice RAG Orchestration."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VoiceLatencyBreakdown(BaseModel):
    """Fine-grained stage-by-stage latency telemetry measured using monotonic timing."""

    # Stage Latencies (ms)
    audio_validation: float = Field(0.0, ge=0.0, description="Audio payload validation latency in ms.")
    stt: float = Field(0.0, ge=0.0, description="Speech-to-Text transcription latency in ms.")
    query_normalization: float = Field(0.0, ge=0.0, description="Query sanitization & language resolution latency in ms.")
    embedding: float = Field(0.0, ge=0.0, description="Dense vector embedding latency in ms.")
    qdrant: float = Field(0.0, ge=0.0, description="Qdrant vector ANN search latency in ms.")
    bm25: float = Field(0.0, ge=0.0, description="BM25 lexical search latency in ms.")
    fusion: float = Field(0.0, ge=0.0, description="Reciprocal Rank Fusion latency in ms.")
    reranking: float = Field(0.0, ge=0.0, description="Neural Cross-Encoder reranking latency in ms.")
    context: float = Field(0.0, ge=0.0, description="Context selection, budget & diversity filtering latency in ms.")
    guardrails: float = Field(0.0, ge=0.0, description="Total pre/post generation guardrails latency in ms.")
    guardrails_pre: float = Field(0.0, ge=0.0, description="Pre-generation input & context guardrail latency in ms.")
    prompt: float = Field(0.0, ge=0.0, description="Grounded prompt construction latency in ms.")
    llm: float = Field(0.0, ge=0.0, description="LLM response generation latency in ms.")
    grounding: float = Field(0.0, ge=0.0, description="Grounding & citation validation latency in ms.")
    guardrails_post: float = Field(0.0, ge=0.0, description="Post-generation grounding guardrail latency in ms.")
    tts: float = Field(0.0, ge=0.0, description="Text-to-Speech audio synthesis latency in ms.")
    serialization: float = Field(0.0, ge=0.0, description="Response encoding & serialization latency in ms.")
    total: float = Field(0.0, ge=0.0, description="End-to-end pipeline latency in ms.")

    # Category-Isolated Latency Subtotals (Mandatory for SLA separation)
    retrieval_total_ms: float = Field(0.0, ge=0.0, description="Total retrieval & reranking latency (SLA bounded to 200ms).")
    generation_total_ms: float = Field(0.0, ge=0.0, description="Total generation, prompting & grounding validation latency.")
    voice_total_ms: float = Field(0.0, ge=0.0, description="Total voice I/O latency (STT + TTS).")
    end_to_end_total_ms: float = Field(0.0, ge=0.0, description="End-to-end total pipeline latency.")

    # Diagnostic execution flags
    cache_hit: bool = Field(False, description="Whether retrieval was served from cache.")
    execution_type: str = Field("warm", description="'cold' or 'warm' execution indicator.")
    timeout_stage: Optional[str] = Field(None, description="Stage identifier where timeout occurred, if any.")
    fallback_used: bool = Field(False, description="Whether fallback pipeline degraded gracefully.")


class AudioOutputMetadata(BaseModel):
    """Metadata describing synthesized speech audio payload."""

    available: bool = Field(False, description="Whether synthesized audio is available in the response.")
    format: str = Field("wav", description="Audio container format ('wav', 'mp3', etc.).")
    sample_rate: Optional[int] = Field(24000, description="Audio sampling rate in Hz.")
    duration_s: Optional[float] = Field(0.0, ge=0.0, description="Estimated audio duration in seconds.")
    provider: Optional[str] = Field(None, description="Name of TTS provider utilized.")
    audio_base64: Optional[str] = Field(None, description="Base64-encoded audio payload if synthesized.")


class VoiceAskResponse(BaseModel):
    """Structured response payload returned by POST /api/voice-ask."""

    status: str = Field("success", description="'success', 'partial_success', or 'error'.")
    request_id: str = Field(..., description="Unique request tracing identifier.")
    transcript: str = Field(..., description="Speech-to-Text transcribed user query.")
    detected_language: Optional[str] = Field(None, description="Language code detected by STT or language detector.")
    language: str = Field("en", description="Canonical BCP-47 / ISO language code applied ('en', 'hi', 'ta', 'te', 'ml').")
    answer: str = Field(..., description="Grounded LLM-generated response text.")
    grounded: bool = Field(False, description="Whether the answer was fully verified and grounded against context.")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="End-to-end confidence score.")
    citations: List[str] = Field(default_factory=list, description="Verified chunk or document citation IDs.")
    citation_provenance: List[Any] = Field(default_factory=list, description="Chunk provenance citations.")
    retrieval_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Retrieval confidence score from adaptive router.")
    reranking_used: bool = Field(False, description="Whether Cross-Encoder neural reranking was executed.")
    reranker_tier: Optional[str] = Field(None, description="Adaptive routing tier ('high', 'medium', 'low', 'skip').")
    guardrail_decision: Optional[str] = Field("allow", description="Guardrail outcome ('allow', 'refuse', 'block').")
    fallback_status: bool = Field(False, description="Whether any subsystem fallback was triggered.")
    audio: AudioOutputMetadata = Field(default_factory=AudioOutputMetadata, description="Synthesized voice audio payload.")
    latency_ms: VoiceLatencyBreakdown = Field(default_factory=VoiceLatencyBreakdown, description="Granular latency telemetry.")
    error: Optional[str] = Field(None, description="Error detail message if any stage degraded gracefully.")
