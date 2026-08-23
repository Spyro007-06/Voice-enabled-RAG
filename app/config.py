"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Core application settings loaded from environment variables or .env file."""

    # Application settings
    APP_NAME: str = "HH Goa 2026 Voice RAG"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS configuration
    CORS_ORIGINS: Union[List[str], str] = ["*"]

    # Phase 2 — Dataset & Ingestion Configuration
    DATASET_NAME: str = "ai4bharat/MSMARCO-XI"
    DATASET_SPLIT: str = "validation"
    DATASET_LANGUAGE: str = "hi"
    DATASET_SAMPLE_SIZE: int = 100
    DATASET_STREAMING: bool = True
    DATASET_CACHE_DIR: str = "data/raw"

    # Phase 2 — Chunking Configuration
    FIXED_CHUNK_SIZE: int = 250
    FIXED_CHUNK_OVERLAP: int = 50
    SENTENCE_MAX_CHUNK_SIZE: int = 300
    SLIDING_WINDOW_SIZE: int = 3
    SLIDING_WINDOW_OVERLAP: int = 1
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.4
    CHUNKING_STRATEGIES: str = "fixed,sentence,sliding_window,semantic,hierarchical"

    # Phase 3 — Embedding & Vector Index Configuration
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    EMBEDDING_DEVICE: str = "auto"  # 'auto', 'cuda', or 'cpu'
    EMBEDDING_BATCH_SIZE: int = 256
    CPU_NUM_THREADS: int = 16
    QDRANT_PATH: str = "data/qdrant"
    QDRANT_COLLECTION_NAME: str = "msmarco_xi"
    QDRANT_UPSERT_BATCH_SIZE: int = 256
    INDEX_STRATEGIES: str = "fixed,sentence,sliding_window,semantic,hierarchical"
    INDEX_SAMPLE_SIZE: int = 1000

    # Phase 4 — Hybrid Retrieval & Fusion Configuration
    DENSE_TOP_K: int = 20
    BM25_TOP_K: int = 20
    HYBRID_ALPHA: float = 0.65  # Weight for dense in weighted score fusion: alpha*dense + (1-alpha)*bm25
    RRF_K: int = 60  # Smoothing constant for Reciprocal Rank Fusion
    RETRIEVAL_STRATEGIES: str = "all"
    RETRIEVAL_CACHE_ENABLED: bool = True
    RETRIEVAL_CACHE_SIZE: int = 1000
    BM25_INDEX_PATH: str = "data/bm25_index.pkl"
    EVAL_SAMPLE_SIZE: int = 200

    # Phase 5 & Phase 6.3 — Production Cross-Encoder Reranking Configuration
    RERANKER_PROVIDER: str = "minilm"  # 'minilm', 'bge', 'none'
    RERANKER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    LIGHTWEIGHT_RERANKER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    HEAVY_RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    HEAVY_RERANKER_ENABLED: bool = False  # Disabled by default on CPU for strict latency budgets
    RERANKER_DEVICE: str = "auto"  # 'auto', 'cuda', or 'cpu'
    RERANK_CANDIDATE_K: int = 5
    RERANKER_DEFAULT_CANDIDATE_K: int = 5
    RERANKER_MAX_LENGTH: int = 128
    RERANKER_BATCH_SIZE: int = 16
    RERANK_BATCH_SIZE: int = 16  # Backward compatibility alias
    RERANKER_HIGH_CONFIDENCE_SKIP: bool = True
    RERANKER_WARMUP_ENABLED: bool = False

    CONTEXT_DIVERSITY_ENABLED: bool = True
    MAX_CHUNKS_PER_DOCUMENT: int = 2
    FINAL_CONTEXT_K: int = 5
    MAX_CONTEXT_CHARS: int = 6000

    # Phase 5.5, 6.3 & 6.4 — Latency Optimization, Adaptive Reranking & RAG Cache
    ADAPTIVE_ENABLED: bool = True
    LIGHTWEIGHT_RERANKER_MAX_LENGTH: int = 128
    ADAPTIVE_CONFIDENCE_W_DENSE: float = 0.45
    ADAPTIVE_CONFIDENCE_W_AGREEMENT: float = 0.35
    ADAPTIVE_CONFIDENCE_W_MARGIN: float = 0.20
    ADAPTIVE_THRESHOLD_HIGH: float = 0.65  # Calibrated Pareto-optimal threshold (skips if conf >= 0.65)
    ADAPTIVE_THRESHOLD_LOW: float = 0.30   # MiniLM K=3 if 0.30 <= conf < 0.65; MiniLM K=5 if < 0.30
    ADAPTIVE_MEDIUM_K: int = 3
    ADAPTIVE_LOW_K: int = 5
    TARGET_RETRIEVAL_MS: float = 75.0
    TARGET_RERANK_MS: float = 50.0
    TARGET_CONTEXT_MS: float = 10.0

    # Bounded RAG LRU Cache
    RAG_CACHE_ENABLED: bool = False
    RAG_CACHE_MAX_SIZE: int = 256
    RAG_CACHE_TTL_SECONDS: int = 300

    # Phase 6.5 & 6.5.1 — Production Latency Hardening, Calibrated Timeouts & Retrieval Stability
    QDRANT_TIMEOUT_MS: float = 2500.0
    BM25_TIMEOUT_MS: float = 2500.0
    RERANKER_TIMEOUT_MS: float = 1500.0
    RETRIEVAL_TIMEOUT_MS: float = 3000.0
    TOTAL_RETRIEVAL_DEADLINE_MS: float = 3000.0
    RERANKER_MIN_BUDGET_MS: float = 20.0  # Measured MiniLM K=5 P50 latency (20.02ms)
    RERANKER_EARLY_EXIT_MARGIN: float = 0.0  # 0.0 prevents quality degradation on edge cases
    PARALLEL_RETRIEVAL_ENABLED: bool = False
    PARALLEL_RETRIEVAL_WORKERS: int = 4

    # Phase 6.1 & Gemini Migration — LLM Generation Architecture Configuration
    LLM_PROVIDER: str = "gemini"  # 'gemini', 'mock', 'sarvam', 'openai'
    LLM_MODEL_NAME: str = "gemini-2.5-flash"

    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048
    LLM_TOP_P: float = 0.9
    LLM_TIMEOUT: float = 60.0  # Generation timeout in seconds
    LLM_STREAMING: bool = False
    LLM_SYSTEM_PROMPT: str = (
        "You are a multilingual retrieval-grounded question answering system.\n"
        "Your task is to answer the user's question using ONLY the retrieved passages supplied in the context.\n"
        "The retrieved passages come from the MSMARCO-XI multilingual corpus.\n"
        "Rules:\n"
        "1. Do not use outside knowledge.\n"
        "2. Do not invent facts.\n"
        "3. Do not infer unsupported information.\n"
        "4. Do not fabricate citations.\n"
        "5. If the retrieved context does not contain enough information to answer the question, clearly state "
        "that the information is not available in the retrieved context.\n"
        "6. Answer in the requested language whenever the retrieved context supports answering in that language.\n"
        "7. Keep the answer concise and directly related to the question.\n"
        "8. Preserve factual meaning from the retrieved passages.\n"
        "9. Never claim that information exists in the corpus unless it is present in the supplied context."
    )

    # Phase 6.5 — RAG Generation Guardrails Configuration
    GUARDRAILS_ENABLED: bool = True
    GUARDRAIL_MIN_CONFIDENCE: float = 0.0   # Configurable threshold (0.0 allows all non-empty development retrieval)
    GUARDRAIL_MIN_RELEVANCE: float = 0.10   # Reject generation if evidence relevance < 0.10
    GUARDRAIL_REPLACE_UNGROUNDED: bool = True  # Replace ungrounded answer with safe fallback
    GUARDRAIL_FALLBACK_MESSAGE: str = "I don't have enough information in the retrieved context to answer that."
    TARGET_GUARDRAIL_MS: float = 5.0

    # Phase 6.1 & 6.6 — Production Provider & Voice Pipeline Configuration
    STT_PROVIDER: str = "sarvam"  # 'mock', 'sarvam'
    TTS_PROVIDER: str = "sarvam"  # 'mock', 'sarvam'
    VECTOR_PROVIDER: str = "qdrant_local"  # 'qdrant_local', 'qdrant_cloud', 'memory'

    # Phase 6.6 — Voice RAG Orchestration & Audio Security Configuration
    STT_TIMEOUT_MS: float = 5000.0
    TTS_TIMEOUT_MS: float = 5000.0
    MAX_AUDIO_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB audio upload limit
    ALLOWED_AUDIO_MIME_TYPES: List[str] = [
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/mpeg",
        "audio/mp3",
        "audio/ogg",
        "audio/webm",
        "audio/m4a",
        "audio/x-m4a",
        "audio/mp4",
        "audio/aac",
        "audio/flac",
    ]

    # Google Gemini Credentials & Configuration
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"

    # Production External Credentials & Endpoints (Never hardcoded, never logged)
    SARVAM_API_KEY: Optional[str] = None
    SARVAM_BASE_URL: str = "https://api.sarvam.ai"
    SARVAM_STT_MODEL: str = "saarika:v2.5"
    SARVAM_TTS_MODEL: str = "bulbul:v2"
    SARVAM_TTS_SPEAKER: str = "anushka"
    SARVAM_LLM_MODEL: str = "sarvam-105b"

    VECTOR_DB_URL: Optional[str] = None
    VECTOR_DB_API_KEY: Optional[str] = None

    LLM_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_LLM_MODEL: str = "gpt-4o-mini"

    # Phase 6.9 — Production Security Hardening & Rate Limiting Configuration
    MAX_JSON_PAYLOAD_SIZE_BYTES: int = 2 * 1024 * 1024  # 2 MB JSON payload limit
    MAX_QUERY_LENGTH: int = 2000  # Safe max query character length
    MAX_LANGUAGE_LENGTH: int = 32  # Max language code length
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 120
    RATE_LIMIT_BURST: int = 30
    SECURITY_HEADERS_ENABLED: bool = True
    STRICT_CONTENT_TYPE_CHECK: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS origins whether provided as a JSON list, comma-separated string, or list."""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return ["*"]
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed]
                except Exception:
                    pass
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


def validate_production_config(settings: Settings) -> List[str]:
    """Validate that production configuration meets strict operational security standards.

    Returns:
        List[str]: List of security warnings or non-compliant configuration notices.
    """
    issues: List[str] = []
    if settings.ENVIRONMENT.lower() == "production":
        if settings.DEBUG:
            issues.append("DEBUG mode must be disabled (False) in production.")
        if "*" in settings.CORS_ORIGINS:
            issues.append("CORS_ORIGINS should not contain wildcard '*' in production when credentials are enabled.")
        if settings.STT_PROVIDER == "sarvam" and not settings.SARVAM_API_KEY:
            issues.append("SARVAM_API_KEY is required when STT_PROVIDER is 'sarvam'.")
        if settings.TTS_PROVIDER == "sarvam" and not settings.SARVAM_API_KEY:
            issues.append("SARVAM_API_KEY is required when TTS_PROVIDER is 'sarvam'.")
        if settings.LLM_PROVIDER == "gemini" and not settings.GEMINI_API_KEY:
            issues.append("GEMINI_API_KEY is required when LLM_PROVIDER is 'gemini'.")
        if settings.LLM_PROVIDER == "sarvam" and not settings.SARVAM_API_KEY:
            issues.append("SARVAM_API_KEY is required when LLM_PROVIDER is 'sarvam'.")
        if settings.LLM_PROVIDER == "openai" and not settings.OPENAI_API_KEY:
            issues.append("OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'.")
        if settings.VECTOR_PROVIDER == "qdrant_cloud" and not settings.VECTOR_DB_API_KEY:
            issues.append("VECTOR_DB_API_KEY is required when VECTOR_PROVIDER is 'qdrant_cloud'.")
    return issues


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings singleton instance."""
    return Settings()
