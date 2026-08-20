# Production Provider Integration & Real Environment Architecture

**Project:** HH Goa 2026 — Multilingual Voice RAG  
**Phase:** 6.12  
**Scope:** Real Provider Architecture, Secrets Management, Staging/Production Separation & Cost Protection  

---

## 1. Provider Selection & Environment Mapping

All external cloud and local providers are resolved dynamically through the thread-safe provider factory system in `app/providers/factory.py`. No provider credentials or implementations are hardcoded.

```mermaid
flowchart TD
    Env[Environment Variables / .env] --> Settings[app.config.Settings Singleton]
    Settings --> Factory[app.providers.factory]
    
    subgraph STT[Speech-to-Text]
        Factory --> MockSTT[MockSTTProvider (Dev/CI)]
        Factory --> SarvamSTT[SarvamSTTProvider (Saarika-v2)]
    end
    
    subgraph LLM[Generation Engine]
        Factory --> MockLLM[MockLLMProvider (Dev/CI)]
        Factory --> SarvamLLM[SarvamLLMProvider (Sarvam-2B)]
        Factory --> OpenAILLM[OpenAILLMProvider (GPT-4o-mini)]
    end
    
    subgraph TTS[Text-to-Speech]
        Factory --> MockTTS[MockTTSProvider (Dev/CI)]
        Factory --> SarvamTTS[SarvamTTSProvider (Bulbul-v1)]
    end
    
    subgraph Vector[Vector Database]
        Factory --> QdrantLocal[QdrantLocal / Disk Mount]
        Factory --> QdrantCloud[QdrantCloudVectorProvider]
    end
```

---

## 2. Environment Configurations

The system strictly enforces three operational environment tiers:

### A. Development (`ENVIRONMENT=development`)
- Mock providers allowed for zero-cost rapid offline iteration.
- Local embedded Qdrant (`data/qdrant`) or in-memory vector store.
- Local debug logging allowed.

### B. Staging (`ENVIRONMENT=staging`)
- Real external providers configured with staging API keys.
- Qdrant container or staging Qdrant Cloud.
- Rate limiting and observability active.

### C. Production (`ENVIRONMENT=production`)
- `DEBUG=false` strictly enforced (`validate_production_config`).
- Non-wildcard CORS origins required.
- Real provider credentials required at startup.
- Full rate limiting (120 RPM default) and security headers active.

---

## 3. Secret Isolation & Zero-Leakage Controls

1. **No Key Ingestion in Version Control:**
   - `.gitignore` and `.dockerignore` exclude `.env`, `*.key`, `*.pem`, `*.crt`, and credential artifacts.
2. **Log Redaction:**
   - `StructuredJSONFormatter` dynamically masks any API keys (`sk_live_...`, `Bearer ...`) before emitting logs to stdout/stderr.
3. **Client-Facing Redaction:**
   - `sanitize_output_text()` strips internal filesystem paths (`C:\...`, `/home/...`) and accidental key traces from all API responses and error messages.

---

## 4. Cost Protection & Smoke Testing Policy

To prevent unintended cloud API billing during automated unit and regression testing:

- Standard `pytest` test runs execute with deterministic mock providers and local offline embedding/reranking.
- Real cloud provider integration tests in `tests/test_phase612_real_provider.py` and `scripts/test_phase612_production.py` are cost-protected:
  ```bash
  # Explicitly enable real provider live calls:
  export PHASE612_REAL_PROVIDER_TESTS=true
  export SARVAM_API_KEY="sk_live_..."
  
  # Run live provider verification:
  uv run pytest tests/test_phase612_real_provider.py -v
  ```
- If `PHASE612_REAL_PROVIDER_TESTS` is not set or API keys are missing, tests skip gracefully without failing the build.

---

## 5. Provider Failure Modes & Fallback Hierarchy

| Component | Failure Mode | Fallback Strategy | Outcome |
|---|---|---|:---:|
| **STT** | Upstream timeout / 5xx | Raises structured `STTError` | Returns HTTP 502 with safe error detail |
| **Qdrant** | Connection loss / timeout | Switches to Lexical BM25 index | Returns HTTP 200 with BM25 results |
| **BM25** | Index corrupted / timeout | Switches to Dense ANN search | Returns HTTP 200 with Dense results |
| **MiniLM** | Reranker timeout / budget cut | Falls back to Reciprocal Rank Fusion | Returns HTTP 200 with RRF results |
| **LLM** | Generation timeout / refusal | Activates structured refusal | Returns HTTP 200 with honest refusal |
| **TTS** | Upstream audio failure | Emits grounded text response | Returns HTTP 200 with `status="partial_success"` |

---

## 6. Real vs Mock Latency Distinctions

| Subsystem Stage | Mock / Local Baseline (ms) | Real Cloud Provider (ms) | Notes |
|---|:---:|:---:|---|
| **STT (Saarika-v2)** | ~1.5 ms | ~250 – 450 ms | Dependent on audio duration & network round-trip. |
| **Dense ANN (Qdrant)** | ~18.5 ms | ~25 – 60 ms | Sub-20ms when hosted in same VPC/network. |
| **BM25 Lexical Search** | ~8.1 ms | ~8.1 ms | Embedded in-process execution. |
| **Adaptive Reranking** | ~20.0 ms | ~20.0 ms | In-process lightweight cross-encoder. |
| **LLM Generation (Sarvam-2B)** | ~2.0 ms (mock) | ~400 – 900 ms | Streaming token generation round-trip. |
| **TTS (Bulbul-v1)** | ~1.0 ms (mock) | ~300 – 600 ms | Audio synthesis and Base64 packaging. |
