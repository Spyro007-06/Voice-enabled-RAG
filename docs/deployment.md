# Production Containerization & Deployment Architecture

**Project:** HH Goa 2026 — Multilingual Voice RAG  
**Phase:** 6.11  
**Architecture Classification:** Containerized Multi-Tier Voice RAG Microservice  

---

## 1. Architectural Topology

```mermaid
flowchart TD
    Client[External Client / Mobile App / Web] -->|HTTPS :443| LB[Reverse Proxy / Ingress / Load Balancer]
    LB -->|HTTP :8000| API[voice-rag-api Container (FastAPI + Uvicorn 4 Workers)]
    
    subgraph DockerBridge[Docker Bridge Network: voice-rag-net]
        API -->|HTTP :6333 / Internal| Qdrant[Qdrant Vector DB Container]
        API -->|Local Memory / Disk| BM25[BM25 Index (9,355 Records)]
        API -->|Mounted Volume| HFModelCache[(Persistent Model Cache: /app/data/cache)]
    end
    
    subgraph ExternalProviders[External Cloud AI Providers (TLS)]
        API -.->|Sarvam API (HTTPS)| SarvamSTT[Sarvam STT / Saarika-v2]
        API -.->|Sarvam API (HTTPS)| SarvamTTS[Sarvam TTS / Bulbul-v1]
        API -.->|Sarvam API (HTTPS)| SarvamLLM[Sarvam LLM / Sarvam-2B]
    end
    
    subgraph Observability[Telemetry & Health Monitoring]
        Prometheus[Prometheus Scraper] -->|GET /metrics| API
        K8sProbe[Health Probes] -->|GET /health| API
    end
```

---

## 2. Container Specifications

### Voice RAG API Container (`voice-rag-api`)
- **Base Image:** `python:3.11-slim` (multi-stage build with `ghcr.io/astral-sh/uv`).
- **User:** Non-root `appuser` (UID: 10001, GID: 10001).
- **Process Manager:** `uvicorn` production ASGI server (`4 workers`, `--proxy-headers`, `--forwarded-allow-ips=*`).
- **Resource Constraints:** 
  - Limit: 4.0 CPU cores, 4 GB RAM.
  - Reservation: 1.0 CPU cores, 1 GB RAM.
- **Port:** Exposed `8000`.
- **Healthcheck:** `GET /health` every 30s (5s timeout, 15s start period, 3 retries).

### Vector Database Container (`qdrant`)
- **Image:** `qdrant/qdrant:v1.9.0`.
- **Port:** `6333` (Internal network only; no public exposure).
- **Persistent Storage:** Named volume `qdrant_storage:/qdrant/storage`.
- **Healthcheck:** `curl -f http://localhost:6333/readyz` every 15s.

---

## 3. Persistent Volumes Strategy

To prevent model redownloads and preserve vector states across container restarts:

1. **`hhgoa_model_cache`:** Mounted at `/app/data/cache/huggingface`. Caches the `multilingual-e5-small` embedding weights and `mmarco-mMiniLMv2` cross-encoder weights.
2. **`hhgoa_app_data`:** Mounted at `/app/data`. Houses the persistent `bm25_index.pkl` (9,355 records) and local dataset partitions.
3. **`hhgoa_qdrant_storage`:** Mounted at `/qdrant/storage`. Retains Qdrant collections, payloads, and HNSW graphs.

---

## 4. Environment & Secrets Management

All secrets are passed at runtime via environment variables (never baked into the container image or committed to source control):

| Variable | Requirement | Purpose |
|---|:---:|---|
| `ENVIRONMENT` | Required | Set to `production` or `development`. |
| `DEBUG` | Required | Set to `false` in production. |
| `SARVAM_API_KEY` | Conditional | Required when using Sarvam STT/TTS/LLM in production. |
| `OPENAI_API_KEY` | Conditional | Required if `LLM_PROVIDER=openai`. |
| `VECTOR_DB_URL` | Required | Points to `http://qdrant:6333` in Docker Compose. |
| `RATE_LIMIT_ENABLED` | Required | Enforces per-client sliding window rate limiting (default: `120 RPM`). |
| `TOTAL_RETRIEVAL_DEADLINE_MS` | Required | Enforces the 200 ms SLA retrieval deadline. |

---

## 5. Security Architecture & Controls

1. **Non-Root Execution:** Container processes run exclusively under UID 10001 (`appuser`).
2. **Secret Isolation:** `.dockerignore` excludes `.env`, `*.key`, `*.pem`, `*.crt`, `credentials.json`, and local test secrets.
3. **HTTP Security Headers:** Injected by middleware:
   - `X-Content-Type-Options: nosniff`
   - `X-Frame-Options: DENY`
   - `Referrer-Policy: strict-origin-when-cross-origin`
   - `Content-Security-Policy: default-src 'self'; frame-ancestors 'none'`
   - `Permissions-Policy: geolocation=(), camera=(), microphone=()`
   - `Cache-Control: no-store` on API endpoints.
4. **Input & Audio Validation:**
   - Max JSON payload: 2 MB.
   - Max Query length: 2,000 characters.
   - Max Audio payload: 15 MB.
   - Binary magic-byte inspection on audio uploads.
   - Filename path traversal stripping.
5. **Output Scrubbing:** Automatic masking of API tokens, private filesystem paths, and raw exception stack traces in client responses.

---

## 6. Health Checks & Observability

- **Liveness & Readiness Endpoint:** `GET /health` returns HTTP 200 `{"status": "healthy", ...}`.
- **Prometheus Metrics:** `GET /metrics` provides real-time latency histograms, retrieval confidence metrics, provider health states, and cache statistics.
- **Request Tracing:** `X-Request-ID` correlation on all request/response cycles and structured JSON logs.

---

## 7. Deployment Commands

### Build Container Image
```bash
docker build -t hh-goa-voice-rag:latest .
```

### Run with Docker Compose
```bash
# Start all services in detached mode
docker compose up -d

# Check service status
docker compose ps

# View live structured logs
docker compose logs -f voice-rag-api

# Execute health probe
curl -f http://localhost:8000/health

# Stop and gracefully shut down
docker compose down
```
