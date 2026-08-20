# HH Goa 2026 — Multilingual Voice RAG System

[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()
[![Tests: 460+ Passing](https://img.shields.io/badge/Tests-460%2B%20Passing-success.svg)]()
[![Languages: 5 Indic](https://img.shields.io/badge/Languages-EN%20%7C%20HI%20%7C%20TA%20%7C%20TE%20%7C%20ML-blue.svg)]()
[![Streaming: Server--Sent Events](https://img.shields.io/badge/Streaming-Native%20SSE-cyan.svg)]()

Enterprise-grade, async-first Multilingual Voice Retrieval-Augmented Generation (RAG) system supporting text and voice queries across 5 Indic languages: **English**, **Hindi (हिंदी)**, **Tamil (தமிழ்)**, **Telugu (తెలుగు)**, and **Malayalam (മലയാളം)** with native Server-Sent Events (SSE) token streaming, sub-50ms hybrid retrieval, neural reranking, pre/post-generation guardrails, and speech-to-text / text-to-speech pipelines.

---

## 1. Architecture Overview

```
Client (Web / Mobile / Voice Ingress)
  │
  ├──► GET /api/ask-stream (Native SSE Token Streaming)
  │      ├──► Stage Events: "retrieving" ──► "ranking" ──► "generating" ──► tokens ──► "done"
  │
  ├──► POST /api/ask (Buffered Grounded QA)
  │      ├──► Central Language Resolver & Query Normalization
  │      ├──► Hybrid Dense + BM25 Retrieval (Qdrant 28k vectors + BM25 48k chunks)
  │      ├──► Reciprocal Rank Fusion (RRF) & Cross-Encoder Adaptive Neural Reranking
  │      ├──► Pre/Post-Generation Security Guardrails & Citation Grounding
  │      └──► Low-latency LLM Generation (Sarvam AI / OpenAI / Mock)
  │
  └──► POST /api/voice-ask (End-to-End Voice QA)
         ├──► Magic-Byte Audio Container Validation (WAV, MP3, OGG, WebM, M4A, FLAC)
         ├──► Speech-to-Text (STT) Transcription (Sarvam Saarika v2)
         ├──► Multilingual RAG Knowledge Retrieval & Generation
         └──► Text-to-Speech (TTS) Shielded Synthesis (Sarvam Bulbul v1)
```

---

## 2. Supported Languages & Canonical Codes

| Language | BCP-47 Code | Script | Native Name | Sample Query |
|:---|:---:|:---:|:---:|:---|
| **English** | `en` | Latin | English | *"What is artificial intelligence?"* |
| **Hindi** | `hi` | Devanagari | हिन्दी | *"कृत्रिम बुद्धिमत्ता क्या है?"* |
| **Tamil** | `ta` | Tamil | தமிழ் | *"செயற்கை நுண்ணறிவு என்றால் என்ன?"* |
| **Telugu** | `te` | Telugu | తెలుగు | *"కృత్రిమ మేధస్సు అంటే ఏమిటి?"* |
| **Malayalam** | `ml` | Malayalam | മലയാളം | *"കൃത്രിമ ബുദ്ധി എന്താണ്?"* |

---

## 3. Quickstart & Installation

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended)

### Local Setup
```bash
# 1. Clone repository
git clone <repo-url>
cd HH-T2

# 2. Install dependencies
uv sync

# 3. Configure environment
cp .env.example .env

# 4. Run test suite
uv run pytest

# 5. Start development backend
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Accessing the Web Application
Open your browser at `http://localhost:8000` (or serve `frontend/` via any static file server).

---

## 4. Environment Configuration (`.env`)

```ini
# Environment Mode
ENVIRONMENT=production
DEBUG=false

# Provider API Keys
SARVAM_API_KEY=your_sarvam_api_key_here
OPENAI_API_KEY=optional_openai_key_here

# Vector Database (Qdrant)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=msmarco_xi

# Latency & Generation Caps
MAX_TOKENS=192
LLM_TIMEOUT_MS=6000.0
TTS_TIMEOUT_MS=5000.0
```

---

## 5. API Endpoints

### 5.1 Streaming Endpoint (Server-Sent Events)
- **`GET /api/ask-stream`**
  - Query parameters: `query` (str), `language` (optional str), `top_k` (int, default 5)
  - Emits SSE events: `stage`, `token`, `done`, `error`

### 5.2 Standard QA Endpoint
- **`POST /api/ask`**
  - Body: `{"query": "...", "language": "en", "top_k": 5}`
  - Returns: `answer`, `grounded`, `confidence`, `citations`, `citation_provenance`, `latency_ms`

### 5.3 Voice Pipeline Endpoint
- **`POST /api/voice-ask`**
  - Multipart form data: `audio` (binary file), `language` (optional), `top_k` (int), `synthesize_speech` (bool)
  - Returns: `transcription`, `answer`, `audio_output`, `grounded`, `citations`, `latency_breakdown`

### 5.4 System & Observability Endpoints
- **`GET /health`** — System health probe (`{"status": "healthy"}`)
- **`GET /metrics`** — Prometheus telemetry counters, gauges, and latency histograms
- **`GET /openapi.json`** — OpenAPI v3 specification

---

## 6. Docker Container Deployment

```bash
# Build & start full container stack (API + Qdrant)
docker-compose up -d --build

# Inspect logs
docker-compose logs -f voice-rag-api
```

---

## 7. Deterministic Demo Verification

To execute the automated 18-step demo verification scenario across all 5 languages, cross-language retrieval, safe refusal, SSE streaming, and voice pipeline:

```bash
uv run python scripts/run_demo_check.py
```

---

## 8. Security & Guardrails

- **Pre-Generation Validation:** Blocks prompt injection, jailbreak attempts, and confidential key extraction.
- **Post-Generation Verification:** Enforces grounding check; replaces ungrounded text with standard refusal.
- **Rate Limiting:** 120 requests/minute token bucket with `Retry-After` headers.
- **Audio Hardening:** Magic-byte sniffing on uploaded audio containers; 60s max recording cap; Page Visibility cleanup.
- **Secret Isolation:** Zero credential or token leakage in logs, error payloads, or frontend bundles.

---

## 9. QA & Benchmarking Artifacts

Detailed Phase 6.22 certification artifacts available under `benchmarks/`:
- `benchmarks/phase622_qa_matrix.json`
- `benchmarks/phase622_qa_report.md`
- `benchmarks/phase622_performance.json`
- `benchmarks/phase622_performance_report.md`
