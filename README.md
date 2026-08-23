# HH Goa 2026 — Multilingual Voice RAG System

[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()
[![Tests: Passing](https://img.shields.io/badge/Tests-100%25%20Passing-success.svg)]()
[![Languages: 5 Indic](https://img.shields.io/badge/Languages-EN%20%7C%20HI%20%7C%20TA%20%7C%20TE%20%7C%20ML-blue.svg)]()
[![Streaming: Server--Sent Events](https://img.shields.io/badge/Streaming-Native%20SSE-cyan.svg)]()

Enterprise-grade, async-first Multilingual Voice Retrieval-Augmented Generation (RAG) system supporting text and voice queries across 5 Indic languages: **English**, **Hindi (हिंदी)**, **Tamil (தமிழ்)**, **Telugu (తెలుగు)**, and **Malayalam (മലയാളം)** with native Server-Sent Events (SSE) token streaming, sub-50ms hybrid retrieval, neural reranking, pre/post-generation guardrails, and speech-to-text / text-to-speech pipelines.

---

## 1. Provider Responsibility Architecture

- **LLM**: Google Gemini API (`gemini-2.5-flash`)
- **STT**: Sarvam AI Saarika v2.5 (`saarika:v2.5`)
- **TTS**: Sarvam AI Bulbul v2 (`bulbul:v2`)
- **Embedding**: `intfloat/multilingual-e5-small` (384-dim)
- **Vector DB**: Qdrant (`data/qdrant` / 28,541 vectors)
- **Lexical Index**: BM25 (`data/bm25_index.pkl` / 48,206 chunks)
- **Reranker**: Cross-Encoder `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- **Corpus**: MSMARCO-XI Multilingual Passage Dataset

```
                    USER
                     │
             ┌───────┴───────┐
             │               │
           TEXT            VOICE
             │               │
             │          Sarvam STT (Saarika v2.5)
             │               │
             └───────┬───────┘
                     ↓
              LANGUAGE RESOLUTION (en, hi, ta, te, ml)
                     ↓
              MULTILINGUAL RAG
                     ↓
          Qdrant + BM25 + RRF (k=60)
                     ↓
            Cross-Encoder Reranker
                     ↓
             PRE-GUARD RELEVANCE CHECK
             ┌───────┴───────┐
      [Below Threshold]  [Relevant Evidence]
             │               │
          REFUSAL            ↓
      (Gemini NOT called)  GOOGLE GEMINI LLM
                             │
                             ↓
                     GROUNDED ANSWER
                             │
                             ↓
                    POST-GUARD CITATION VALIDATION
                             │
             ┌───────────────┴───────────────┐
             │                               │
         TEXT MODE                      VOICE MODE
             │                               │
             ↓                               ↓
      Frontend Answer                 Sarvam TTS (Bulbul v2)
      + Expandable Sources                   │
                                             ↓
                                      Synthesized Audio
```

---

## 2. Supported Languages & Canonical Codes

| Language | BCP-47 Code | Script | Native Name | Sample Query |
|:---|:---:|:---:|:---:|:---|
| **English** | `en` | Latin | English | *"What is a computer?"* |
| **Hindi** | `hi` | Devanagari | हिन्दी | *"कंप्यूटर क्या है?"* |
| **Tamil** | `ta` | Tamil | தமிழ் | *"கணினி என்றால் என்ன?"* |
| **Telugu** | `te` | Telugu | తెలుగు | *"కంప్యూటర్ అంటే ఏమిటి?"* |
| **Malayalam** | `ml` | Malayalam | മലയാളം | *"കമ്പ്യൂട്ടർ എന്താണ്?"* |

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
Open your browser at `http://localhost:8000`.

---

## 4. Environment Configuration (`.env`)

```ini
# Environment Mode
ENVIRONMENT=production
DEBUG=false

# LLM Provider Configuration (Google Gemini)
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# Speech Provider Configuration (Sarvam AI)
STT_PROVIDER=sarvam
TTS_PROVIDER=sarvam
SARVAM_API_KEY=your_sarvam_api_key_here
SARVAM_STT_MODEL=saarika:v2.5
SARVAM_TTS_MODEL=bulbul:v2
SARVAM_TTS_SPEAKER=anushka

# Vector Database (Qdrant)
VECTOR_PROVIDER=qdrant_local
QDRANT_PATH=data/qdrant
QDRANT_COLLECTION_NAME=msmarco_xi
```
