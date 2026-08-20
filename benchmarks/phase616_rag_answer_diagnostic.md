# Phase 6.16 RAG Answer Pipeline Diagnostic Report

**System:** HH Goa 2026 Multilingual Voice RAG
**Execution Timestamp:** 2026-08-18 16:26:28 UTC

## 1. Executive Summary & Root Cause

```
FINAL STATUS:
RAG ANSWER PIPELINE: DATASET / LANGUAGE-FILTER MISMATCH & DATASET COVERAGE

ROOT CAUSE:
1. Metadata Filter Mismatch (Code F/G): The MSMARCO-XI indexed collection contains chunks tagged with language='hin_Deva'. When frontend / API sends language='en', the strict Qdrant/BM25 language filter requires language='en', yielding 0 results.
2. Dataset Boundary (Code A): The sample dataset (hinval.parquet) contains specific passages (e.g. corporations, Rachel Carson, genetics, French translations) and lacks full encyclopedic articles for 'What is a computer?' or 'What is natural language processing?'.
3. Guardrail Behavior: Pre-generation guardrail correctly refuses empty/unrelated context to prevent hallucination.
```

## 2. Qdrant & BM25 Index Status

- **Qdrant Collection:** `msmarco_xi` (Points: `8467`, Dim: `384`)
- **Indexed Languages:** `['hin_Deva']`
- **BM25 Documents:** `9355`
- **BM25 Vocab Size:** `15866`

## 3. Query Traces Across 4 Diagnostic Queries

| Query | `language='en'` Chunks | `language='en'` Refusal | `language=None` Chunks | `language=None` Grounded |
|---|:---:|:---:|:---:|:---:|
| **What is a computer?** | `0` | `Yes (Refusal)` | `5` | `True` |
| **What is machine learning?** | `0` | `Yes (Refusal)` | `5` | `True` |
| **What is artificial intelligence?** | `0` | `Yes (Refusal)` | `5` | `True` |
| **What is the internet?** | `0` | `Yes (Refusal)` | `5` | `True` |

## 4. Pipeline Trace per Stage for 'What is a computer?'

- **Query:** `What is a computer?`
- **Query Validation:** PASS
- **Language Resolution:** `en` resolved to `['en', 'eng', 'eng_Latn']`
- **Embedding Generation:** PASS (384-d, latency ~9959.29ms)
- **Dense Retrieval (with `language='en'`):** 0 chunks (no chunks in Qdrant with `language='eng_Latn'`)
- **Dense Retrieval (with `language=None`):** 5 chunks retrieved cross-lingually
- **BM25 Retrieval (with `language='en'`):** 0 chunks
- **RRF Fusion:** 0 chunks with filter, 5 chunks without filter
- **Guardrail Evaluation:** `allowed=False`, `reason='empty_retrieval_context'`
- **LLM Generation:** `called=False`
- **Frontend Mapping:** PASS (maps `answer`, `grounded`, `confidence`, `citations`, `latency_ms`)
