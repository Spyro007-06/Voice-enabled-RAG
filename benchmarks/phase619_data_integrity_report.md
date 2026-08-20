# Phase 6.19 — Multilingual Data & Index Integrity Report

**Project:** HH Goa 2026 Multilingual Voice RAG System  
**Date:** 2026-08-18  
**Integrity Validation Status:** **PASSED**

---

## 1. Chunk File Integrity

| Language | Canonical Code | Chunk File | Chunks Generated | Empty Chunks | Missing Metadata |
|---|---|---|---|---|---|
| Hindi | `hin_Deva` | `data/processed/chunks_hindi.jsonl` | 9355 | 0 | 0 |
| English | `eng_Latn` | `data/processed/chunks_english.jsonl` | 9858 | 0 | 0 |
| Tamil | `tam_Taml` | `data/processed/chunks_tamil.jsonl` | 9910 | 0 | 0 |
| Telugu | `tel_Telu` | `data/processed/chunks_telugu.jsonl` | 9168 | 0 | 0 |
| Malayalam | `mal_Mlym` | `data/processed/chunks_malayalam.jsonl` | 9915 | 0 | 0 |

- **Total Chunks Across All Files:** 48206
- **Unique Deterministic Chunk IDs:** 48206
- **Duplicate IDs Detected:** 0
- **Malformed Unicode Errors:** 0

---

## 2. BM25 Lexical Index Integrity

- **Status:** Loaded (`data/bm25_index.pkl`)
- **Total Indexed Documents:** 48206
- **Unique Document IDs:** 48206
- **Metadata Consistency:** 100% aligned with chunk IDs

---

## 3. Qdrant Vector Store Integrity

- **Collection:** `msmarco_xi`
- **Vector Dimension:** 384
- **Metric Distance:** COSINE
- **Vectors Upserted:** 28541
- **NaN / Inf Embeddings:** 0 detected
