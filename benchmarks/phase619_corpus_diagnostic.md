# Phase 6.19 — Corpus & Ingestion Diagnostic Report

**Project:** HH Goa 2026 Multilingual Voice RAG System  
**Date:** 2026-08-18  
**Objective:** Diagnose and document the exact root cause of the language imbalance and define the canonical multilingual ingestion architecture.

---

## 1. Executive Summary & Exact Root Cause

### Root Cause Analysis
1. **Initial Dataset Ingestion Target:**
   In early development phases (Phase 2 & Phase 3), `app/config.py` configured:
   ```python
   DATASET_LANGUAGE: str = "hi"
   DATASET_SPLIT: str = "validation"
   ```
   This led the offline ingestion pipeline (`scripts/build_chunks.py` & `scripts/build_index.py`) to download only `data/raw/hinval.parquet` (validation split for Hindi).

2. **Passage Extraction Behavior:**
   In `app/ingestion/normalizer.py`, `DocumentNormalizer.normalize_record()` extracts:
   ```python
   target_lang = record.get("target_lang")  # "hin_Deva"
   ...
   document = Document(
       document_id=doc_id,
       text=normalized_text,  # Preferred translated_passages (Hindi)
       language=target_lang or source_lang,  # Stored as "hin_Deva"
       source="ai4bharat/MSMARCO-XI",
       metadata=meta,  # Stored eng_passage in metadata
   )
   ```
   Thus, all generated chunks from the 5 chunking strategies received `language="hin_Deva"`.

3. **Phase 6.16 Additive Extraction:**
   In Phase 6.16, `scripts/generate_english_chunks.py` extracted the `eng_passage` metadata from the existing Hindi records, producing 4,985 English chunks (`language="eng_Latn"`), bringing total indexed points to:
   - Hindi (`hin_Deva`): 8,467 points
   - English (`eng_Latn`): 4,985 points
   - Total: 13,452 points

4. **Missing Languages:**
   Tamil (`tam_Taml`), Telugu (`tel_Telu`), and Malayalam (`mal_Mlym`) were never downloaded into `data/raw/` or ingested through the chunking pipeline. Consequently, strict language filters for `ta`, `te`, and `ml` return 0 candidates, while cross-lingual retrieval (`language=None`) falls back to Hindi and English documents.

---

## 2. Dataset & Index Diagnostic Metrics

| Language | ISO Code | Canonical BCP-47 | Source Parquet | Available Raw Records | Current Indexed Chunks | Target Chunks (Phase 6.19) |
|---|---|---|---|---|---|---|
| **English** | `en` | `eng_Latn` | `hinval.parquet` (`English_passages`) | 97,941 | 4,985 | ~5,000 |
| **Hindi** | `hi` | `hin_Deva` | `hinval.parquet` (`Translated_passages`) | 97,941 | 8,467 | ~8,467 |
| **Tamil** | `ta` | `tam_Taml` | `tamval.parquet` (`Translated_passages`) | 97,941 | 0 | ~5,000 |
| **Telugu** | `te` | `tel_Telu` | `telval.parquet` (`Translated_passages`) | 97,941 | 0 | ~5,000 |
| **Malayalam** | `ml` | `mal_Mlym` | `malval.parquet` (`Translated_passages`) | 97,941 | 0 | ~5,000 |

---

## 3. Canonical Language Mapping & Normalization Architecture

To eliminate any ambiguity across ingestion, chunking, indexing, and retrieval, the central language resolver is formalized as:

```python
CANONICAL_LANGUAGE_MAP = {
    # English
    "en": "eng_Latn", "eng": "eng_Latn", "eng_latn": "eng_Latn", "eng_Latn": "eng_Latn", "english": "eng_Latn",
    # Hindi
    "hi": "hin_Deva", "hin": "hin_Deva", "hin_deva": "hin_Deva", "hin_Deva": "hin_Deva", "hindi": "hin_Deva",
    # Tamil
    "ta": "tam_Taml", "tam": "tam_Taml", "tam_taml": "tam_Taml", "tam_Taml": "tam_Taml", "tamil": "tam_Taml",
    # Telugu
    "te": "tel_Telu", "tel": "tel_Telu", "tel_telu": "tel_Telu", "tel_Telu": "tel_Telu", "telugu": "tel_Telu",
    # Malayalam
    "ml": "mal_Mlym", "mal": "mal_Mlym", "mal_mlym": "mal_Mlym", "mal_Mlym": "mal_Mlym", "malayalam": "mal_Mlym",
}
```

---

## 4. Phase 6.19 Ingestion & Rebuild Strategy

1. **Ingest Tamil, Telugu, and Malayalam Parquets:**
   Download and sample 100 queries each from `tamval.parquet`, `telval.parquet`, `malval.parquet`.
2. **Chunk Generation:**
   Process each language through the 5 validated chunking strategies (Fixed, Sentence, Sliding Window, Semantic, Hierarchical) to generate `chunks_tamil.jsonl`, `chunks_telugu.jsonl`, and `chunks_malayalam.jsonl`.
3. **Safe Qdrant Rebuild (`msmarco_xi_v619`):**
   Create new collection, upsert all 5 language chunk sets with deterministic UUIDs and 384-dim normalized E5 embeddings, validate recall, and point active collection alias to the new index.
4. **Unified BM25 Rebuild:**
   Index all 5 language chunk files using `IndicUnicodeTokenizer` with word and char n-grams.
5. **Quality & SLA Validation:**
   Execute 25 factual query matrix (5 queries × 5 languages) to verify strict filtering, citations, grounding, and sub-200ms retrieval SLA.
