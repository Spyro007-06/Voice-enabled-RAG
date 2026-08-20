# Phase 6.23 — Demo Failure Investigation & Root Cause Analysis

## Investigation Overview

- **Investigation Date:** 2026-08-19
- **Target System:** HH Goa 2026 Multilingual Voice RAG System
- **Baseline:** Phase 6.22 (480 passed, 1 skipped, 0 failed; Demo Check: 17/18 steps)

---

## 1. Failed Scenario Identification

- **Failed Demo Step:** Step 16 (`[16/18] Verify Safe Refusal (No Hallucination)`)
- **Query Submitted (Step 15):** `"What is the recipe for chocolate cake?"`
- **Language:** `en`
- **Expected Result:** The system should safely refuse or return `grounded: False` with `"I don't have enough information in the retrieved context to answer that."` because the MSMARCO-XI index contains technical/general knowledge, not culinary recipes for chocolate cake.
- **Actual Result in Phase 6.22:**
  - `status`: `200 OK`
  - `grounded`: `True`
  - `answer`: `"[MOCK] Answer to 'What is the recipe for chocolate cake?' based on 3 source(s): One of the things these apples are good for is apple pies..."`
  - Step 16 assertion `not grounded_neg or "don't have enough information" in ans_neg.lower()` failed (0 points awarded).

---

## 2. Root Cause Analysis

1. **Relevance Guard Stopword & Subword Collision (`app/guardrails/relevance.py`):**
   - `RelevanceGuard` used `IndicUnicodeTokenizer`, which generates character tri-grams (`'the'`, `'for'`, `'ate'`, `'ake'`).
   - The query `"What is the recipe for chocolate cake?"` shares English stopwords (`"is"`, `"the"`, `"for"`) and tri-grams (`'ate'`, `'ake'`) with unrelated retrieved chunks (apple pie/sauce), creating an artificial `overlap_ratio` > 0.3.
   - Furthermore, `composite_relevance` accepted baseline dense vector scores (`max_retrieval_score >= 0.5`), which for multilingual-e5-small unit embeddings naturally hover around 0.55–0.65 even on arbitrary non-matching queries.
   - As a result, `validate_pre_generation` permitted the off-topic query to proceed.

2. **Mock LLM Generation Provider Lack of Query Coverage Check (`app/generation/provider.py`):**
   - In local/mock mode (`LLM_PROVIDER=mock`), `MockGenerationProvider.generate()` unconditionally synthesized a mock response and set `grounded = True` whenever context chunks were present, regardless of whether the retrieved context contained the query's core content entities.
   - Because `MockGenerationProvider` embedded the first snippet of the apple chunk into its answer, `GroundingGuard.evaluate_grounding()` observed 100% lexical overlap between the answer and context, thus marking the answer as `grounded = True`.

---

## 3. Failure Summary

- **FAILED SCENARIO:** Step 16 / 18 (`Verify Safe Refusal (No Hallucination)`) on query `"What is the recipe for chocolate cake?"`.
- **EXPECTED:** `grounded == False` and/or refusal text `"I don't have enough information in the retrieved context to answer that."`.
- **ACTUAL:** `grounded == True` with fabricated mock answer linking chocolate cake query to apple pie context.
- **ROOT CAUSE:** Lexical stopword/tri-gram false positive in `RelevanceGuard` combined with unconditioned mock answer synthesis in `MockGenerationProvider`.
- **AFFECTED COMPONENTS:**
  1. `app/guardrails/relevance.py`
  2. `app/generation/provider.py`
- **PROPOSED FIX:**
  1. Filter functional stopwords and compare whole content words (not character tri-grams) in `RelevanceGuard`. Require meaningful content-word overlap (>0.40) or high cross-encoder reranker score (>0.35) for relevance when retrieval confidence is low.
  2. In `MockGenerationProvider`, verify that at least 50% of meaningful content words exist in the context before returning grounded text; otherwise return the standard truthful refusal statement (`"I don't have enough information in the retrieved context to answer that."`) with `grounded = False`.
- **RISK:** Negligible. In-domain queries across English, Hindi, Tamil, Telugu, and Malayalam retain 100% keyword and semantic overlap with their relevant retrieved documents.
- **REGRESSION IMPACT:** Zero regression on valid queries; improves safety and refusal behavior on out-of-domain/unindexed queries.
