# Phase 6.6 End-to-End RAG Generation Benchmark Report

**Date:** 2026-08-17T06:35:36Z  
**Total Queries Evaluated:** 37 across 5 languages (EN, HI, TA, TE, ML)  
**Cache Mode:** DISABLED (Strict cold-path evaluation)  

## Summary Latency Table (Milliseconds)

| Stage | Mean (ms) | P50 (ms) | P70 (ms) | P95 (ms) | P100 (Max) | Share of Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Query Embedding | 25.16 | 15.15 | 39.56 | 52.34 | 60.68 | 1.1% |
| Dense/BM25 Retrieval | 109.79 | 80.04 | 131.56 | 207.59 | 225.28 | 4.8% |
| Cross-Encoder Reranking | 2150.02 | 1505.36 | 2417.25 | 4422.71 | 5379.16 | 94.0% |
| Context Selection | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.0% |
| Prompt Construction | 0.07 | 0.07 | 0.08 | 0.13 | 0.16 | 0.0% |
| LLM Generation | 0.23 | 0.2 | 0.24 | 0.42 | 0.53 | 0.0% |
| Guardrails & Grounding | 0.41 | 0.42 | 0.46 | 0.88 | 0.99 | 0.0% |
| **Total End-to-End Pipeline** | 2286.61 | 1595.04 | 2632.9 | 4677.71 | 5644.95 | 100.0% |

## Pipeline Bottleneck Analysis

1. **Embedding & Dense Search**: Contributes **5.9%** of overall runtime.
2. **Adaptive Reranking**: Contributes **94.0%** of total time, executing cross-encoders dynamically when retriever confidence is below certainty margins.
3. **Guardrails & Grounding**: Consumes only **0.41 ms** (0.0%), adding zero noticeable overhead.
4. **Prompt Construction & Formatting**: Consumes **0.07 ms**.
5. **LLM Generation**: In the mock/deterministic provider tier, generation takes **0.23 ms**.

## Target SLA Assessment (200 ms)

- **P50 Latency:** 1595.04 ms
- **P70 Latency:** 2632.9 ms
- **P95 Latency:** 4677.71 ms
- **P100 Latency:** 5644.95 ms
- **200 ms SLA Target Compliance Rate:** 0.00%
- **200 ms Target Achieved?**: NO (Exceeds 200 ms target on cold paths)
