# Phase 6.5.1 — Latency SLA & Quality Calibration Benchmark Report


## Baseline Configurations

| Config | P50 | P95 | P99 | R@1 | R@5 | R@10 | MRR@10 | Skip% |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A. RRF Only | 171.2 | 213.67 | 227.56 | 0.3667 | 0.5333 | 0.6667 | 0.4443 | 100.0% |
| B. MiniLM K=3 | 161.49 | 215.67 | 4980.54 | 0.4 | 0.5333 | 0.6667 | 0.4609 | 0.0% |
| C. MiniLM K=5 | 157.43 | 213.71 | 236.8 | 0.4333 | 0.5333 | 0.6667 | 0.4776 | 0.0% |
| D. Adaptive (0.70/0.40) | 168.71 | 224.41 | 228.36 | 0.4 | 0.5333 | 0.6667 | 0.4609 | 0.0% |

## Per-Query Quality Diagnostic

- Queries where MiniLM K=5 succeeds but Adaptive fails: **0/30**


## Component Latency Profile

| Component | P50 | P95 | P99 |
|:---|:---:|:---:|:---:|
| Embedding | 10.61ms | 11.79ms | 11.98ms |
| Qdrant ANN | 91.41ms | 99.08ms | 104.67ms |
| BM25 Search | 66.57ms | 87.02ms | 90.7ms |
| MiniLM K=5 | 20.02ms | 84.33ms | 88.9ms |

## Recommended Configuration

- `ADAPTIVE_THRESHOLD_HIGH`: **0.65**
- `ADAPTIVE_THRESHOLD_LOW`: **0.3**
- `RERANKER_EARLY_EXIT_MARGIN`: **0.0**
- `PARALLEL_RETRIEVAL_ENABLED`: **True**
- `TOTAL_RETRIEVAL_DEADLINE_MS`: **200**
- `QDRANT_TIMEOUT_MS`: **157.0**
- `BM25_TIMEOUT_MS`: **136.0**
- `RERANKER_TIMEOUT_MS`: **133.0**

## Final Configuration Results

- P50: **154.78ms** | P95: **190.41ms** | P99: **204.19ms**
- R@1: **0.4333** | R@5: **0.5333** | R@10: **0.6667** | MRR@10: **0.4776**
- Skip: **25.0%** | K=3: **75.0%** | K=5: **0.0%**

## Quality Safety Gate

- MiniLM K=5 Reference: R@10=0.6667, MRR@10=0.4776
- Final Config: R@10=0.6667, MRR@10=0.4776
- Gap: R@10=+0.0000, MRR@10=+0.0000
