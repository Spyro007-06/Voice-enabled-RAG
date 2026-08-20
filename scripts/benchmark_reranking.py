"""Reranking latency and context budget benchmarking script profiling stage latencies across candidate pool sizes."""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.reranking.bge_reranker import get_reranker
from app.reranking.service import RerankingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark_reranking")


def benchmark_reranking(
    num_queries: int = 50,
    report_path: str = "benchmarks/reranking_latency.json",
) -> Dict[str, Any]:
    """Profile candidate retrieval, cross-encoder inference, and context selection latencies across K=10, 20, 30."""
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 70)
    print("PHASE 5: CROSS-ENCODER RERANKING LATENCY & BUDGET BENCHMARK")
    print("=" * 70)

    # 1. Warmup Reranker Model
    print("\n[1/3] Warming up Cross-Encoder Reranker...")
    reranker = get_reranker()
    t_warmup = time.perf_counter()
    _ = reranker._ensure_model()
    init_ms = (time.perf_counter() - t_warmup) * 1000
    print(f"  - Model loaded: {reranker.model_name} (Device: {reranker.device}) in {init_ms:.2f} ms")

    service = RerankingService(reranker=reranker)

    # Multilingual benchmark queries
    query_set = [
        "कॉर्पोरेशन क्या है?",
        "What is a corporation?",
        "भारत की राजधानी और शासन व्यवस्था",
        "कंपनी और निगम के बीच अंतर",
        "व्यापार और वित्तीय बाजार",
        "Government structure and legal framework",
        "अर्थव्यवस्था और विकास योजना",
        "How to collect followers on wow",
    ]
    queries = [query_set[i % len(query_set)] for i in range(num_queries)]

    # Initial inference warmup
    _ = service.rerank_and_select("warmup query", top_k=5, candidate_k=10)

    print(f"\n[2/3] Benchmarking {num_queries} queries across Candidate K pools (10, 20, 30)...")

    candidate_k_values = [10, 20, 30]
    benchmark_results: Dict[str, Any] = {}

    for cand_k in candidate_k_values:
        print(f"\n  Running Candidate K = {cand_k}...")
        retrieval_times = []
        reranking_times = []
        context_times = []
        total_times = []
        context_chars = []
        selected_counts = []

        for q in queries:
            results, stats, lat = service.rerank_and_select(
                query=q,
                top_k=5,
                candidate_k=cand_k,
                retrieval_mode="rrf_hybrid",
            )
            retrieval_times.append(lat.retrieval_ms)
            reranking_times.append(lat.reranking_ms)
            context_times.append(lat.context_selection_ms)
            total_times.append(lat.total_ms)
            context_chars.append(stats.total_characters)
            selected_counts.append(stats.selected_chunks)

        def calc_stats(series: List[float]) -> Dict[str, float]:
            return {
                "mean_ms": round(float(np.mean(series)), 2),
                "p50_ms": round(float(np.percentile(series, 50)), 2),
                "p70_ms": round(float(np.percentile(series, 70)), 2),
                "p95_ms": round(float(np.percentile(series, 95)), 2),
                "p100_ms": round(float(np.percentile(series, 100)), 2),
            }

        benchmark_results[f"candidate_k_{cand_k}"] = {
            "candidate_k": cand_k,
            "final_k": 5,
            "retrieval_latency": calc_stats(retrieval_times),
            "reranking_latency": calc_stats(reranking_times),
            "context_selection_latency": calc_stats(context_times),
            "total_latency": calc_stats(total_times),
            "context_stats": {
                "avg_characters": round(float(np.mean(context_chars)), 1),
                "p95_characters": round(float(np.percentile(context_chars, 95)), 1),
                "max_characters": int(np.max(context_chars)),
                "avg_chunks_selected": round(float(np.mean(selected_counts)), 1),
            },
        }

    report = {
        "model_name": reranker.model_name,
        "device": reranker.device,
        "model_initialization_ms": round(init_ms, 2),
        "num_queries_evaluated": num_queries,
        "evaluations": benchmark_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("RERANKING LATENCY & BUDGET PROFILE SUMMARY")
    print("=" * 70)
    print(f"{'Candidate Pool':<16} | {'Retrieval P50':<14} | {'Rerank P50':<12} | {'Combined P50':<14} | {'Combined P100':<14}")
    print("-" * 70)
    for key, data in benchmark_results.items():
        ret_p50 = data["retrieval_latency"]["p50_ms"]
        rerank_p50 = data["reranking_latency"]["p50_ms"]
        tot_p50 = data["total_latency"]["p50_ms"]
        tot_p100 = data["total_latency"]["p100_ms"]
        print(f"{key:<16} | {ret_p50:<14.2f} | {rerank_p50:<12.2f} | {tot_p50:<14.2f} | {tot_p100:<14.2f}")
    print("=" * 70)
    print(f"Report saved to: {report_path}")

    return report


if __name__ == "__main__":
    benchmark_reranking()
