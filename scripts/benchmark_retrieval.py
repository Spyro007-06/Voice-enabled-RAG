"""Retrieval latency benchmarking script profiling stage-by-stage execution times."""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retrieval.service import RetrievalService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark_retrieval")


def benchmark_retrieval(
    num_queries: int = 50,
    top_k: int = 10,
    report_path: str = "benchmarks/retrieval_latency.json",
) -> Dict[str, Any]:
    """Profile dense, BM25, and hybrid fusion stage-by-stage latencies across multilingual queries."""
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 70)
    print("HYBRID RETRIEVAL LATENCY & TELEMETRY BENCHMARK")
    print("=" * 70)

    service = RetrievalService()

    # Multilingual query benchmark set
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

    # Warmup
    print("\n[1/2] Running Warmup Inferences...")
    _ = service.retrieve("warmup query", top_k=top_k, fusion_method="rrf", use_cache=False)
    _ = service.retrieve("warmup query", top_k=top_k, fusion_method="weighted", use_cache=False)

    print(f"\n[2/2] Benchmarking {num_queries} queries across retrieval modes (Cache DISABLED)...")

    results_by_mode: Dict[str, Dict[str, List[float]]] = {
        "dense_only": {"embedding": [], "dense": [], "total": []},
        "bm25_only": {"bm25": [], "total": []},
        "weighted_hybrid": {"embedding": [], "dense": [], "bm25": [], "fusion": [], "total": []},
        "rrf_hybrid": {"embedding": [], "dense": [], "bm25": [], "fusion": [], "total": []},
    }

    for mode in ("dense_only", "bm25_only", "weighted", "rrf"):
        mode_key = "weighted_hybrid" if mode == "weighted" else ("rrf_hybrid" if mode == "rrf" else mode)
        for q in queries:
            _, lat = service.retrieve(
                query=q,
                top_k=top_k,
                fusion_method=mode,
                use_cache=False,
            )
            if mode == "dense_only":
                results_by_mode[mode_key]["embedding"].append(lat.embedding_ms)
                results_by_mode[mode_key]["dense"].append(lat.dense_retrieval_ms)
                results_by_mode[mode_key]["total"].append(lat.total_ms)
            elif mode == "bm25_only":
                results_by_mode[mode_key]["bm25"].append(lat.bm25_retrieval_ms)
                results_by_mode[mode_key]["total"].append(lat.total_ms)
            else:
                results_by_mode[mode_key]["embedding"].append(lat.embedding_ms)
                results_by_mode[mode_key]["dense"].append(lat.dense_retrieval_ms)
                results_by_mode[mode_key]["bm25"].append(lat.bm25_retrieval_ms)
                results_by_mode[mode_key]["fusion"].append(lat.fusion_ms)
                results_by_mode[mode_key]["total"].append(lat.total_ms)

    # Compute percentiles
    def get_stats(series: List[float]) -> Dict[str, float]:
        if not series:
            return {}
        return {
            "mean_ms": round(float(np.mean(series)), 2),
            "p50_ms": round(float(np.percentile(series, 50)), 2),
            "p70_ms": round(float(np.percentile(series, 70)), 2),
            "p95_ms": round(float(np.percentile(series, 95)), 2),
            "p100_ms": round(float(np.percentile(series, 100)), 2),
        }

    report = {
        "num_queries_evaluated": num_queries,
        "top_k": top_k,
        "cache_enabled": False,
        "modes": {
            mode: {stage: get_stats(vals) for stage, vals in data.items()}
            for mode, data in results_by_mode.items()
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("RETRIEVAL LATENCY BENCHMARK SUMMARY (milliseconds)")
    print("=" * 70)
    print(f"{'Mode':<18} | {'P50':<8} | {'P70':<8} | {'P95':<8} | {'P100 (Max)':<10}")
    print("-" * 70)
    for mode, data in report["modes"].items():
        tot = data.get("total", {})
        print(f"{mode:<18} | {tot.get('p50_ms', 0):<8.2f} | {tot.get('p70_ms', 0):<8.2f} | {tot.get('p95_ms', 0):<8.2f} | {tot.get('p100_ms', 0):<10.2f}")
    print("=" * 70)
    print(f"Report saved to: {report_path}")

    return report


if __name__ == "__main__":
    benchmark_retrieval()
