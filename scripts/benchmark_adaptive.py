"""Comprehensive latency and quality benchmark for Adaptive Retrieval and Latency Optimization (Phase 5.5)."""

import csv
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Set, Tuple
import numpy as np

# Ensure app root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings
from app.ingestion.dataset_loader import DatasetLoader
from app.reranking.adaptive import (
    AdaptiveRetrievalService,
    calculate_retrieval_confidence,
    get_adaptive_retrieval_service,
)
from app.reranking.bge_reranker import get_reranker
from app.reranking.context_selector import ContextSelector
from app.reranking.lightweight_reranker import get_lightweight_reranker
from app.reranking.models import RerankResult
from app.reranking.service import RerankingService
from app.retrieval.evaluation import compute_mrr_at_k, compute_recall_at_k
from app.retrieval.models import RetrievalResult
from app.retrieval.service import get_retrieval_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("benchmark_adaptive")


def calculate_percentiles(times_ms: List[float]) -> Dict[str, float]:
    """Calculate mean, P50, P70, P95, P100 from millisecond timings."""
    if not times_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p70_ms": 0.0, "p95_ms": 0.0, "p100_ms": 0.0}
    arr = np.array(times_ms)
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p70_ms": round(float(np.percentile(arr, 70)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p100_ms": round(float(np.max(arr)), 2),
    }


def load_eval_queries(limit: int = 100) -> List[Tuple[str, int, Set[str]]]:
    """Load query records with ground-truth relevant document IDs."""
    loader = DatasetLoader(
        dataset_name="ai4bharat/MSMARCO-XI",
        split="validation",
        language="hi",
        sample_size=limit,
    )
    records = list(loader.load_records(limit=limit))

    eval_items: List[Tuple[str, int, Set[str]]] = []
    for rec in records:
        query_id = rec.get("query_id")
        query_text = rec.get("query", "").strip()
        passages_data = rec.get("passages", {}) or {}
        is_selected = passages_data.get("is_selected", []) or []

        if not query_text or query_id is None:
            continue

        relevant_docs = {
            f"msmarco_{query_id}_{idx}"
            for idx, sel in enumerate(is_selected)
            if sel == 1
        }

        if relevant_docs:
            eval_items.append((query_text, query_id, relevant_docs))

    return eval_items


def compute_metrics(retrieved_items: List[Any], relevant_docs: Set[str]) -> Dict[str, float]:
    """Compute Recall@1, Recall@5, Recall@10, and MRR@10 for ground-truth relevance."""
    if not relevant_docs:
        return {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr@10": 0.0}

    def is_rel(item) -> bool:
        doc_id = getattr(item, "document_id", "")
        meta = getattr(item, "metadata", {}) or {}
        return (
            doc_id in relevant_docs
            or meta.get("is_selected") == 1
            or meta.get("is_selected") == "1"
        )

    r1 = 1.0 if any(is_rel(item) for item in retrieved_items[:1]) else 0.0
    r5 = 1.0 if any(is_rel(item) for item in retrieved_items[:5]) else 0.0
    r10 = 1.0 if any(is_rel(item) for item in retrieved_items[:10]) else 0.0

    mrr = 0.0
    for rank_idx, item in enumerate(retrieved_items[:10], start=1):
        if is_rel(item):
            mrr = 1.0 / rank_idx
            break

    return {"recall@1": r1, "recall@5": r5, "recall@10": r10, "mrr@10": mrr}


def run_adaptive_benchmark(eval_query_count: int = 20):
    """Execute rigorous benchmark comparing Dense, RRF, BGE, Lightweight, and Adaptive approaches."""
    settings = get_settings()
    logger.info("Initializing services and pre-warming models...")

    ret_service = get_retrieval_service()
    bge_reranker = get_reranker()
    lightweight_reranker = get_lightweight_reranker()
    context_selector = ContextSelector(final_top_k=5, max_chunks_per_doc=2, max_context_chars=6000)
    adaptive_service = get_adaptive_retrieval_service()

    # Pre-warm models (separated from benchmark measurements)
    logger.info("Pre-warming models...")
    _ = bge_reranker.rerank("warmup query", [
        RetrievalResult(chunk_id="w1", document_id="d1", text="Warmup passage 1", chunk_type="fixed", dense_score=0.8, rank=1)
    ])
    _ = lightweight_reranker.rerank("warmup query", [
        RetrievalResult(chunk_id="w1", document_id="d1", text="Warmup passage 1", chunk_type="fixed", dense_score=0.8, rank=1)
    ])

    # Load MSMARCO-XI Ground Truth Queries
    eval_items = load_eval_queries(limit=eval_query_count * 2)
    test_items = eval_items[:eval_query_count]
    logger.info("Loaded %d evaluation queries with ground truth.", len(test_items))

    # Define Configurations to Benchmark
    configs = [
        "dense",
        "rrf",
        "bge_k3",
        "bge_k5",
        "bge_k10",
        "lightweight_k3",
        "lightweight_k5",
        "adaptive",
    ]

    results_data: Dict[str, Any] = {
        "hardware": "Intel 20-Core (Intel64 Family 6 Model 186)",
        "eval_queries_count": len(test_items),
        "configurations": {},
    }

    print("\n" + "=" * 80)
    print("STARTING PHASE 5.5 ADAPTIVE RETRIEVAL & LATENCY OPTIMIZATION BENCHMARK")
    print("=" * 80)

    for cfg_name in configs:
        logger.info("Benchmarking configuration: %s ...", cfg_name)
        timings: List[float] = []
        metrics_list: List[Dict[str, float]] = []

        for q_text, q_id, rel_docs in test_items:
            t0 = time.perf_counter()

            if cfg_name == "dense":
                res, _, _ = ret_service.dense_retriever.retrieve(query=q_text, top_k=5)
                t_elapsed = (time.perf_counter() - t0) * 1000
                timings.append(t_elapsed)
                metrics_list.append(compute_metrics(res, rel_docs))

            elif cfg_name == "rrf":
                d_res, _, _ = ret_service.dense_retriever.retrieve(query=q_text, top_k=10)
                b_res, _ = ret_service.bm25_retriever.retrieve(query=q_text, top_k=10)
                f_res = ret_service.fusion.fuse_results(d_res, b_res, top_k=5, method="rrf")
                t_elapsed = (time.perf_counter() - t0) * 1000
                timings.append(t_elapsed)
                metrics_list.append(compute_metrics(f_res, rel_docs))

            elif cfg_name.startswith("bge_"):
                k_val = int(cfg_name.split("_k")[1])
                d_res, _, _ = ret_service.dense_retriever.retrieve(query=q_text, top_k=k_val)
                b_res, _ = ret_service.bm25_retriever.retrieve(query=q_text, top_k=k_val)
                f_res = ret_service.fusion.fuse_results(d_res, b_res, top_k=k_val, method="rrf")
                reranked, _ = bge_reranker.rerank(query=q_text, candidates=f_res)
                sel, _, _ = context_selector.select_context(reranked, top_k=5)
                t_elapsed = (time.perf_counter() - t0) * 1000
                timings.append(t_elapsed)
                metrics_list.append(compute_metrics(sel, rel_docs))

            elif cfg_name.startswith("lightweight_"):
                k_val = int(cfg_name.split("_k")[1])
                d_res, _, _ = ret_service.dense_retriever.retrieve(query=q_text, top_k=k_val)
                b_res, _ = ret_service.bm25_retriever.retrieve(query=q_text, top_k=k_val)
                f_res = ret_service.fusion.fuse_results(d_res, b_res, top_k=k_val, method="rrf")
                reranked, _ = lightweight_reranker.rerank(query=q_text, candidates=f_res)
                sel, _, _ = context_selector.select_context(reranked, top_k=5)
                t_elapsed = (time.perf_counter() - t0) * 1000
                timings.append(t_elapsed)
                metrics_list.append(compute_metrics(sel, rel_docs))

            elif cfg_name == "adaptive":
                sel, decision, _ = adaptive_service.adaptive_retrieve(query=q_text, top_k=5)
                t_elapsed = (time.perf_counter() - t0) * 1000
                timings.append(t_elapsed)
                metrics_list.append(compute_metrics(sel, rel_docs))

        pcts = calculate_percentiles(timings)
        avg_r1 = round(float(np.mean([m["recall@1"] for m in metrics_list])), 3)
        avg_r5 = round(float(np.mean([m["recall@5"] for m in metrics_list])), 3)
        avg_r10 = round(float(np.mean([m["recall@10"] for m in metrics_list])), 3)
        avg_mrr = round(float(np.mean([m["mrr@10"] for m in metrics_list])), 3)

        results_data["configurations"][cfg_name] = {
            "recall@1": avg_r1,
            "recall@5": avg_r5,
            "recall@10": avg_r10,
            "mrr@10": avg_mrr,
            "latency": pcts,
        }

        logger.info(
            "  [%s] Recall@10: %.3f | MRR@10: %.3f | P50: %.2f ms | P95: %.2f ms | P100: %.2f ms",
            cfg_name.ljust(15), avg_r10, avg_mrr, pcts["p50_ms"], pcts["p95_ms"], pcts["p100_ms"]
        )

    # Save JSON Report
    os.makedirs("benchmarks", exist_ok=True)
    json_path = "benchmarks/adaptive_retrieval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    logger.info("Saved benchmark JSON to %s", json_path)

    # Save CSV Report
    csv_path = "benchmarks/adaptive_retrieval.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Configuration",
            "Recall@1",
            "Recall@5",
            "Recall@10",
            "MRR@10",
            "P50 (ms)",
            "P70 (ms)",
            "P95 (ms)",
            "P100 (ms)",
        ])
        for name, data in results_data["configurations"].items():
            lat = data["latency"]
            writer.writerow([
                name,
                data["recall@1"],
                data["recall@5"],
                data["recall@10"],
                data["mrr@10"],
                lat["p50_ms"],
                lat["p70_ms"],
                lat["p95_ms"],
                lat["p100_ms"],
            ])
    logger.info("Saved benchmark CSV to %s", csv_path)

    # Print Formatted Table
    print("\n" + "=" * 90)
    print(f"{'Configuration':<18} | {'R@1':<5} | {'R@5':<5} | {'R@10':<5} | {'MRR@10':<6} | {'P50 (ms)':<8} | {'P70 (ms)':<8} | {'P95 (ms)':<8} | {'P100 (ms)':<9}")
    print("-" * 90)
    for name, data in results_data["configurations"].items():
        lat = data["latency"]
        print(
            f"{name:<18} | {data['recall@1']:<5.3f} | {data['recall@5']:<5.3f} | {data['recall@10']:<5.3f} | {data['mrr@10']:<6.3f} | "
            f"{lat['p50_ms']:<8.2f} | {lat['p70_ms']:<8.2f} | {lat['p95_ms']:<8.2f} | {lat['p100_ms']:<9.2f}"
        )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_adaptive_benchmark(eval_query_count=count)
