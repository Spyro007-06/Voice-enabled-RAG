"""Retrieval quality evaluation script measuring Recall@K and MRR@10 on MSMARCO-XI ground truth."""

import csv
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Set, Tuple

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings
from app.ingestion.dataset_loader import DatasetLoader
from app.retrieval.evaluation import RetrievalEvaluator
from app.retrieval.service import RetrievalService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate_retrieval")


def load_eval_queries(limit: int = 100) -> List[Tuple[str, int, Set[str], str]]:
    """Load query records with ground-truth relevant document IDs."""
    settings = get_settings()
    loader = DatasetLoader(
        dataset_name=settings.DATASET_NAME,
        split=settings.DATASET_SPLIT,
        language=settings.DATASET_LANGUAGE,
        sample_size=limit,
    )
    records = list(loader.load_records(limit=limit))

    eval_items: List[Tuple[str, int, Set[str], str]] = []
    for rec in records:
        query_id = rec.get("query_id")
        query_text = rec.get("query", "").strip()
        target_lang = rec.get("target_lang", "hin_Deva")
        passages_data = rec.get("passages", {}) or {}
        is_selected = passages_data.get("is_selected", []) or []

        if not query_text or query_id is None:
            continue

        relevant_docs = {
            f"msmarco_{query_id}_{idx}"
            for idx, sel in enumerate(is_selected)
            if sel == 1
        }

        # Keep queries that have at least one relevant passage
        if relevant_docs:
            eval_items.append((query_text, query_id, relevant_docs, target_lang))

    return eval_items


def evaluate_retrieval(
    sample_size: int = 100,
    json_report_path: str = "benchmarks/retrieval_quality.json",
    csv_report_path: str = "benchmarks/retrieval_quality.csv",
) -> Dict[str, Any]:
    """Run retrieval quality evaluation across chunk strategies and fusion methods."""
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 70)
    print(f"MSMARCO-XI RETRIEVAL QUALITY EVALUATION (Queries: {sample_size})")
    print("=" * 70)

    eval_queries = load_eval_queries(limit=sample_size)
    print(f"Loaded {len(eval_queries)} evaluation queries with ground-truth relevant passages.")

    if not eval_queries:
        print("[ERROR] No valid evaluation queries loaded.")
        return {}

    service = RetrievalService()
    evaluator = RetrievalEvaluator()

    # 1. Strategy Comparison Matrix (Using RRF fusion)
    print("\n[1/2] Evaluating Retrieval by Chunking Strategy...")
    strategies_to_test = ["fixed", "sentence", "sliding_window", "semantic", "hierarchical", "all"]
    strategy_metrics: Dict[str, Dict[str, float]] = {}

    for strat in strategies_to_test:
        evaluator.reset()
        strat_arg = None if strat == "all" else [strat]

        for query_text, _, relevant_docs, _ in eval_queries:
            results, lat = service.retrieve(
                query=query_text,
                top_k=10,
                strategies=strat_arg,
                fusion_method="rrf",
                use_cache=False,
            )
            evaluator.evaluate_query(
                retrieved_results=results,
                relevant_doc_ids=relevant_docs,
                latency_ms=lat.total_ms,
            )

        summary = evaluator.get_summary()
        strategy_metrics[strat] = summary
        print(
            f"  - Strategy '{strat:<14}': Recall@1={summary['recall@1']:.3f} | "
            f"Recall@5={summary['recall@5']:.3f} | Recall@10={summary['recall@10']:.3f} | "
            f"MRR@10={summary['mrr@10']:.3f} | Avg Latency={summary['avg_latency_ms']:.1f}ms"
        )

    # 2. Retriever & Fusion Comparison Matrix (All strategies active)
    print("\n[2/2] Evaluating Retrievers & Fusion Methods...")
    fusion_methods = ["dense_only", "bm25_only", "weighted", "rrf"]
    fusion_metrics: Dict[str, Dict[str, float]] = {}

    for f_method in fusion_methods:
        evaluator.reset()
        method_label = "weighted_hybrid" if f_method == "weighted" else ("rrf_hybrid" if f_method == "rrf" else f_method)

        for query_text, _, relevant_docs, _ in eval_queries:
            results, lat = service.retrieve(
                query=query_text,
                top_k=10,
                strategies=None,
                fusion_method=f_method,
                use_cache=False,
            )
            evaluator.evaluate_query(
                retrieved_results=results,
                relevant_doc_ids=relevant_docs,
                latency_ms=lat.total_ms,
            )

        summary = evaluator.get_summary()
        fusion_metrics[method_label] = summary
        print(
            f"  - Retriever '{method_label:<15}': Recall@1={summary['recall@1']:.3f} | "
            f"Recall@5={summary['recall@5']:.3f} | Recall@10={summary['recall@10']:.3f} | "
            f"MRR@10={summary['mrr@10']:.3f} | Avg Latency={summary['avg_latency_ms']:.1f}ms"
        )

    # 3. Save JSON and CSV Reports
    os.makedirs(os.path.dirname(json_report_path), exist_ok=True)
    report = {
        "evaluation_dataset": "ai4bharat/MSMARCO-XI",
        "queries_evaluated": len(eval_queries),
        "chunking_strategy_comparison": strategy_metrics,
        "fusion_method_comparison": fusion_metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(json_report_path, "w", encoding="utf-8") as f_json:
        json.dump(report, f_json, indent=2)

    # Save CSV Report
    csv_rows = []
    for strat, data in strategy_metrics.items():
        csv_rows.append({
            "Category": "Strategy",
            "Name": strat,
            "Recall@1": data["recall@1"],
            "Recall@5": data["recall@5"],
            "Recall@10": data["recall@10"],
            "MRR@10": data["mrr@10"],
            "Avg Latency (ms)": data["avg_latency_ms"],
        })
    for method, data in fusion_metrics.items():
        csv_rows.append({
            "Category": "Fusion/Retriever",
            "Name": method,
            "Recall@1": data["recall@1"],
            "Recall@5": data["recall@5"],
            "Recall@10": data["recall@10"],
            "MRR@10": data["mrr@10"],
            "Avg Latency (ms)": data["avg_latency_ms"],
        })

    with open(csv_report_path, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n" + "=" * 70)
    print("REPORTS SAVED SUCCESSFULLY")
    print("=" * 70)
    print(f"JSON Report: {json_report_path}")
    print(f"CSV Report : {csv_report_path}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    sample = 100
    if len(sys.argv) > 1:
        try:
            sample = int(sys.argv[1])
        except ValueError:
            pass
    evaluate_retrieval(sample_size=sample)
