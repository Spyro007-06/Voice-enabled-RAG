"""Evaluation script measuring Recall@K and MRR@10 before and after cross-encoder reranking."""

import csv
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Set, Tuple

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.dataset_loader import DatasetLoader
from app.reranking.service import RerankingService
from app.retrieval.evaluation import compute_mrr_at_k, compute_recall_at_k
from app.retrieval.service import RetrievalService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate_reranking")


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


def evaluate_reranking(
    sample_size: int = 50,
    json_path: str = "benchmarks/reranking_quality.json",
    csv_path: str = "benchmarks/reranking_quality.csv",
) -> Dict[str, Any]:
    """Compare retrieval accuracy before and after cross-encoder reranking."""
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 70)
    print(f"MSMARCO-XI RERANKING QUALITY EVALUATION (Queries: {sample_size})")
    print("=" * 70)

    eval_queries = load_eval_queries(limit=sample_size)
    print(f"Loaded {len(eval_queries)} ground-truth evaluation queries.")

    retrieval_service = RetrievalService()
    rerank_service = RerankingService()

    comparisons: Dict[str, Dict[str, Any]] = {}
    csv_rows = []

    def run_eval_pipeline(mode_name: str, strat_filter: Any = None):
        # 1. Base Retrieval
        r1_list_base, r5_list_base, r10_list_base, mrr_list_base = [], [], [], []
        # 2. Reranked
        r1_list_rerank, r5_list_rerank, r10_list_rerank, mrr_list_rerank = [], [], [], []

        for q_text, _, relevant_docs in eval_queries:
            # Base retrieval
            fusion_arg = "rrf" if "rrf" in mode_name else ("weighted" if "weighted" in mode_name else mode_name)
            base_results, _ = retrieval_service.retrieve(
                query=q_text,
                top_k=10,
                strategies=strat_filter,
                fusion_method=fusion_arg,
                use_cache=False,
            )
            r1_list_base.append(compute_recall_at_k(base_results, relevant_docs, k=1))
            r5_list_base.append(compute_recall_at_k(base_results, relevant_docs, k=5))
            r10_list_base.append(compute_recall_at_k(base_results, relevant_docs, k=10))
            mrr_list_base.append(compute_mrr_at_k(base_results, relevant_docs, k=10))

            # Reranked retrieval
            rerank_results, _, _ = rerank_service.rerank_and_select(
                query=q_text,
                top_k=10,
                candidate_k=20,
                strategies=strat_filter,
                retrieval_mode=mode_name,
            )
            r1_list_rerank.append(compute_recall_at_k(rerank_results, relevant_docs, k=1))
            r5_list_rerank.append(compute_recall_at_k(rerank_results, relevant_docs, k=5))
            r10_list_rerank.append(compute_recall_at_k(rerank_results, relevant_docs, k=10))
            mrr_list_rerank.append(compute_mrr_at_k(rerank_results, relevant_docs, k=10))

        n = len(eval_queries)
        summary = {
            "before_rerank": {
                "recall@1": round(sum(r1_list_base) / n, 3),
                "recall@5": round(sum(r5_list_base) / n, 3),
                "recall@10": round(sum(r10_list_base) / n, 3),
                "mrr@10": round(sum(mrr_list_base) / n, 3),
            },
            "after_rerank": {
                "recall@1": round(sum(r1_list_rerank) / n, 3),
                "recall@5": round(sum(r5_list_rerank) / n, 3),
                "recall@10": round(sum(r10_list_rerank) / n, 3),
                "mrr@10": round(sum(mrr_list_rerank) / n, 3),
            },
        }
        return summary

    # Evaluate across retrieval modes
    modes_to_test = ["dense_only", "bm25_only", "rrf_hybrid"]
    print("\n[1/2] Evaluating Retrieval Modes Before vs. After Reranking...")
    for mode in modes_to_test:
        res = run_eval_pipeline(mode_name=mode, strat_filter=None)
        comparisons[f"mode_{mode}"] = res
        b = res["before_rerank"]
        a = res["after_rerank"]
        print(
            f"  - {mode:<15}: Before (R@10={b['recall@10']:.3f}, MRR={b['mrr@10']:.3f}) | "
            f"After (R@10={a['recall@10']:.3f}, MRR={a['mrr@10']:.3f})"
        )
        csv_rows.append({
            "Evaluation Type": "Mode",
            "Configuration": mode,
            "Before Recall@1": b["recall@1"],
            "After Recall@1": a["recall@1"],
            "Before Recall@10": b["recall@10"],
            "After Recall@10": a["recall@10"],
            "Before MRR@10": b["mrr@10"],
            "After MRR@10": a["mrr@10"],
        })

    # Evaluate across chunking strategies
    strategies_to_test = ["fixed", "sentence", "sliding_window", "semantic", "hierarchical"]
    print("\n[2/2] Evaluating Chunk Strategies Under RRF + Reranker...")
    for strat in strategies_to_test:
        res = run_eval_pipeline(mode_name="rrf_hybrid", strat_filter=[strat])
        comparisons[f"strategy_{strat}"] = res
        b = res["before_rerank"]
        a = res["after_rerank"]
        print(
            f"  - Strategy '{strat:<14}': Before (R@10={b['recall@10']:.3f}, MRR={b['mrr@10']:.3f}) | "
            f"After (R@10={a['recall@10']:.3f}, MRR={a['mrr@10']:.3f})"
        )
        csv_rows.append({
            "Evaluation Type": "Strategy",
            "Configuration": strat,
            "Before Recall@1": b["recall@1"],
            "After Recall@1": a["recall@1"],
            "Before Recall@10": b["recall@10"],
            "After Recall@10": a["recall@10"],
            "Before MRR@10": b["mrr@10"],
            "After MRR@10": a["mrr@10"],
        })

    # Save outputs
    report = {
        "evaluation_dataset": "ai4bharat/MSMARCO-XI",
        "queries_evaluated": len(eval_queries),
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "comparisons": comparisons,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n" + "=" * 70)
    print("RERANKING EVALUATION COMPLETE")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    sample = 50
    if len(sys.argv) > 1:
        try:
            sample = int(sys.argv[1])
        except ValueError:
            pass
    evaluate_reranking(sample_size=sample)
