"""
Phase 6.26 Retrieval Relevance and Grounding Evaluation Script.
Evaluates multi-signal relevance calibration against in-domain, off-topic, and adversarial test cases.

Outputs:
- benchmarks/phase626_relevance_results.json
- benchmarks/phase626_relevance_report.md
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List
import numpy as np
from dotenv import load_dotenv

# Ensure UTF-8 output encoding on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
load_dotenv(os.path.join(BASE_DIR, ".env"))

from app.config import get_settings
from app.guardrails.relevance import validate_answerability, RelevanceGuard
from app.reranking.adaptive import get_adaptive_retrieval_service
from app.reranking.models import RerankResult


def make_synthetic_rerank_result(
    chunk_id: str,
    text: str,
    reranker_score: float = 0.10,
    dense_score: float = 0.20,
    fusion_score: float = 0.05,
    language: str = "eng_Latn",
) -> RerankResult:
    return RerankResult(
        chunk_id=chunk_id,
        document_id="doc_synthetic",
        text=text,
        chunk_type="semantic",
        language=language,
        dense_score=dense_score,
        bm25_score=dense_score * 10,
        fusion_score=fusion_score,
        reranker_score=reranker_score,
        original_rank=1,
        rank=1,
    )


async def run_evaluation():
    print("=" * 70)
    print("PHASE 6.26 — RETRIEVAL RELEVANCE & GROUNDING QUALITY EVALUATION")
    print("=" * 70)

    dataset_path = os.path.join(BASE_DIR, "data", "evaluation", "phase626_relevance_cases.json")
    if not os.path.exists(dataset_path):
        print(f"Error: Evaluation dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    print(f"Loaded {len(cases)} evaluation cases from {dataset_path}")

    adaptive_service = get_adaptive_retrieval_service()
    guard = RelevanceGuard()

    results: List[Dict[str, Any]] = []
    t_start_total = time.perf_counter()

    for idx, case in enumerate(cases, 1):
        cid = case["id"]
        category = case["category"]
        lang = case.get("language", "en")
        query = case["query"]
        expected_allowed = case["expected_allowed"]
        synthetic_ctx = case.get("synthetic_context")

        t0 = time.perf_counter()

        retrieved_items = []
        if synthetic_ctx:
            # Synthetic adversarial evaluation
            dummy_item = make_synthetic_rerank_result(
                chunk_id="synthetic_adv_01",
                text=synthetic_ctx,
                reranker_score=0.08,
                dense_score=0.15,
                fusion_score=0.02,
                language=lang,
            )
            retrieved_items = [dummy_item]
        else:
            # Real live retrieval pipeline evaluation
            try:
                ret_results, decision, ret_latency = adaptive_service.adaptive_retrieve(
                    query=query,
                    language=lang,
                    top_k=5,
                )
                retrieved_items = ret_results
            except Exception as e:
                print(f"[{idx}/{len(cases)}] Error during retrieval for '{query}': {e}")
                retrieved_items = []

        # Run multi-signal relevance guardrail
        ans_res = validate_answerability(query, retrieved_items, language=lang)
        relevance_eval = guard.evaluate_relevance(query, retrieved_items)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        is_match = (ans_res.allowed == expected_allowed)
        status_sym = "✓ PASS" if is_match else "✕ FAIL"

        top_chunk_snippet = ""
        if retrieved_items and hasattr(retrieved_items[0], "text"):
            top_chunk_snippet = retrieved_items[0].text[:90].replace("\n", " ") + "..."

        res_entry = {
            "id": cid,
            "category": category,
            "language": lang,
            "query": query,
            "expected_allowed": expected_allowed,
            "actual_allowed": ans_res.allowed,
            "decision": ans_res.decision,
            "confidence": ans_res.confidence,
            "reason": ans_res.reason,
            "signals": ans_res.signals or {},
            "is_match": is_match,
            "latency_ms": round(elapsed_ms, 2),
            "retrieved_count": len(retrieved_items),
            "top_snippet": top_chunk_snippet,
        }
        results.append(res_entry)

        print(
            f"[{idx:02d}/{len(cases):02d}] {status_sym} | {category:24s} | {lang:2s} | "
            f"Allowed: {str(ans_res.allowed):5s} (Exp: {str(expected_allowed):5s}) | "
            f"Score: {ans_res.confidence:.2f} ({ans_res.decision}) | {query[:35]}"
        )

    total_time_s = time.perf_counter() - t_start_total

    # =========================================================================
    # Compute Aggregate Benchmark Metrics
    # =========================================================================
    in_domain = [r for r in results if r["category"] == "in_domain_relevant"]
    off_topic = [r for r in results if r["category"] == "off_topic_unanswerable"]
    adversarial = [r for r in results if r["category"] == "adversarial_relevance"]

    tp = sum(1 for r in in_domain if r["actual_allowed"])
    fn = sum(1 for r in in_domain if not r["actual_allowed"])
    tn_off = sum(1 for r in off_topic if not r["actual_allowed"])
    fp_off = sum(1 for r in off_topic if r["actual_allowed"])
    tn_adv = sum(1 for r in adversarial if not r["actual_allowed"])
    fp_adv = sum(1 for r in adversarial if r["actual_allowed"])

    total_fp = fp_off + fp_adv
    total_tn = tn_off + tn_adv

    precision = (tp / (tp + total_fp)) if (tp + total_fp) > 0 else 1.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    relevant_acc_rate = (tp / len(in_domain) * 100.0) if in_domain else 0.0
    off_topic_rej_rate = (tn_off / len(off_topic) * 100.0) if off_topic else 0.0
    adversarial_rej_rate = (tn_adv / len(adversarial) * 100.0) if adversarial else 0.0

    latencies = [r["latency_ms"] for r in results]
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))

    # Language breakdown for in-domain
    lang_stats = {}
    for lang in ["en", "hi", "ta", "te", "ml"]:
        lang_items = [r for r in in_domain if r["language"] == lang]
        l_tp = sum(1 for r in lang_items if r["actual_allowed"])
        lang_stats[lang] = {
            "total": len(lang_items),
            "passed": l_tp,
            "accuracy": (l_tp / len(lang_items) * 100.0) if lang_items else 0.0,
        }

    summary_metrics = {
        "timestamp": datetime.now().isoformat(),
        "total_cases": len(results),
        "in_domain_total": len(in_domain),
        "in_domain_accepted": tp,
        "in_domain_acceptance_rate_pct": round(relevant_acc_rate, 2),
        "off_topic_total": len(off_topic),
        "off_topic_rejected": tn_off,
        "off_topic_rejection_rate_pct": round(off_topic_rej_rate, 2),
        "adversarial_total": len(adversarial),
        "adversarial_rejected": tn_adv,
        "adversarial_rejection_rate_pct": round(adversarial_rej_rate, 2),
        "false_acceptances": total_fp,
        "false_refusals": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "latency_p50_ms": round(p50_lat, 2),
        "latency_p95_ms": round(p95_lat, 2),
        "total_duration_sec": round(total_time_s, 2),
        "language_breakdown": lang_stats,
    }

    # =========================================================================
    # Output Files Generation
    # =========================================================================
    os.makedirs(os.path.join(BASE_DIR, "benchmarks"), exist_ok=True)

    json_output_path = os.path.join(BASE_DIR, "benchmarks", "phase626_relevance_results.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary_metrics, "cases": results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved benchmark results to {json_output_path}")

    md_output_path = os.path.join(BASE_DIR, "benchmarks", "phase626_relevance_report.md")
    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write("# Phase 6.26 Retrieval Relevance & Grounding Quality Benchmark Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Corpus:** MSMARCO-XI (48,206 chunks, 28,541 vectors)  \n")
        f.write(f"**Models:** `multilingual-e5-small` · `mmarco-mMiniLMv2-L12-H384-v1` · `Sarvam 105B`\n\n")
        f.write("---\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write("| Metric | Result | Target |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **Relevant Acceptance Rate** | **{relevant_acc_rate:.1f}%** ({tp}/{len(in_domain)}) | >= 90.0% |\n")
        f.write(f"| **Off-Topic Rejection Rate** | **{off_topic_rej_rate:.1f}%** ({tn_off}/{len(off_topic)}) | 100.0% |\n")
        f.write(f"| **Adversarial Rejection Rate** | **{adversarial_rej_rate:.1f}%** ({tn_adv}/{len(adversarial)}) | 100.0% |\n")
        f.write(f"| **False Acceptances (Hallucination Risk)** | **{total_fp}** | 0 |\n")
        f.write(f"| **False Refusals** | **{fn}** | <= 2 |\n")
        f.write(f"| **Precision** | **{precision:.4f}** | >= 0.9500 |\n")
        f.write(f"| **Recall** | **{recall:.4f}** | >= 0.9000 |\n")
        f.write(f"| **F1 Score** | **{f1:.4f}** | >= 0.9200 |\n")
        f.write(f"| **Latency (P50 / P95)** | **{p50_lat:.1f} ms / {p95_lat:.1f} ms** | <= 250 ms |\n\n")
        f.write("---\n\n")
        f.write("## 2. Multilingual In-Domain Performance\n\n")
        f.write("| Language | Code | Queries | Accepted | Accuracy |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for l, stat in lang_stats.items():
            name_map = {"en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "ml": "Malayalam"}
            f.write(f"| {name_map.get(l, l)} | `{l}` | {stat['total']} | {stat['passed']} | **{stat['accuracy']:.1f}%** |\n")
        f.write("\n---\n\n")
        f.write("## 3. Case-by-Case Evaluation Matrix\n\n")
        f.write("| ID | Category | Language | Query | Decision | Allowed | Expected | Match |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results:
            m_icon = "✓" if r["is_match"] else "✕"
            f.write(f"| `{r['id']}` | {r['category']} | `{r['language']}` | {r['query']} | `{r['decision']}` | {r['actual_allowed']} | {r['expected_allowed']} | {m_icon} |\n")
    print(f"Saved benchmark markdown report to {md_output_path}")

    # =========================================================================
    # Final Terminal Summary Display
    # =========================================================================
    print("\n" + "=" * 50)
    print("PHASE 6.26 BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"In-Domain Acceptance: {tp}/{len(in_domain)} ({relevant_acc_rate:.1f}%)")
    print(f"Off-Topic Rejection:  {tn_off}/{len(off_topic)} ({off_topic_rej_rate:.1f}%)")
    print(f"Adversarial Rejection:{tn_adv}/{len(adversarial)} ({adversarial_rej_rate:.1f}%)")
    print(f"False Acceptances:    {total_fp}")
    print(f"False Refusals:       {fn}")
    print(f"Precision:            {precision:.4f}")
    print(f"Recall:               {recall:.4f}")
    print(f"F1 Score:             {f1:.4f}")
    print(f"Latency P50:          {p50_lat:.1f} ms")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(run_evaluation())
