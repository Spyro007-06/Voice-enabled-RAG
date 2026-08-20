"""Embedding benchmark measuring latency, throughput, and percentiles for Multilingual E5."""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings
from app.embeddings.multilingual_e5 import MultilingualE5EmbeddingProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark_embeddings")


def run_embedding_benchmark(
    num_sentences: int = 256,
    batch_size: int = 32,
    num_runs: int = 5,
    report_path: str = "benchmarks/embedding_report.json",
) -> Dict[str, Any]:
    """Execute comprehensive embedding benchmark across multilingual sentences."""
    sys.stdout.reconfigure(encoding="utf-8")
    settings = get_settings()

    print("=" * 70)
    print("EMBEDDING MODEL LATENCY & THROUGHPUT BENCHMARK")
    print("=" * 70)

    # Representative multilingual evaluation corpus (English, Hindi, Tamil, Telugu, Malayalam, Bengali)
    sample_corpus = [
        "What is the capital of India and how is the government structured?",
        "भारत की राजधानी नई दिल्ली है और यहाँ राष्ट्रपति भवन स्थित है।",
        "இந்தியாவின் தலைநகரம் புதுதில்லி ஆகும், இங்கு மத்திய அரசு செயல்படுகிறது.",
        "భారతదేశ రాజధాని న్యూఢిల్లీ, ఇక్కడ పార్లమెంటు ఉంది.",
        "ഇന്ത്യയുടെ തലസ്ഥാനം ന്യൂഡൽഹിയാണ്, ഇവിടെ രാഷ്ട്രപതി ഭവൻ സ്ഥിതി ചെയ്യുന്നു.",
        "ভারতের রাজধানী নতুন দিল্লি এবং এটি একটি গুরুত্বপূর্ণ ঐতিহাসিক শহর।",
        "The retrieval-augmented generation system combines dense vector search with neural generation.",
        "मल्टीलिंगुअल एम्बेडिंग मॉडल विभिन्न भारतीय भाषाओं में प्रश्नों का सटीक उत्तर खोजने में सक्षम है।",
    ]
    test_texts = [sample_corpus[i % len(sample_corpus)] for i in range(num_sentences)]

    # 1. Model Loading Benchmark
    print(f"\n[1/3] Benchmarking Model Loading for '{settings.EMBEDDING_MODEL}'...")
    t_load_start = time.perf_counter()
    embedder = MultilingualE5EmbeddingProvider(
        model_name=settings.EMBEDDING_MODEL,
        device=settings.EMBEDDING_DEVICE,
        batch_size=batch_size,
    )
    dim = embedder.embedding_dimension
    t_load_sec = time.perf_counter() - t_load_start
    print(f"  Model loaded in {t_load_sec:.3f}s (Dimension: {dim}, Device: {embedder.device})")

    # 2. Warmup Run
    print("\n[2/3] Performing Warmup Inferences...")
    _ = embedder.embed_query("warmup search query")
    _ = embedder.embed_documents(test_texts[:batch_size])

    # 3. Document Embedding Latency & Throughput Benchmark
    print(f"\n[3/3] Running {num_runs} evaluation passes across {num_sentences} multilingual texts...")
    batch_latencies_ms: List[float] = []
    all_pass_times_sec: List[float] = []

    for run_idx in range(num_runs):
        t_pass_start = time.perf_counter()
        for i in range(0, num_sentences, batch_size):
            batch = test_texts[i : i + batch_size]
            t_b_start = time.perf_counter()
            _ = embedder.embed_documents(batch)
            b_ms = (time.perf_counter() - t_b_start) * 1000
            batch_latencies_ms.append(b_ms)
        pass_duration = time.perf_counter() - t_pass_start
        all_pass_times_sec.append(pass_duration)

    # 4. Query Embedding Latency Benchmark (Single query online simulation)
    query_latencies_ms: List[float] = []
    test_queries = [
        "What is a corporation?",
        "कॉर्पोरेशन क्या है?",
        "பொருளாதாரம் என்றால் என்ன?",
        "వ్యాపార నిర్వహణ అంటే ఏమిటి?",
    ]
    for _ in range(20):
        for q in test_queries:
            t_q_start = time.perf_counter()
            _ = embedder.embed_query(q)
            q_ms = (time.perf_counter() - t_q_start) * 1000
            query_latencies_ms.append(q_ms)

    # Compute Statistics
    avg_batch_ms = float(np.mean(batch_latencies_ms))
    p50_batch_ms = float(np.percentile(batch_latencies_ms, 50))
    p70_batch_ms = float(np.percentile(batch_latencies_ms, 70))
    p100_batch_ms = float(np.percentile(batch_latencies_ms, 100))

    total_texts_evaluated = num_sentences * num_runs
    total_eval_time = sum(all_pass_times_sec)
    texts_per_sec = round(total_texts_evaluated / max(0.0001, total_eval_time), 2)

    avg_query_ms = float(np.mean(query_latencies_ms))
    p50_query_ms = float(np.percentile(query_latencies_ms, 50))
    p95_query_ms = float(np.percentile(query_latencies_ms, 95))
    p100_query_ms = float(np.percentile(query_latencies_ms, 100))

    report = {
        "embedding_model": embedder.model_name,
        "vector_dimension": dim,
        "device": embedder.device,
        "batch_size": batch_size,
        "num_texts_per_pass": num_sentences,
        "num_runs": num_runs,
        "model_loading_time_sec": round(t_load_sec, 4),
        "document_embedding_benchmarks": {
            "throughput_texts_per_sec": texts_per_sec,
            "avg_batch_latency_ms": round(avg_batch_ms, 2),
            "p50_batch_latency_ms": round(p50_batch_ms, 2),
            "p70_batch_latency_ms": round(p70_batch_ms, 2),
            "p100_batch_latency_ms": round(p100_batch_ms, 2),
        },
        "query_embedding_benchmarks": {
            "avg_query_latency_ms": round(avg_query_ms, 2),
            "p50_query_latency_ms": round(p50_query_ms, 2),
            "p95_query_latency_ms": round(p95_query_ms, 2),
            "p100_query_latency_ms": round(p100_query_ms, 2),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("EMBEDDING BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Model                  : {embedder.model_name}")
    print(f"Dimension              : {dim}")
    print(f"Device                 : {embedder.device}")
    print(f"Throughput             : {texts_per_sec} texts/sec")
    print(f"Batch Latency (P50/P70): {p50_batch_ms:.2f} ms / {p70_batch_ms:.2f} ms (P100: {p100_batch_ms:.2f} ms)")
    print(f"Query Latency (Avg/P95): {avg_query_ms:.2f} ms / {p95_query_ms:.2f} ms (P100: {p100_query_ms:.2f} ms)")
    print(f"Report Saved           : {report_path}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    run_embedding_benchmark()
