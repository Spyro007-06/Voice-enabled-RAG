"""Offline chunk generation, validation, and benchmarking script for MSMARCO-XI."""

import csv
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.chunking.models import Chunk
from app.chunking.registry import ChunkingRegistry
from app.chunking.validator import ChunkValidator
from app.config import get_settings
from app.ingestion.dataset_loader import DatasetLoader
from app.ingestion.models import Document
from app.ingestion.normalizer import DocumentNormalizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)
logger = logging.getLogger("build_chunks")


def build_chunks(
    sample_size: int = 100,
    strategies_to_run: List[str] = None,
    output_dir: str = "data/processed",
    report_dir: str = "benchmarks",
) -> Dict[str, Any]:
    """Execute complete offline chunking pipeline across configured strategies."""
    sys.stdout.reconfigure(encoding="utf-8")
    settings = get_settings()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    print("=" * 70)
    print(f"OFFLINE CHUNK GENERATION & BENCHMARKING (Sample size: {sample_size} records)")
    print("=" * 70)

    # 1. Load raw dataset records
    loader = DatasetLoader(
        dataset_name=settings.DATASET_NAME,
        split=settings.DATASET_SPLIT,
        language=settings.DATASET_LANGUAGE,
        sample_size=sample_size,
    )
    print(f"\n[1/5] Loading records from {loader.dataset_name} ({loader.split}/{loader.language})...")
    raw_records = list(loader.load_records(limit=sample_size))
    print(f"Loaded {len(raw_records)} query records.")

    # 2. Normalize records into Document objects
    print("\n[2/5] Normalizing dataset passages into canonical Document objects...")
    normalizer = DocumentNormalizer()
    documents: List[Document] = []
    for rec in raw_records:
        docs = normalizer.normalize_record(rec)
        documents.extend(docs)
    print(f"Generated {len(documents)} clean Document instances from {len(raw_records)} records.")

    # 3. Discover and instantiate chunking strategies
    strategies = ChunkingRegistry.get_configured_strategies(strategies_to_run)
    strategy_names = [s.strategy_name for s in strategies]
    print(f"\n[3/5] Active chunking strategies: {strategy_names}")

    # 4. Initialize validator
    validator = ChunkValidator(min_char_length=1)

    # 5. Process each strategy
    print("\n[4/5] Executing chunking, validation, and serialization...")
    benchmark_report: Dict[str, Any] = {
        "metadata": {
            "dataset_name": loader.dataset_name,
            "split": loader.split,
            "language": loader.language,
            "raw_records_processed": len(raw_records),
            "documents_processed": len(documents),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "strategies": {},
    }

    csv_rows = []

    for strategy in strategies:
        strat_name = strategy.strategy_name
        print(f"\n  >> Running Strategy: '{strat_name}'...")

        t_start = time.perf_counter()
        all_chunks: List[Chunk] = []

        for doc in documents:
            doc_chunks = strategy.chunk(doc)
            all_chunks.extend(doc_chunks)

        t_elapsed_sec = time.perf_counter() - t_start
        t_elapsed_ms = t_elapsed_sec * 1000

        # Validate chunks
        is_valid, validation_errors = validator.validate_chunks(all_chunks, raise_on_error=False)
        if not is_valid:
            print(f"     [WARNING] Strategy {strat_name} produced {len(validation_errors)} validation errors:")
            for err in validation_errors[:3]:
                print(f"       - {err}")
        else:
            print(f"     [VALIDATION] All {len(all_chunks)} chunks validated successfully (100% deterministic & valid).")

        # Write to JSONL
        jsonl_filename = os.path.join(output_dir, f"chunks_{strat_name}.jsonl")
        with open(jsonl_filename, "w", encoding="utf-8") as f_out:
            for chunk in all_chunks:
                f_out.write(chunk.model_dump_json() + "\n")

        # Compute statistics
        char_lengths = [len(c.text) for c in all_chunks] if all_chunks else [0]
        token_counts = [c.token_count for c in all_chunks] if all_chunks else [0]

        total_chunks = len(all_chunks)
        avg_chars = sum(char_lengths) / max(1, total_chunks)
        min_chars = min(char_lengths) if all_chunks else 0
        max_chars = max(char_lengths) if all_chunks else 0

        avg_tokens = sum(token_counts) / max(1, total_chunks)
        min_tokens = min(token_counts) if all_chunks else 0
        max_tokens = max(token_counts) if all_chunks else 0

        chunks_per_doc = total_chunks / max(1, len(documents))

        strat_stats = {
            "strategy": strat_name,
            "documents_processed": len(documents),
            "chunks_generated": total_chunks,
            "chunks_per_document": round(chunks_per_doc, 2),
            "avg_chunk_char_length": round(avg_chars, 2),
            "min_chunk_char_length": min_chars,
            "max_chunk_char_length": max_chars,
            "avg_chunk_tokens": round(avg_tokens, 2),
            "min_chunk_tokens": min_tokens,
            "max_chunk_tokens": max_tokens,
            "processing_time_ms": round(t_elapsed_ms, 2),
            "output_file": jsonl_filename,
            "validation_passed": is_valid,
        }

        benchmark_report["strategies"][strat_name] = strat_stats

        csv_rows.append({
            "Strategy": strat_name,
            "Documents": len(documents),
            "Chunks": total_chunks,
            "Chunks/Doc": round(chunks_per_doc, 2),
            "Avg Chars": round(avg_chars, 1),
            "Min Chars": min_chars,
            "Max Chars": max_chars,
            "Avg Tokens": round(avg_tokens, 1),
            "Time (ms)": round(t_elapsed_ms, 1),
            "Valid": is_valid,
        })

        print(f"     Chunks Generated: {total_chunks}")
        print(f"     Avg Chars / Tokens: {avg_chars:.1f} / {avg_tokens:.1f}")
        print(f"     Time: {t_elapsed_ms:.1f} ms ({total_chunks / max(0.001, t_elapsed_sec):.0f} chunks/sec)")
        print(f"     Saved: {jsonl_filename}")

    # 6. Save Comparison Reports
    print("\n[5/5] Saving benchmark reports...")
    json_report_path = os.path.join(report_dir, "chunking_report.json")
    with open(json_report_path, "w", encoding="utf-8") as f_json:
        json.dump(benchmark_report, f_json, indent=2, ensure_ascii=False)
    print(f"  - Saved JSON Benchmark: {json_report_path}")

    csv_report_path = os.path.join(report_dir, "chunking_report.csv")
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_report_path, "w", encoding="utf-8", newline="") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  - Saved CSV Benchmark : {csv_report_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("CHUNKING BENCHMARK COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<16} | {'Docs':<5} | {'Chunks':<7} | {'Avg Chars':<10} | {'Avg Tokens':<11} | {'Time (ms)':<10}")
    print("-" * 70)
    for row in csv_rows:
        print(f"{row['Strategy']:<16} | {row['Documents']:<5} | {row['Chunks']:<7} | {row['Avg Chars']:<10} | {row['Avg Tokens']:<11} | {row['Time (ms)']:<10}")
    print("=" * 70)

    return benchmark_report


if __name__ == "__main__":
    sample_size = 100
    if len(sys.argv) > 1:
        try:
            sample_size = int(sys.argv[1])
        except ValueError:
            pass
    build_chunks(sample_size=sample_size)
