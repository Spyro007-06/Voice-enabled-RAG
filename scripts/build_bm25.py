"""Offline BM25 index construction script loading processed chunks and serializing the BM25 model."""

import json
import logging
import os
import sys
import time
from typing import List

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.chunking.models import Chunk
from app.config import get_settings
from app.retrieval.bm25 import BM25Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)
logger = logging.getLogger("build_bm25")


def load_all_chunks(
    processed_dir: str = "data/processed",
    strategies: List[str] = None,
) -> List[Chunk]:
    """Load chunks from all available JSONL files including English chunks (Phase 6.16)."""
    if strategies is None:
        settings = get_settings()
        raw = settings.INDEX_STRATEGIES
        strategies = [s.strip() for s in raw.split(",") if s.strip()] if isinstance(raw, str) else list(raw)

    chunks: List[Chunk] = []
    for strat in strategies:
        jsonl_path = os.path.join(processed_dir, f"chunks_{strat}.jsonl")
        if not os.path.exists(jsonl_path):
            logger.warning("JSONL file not found for strategy '%s': %s", strat, jsonl_path)
            continue

        with open(jsonl_path, "r", encoding="utf-8") as f:
            count = 0
            for line in f:
                stripped = line.strip()
                if stripped:
                    try:
                        chunks.append(Chunk(**json.loads(stripped)))
                        count += 1
                    except Exception as e:
                        logger.warning("Failed parsing chunk JSON: %s", e)
            logger.info("Loaded %d chunks from %s", count, jsonl_path)

    # Phase 6.16: also load English chunks for multilingual BM25 support
    english_path = os.path.join(processed_dir, "chunks_english.jsonl")
    if os.path.exists(english_path):
        with open(english_path, "r", encoding="utf-8") as f:
            count = 0
            for line in f:
                stripped = line.strip()
                if stripped:
                    try:
                        chunks.append(Chunk(**json.loads(stripped)))
                        count += 1
                    except Exception as e:
                        logger.warning("Failed parsing English chunk JSON: %s", e)
            logger.info("Loaded %d English chunks from %s (Phase 6.16)", count, english_path)
    else:
        logger.warning(
            "English chunks file not found at %s. Run scripts/generate_english_chunks.py first.",
            english_path,
        )

    return chunks


def build_bm25(output_path: str = "data/bm25_index.pkl") -> None:
    """Build and serialize the multilingual BM25 index."""
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 70)
    print("BUILDING MULTILINGUAL BM25 LEXICAL INDEX (OFFLINE)")
    print("=" * 70)

    t0 = time.perf_counter()
    chunks = load_all_chunks()
    if not chunks:
        print("[ERROR] No chunks found in data/processed/. Run `python scripts/build_chunks.py` first.")
        return

    print(f"Total Chunks to Index: {len(chunks):,}")
    retriever = BM25Retriever(index_path=output_path)
    indexed_count = retriever.build_index_from_chunks(chunks)
    saved_path = retriever.save_index(output_path)
    t_total = time.perf_counter() - t0

    print("\n" + "=" * 70)
    print("BM25 INDEXING SUMMARY")
    print("=" * 70)
    print(f"Total Chunks Indexed : {indexed_count:,}")
    print(f"Index File Size      : {os.path.getsize(saved_path) / (1024*1024):.2f} MB")
    print(f"Saved Path           : {saved_path}")
    print(f"Total Duration       : {t_total:.2f} s ({indexed_count / max(0.001, t_total):.0f} chunks/sec)")
    print("=" * 70)


if __name__ == "__main__":
    build_bm25()
