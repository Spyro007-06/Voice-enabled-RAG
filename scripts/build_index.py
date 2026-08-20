"""Batch vector indexing script loading JSONL chunks, generating multilingual embeddings, and upserting to Qdrant."""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.chunking.models import Chunk
from app.config import get_settings
from app.embeddings.multilingual_e5 import MultilingualE5EmbeddingProvider
from app.retrieval.qdrant_store import QdrantVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)
logger = logging.getLogger("build_index")


def load_chunks_from_jsonl(
    jsonl_path: str,
    limit: Optional[int] = None,
) -> List[Chunk]:
    """Read Chunk models from a JSONL file up to limit."""
    chunks: List[Chunk] = []
    if not os.path.exists(jsonl_path):
        logger.warning("JSONL file not found at %s", jsonl_path)
        return chunks

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if limit and len(chunks) >= limit:
                break
            stripped = line.strip()
            if stripped:
                try:
                    data = json.loads(stripped)
                    chunks.append(Chunk(**data))
                except Exception as e:
                    logger.warning("Failed to parse chunk JSON: %s", e)
    return chunks


def build_index(
    sample_size: Optional[int] = None,
    strategies_to_index: Optional[List[str]] = None,
    recreate_collection: bool = False,
    report_path: str = "benchmarks/indexing_report.json",
) -> Dict[str, Any]:
    """Execute complete offline batch vector indexing pipeline."""
    sys.stdout.reconfigure(encoding="utf-8")
    settings = get_settings()

    target_sample_size = sample_size if sample_size is not None else settings.INDEX_SAMPLE_SIZE
    collection_name = settings.QDRANT_COLLECTION_NAME

    if strategies_to_index is None:
        raw_strat = settings.INDEX_STRATEGIES
        selected_strategies = [s.strip() for s in raw_strat.split(",") if s.strip()] if isinstance(raw_strat, str) else list(raw_strat)
    else:
        selected_strategies = strategies_to_index

    print("=" * 70)
    print("PHASE 3: MULTILINGUAL EMBEDDINGS & QDRANT VECTOR INDEXING")
    print("=" * 70)
    print(f"Collection Name    : {collection_name}")
    print(f"Qdrant Storage Path: {settings.QDRANT_PATH}")
    print(f"Embedding Model    : {settings.EMBEDDING_MODEL}")
    print(f"Index Strategies   : {selected_strategies}")
    print(f"Index Sample Target: {target_sample_size}")

    # 1. Initialize Embedding Provider
    print("\n[1/4] Initializing Multilingual E5 Embedding Provider...")
    t_embed_init_start = time.perf_counter()
    embedder = MultilingualE5EmbeddingProvider(
        model_name=settings.EMBEDDING_MODEL,
        device=settings.EMBEDDING_DEVICE,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )
    dim = embedder.embedding_dimension
    t_embed_init = time.perf_counter() - t_embed_init_start
    print(f"  - Model initialized: {embedder.model_name} (dim={dim}, device={embedder.device}) in {t_embed_init:.2f}s")

    # 2. Initialize Qdrant Vector Store & Collection
    print("\n[2/4] Initializing Qdrant Collection...")
    store = QdrantVectorStore(location=settings.QDRANT_PATH)
    if recreate_collection:
        store.recreate_collection(collection_name=collection_name, vector_size=dim)
        print(f"  - Recreated collection '{collection_name}' (dim={dim}, distance=COSINE)")
    else:
        created = store.create_collection_if_not_exists(collection_name=collection_name, vector_size=dim)
        print(f"  - Collection '{collection_name}': {'Created newly' if created else 'Existing collection connected'}")

    # 3. Load Chunks from Processed JSONL files
    print("\n[3/4] Loading Chunks from data/processed/...")
    t_load_start = time.perf_counter()
    all_chunks_to_index: List[Chunk] = []
    chunks_per_strategy_target = max(1, target_sample_size // max(1, len(selected_strategies)))

    for strat in selected_strategies:
        jsonl_path = os.path.join("data", "processed", f"chunks_{strat}.jsonl")
        strat_chunks = load_chunks_from_jsonl(jsonl_path, limit=chunks_per_strategy_target)
        print(f"  - Loaded {len(strat_chunks)} chunks for strategy '{strat}' from {jsonl_path}")
        all_chunks_to_index.extend(strat_chunks)

    t_load = time.perf_counter() - t_load_start
    print(f"Total chunks selected for indexing: {len(all_chunks_to_index)} (loaded in {t_load*1000:.1f}ms)")

    if not all_chunks_to_index:
        print("[WARNING] No chunks found to index. Make sure Phase 2 chunk files exist in data/processed/.")
        return {}

    # 4. Batch Embed and Upsert
    print("\n[4/4] Generating Embeddings and Upserting Vectors to Qdrant...")
    t_index_start = time.perf_counter()
    total_embeddings_time = 0.0
    total_upsert_time = 0.0
    total_vectors_indexed = 0

    embed_batch_size = settings.EMBEDDING_BATCH_SIZE
    upsert_batch_size = settings.QDRANT_UPSERT_BATCH_SIZE

    # Process in chunks of upsert_batch_size
    for i in range(0, len(all_chunks_to_index), upsert_batch_size):
        chunk_batch = all_chunks_to_index[i : i + upsert_batch_size]
        raw_texts = [c.text for c in chunk_batch]

        # Batch encode
        t_batch_embed_start = time.perf_counter()
        vectors = embedder.embed_documents(raw_texts)
        t_batch_embed = time.perf_counter() - t_batch_embed_start
        total_embeddings_time += t_batch_embed

        # Batch upsert
        t_batch_upsert_start = time.perf_counter()
        upserted_count = store.upsert_chunks(
            collection_name=collection_name,
            chunks=chunk_batch,
            vectors=vectors,
            batch_size=upsert_batch_size,
        )
        t_batch_upsert = time.perf_counter() - t_batch_upsert_start
        total_upsert_time += t_batch_upsert
        total_vectors_indexed += upserted_count

        print(
            f"  Batch {i // upsert_batch_size + 1}/{(len(all_chunks_to_index) + upsert_batch_size - 1) // upsert_batch_size}: "
            f"Indexed {upserted_count} vectors (Embed: {t_batch_embed*1000:.1f}ms | Upsert: {t_batch_upsert*1000:.1f}ms)"
        )

    t_total_index = time.perf_counter() - t_index_start
    stats = store.get_collection_stats(collection_name)

    # 5. Compile Indexing Report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report = {
        "collection_name": collection_name,
        "embedding_model": embedder.model_name,
        "embedding_dimension": dim,
        "device": embedder.device,
        "strategies_indexed": selected_strategies,
        "total_chunks_processed": len(all_chunks_to_index),
        "total_vectors_indexed": total_vectors_indexed,
        "collection_points_count": stats.get("points_count", total_vectors_indexed),
        "timing": {
            "jsonl_loading_sec": round(t_load, 4),
            "model_init_sec": round(t_embed_init, 4),
            "total_embedding_sec": round(total_embeddings_time, 4),
            "total_upsert_sec": round(total_upsert_time, 4),
            "total_pipeline_sec": round(t_total_index, 4),
            "indexing_throughput_vectors_per_sec": round(total_vectors_indexed / max(0.001, t_total_index), 2),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(report_path, "w", encoding="utf-8") as f_out:
        json.dump(report, f_out, indent=2)

    print("\n" + "=" * 70)
    print("INDEXING SUMMARY & STATS")
    print("=" * 70)
    print(f"Total Vectors in Qdrant   : {stats.get('points_count', total_vectors_indexed)}")
    print(f"Total Chunks Indexed      : {total_vectors_indexed}")
    print(f"Embedding Time            : {total_embeddings_time:.2f}s")
    print(f"Qdrant Upsert Time        : {total_upsert_time:.2f}s")
    print(f"Total Indexing Duration   : {t_total_index:.2f}s ({report['timing']['indexing_throughput_vectors_per_sec']} vectors/sec)")
    print(f"Indexing Report Saved     : {report_path}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    sample = 1000
    if len(sys.argv) > 1:
        try:
            sample = int(sys.argv[1])
        except ValueError:
            pass
    build_index(sample_size=sample, recreate_collection=True)
