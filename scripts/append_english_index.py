"""
Phase 6.16 - Append English chunks to existing Qdrant collection.
Does NOT recreate the collection - preserves existing Hindi chunks.
"""
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)-8s | %(message)s")

from app.chunking.models import Chunk
from app.config import get_settings
from app.embeddings.multilingual_e5 import MultilingualE5EmbeddingProvider
from app.retrieval.qdrant_store import QdrantVectorStore

def append_english_to_index():
    settings = get_settings()
    collection_name = settings.QDRANT_COLLECTION_NAME
    english_jsonl = os.path.join("data", "processed", "chunks_english.jsonl")

    print("=" * 70)
    print("PHASE 6.16: APPENDING ENGLISH CHUNKS TO QDRANT")
    print("=" * 70)

    if not os.path.exists(english_jsonl):
        print(f"[ERROR] {english_jsonl} not found. Run generate_english_chunks.py first.")
        sys.exit(1)

    # Load English chunks
    print(f"\n[1/4] Loading English chunks from {english_jsonl}...")
    chunks = []
    with open(english_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    chunks.append(Chunk(**data))
                except Exception as e:
                    print(f"  [WARN] {e}")
    print(f"  Loaded {len(chunks):,} English chunks")

    # Init embedding model
    print(f"\n[2/4] Initializing embedding model: {settings.EMBEDDING_MODEL}...")
    t0 = time.perf_counter()
    embedder = MultilingualE5EmbeddingProvider(
        model_name=settings.EMBEDDING_MODEL,
        device=settings.EMBEDDING_DEVICE,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )
    dim = embedder.embedding_dimension
    print(f"  Model ready: dim={dim}, device={embedder.device}, took {time.perf_counter()-t0:.2f}s")

    # Connect to Qdrant (append mode - no recreate)
    print(f"\n[3/4] Connecting to Qdrant collection '{collection_name}' (append mode)...")
    store = QdrantVectorStore(location=settings.QDRANT_PATH)
    
    # Verify collection exists
    if not store.collection_exists(collection_name):
        print(f"  [ERROR] Collection '{collection_name}' does not exist!")
        sys.exit(1)
    
    stats_before = store.get_collection_stats(collection_name)
    print(f"  Points before: {stats_before.get('points_count', '?'):,}")

    # Batch embed + upsert
    print(f"\n[4/4] Embedding and upserting {len(chunks):,} English chunks...")
    t_start = time.perf_counter()
    total_indexed = 0
    batch_size = settings.QDRANT_UPSERT_BATCH_SIZE

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.text for c in batch]
        vectors = embedder.embed_documents(texts)
        upserted = store.upsert_chunks(
            collection_name=collection_name,
            chunks=batch,
            vectors=vectors,
            batch_size=batch_size,
        )
        total_indexed += upserted
        print(f"  Batch {i//batch_size + 1}/{(len(chunks)+batch_size-1)//batch_size}: {upserted} vectors upserted (total: {total_indexed:,})")

    elapsed = time.perf_counter() - t_start
    stats_after = store.get_collection_stats(collection_name)
    
    print()
    print("=" * 70)
    print("INDEXING SUMMARY")
    print("=" * 70)
    print(f"English chunks indexed : {total_indexed:,}")
    print(f"Points before          : {stats_before.get('points_count', '?'):,}")
    print(f"Points after           : {stats_after.get('points_count', '?'):,}")
    print(f"Total indexing time    : {elapsed:.2f}s ({total_indexed/max(0.001,elapsed):.0f} vectors/sec)")
    print("=" * 70)
    return total_indexed

if __name__ == "__main__":
    n = append_english_to_index()
    sys.exit(0 if n > 0 else 1)
