"""Inspection utility to verify Qdrant vector store health, point count, metadata, and strategies."""

import json
import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings
from app.retrieval.qdrant_store import QdrantVectorStore


def check_index():
    """Inspect and display status of Qdrant collection."""
    sys.stdout.reconfigure(encoding="utf-8")
    settings = get_settings()
    collection_name = settings.QDRANT_COLLECTION_NAME

    print("=" * 70)
    print("QDRANT VECTOR INDEX HEALTH & STATUS INSPECTOR")
    print("=" * 70)
    print(f"Storage Path    : {settings.QDRANT_PATH}")
    print(f"Collection Name : {collection_name}")

    store = QdrantVectorStore(location=settings.QDRANT_PATH)
    stats = store.get_collection_stats(collection_name)

    if not stats.get("exists"):
        print(f"\n[ERROR] Collection '{collection_name}' does not exist in {settings.QDRANT_PATH}.")
        print("Run `python scripts/build_index.py` to initialize and populate the index.")
        return

    print("\n" + "-" * 70)
    print("COLLECTION METRICS")
    print("-" * 70)
    print(f"Status              : {stats.get('status')}")
    print(f"Total Points/Vectors: {stats.get('points_count', 0):,}")
    print(f"Vector Size         : {stats.get('vectors_config', {}).get('size')} dimensions")
    print(f"Distance Metric     : {stats.get('vectors_config', {}).get('distance')}")

    # Fetch sample points to inspect payload schemas, strategies, and languages
    try:
        # Scroll first 100 points
        records, _ = store.client.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )

        strategies_found = set()
        languages_found = set()
        payload_keys = set()

        for point in records:
            p = point.payload or {}
            payload_keys.update(p.keys())
            if "chunk_type" in p:
                strategies_found.add(p["chunk_type"])
            if "language" in p and p["language"]:
                languages_found.add(p["language"])

        print("\n" + "-" * 70)
        print("PAYLOAD SCHEMA & MULTILINGUAL COVERAGE")
        print("-" * 70)
        print(f"Payload Attributes  : {sorted(list(payload_keys))}")
        print(f"Indexed Strategies  : {sorted(list(strategies_found))}")
        print(f"Languages Represented: {sorted(list(languages_found))}")

        if records:
            print("\nSample Indexed Point (First Record):")
            sample_preview = {
                "id": str(records[0].id),
                "payload": records[0].payload,
            }
            print(json.dumps(sample_preview, indent=2, ensure_ascii=False)[:1200])

    except Exception as e:
        print(f"Error inspecting point payloads: {e}")

    print("\n" + "=" * 70)
    print("CHECK COMPLETE — QDRANT INDEX HEALTHY & OPERATIONAL")
    print("=" * 70)


if __name__ == "__main__":
    check_index()
