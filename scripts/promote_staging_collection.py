"""Promote validated multilingual staging collection msmarco_xi_v619 to production msmarco_xi."""

import logging
import os
import sys
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s")
logger = logging.getLogger("promote_staging_collection")


def promote():
    qdrant_path = "data/qdrant"
    client = QdrantClient(path=qdrant_path)
    source_coll = "msmarco_xi_v619"
    target_coll = "msmarco_xi"

    logger.info("Verifying source collection %s...", source_coll)
    total_count = client.count(source_coll).count
    logger.info("Source collection %s has %d points.", source_coll, total_count)

    # Validate distribution and non-null language across points
    scroll_offset = None
    seen_ids = set()
    lang_dist = {}
    valid_dim_count = 0

    while True:
        records, next_offset = client.scroll(
            collection_name=source_coll,
            limit=500,
            offset=scroll_offset,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            break

        for r in records:
            seen_ids.add(r.id)
            if len(r.vector) == 384:
                valid_dim_count += 1
            lang = r.payload.get("language")
            if not lang:
                raise ValueError(f"Null or empty language in point {r.id}")
            lang_dist[lang] = lang_dist.get(lang, 0) + 1

        if next_offset is None:
            break
        scroll_offset = next_offset

    logger.info("Validation complete:")
    logger.info("  Total points verified: %d", len(seen_ids))
    logger.info("  384-dim vectors: %d", valid_dim_count)
    logger.info("  Language distribution:")
    for l, cnt in sorted(lang_dist.items()):
        logger.info("    - %s: %d points", l, cnt)

    # Recreate production target collection msmarco_xi
    logger.info("Promoting to production collection %s...", target_coll)
    client.recreate_collection(
        collection_name=target_coll,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

    # Copy verified points into target_coll in batches of 500
    scroll_offset = None
    copied = 0
    t0 = time.time()

    while True:
        records, next_offset = client.scroll(
            collection_name=source_coll,
            limit=500,
            offset=scroll_offset,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            break

        points = [
            models.PointStruct(
                id=r.id,
                vector=r.vector,
                payload=r.payload,
            )
            for r in records
        ]
        client.upsert(
            collection_name=target_coll,
            points=points,
        )
        copied += len(points)
        if next_offset is None:
            break
        scroll_offset = next_offset

    final_count = client.count(target_coll).count
    logger.info("Promotion successful! %s now contains %d production points (Copied in %.1fs).", target_coll, final_count, time.time() - t0)

    # Delete staging collection
    try:
        client.delete_collection(source_coll)
        logger.info("Staging collection %s successfully cleaned up.", source_coll)
    except Exception as e:
        logger.warning("Could not delete staging collection: %s", e)


if __name__ == "__main__":
    promote()
