"""Rebuild safe Qdrant vector index and BM25 index across all 5 languages."""

import json
import logging
import os
import random
import sys
import time
import uuid
from typing import Any, Dict, List, Set

import torch
torch.set_num_threads(16)

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.chunking.models import Chunk
from app.config import get_settings
from app.embeddings.multilingual_e5 import MultilingualE5EmbeddingProvider
from app.retrieval.bm25 import BM25Retriever
from qdrant_client import QdrantClient
from qdrant_client.http import models

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s")
logger = logging.getLogger("rebuild_multilingual_index")

CHUNK_FILES = [
    ("data/processed/chunks_hindi.jsonl", "hin_Deva", "Hindi"),
    ("data/processed/chunks_english.jsonl", "eng_Latn", "English"),
    ("data/processed/chunks_tamil.jsonl", "tam_Taml", "Tamil"),
    ("data/processed/chunks_telugu.jsonl", "tel_Telu", "Telugu"),
    ("data/processed/chunks_malayalam.jsonl", "mal_Mlym", "Malayalam"),
]


def load_all_multilingual_chunks(max_per_lang: int = 2500) -> tuple[List[Chunk], List[Chunk]]:
    """Load chunks. Returns (all_chunks_for_bm25, balanced_chunks_for_qdrant)."""
    all_chunks: List[Chunk] = []
    qdrant_chunks: List[Chunk] = []
    seen_ids: Set[str] = set()
    lang_counts: Dict[str, int] = {}
    qdrant_lang_counts: Dict[str, int] = {}

    for f_path, canon_lang, lang_name in CHUNK_FILES:
        if not os.path.exists(f_path):
            logger.warning("Chunk file %s does not exist. Skipping.", f_path)
            continue
        logger.info("Loading %s chunks from %s...", lang_name, f_path)
        lang_loaded = 0
        with open(f_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                c = Chunk.model_validate(data)
                if c.chunk_id in seen_ids:
                    continue
                seen_ids.add(c.chunk_id)
                all_chunks.append(c)
                lang = c.metadata.get("language") or getattr(c, "language", canon_lang)
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

                if lang_loaded < max_per_lang:
                    qdrant_chunks.append(c)
                    qdrant_lang_counts[lang] = qdrant_lang_counts.get(lang, 0) + 1
                    lang_loaded += 1

    logger.info("Total BM25 chunks loaded: %d", len(all_chunks))
    for l, cnt in sorted(lang_counts.items()):
        logger.info("  BM25 %s: %d chunks", l, cnt)

    logger.info("Total Qdrant balanced chunks loaded: %d", len(qdrant_chunks))
    for l, cnt in sorted(qdrant_lang_counts.items()):
        logger.info("  Qdrant %s: %d chunks", l, cnt)

    return all_chunks, qdrant_chunks


def rebuild_bm25(chunks: List[Chunk], output_path: str = "data/bm25_index.pkl"):
    logger.info("Rebuilding BM25 lexical index with %d chunks...", len(chunks))
    t0 = time.time()
    retriever = BM25Retriever()
    retriever.build_index_from_chunks(chunks)
    retriever.save_index(output_path)
    logger.info("BM25 index built and saved to %s in %.1fs (Size: %.2f MB)", output_path, time.time() - t0, os.path.getsize(output_path) / (1024 * 1024))


def rebuild_qdrant(chunks: List[Chunk], qdrant_path: str = "data/qdrant", target_collection: str = "msmarco_xi"):
    logger.info("Connecting to Qdrant storage at %s...", qdrant_path)
    client = QdrantClient(path=qdrant_path)
    embedder = MultilingualE5EmbeddingProvider(batch_size=256)

    # Step 1: Create safe target collection msmarco_xi_v619
    temp_collection = f"{target_collection}_v619"
    logger.info("Creating clean collection: %s (384-dim Cosine)...", temp_collection)
    client.recreate_collection(
        collection_name=temp_collection,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

    batch_size = 256
    total_chunks = len(chunks)
    logger.info("Embedding and upserting %d chunks in batches of %d...", total_chunks, batch_size)
    t0 = time.time()
    upserted = 0

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        embeddings = embedder.embed_documents(texts)

        points = []
        for c, emb in zip(batch, embeddings):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, c.chunk_id))
            language = c.metadata.get("language") or getattr(c, "language", "unknown")
            payload = {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "chunk_type": c.chunk_type,
                "language": language,
                "text": c.text,
                **c.metadata,
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=emb,
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=temp_collection,
            points=points,
        )
        upserted += len(points)
        if upserted % 1024 == 0 or upserted == total_chunks:
            elapsed = time.time() - t0
            rate = upserted / elapsed if elapsed > 0 else 0
            logger.info("  Upserted %d / %d points (%.1f%%) | Rate: %.1f pts/sec", upserted, total_chunks, (upserted / total_chunks) * 100, rate)

    logger.info("Finished embedding and upserting %d points to %s in %.1fs.", upserted, temp_collection, time.time() - t0)

    # Step 2: Validate count in staging collection
    count_res = client.count(collection_name=temp_collection)
    logger.info("Validation: %s point count = %d", temp_collection, count_res.count)
    if count_res.count != total_chunks:
        raise RuntimeError(f"Point count mismatch! Expected {total_chunks}, got {count_res.count}")

    # Step 3: Promote staging collection to primary msmarco_xi
    logger.info("Promoting %s -> %s...", temp_collection, target_collection)
    client.recreate_collection(
        collection_name=target_collection,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

    # Copy points from staging to primary in batches
    scroll_offset = None
    copied = 0
    while True:
        records, next_offset = client.scroll(
            collection_name=temp_collection,
            limit=500,
            offset=scroll_offset,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            break
        copy_points = [
            models.PointStruct(
                id=r.id,
                vector=r.vector,
                payload=r.payload,
            )
            for r in records
        ]
        client.upsert(
            collection_name=target_collection,
            points=copy_points,
        )
        copied += len(copy_points)
        if next_offset is None:
            break
        scroll_offset = next_offset

    final_count = client.count(collection_name=target_collection).count
    logger.info("Promotion complete: %s has %d verified points.", target_collection, final_count)

    # Clean up staging collection
    try:
        client.delete_collection(temp_collection)
        logger.info("Cleaned up staging collection %s.", temp_collection)
    except Exception as e:
        logger.warning("Could not delete staging collection: %s", e)


def main():
    all_chunks, qdrant_chunks = load_all_multilingual_chunks(max_per_lang=2500)
    if not all_chunks:
        logger.error("No chunks loaded. Aborting index rebuild.")
        return

    # Rebuild BM25 across all 48,206 chunks
    rebuild_bm25(all_chunks)

    # Rebuild Qdrant across 12,500 balanced multilingual chunks
    rebuild_qdrant(qdrant_chunks)
    logger.info("Multilingual Index Rebuild complete for all 5 languages!")


if __name__ == "__main__":
    main()
