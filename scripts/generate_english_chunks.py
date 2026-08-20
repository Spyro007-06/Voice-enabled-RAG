"""
Phase 6.16 - Generate English-language chunks from existing Hindi MSMARCO-XI chunk files.

Each Hindi chunk in data/processed/ already stores the original English passage text
in metadata.eng_passage. This script extracts those English passages and creates
proper Chunk objects with language="eng_Latn" so they can be indexed in Qdrant
and retrieved by English-language queries.

Output: data/processed/chunks_english.jsonl
"""

import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STRATEGY_FILES = [
    "fixed",
    "sentence",
    "sliding_window",
    "semantic",
    "hierarchical",
]

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "chunks_english.jsonl")


def make_english_chunk_id(original_chunk_id: str) -> str:
    raw_key = f"{original_chunk_id}:eng_Latn"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:8]
    return f"{original_chunk_id}_en_{digest}"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def generate_english_chunks():
    t_start = time.perf_counter()
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    total_read = 0
    total_skipped_no_eng = 0
    total_written = 0
    seen_doc_passage_ids = set()

    print("=" * 70)
    print("PHASE 6.16: GENERATING ENGLISH CHUNKS FROM HINDI MSMARCO-XI DATA")
    print("=" * 70)
    print(f"Reading strategy files: {STRATEGY_FILES}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for strategy in STRATEGY_FILES:
            jsonl_path = os.path.join(PROCESSED_DIR, f"chunks_{strategy}.jsonl")
            if not os.path.exists(jsonl_path):
                print(f"  [SKIP] {jsonl_path} not found")
                continue

            file_read = 0
            file_written = 0
            file_skipped_no_eng = 0
            file_skipped_dup = 0

            with open(jsonl_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    total_read += 1
                    file_read += 1

                    metadata = data.get("metadata", {}) or {}
                    eng_passage = (metadata.get("eng_passage") or "").strip()

                    if not eng_passage or len(eng_passage) < 20:
                        total_skipped_no_eng += 1
                        file_skipped_no_eng += 1
                        continue

                    doc_id = data.get("document_id", "")
                    dedup_key = f"{doc_id}:{strategy}"
                    if dedup_key in seen_doc_passage_ids:
                        file_skipped_dup += 1
                        continue
                    seen_doc_passage_ids.add(dedup_key)

                    original_chunk_id = data.get("chunk_id", "")
                    eng_chunk_id = make_english_chunk_id(original_chunk_id)

                    eng_metadata = dict(metadata)
                    eng_metadata["language"] = "eng_Latn"
                    eng_metadata["source_chunk_id"] = original_chunk_id
                    eng_metadata["source_language"] = metadata.get("language", "hin_Deva")
                    eng_metadata["strategy"] = strategy
                    if metadata.get("eng_answer"):
                        eng_metadata["answer"] = metadata["eng_answer"]
                    if metadata.get("eng_query"):
                        eng_metadata["query"] = metadata["eng_query"]

                    token_count = estimate_tokens(eng_passage)

                    english_chunk = {
                        "chunk_id": eng_chunk_id,
                        "document_id": doc_id,
                        "text": eng_passage,
                        "chunk_type": strategy,
                        "chunk_index": data.get("chunk_index", 0),
                        "start_position": 0,
                        "end_position": len(eng_passage),
                        "token_count": token_count,
                        "metadata": eng_metadata,
                    }

                    f_out.write(json.dumps(english_chunk, ensure_ascii=False) + "\n")
                    total_written += 1
                    file_written += 1

            print(f"  [{strategy:>16}] read={file_read:>5} | written={file_written:>5} | skipped_no_eng={file_skipped_no_eng:>4} | skipped_dup={file_skipped_dup:>4}")

    elapsed = time.perf_counter() - t_start
    output_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024) if os.path.exists(OUTPUT_FILE) else 0

    print()
    print("=" * 70)
    print("ENGLISH CHUNK GENERATION SUMMARY")
    print("=" * 70)
    print(f"Total Hindi chunks read      : {total_read:,}")
    print(f"Total English chunks written : {total_written:,}")
    print(f"Skipped (no eng_passage)     : {total_skipped_no_eng:,}")
    print(f"Output file                  : {OUTPUT_FILE}")
    print(f"Output size                  : {output_size_mb:.2f} MB")
    print(f"Elapsed                      : {elapsed:.2f}s")
    print("=" * 70)
    return total_written


if __name__ == "__main__":
    n = generate_english_chunks()
    print(f"\nDone. Generated {n:,} English chunks.")
    sys.exit(0 if n > 0 else 1)
