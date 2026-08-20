"""Generate canonical multilingual chunks for English, Hindi, Tamil, Telugu, and Malayalam."""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List
import pyarrow.parquet as pq

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.chunking.registry import ChunkingRegistry
from app.ingestion.models import Document
from app.ingestion.normalizer import normalize_text

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s")
logger = logging.getLogger("generate_multilingual_chunks")

LANGUAGE_CONFIGS = [
    {
        "lang_code": "hi",
        "canonical_lang": "hin_Deva",
        "lang_name": "Hindi",
        "parquet_file": "data/raw/hinval.parquet",
        "passage_key": "Translated_passages",
        "query_key": "query",
        "answer_key": "Answer",
        "output_file": "data/processed/chunks_hindi.jsonl",
    },
    {
        "lang_code": "en",
        "canonical_lang": "eng_Latn",
        "lang_name": "English",
        "parquet_file": "data/raw/hinval.parquet",
        "passage_key": "English_passages",
        "query_key": "Eng_Query",
        "answer_key": "Eng_Answer",
        "output_file": "data/processed/chunks_english.jsonl",
    },
    {
        "lang_code": "ta",
        "canonical_lang": "tam_Taml",
        "lang_name": "Tamil",
        "parquet_file": "data/raw/tamval.parquet",
        "passage_key": "Translated_passages",
        "query_key": "query",
        "answer_key": "Answer",
        "output_file": "data/processed/chunks_tamil.jsonl",
    },
    {
        "lang_code": "te",
        "canonical_lang": "tel_Telu",
        "lang_name": "Telugu",
        "parquet_file": "data/raw/telval.parquet",
        "passage_key": "Translated_passages",
        "query_key": "query",
        "answer_key": "Answer",
        "output_file": "data/processed/chunks_telugu.jsonl",
    },
    {
        "lang_code": "ml",
        "canonical_lang": "mal_Mlym",
        "lang_name": "Malayalam",
        "parquet_file": "data/raw/malval.parquet",
        "passage_key": "Translated_passages",
        "query_key": "query",
        "answer_key": "Answer",
        "output_file": "data/processed/chunks_malayalam.jsonl",
    },
]


def process_language(cfg: Dict[str, Any], sample_size: int = 100) -> int:
    parquet_path = cfg["parquet_file"]
    if not os.path.exists(parquet_path):
        logger.error("Parquet file %s not found. Skipping %s.", parquet_path, cfg["lang_name"])
        return 0

    canonical_lang = cfg["canonical_lang"]
    lang_name = cfg["lang_name"]
    passage_key = cfg["passage_key"]
    query_key = cfg["query_key"]
    answer_key = cfg["answer_key"]
    output_path = cfg["output_file"]

    logger.info("Processing %s (%s) from %s...", lang_name, canonical_lang, parquet_path)
    pf = pq.ParquetFile(parquet_path)
    
    documents: List[Document] = []
    yielded = 0

    for rg_idx in range(pf.num_row_groups):
        if yielded >= sample_size:
            break
        rg = pf.read_row_group(rg_idx)
        cols = rg.column_names
        num_rows = len(rg)

        for i in range(num_rows):
            if yielded >= sample_size:
                break
            record = {c: rg[c][i].as_py() for c in cols}
            query_id = record.get("query_id")
            if query_id is None:
                continue

            query = record.get(query_key) or record.get("query") or record.get("Eng_Query") or ""
            answer = record.get(answer_key) or record.get("Answer") or record.get("Eng_Answer") or ""
            query_type = record.get("query_type", "")
            passages_data = record.get("passages", {}) or {}
            target_passages = passages_data.get(passage_key, []) or []
            is_selected_list = passages_data.get("is_selected", []) or []

            for p_idx, p_text in enumerate(target_passages):
                norm_text = normalize_text(p_text)
                if not norm_text:
                    continue
                is_selected = int(is_selected_list[p_idx]) if p_idx < len(is_selected_list) else 0
                doc_id = f"msmarco_{canonical_lang}_{query_id}_{p_idx}"

                meta = {
                    "query_id": query_id,
                    "query": query,
                    "answer": answer,
                    "query_type": query_type,
                    "passage_index": p_idx,
                    "is_selected": is_selected,
                    "language": canonical_lang,
                    "language_name": lang_name,
                    "source": "ai4bharat/MSMARCO-XI",
                }

                doc = Document(
                    document_id=doc_id,
                    text=norm_text,
                    title=None,
                    language=canonical_lang,
                    source="ai4bharat/MSMARCO-XI",
                    metadata=meta,
                )
                documents.append(doc)

            yielded += 1

    logger.info("Generated %d Document objects from %d query records for %s.", len(documents), yielded, lang_name)

    # Run Chunking Strategies
    strategies = ChunkingRegistry.get_configured_strategies()
    total_chunks = 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f_out:
        for strat in strategies:
            strat_chunks = 0
            for doc in documents:
                chunks = strat.chunk(doc)
                for c in chunks:
                    c.metadata["language"] = canonical_lang
                    c.metadata["language_name"] = lang_name
                    line = json.dumps(c.model_dump(), ensure_ascii=False)
                    f_out.write(line + "\n")
                    strat_chunks += 1
            total_chunks += strat_chunks
            logger.info("  Strategy '%s': %d chunks", strat.strategy_name, strat_chunks)

    logger.info("Saved %d total chunks for %s -> %s", total_chunks, lang_name, output_path)
    return total_chunks


def main():
    t0 = time.time()
    total = 0
    for cfg in LANGUAGE_CONFIGS:
        cnt = process_language(cfg, sample_size=100)
        total += cnt
    logger.info("Multilingual chunk generation complete: %d total chunks in %.1fs", total, time.time() - t0)


if __name__ == "__main__":
    main()
