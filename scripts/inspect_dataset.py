"""Inspection script for ai4bharat/MSMARCO-XI dataset schema, columns, splits, and records."""

import json
import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pyarrow.parquet as pq
from app.ingestion.dataset_loader import DatasetLoader, LANGUAGE_CODE_MAP


def run_inspection():
    """Inspect and display MSMARCO-XI dataset schema and sample records."""
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 70)
    print("MSMARCO-XI DATASET SCHEMA & ARCHITECTURE INSPECTION")
    print("=" * 70)

    loader = DatasetLoader(
        dataset_name="ai4bharat/MSMARCO-XI",
        split="validation",
        language="hi",
        sample_size=5,
    )

    print(f"\nDataset Identifier : {loader.dataset_name}")
    print(f"Configured Split   : {loader.split}")
    print(f"Target Language    : {loader.language} (Hindi)")
    print(f"Supported Languages: {list(LANGUAGE_CODE_MAP.keys())}")
    print(f"Available Splits   : ['train', 'validation']")

    print("\nEnsuring data file availability...")
    data_file = loader.ensure_data_file()
    print(f"Loaded Parquet File: {data_file} ({os.path.getsize(data_file) / (1024*1024):.2f} MB)")

    # Read Parquet schema
    pq_file = pq.ParquetFile(data_file)
    schema = pq_file.schema_arrow

    print("\n" + "-" * 70)
    print("ACTUAL PARQUET SCHEMA & DATA TYPES")
    print("-" * 70)
    for field in schema:
        print(f"  - Field Name : {field.name}")
        print(f"    Data Type  : {field.type}")
        print(f"    Nullable   : {field.nullable}")

    print(f"\nTotal Records in File: {pq_file.metadata.num_rows:,}")
    print(f"Row Groups Count     : {pq_file.num_row_groups}")

    print("\n" + "-" * 70)
    print("DETAILED FIELD CLASSIFICATION")
    print("-" * 70)
    print("  [Document/Passage Fields]:")
    print("    - passages.Translated_passages (list<string>): Translated Indic passage texts")
    print("    - passages.English_passages (list<string>)   : Original English passage texts")
    print("  [Query Fields]:")
    print("    - query (string)     : Translated user query")
    print("    - Eng_Query (string) : Original English query")
    print("  [Relevance / Target Fields]:")
    print("    - passages.is_selected (list<int32>): Binary relevance label per passage (1=selected, 0=not)")
    print("    - Answer (string)    : Translated reference answer")
    print("    - Eng_Answer (string): Original English reference answer")
    print("  [Identifiers]:")
    print("    - query_id (int32)   : Unique query cluster identifier")
    print("  [Language / Metadata]:")
    print("    - source_lang (string): e.g. 'eng_Latn'")
    print("    - target_lang (string): e.g. 'hin_Deva'")
    print("    - query_type (string) : Query category (e.g., 'DESCRIPTION', 'NUMERIC')")
    print("    - meta (struct)       : Translation parameters (model_name, temperature, etc.)")

    print("\n" + "-" * 70)
    print("SAMPLE RECORD INSPECTION (Record 0)")
    print("-" * 70)
    records = list(loader.load_records(limit=1))
    if records:
        rec = records[0]
        preview = {
            "query_id": rec.get("query_id"),
            "query": rec.get("query"),
            "query_type": rec.get("query_type"),
            "source_lang": rec.get("source_lang"),
            "target_lang": rec.get("target_lang"),
            "Eng_Query": rec.get("Eng_Query"),
            "num_translated_passages": len(rec.get("passages", {}).get("Translated_passages", [])),
            "is_selected": rec.get("passages", {}).get("is_selected", []),
            "first_passage_sample": (
                rec.get("passages", {}).get("Translated_passages", [""])[0][:150] + "..."
                if rec.get("passages", {}).get("Translated_passages")
                else ""
            ),
        }
        print(json.dumps(preview, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE — READY FOR NORMALIZATION & CHUNKING")
    print("=" * 70)


if __name__ == "__main__":
    run_inspection()
