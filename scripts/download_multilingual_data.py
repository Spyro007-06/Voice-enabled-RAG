"""Download Tamil, Telugu, and Malayalam validation parquet files in parallel from Hugging Face."""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download

sys.stdout.reconfigure(encoding="utf-8")

TASKS = [
    ("Tamil", "validation/tamval.parquet"),
    ("Telugu", "validation/telval.parquet"),
    ("Malayalam", "validation/malval.parquet"),
]

def download_one(task):
    lang, filename = task
    print(f"Starting download for {lang} ({filename})...")
    t0 = time.time()
    path = hf_hub_download(
        repo_id="ai4bharat/MSMARCO-XI",
        repo_type="dataset",
        filename=filename,
        local_dir="data/raw",
    )
    elapsed = time.time() - t0
    print(f"Completed {lang} in {elapsed:.1f}s -> {path}")
    return lang, path

if __name__ == "__main__":
    t_start = time.time()
    print("Starting concurrent download of Tamil, Telugu, Malayalam splits...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(download_one, TASKS))
    print(f"\nAll downloads completed in {time.time() - t_start:.1f}s:")
    for lang, path in results:
        print(f"  - {lang}: {path} ({os.path.getsize(path)/(1024*1024):.1f} MB)")
