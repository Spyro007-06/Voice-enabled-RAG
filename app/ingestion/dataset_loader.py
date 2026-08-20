"""Dataset loader for ai4bharat/MSMARCO-XI with streaming and batching support."""

import logging
import os
from typing import Any, Dict, Iterator, List, Optional
import pyarrow.parquet as pq
import requests

from app.config import get_settings

logger = logging.getLogger(__name__)

# Standard language code map for MSMARCO-XI
LANGUAGE_CODE_MAP = {
    "as": ("asm", "Assamese"),
    "bn": ("ben", "Bengali"),
    "gu": ("guj", "Gujarati"),
    "hi": ("hin", "Hindi"),
    "kn": ("kan", "Kannada"),
    "ml": ("mal", "Malayalam"),
    "mr": ("mar", "Marathi"),
    "ne": ("nep", "Nepali"),
    "or": ("ori", "Odia"),
    "pa": ("pan", "Punjabi"),
    "sa": ("san", "Sanskrit"),
    "ta": ("tam", "Tamil"),
    "te": ("tel", "Telugu"),
    "ur": ("urd", "Urdu"),
}


class DatasetLoader:
    """Loader for MSMARCO-XI dataset records."""

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        split: Optional[str] = None,
        language: Optional[str] = None,
        sample_size: Optional[int] = None,
        streaming: Optional[bool] = None,
        cache_dir: Optional[str] = None,
    ):
        settings = get_settings()
        self.dataset_name = dataset_name or settings.DATASET_NAME
        self.split = split or settings.DATASET_SPLIT
        self.language = language or getattr(settings, "DATASET_LANGUAGE", "hi")
        self.sample_size = sample_size if sample_size is not None else settings.DATASET_SAMPLE_SIZE
        self.streaming = streaming if streaming is not None else settings.DATASET_STREAMING
        self.cache_dir = cache_dir or getattr(settings, "DATASET_CACHE_DIR", "data/raw")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_filename_for_split(self) -> str:
        """Determine parquet filename in repo for given split and language."""
        lang_prefix = LANGUAGE_CODE_MAP.get(self.language, (self.language, self.language))[0]
        split_suffix = "val" if "val" in self.split.lower() else "train"
        folder = "validation" if "val" in self.split.lower() else "train"
        return f"{folder}/{lang_prefix}{split_suffix}.parquet"

    def _get_local_cache_path(self) -> str:
        """Return local filepath for cached parquet."""
        lang_prefix = LANGUAGE_CODE_MAP.get(self.language, (self.language, self.language))[0]
        split_suffix = "val" if "val" in self.split.lower() else "train"
        return os.path.join(self.cache_dir, f"{lang_prefix}{split_suffix}.parquet")

    def ensure_data_file(self) -> str:
        """Ensure the target parquet file is available locally."""
        local_path = self._get_local_cache_path()
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            logger.info("Found cached dataset file at %s (%d bytes)", local_path, os.path.getsize(local_path))
            return local_path

        rel_path = self._get_filename_for_split()
        url = f"https://huggingface.co/datasets/{self.dataset_name}/resolve/main/{rel_path}"
        logger.info("Downloading dataset from %s to %s", url, local_path)

        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            logger.info("Successfully downloaded %s (%.2f MB)", rel_path, downloaded / (1024 * 1024))
            return local_path
        except Exception as e:
            if os.path.exists(local_path):
                os.remove(local_path)
            raise RuntimeError(
                f"Failed to download MSMARCO-XI dataset file '{rel_path}' from {url}. Error: {e}"
            ) from e

    def load_records(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Yield raw records from the dataset up to limit or self.sample_size."""
        max_records = limit if limit is not None else self.sample_size
        file_path = self.ensure_data_file()

        logger.info(
            "Loading records from %s [split=%s, language=%s, limit=%s]",
            file_path,
            self.split,
            self.language,
            max_records,
        )

        try:
            parquet_file = pq.ParquetFile(file_path)
            arrow_schema = parquet_file.schema_arrow
            self._validate_schema(arrow_schema)

            yielded = 0
            for row_group_idx in range(parquet_file.num_row_groups):
                if max_records and yielded >= max_records:
                    break

                row_group = parquet_file.read_row_group(row_group_idx)
                cols = row_group.column_names
                num_rows = len(row_group)

                for i in range(num_rows):
                    if max_records and yielded >= max_records:
                        break

                    record = {col: row_group[col][i].as_py() for col in cols}
                    yield record
                    yielded += 1

            logger.info("Successfully yielded %d records", yielded)

        except Exception as e:
            raise RuntimeError(f"Error reading records from {file_path}: {e}") from e

    def _validate_schema(self, schema: Any) -> None:
        """Validate that the parquet file contains expected MSMARCO-XI columns."""
        required_cols = {"query_id", "passages"}
        found_cols = set(schema.names)
        missing = required_cols - found_cols
        if missing:
            raise ValueError(
                f"Invalid MSMARCO-XI schema in {self.dataset_name}. Missing required columns: {missing}. Found: {found_cols}"
            )
