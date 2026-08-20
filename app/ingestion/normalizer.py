"""Text and record normalization logic for MSMARCO-XI dataset."""

import re
from typing import Any, Dict, List
from app.ingestion.models import Document


def normalize_text(text: str) -> str:
    """Safely normalize text by standardizing whitespace and line breaks.
    
    Preserves all unicode characters, Indic scripts, punctuation, and semantic meaning.
    Does not summarize, rewrite, or alter text tokens.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Normalize carriage returns
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Normalize consecutive tabs and non-breaking spaces to standard space
    normalized = re.sub(r"[^\S\n]+", " ", normalized)
    
    # Normalize excessive consecutive newlines (more than 2 to 2)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    
    # Strip leading/trailing whitespace
    return normalized.strip()


class DocumentNormalizer:
    """Normalizes raw MSMARCO-XI records into Document objects."""

    def normalize_record(self, record: Dict[str, Any]) -> List[Document]:
        """Convert a single MSMARCO-XI raw record into a list of Document objects (one per passage).
        
        MSMARCO-XI Record Schema:
        - query_id (int)
        - query (str)
        - Answer (str)
        - query_type (str)
        - source_lang (str)
        - target_lang (str)
        - meta (dict: model_name, temperature, etc.)
        - passages (dict: is_selected, English_passages, Translated_passages)
        - Eng_Query (str)
        - Eng_Answer (str)
        """
        documents: List[Document] = []

        query_id = record.get("query_id")
        if query_id is None:
            return documents

        target_lang = record.get("target_lang")
        source_lang = record.get("source_lang")
        query = record.get("query", "")
        answer = record.get("Answer", "")
        query_type = record.get("query_type", "")
        eng_query = record.get("Eng_Query", "")
        eng_answer = record.get("Eng_Answer", "")
        model_meta = record.get("meta", {}) or {}

        passages_data = record.get("passages", {}) or {}
        translated_passages = passages_data.get("Translated_passages", []) or []
        english_passages = passages_data.get("English_passages", []) or []
        is_selected_list = passages_data.get("is_selected", []) or []

        # Process each passage
        num_passages = max(len(translated_passages), len(english_passages))
        for idx in range(num_passages):
            # Prefer translated passage in target language, fallback to English passage if missing
            passage_text = ""
            if idx < len(translated_passages) and translated_passages[idx]:
                passage_text = translated_passages[idx]
            elif idx < len(english_passages) and english_passages[idx]:
                passage_text = english_passages[idx]

            normalized_text = normalize_text(passage_text)
            if not normalized_text:
                continue

            eng_passage_text = english_passages[idx] if idx < len(english_passages) else ""
            is_selected = int(is_selected_list[idx]) if idx < len(is_selected_list) else 0

            # Deterministic document ID
            doc_id = f"msmarco_{query_id}_{idx}"

            # Preserve all relevant metadata from the dataset
            meta = {
                "query_id": query_id,
                "query": query,
                "answer": answer,
                "query_type": query_type,
                "passage_index": idx,
                "is_selected": is_selected,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "eng_query": eng_query,
                "eng_answer": eng_answer,
                "eng_passage": normalize_text(eng_passage_text) if eng_passage_text else "",
                "translation_meta": model_meta,
            }

            document = Document(
                document_id=doc_id,
                text=normalized_text,
                title=None,  # MSMARCO-XI passages do not contain independent title fields
                language=target_lang or source_lang,
                source="ai4bharat/MSMARCO-XI",
                metadata=meta,
            )
            documents.append(document)

        return documents
