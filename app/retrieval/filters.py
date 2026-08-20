"""Filtering utilities for chunking strategies and language criteria using centralized language resolver."""

from typing import List, Optional, Union
from qdrant_client.http import models
from app.retrieval.language import get_language_filter_synonyms, normalize_language_code


def get_language_synonyms(lang: str) -> List[str]:
    """Resolve a language code to all known alias variants."""
    return get_language_filter_synonyms(lang)


def build_qdrant_filter(
    strategies: Optional[Union[str, List[str]]] = None,
    language: Optional[str] = None,
) -> Optional[models.Filter]:
    """Construct a Qdrant Filter object matching strategy and language conditions."""
    conditions = []

    # 1. Strategy condition
    if strategies is not None:
        if isinstance(strategies, str):
            strat_list = [s.strip().lower() for s in strategies.split(",") if s.strip()]
        else:
            strat_list = [s.strip().lower() for s in strategies if s.strip()]

        if strat_list and "all" not in strat_list:
            if len(strat_list) == 1:
                conditions.append(
                    models.FieldCondition(
                        key="chunk_type",
                        match=models.MatchValue(value=strat_list[0]),
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key="chunk_type",
                        match=models.MatchAny(any=strat_list),
                    )
                )

    # 2. Language condition (with canonical alias expansion)
    if language and language.strip().lower() not in ("all", "none", "*", "auto"):
        synonyms = get_language_filter_synonyms(language)
        if synonyms:
            if len(synonyms) == 1:
                conditions.append(
                    models.FieldCondition(
                        key="language",
                        match=models.MatchValue(value=synonyms[0]),
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key="language",
                        match=models.MatchAny(any=synonyms),
                    )
                )

    if not conditions:
        return None

    return models.Filter(must=conditions)


def matches_filter(
    metadata: dict,
    chunk_type: str,
    strategies: Optional[Union[str, List[str]]] = None,
    language: Optional[str] = None,
) -> bool:
    """Check if a chunk record matches the strategy and language criteria (for BM25 / in-memory)."""
    # Strategy match
    if strategies is not None:
        if isinstance(strategies, str):
            strat_list = [s.strip().lower() for s in strategies.split(",") if s.strip()]
        else:
            strat_list = [s.strip().lower() for s in strategies if s.strip()]

        if strat_list and "all" not in strat_list:
            if chunk_type.lower() not in strat_list:
                return False

    # Language match
    if language and language.strip().lower() not in ("all", "none", "*", "auto"):
        chunk_lang = (metadata.get("language") or "").strip()
        synonyms = [s.lower() for s in get_language_filter_synonyms(language)]
        if chunk_lang and chunk_lang.lower() not in synonyms and not any(chunk_lang.lower().startswith(s) for s in synonyms):
            return False

    return True
