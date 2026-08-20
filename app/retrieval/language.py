"""Centralized Language Resolver and Canonical Normalization for Multilingual RAG."""

from typing import List, Optional

# Canonical 5-language mapping for HH Goa 2026 Multilingual Voice RAG
CANONICAL_LANGUAGE_MAP = {
    # English
    "en": "eng_Latn",
    "eng": "eng_Latn",
    "eng_latn": "eng_Latn",
    "eng_Latn": "eng_Latn",
    "english": "eng_Latn",
    "en-in": "eng_Latn",
    "en-us": "eng_Latn",

    # Hindi
    "hi": "hin_Deva",
    "hin": "hin_Deva",
    "hin_deva": "hin_Deva",
    "hin_Deva": "hin_Deva",
    "hindi": "hin_Deva",
    "hi-in": "hin_Deva",

    # Tamil
    "ta": "tam_Taml",
    "tam": "tam_Taml",
    "tam_taml": "tam_Taml",
    "tam_Taml": "tam_Taml",
    "tamil": "tam_Taml",
    "ta-in": "tam_Taml",

    # Telugu
    "te": "tel_Telu",
    "tel": "tel_Telu",
    "tel_telu": "tel_Telu",
    "tel_Telu": "tel_Telu",
    "telugu": "tel_Telu",
    "te-in": "tel_Telu",

    # Malayalam
    "ml": "mal_Mlym",
    "mal": "mal_Mlym",
    "mal_mlym": "mal_Mlym",
    "mal_Mlym": "mal_Mlym",
    "malayalam": "mal_Mlym",
    "ml-in": "mal_Mlym",
}

BCP47_MAP = {
    # Flores codes
    "eng_latn": "en",
    "hin_deva": "hi",
    "tam_taml": "ta",
    "tel_telu": "te",
    "mal_mlym": "ml",
    "eng-latn": "en",
    "hin-deva": "hi",
    "tam-taml": "ta",
    "tel-telu": "te",
    "mal-mlym": "ml",
    # 2-letter codes
    "en": "en",
    "hi": "hi",
    "ta": "ta",
    "te": "te",
    "ml": "ml",
    # Full names
    "english": "en",
    "hindi": "hi",
    "tamil": "ta",
    "telugu": "te",
    "malayalam": "ml",
    # Regional locale tags
    "en-in": "en",
    "hi-in": "hi",
    "ta-in": "ta",
    "te-in": "te",
    "ml-in": "ml",
    "en_in": "en",
    "hi_in": "hi",
    "ta_in": "ta",
    "te_in": "te",
    "ml_in": "ml",
}

SARVAM_LANG_MAP = {
    "en": "en-IN",
    "hi": "hi-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "ml": "ml-IN",
    "eng_Latn": "en-IN",
    "hin_Deva": "hi-IN",
    "tam_Taml": "ta-IN",
    "tel_Telu": "te-IN",
    "mal_Mlym": "ml-IN",
}

LANGUAGE_DISPLAY_NAMES = {
    "eng_Latn": "English",
    "hin_Deva": "हिन्दी — Hindi",
    "tam_Taml": "தமிழ் — Tamil",
    "tel_Telu": "తెలుగు — Telugu",
    "mal_Mlym": "മലയാളം — Malayalam",
    "en": "English",
    "hi": "हिन्दी — Hindi",
    "ta": "தமிழ் — Tamil",
    "te": "తెలుగు — Telugu",
    "ml": "മലയാളം — Malayalam",
}

# Alias groupings for search filtering
LANGUAGE_ALIASES = {
    "eng_Latn": ["eng_Latn", "eng", "en", "english", "eng_latn", "en-in", "en-us"],
    "hin_Deva": ["hin_Deva", "hin", "hi", "hindi", "hin_deva", "hi-in"],
    "tam_Taml": ["tam_Taml", "tam", "ta", "tamil", "tam_taml", "ta-in"],
    "tel_Telu": ["tel_Telu", "tel", "te", "telugu", "tel_telu", "te-in"],
    "mal_Mlym": ["mal_Mlym", "mal", "ml", "malayalam", "mal_mlym", "ml-in"],
}


def detect_script_language(text: str) -> Optional[str]:
    """Detect the dominant language from Unicode script ranges.

    Ranges:
    - Devanagari (Hindi): U+0900 - U+097F
    - Tamil: U+0B80 - U+0BFF
    - Telugu: U+0C00 - U+0C7F
    - Malayalam: U+0D00 - U+0D7F
    - Latin (English): U+0041 - U+007A
    """
    if not text or not isinstance(text, str):
        return None

    counts = {"hi": 0, "ta": 0, "te": 0, "ml": 0, "en": 0}
    for char in text:
        cp = ord(char)
        if 0x0900 <= cp <= 0x097F:
            counts["hi"] += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            counts["ta"] += 1
        elif 0x0C00 <= cp <= 0x0C7F:
            counts["te"] += 1
        elif 0x0D00 <= cp <= 0x0D7F:
            counts["ml"] += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            counts["en"] += 1

    # Check for non-Latin Indic scripts first
    indic_max = max(counts["hi"], counts["ta"], counts["te"], counts["ml"])
    if indic_max > 0:
        for lang in ("hi", "ta", "te", "ml"):
            if counts[lang] == indic_max:
                return lang

    if counts["en"] > 0:
        return "en"

    return None


def normalize_language_code(lang: Optional[str]) -> Optional[str]:
    """Resolve any input language string or alias to its canonical Flores code (e.g. 'hin_Deva')."""
    if not lang:
        return None
    clean = lang.strip().lower()
    if clean in ("all", "none", "*", "auto"):
        return None
    return CANONICAL_LANGUAGE_MAP.get(clean, CANONICAL_LANGUAGE_MAP.get(clean.replace("_", "-"), clean))


def to_bcp47(lang: Optional[str], default: str = "en") -> str:
    """Normalize any language code into a 2-letter canonical BCP-47 code ('en', 'hi', 'ta', 'te', 'ml')."""
    if not lang:
        return default
    clean = str(lang).strip().lower()
    return BCP47_MAP.get(clean, BCP47_MAP.get(clean.replace("_", "-"), default))


def to_sarvam_code(lang: Optional[str], default: str = "en-IN") -> str:
    """Resolve language code for Sarvam API (e.g. 'hi' -> 'hi-IN', 'unknown' for auto-detect)."""
    if not lang:
        return default
    bcp = to_bcp47(lang, default=default)
    return SARVAM_LANG_MAP.get(bcp, default)


def get_language_filter_synonyms(lang: Optional[str]) -> List[str]:
    """Return all synonym forms for Qdrant and BM25 filter matches."""
    if not lang:
        return []
    canonical = normalize_language_code(lang)
    if canonical and canonical in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[canonical]
    clean = lang.strip().lower()
    return [canonical or lang, clean]

