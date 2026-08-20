"""Off-topic query detection, multi-signal evidence relevance evaluation, and answerability validation (Phase 6.26)."""

import logging
import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from app.guardrails.models import RelevanceAssessment
from app.retrieval.bm25 import IndicUnicodeTokenizer
from app.reranking.models import RerankResult

logger = logging.getLogger(__name__)


# Multilingual functional stopword list across English, Hindi, Tamil, Telugu, and Malayalam
MULTILINGUAL_STOPWORDS = {
    # English
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "can", "will", "just", "should", "now", "tell", "me", "define",
    "explain", "give", "please", "meaning", "name", "list", "today", "current", "latest",
    "my", "near", "best", "it", "its", "it's", "they", "them", "their", "theirs", "he",
    "him", "his", "she", "her", "hers", "we", "us", "our", "ours", "you", "your", "yours",
    # Hindi
    "क्या", "है", "हैं", "था", "थी", "थे", "का", "की", "के", "में", "पर", "से", "को",
    "और", "या", "एक", "यह", "वह", "जो", "तो", "भी", "बताओ", "दीजिए", "मुझे", "अर्थ",
    "किसे", "कहते", "बारे", "किस", "कहा", "जाता", "होता", "होती", "होते", "मेरा", "मेरी",
    "आज", "पास",
    # Tamil
    "என்ன", "எப்படி", "எங்கே", "யார்", "எது", "ஆகும்", "உள்ளது", "மற்றும்", "ஒரு",
    "இந்த", "அந்த", "இல்", "க்கு", "என்றால்", "சொல்லுங்கள்", "விளக்குங்கள்", "பற்றி",
    "விளக்கம்", "ஆகியவை", "என்பது", "எனப்படுவது", "இன்று", "என்",
    # Telugu
    "ఏమిటి", "ఎలా", "ఎక్కడ", "ఎవరు", "మరియు", "ఒక", "ఈ", "ఆ", "లో", "కి", "కు",
    "ఉంది", "అంటే", "చెప్పండి", "వివరించండి", "గురించి", "అర్థం", "అంటారు", "ఈరోజు", "నా",
    # Malayalam
    "എന്ത്", "എങ്ങനെ", "എവിടെ", "ആര്", "ആണ്", "ഉണ്ട്", "കൂടാതെ", "ഒരു", "ഈ", "ആ",
    "ഇൽ", "ക്ക്", "ആയി", "എന്താണ്", "ഏതാണ്", "ഏത്", "പറയുക", "വ്യക്തമാക്കുക", "കുറിച്ച്",
    "അർത്ഥം", "എന്നാൽ", "ആകുന്നു", "ഇന്ന്", "എന്റെ",
}


class AnswerabilityResult(BaseModel):
    """Result of pre-generation answerability validation."""

    allowed: bool = Field(..., description="Whether the retrieved context provides sufficient evidence to answer the query.")
    reason: str = Field(..., description="Categorization code: answerable, empty_retrieval_context, low_semantic_relevance, insufficient_content_overlap, off_topic_query, insufficient_content_words.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall answerability confidence score.")
    decision: str = Field(default="STRONG", description="Decision bucket: STRONG, LIMITED, INSUFFICIENT, OFF_TOPIC.")
    signals: Optional[Dict[str, float]] = Field(default=None, description="Multi-signal relevance breakdown.")


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert value to float without triggering MagicMock auto-creation or type errors."""
    if val is None:
        return default
    # Handle MagicMock instances to avoid returning mock objects
    if hasattr(val, "_mock_name") or hasattr(val, "_mock_methods"):
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        if isinstance(val, str):
            return float(val)
    except (ValueError, TypeError):
        pass
    return default


def extract_content_words(text: str) -> set:
    """Extract lowercase content-word tokens excluding common functional stopwords and punctuation."""
    if not text or not isinstance(text, str):
        return set()
    cleaned = text.lower().strip()
    raw_words = cleaned.split()
    words = [w.strip(".,!?:;\"'()[]{}—–-।|/\\`~@#$%^&*+=<>") for w in raw_words]
    return {w for w in words if w and w not in MULTILINGUAL_STOPWORDS and len(w) > 1}


def is_meaningful_word_match(qw: str, cw: str) -> bool:
    """Check if query content word meaningfully matches context word without false fragment overlap."""
    if qw == cw:
        return True
    # Exact stem / inflection match for words with length >= 4
    if len(qw) >= 4 and len(cw) >= 4:
        if qw.startswith(cw) or cw.startswith(qw) or qw.endswith(cw) or cw.endswith(qw):
            shorter_len = min(len(qw), len(cw))
            longer_len = max(len(qw), len(cw))
            if shorter_len / longer_len >= 0.60:
                return True
    return False


def compute_content_overlap(query_text: str, context_text: str) -> float:
    """Compute strict content-word overlap ratio between query and context."""
    q_words = extract_content_words(query_text)
    if not q_words:
        return 0.0
    c_words = extract_content_words(context_text)
    if not c_words:
        return 0.0

    matched = 0
    for qw in q_words:
        if any(is_meaningful_word_match(qw, cw) for cw in c_words):
            matched += 1
    return matched / len(q_words)


UNANSWERABLE_CORPUS_PATTERNS = [
    re.compile(r"\bmy\s+(bank|account|balance|timetable|schedule|college|exam|password|email|phone|cart|order|name)\b", re.IGNORECASE),
    re.compile(r"\bnear\s+me\b", re.IGNORECASE),
    re.compile(r"\b(current|latest|today['s]?)\s+(weather|temperature|petrol|diesel|cricket|score|stock|gold|time)\b", re.IGNORECASE),
    re.compile(r"\bweather\s+in\s+[a-zA-Z]+\s+today\b", re.IGNORECASE),
    re.compile(r"\bprice\s+of\s+petrol\s+today\b", re.IGNORECASE),
    re.compile(r"\blatest\s+cricket\s+score\b", re.IGNORECASE),
    re.compile(r"\bprime\s+minister\s+of\s+india\b", re.IGNORECASE),
]


def validate_answerability(
    query: str,
    retrieved_context: Union[str, List[Any]],
    language: Optional[str] = None,
) -> AnswerabilityResult:
    """Determine whether the retrieved context contains sufficient evidence to answer the question.

    Signals Evaluated:
    1. Cross-encoder relevance score (reranker_score)
    2. Content-word lexical overlap (excluding functional stopwords)
    3. Query-document dense semantic similarity (dense_score)
    4. Retrieval agreement between dense & BM25 retrieval (fusion_score)
    5. Evidence concentration across top-ranked chunks

    Decisions:
    - STRONG: High multi-signal confidence; allow grounded generation
    - LIMITED: Moderate evidence; allow only with conservative generation
    - INSUFFICIENT: Low multi-signal relevance; refuse generation
    - OFF_TOPIC: Zero lexical overlap and low semantic relevance; refuse generation
    """
    clean_query = query.strip() if query else ""
    if not clean_query:
        return AnswerabilityResult(
            allowed=False,
            reason="empty_retrieval_context",
            confidence=0.0,
            decision="OFF_TOPIC",
            signals={"reranker": 0.0, "lexical": 0.0, "semantic": 0.0, "agreement": 0.0},
        )

    # Check for known unanswerable query types (personal data, real-time live data, location-dependent)
    for pat in UNANSWERABLE_CORPUS_PATTERNS:
        if pat.search(clean_query):
            return AnswerabilityResult(
                allowed=False,
                reason="off_topic_query",
                confidence=0.0,
                decision="OFF_TOPIC",
                signals={"reranker": 0.0, "lexical": 0.0, "semantic": 0.0, "agreement": 0.0},
            )

    q_words = extract_content_words(clean_query)
    if not q_words:
        # Queries with only stopwords (e.g. "Is it in it?") cannot be grounded
        return AnswerabilityResult(
            allowed=False,
            reason="insufficient_content_words",
            confidence=0.0,
            decision="OFF_TOPIC",
            signals={"reranker": 0.0, "lexical": 0.0, "semantic": 0.0, "agreement": 0.0},
        )

    if not retrieved_context:
        return AnswerabilityResult(
            allowed=False,
            reason="empty_retrieval_context",
            confidence=0.0,
            decision="INSUFFICIENT",
            signals={"reranker": 0.0, "lexical": 0.0, "semantic": 0.0, "agreement": 0.0},
        )

    context_texts: List[str] = []
    reranker_scores: List[float] = []
    dense_scores: List[float] = []
    fusion_scores: List[float] = []
    context_languages = set()

    if isinstance(retrieved_context, str):
        if retrieved_context.strip():
            context_texts.append(retrieved_context.strip())
            dense_scores.append(0.5)
    elif isinstance(retrieved_context, list):
        for item in retrieved_context:
            if isinstance(item, RerankResult):
                if item.text:
                    context_texts.append(item.text)
                if item.language:
                    context_languages.add(item.language)
                r = _safe_float(item.reranker_score)
                d = _safe_float(item.dense_score)
                f = _safe_float(item.fusion_score)
                if r > 0:
                    reranker_scores.append(r)
                if d > 0:
                    dense_scores.append(d)
                if f > 0:
                    fusion_scores.append(f)
            elif isinstance(item, dict):
                txt = str(item.get("text") or item.get("content") or "")
                if txt:
                    context_texts.append(txt)
                if item.get("language"):
                    context_languages.add(item.get("language"))
                r = _safe_float(item.get("reranker_score"))
                d = _safe_float(item.get("dense_score") or item.get("score"))
                f = _safe_float(item.get("fusion_score"))
                if r > 0:
                    reranker_scores.append(r)
                if d > 0:
                    dense_scores.append(d)
                if f > 0:
                    fusion_scores.append(f)
            elif isinstance(item, str) and item.strip():
                context_texts.append(item.strip())
            else:
                txt = getattr(item, "text", None)
                if isinstance(txt, str) and txt:
                    context_texts.append(txt)
                lang = getattr(item, "language", None)
                if isinstance(lang, str):
                    context_languages.add(lang)
                r = _safe_float(getattr(item, "reranker_score", None))
                d = _safe_float(getattr(item, "dense_score", None))
                f = _safe_float(getattr(item, "fusion_score", None))
                if r == 0.0 and d == 0.0:
                    gen_score = _safe_float(getattr(item, "score", None))
                    d = gen_score
                if r > 0:
                    reranker_scores.append(r)
                if d > 0:
                    dense_scores.append(d)
                if f > 0:
                    fusion_scores.append(f)

    if not context_texts or not "".join(context_texts).strip():
        return AnswerabilityResult(
            allowed=False,
            reason="empty_retrieval_context",
            confidence=0.0,
            decision="INSUFFICIENT",
            signals={"reranker": 0.0, "lexical": 0.0, "semantic": 0.0, "agreement": 0.0},
        )

    # 1. Content-Word Overlap Signal
    combined_context = " ".join(context_texts)
    lexical_overlap = compute_content_overlap(clean_query, combined_context)

    # 2. Cross-Encoder Reranker Signal
    max_reranker = max(reranker_scores) if reranker_scores else 0.0

    # 3. Dense Semantic Similarity Signal (Normalized for E5-small cosine: 0.60..0.90 -> 0.0..1.0)
    max_dense = max(dense_scores) if dense_scores else 0.0
    semantic_norm = max(0.0, min(1.0, (max_dense - 0.60) / 0.30)) if max_dense > 0 else 0.0

    # 4. Retrieval Agreement / Fusion Signal
    max_fusion = max(fusion_scores) if fusion_scores else 0.0
    agreement_signal = min(1.0, max_fusion * 40.0) if max_fusion > 0 else (0.5 if (max_reranker > 0.3 and max_dense > 0.7) else 0.0)

    # 5. Multi-Signal Weighted Relevance Score
    # Cross-encoder: 40%, Lexical overlap: 30%, Dense semantic: 20%, Agreement: 10%
    relevance_score = (
        0.40 * max_reranker
        + 0.30 * lexical_overlap
        + 0.20 * semantic_norm
        + 0.10 * agreement_signal
    )

    signals_dict = {
        "reranker": round(max_reranker, 4),
        "lexical": round(lexical_overlap, 4),
        "semantic": round(semantic_norm, 4),
        "agreement": round(agreement_signal, 4),
        "composite": round(relevance_score, 4),
    }

    # -------------------------------------------------------------------------
    # Decision Engine: Transparent Multi-Signal Thresholds
    # -------------------------------------------------------------------------

    # Condition 1: Completely off-topic or unrelated context
    if lexical_overlap == 0.0 and max_reranker < 0.20 and max_dense < 0.72:
        return AnswerabilityResult(
            allowed=False,
            reason="insufficient_content_overlap",
            confidence=round(max(max_reranker, max_dense * 0.2), 4),
            decision="OFF_TOPIC",
            signals=signals_dict,
        )

    # Condition 2: Off-topic query or incidental mention with low semantic and reranker relevance
    if max_reranker < 0.15 and max_dense < 0.65:
        return AnswerabilityResult(
            allowed=False,
            reason="low_semantic_relevance",
            confidence=round(max(relevance_score, 0.08), 4),
            decision="INSUFFICIENT",
            signals=signals_dict,
        )

    # Condition 3: Strong relevant evidence
    if (
        max_reranker >= 0.30
        or (lexical_overlap >= 0.50 and max_reranker >= 0.18 and max_dense >= 0.68)
        or (max_dense >= 0.80 and max_reranker >= 0.15)
        or (lexical_overlap >= 0.80 and max_dense >= 0.68)
        or relevance_score >= 0.48
    ):
        conf = max(0.65, relevance_score, max_reranker)
        return AnswerabilityResult(
            allowed=True,
            reason="answerable",
            confidence=min(1.0, round(conf, 4)),
            decision="STRONG",
            signals=signals_dict,
        )

    # Condition 4: Limited / partial evidence
    if (
        (lexical_overlap >= 0.40 and max_dense >= 0.68)
        or (max_reranker >= 0.18 and max_dense >= 0.70)
        or (max_dense >= 0.72)  # Multilingual cross-lingual semantic alignment
        or relevance_score >= 0.30
    ):
        conf = max(0.45, relevance_score, 0.5 * max_reranker + 0.5 * lexical_overlap)
        return AnswerabilityResult(
            allowed=True,
            reason="answerable",
            confidence=min(1.0, round(conf, 4)),
            decision="LIMITED",
            signals=signals_dict,
        )

    # Condition 5: Insufficient evidence default
    return AnswerabilityResult(
        allowed=False,
        reason="low_semantic_relevance",
        confidence=round(relevance_score, 4),
        decision="INSUFFICIENT",
        signals=signals_dict,
    )


class RelevanceGuard:
    """Evaluates whether the user's query is relevant to the retrieved evidence, detecting off-topic questions."""

    def __init__(self, min_relevance_threshold: float = 0.20):
        self.min_relevance_threshold = min_relevance_threshold

    def evaluate_relevance(
        self,
        query: str,
        retrieved_context: Union[str, List[Any]],
    ) -> RelevanceAssessment:
        """Assess query-evidence relevance using retrieval scores and multilingual content word overlap.

        Args:
            query: User question string.
            retrieved_context: Retrieved chunk list or raw context text.

        Returns:
            RelevanceAssessment with relevance_score, is_relevant flag, decision, signals, and rationale.
        """
        ans_res = validate_answerability(query, retrieved_context)

        reason_map = {
            "answerable": "Query has strong semantic and lexical overlap with retrieved evidence.",
            "empty_retrieval_context": "No retrieved context available to evaluate relevance.",
            "insufficient_content_overlap": "Query appears off-topic; evidence content-word overlap is 0.00.",
            "insufficient_content_words": "Query contains only functional stopwords without substantive content terms.",
            "low_semantic_relevance": "Query evidence relevance is below calibrated threshold.",
            "language_mismatch": "Retrieved context language differs from user query without cross-lingual alignment.",
        }

        # Calculate overlap ratio for telemetry
        clean_query = query.strip() if query else ""
        combined_context = ""
        if isinstance(retrieved_context, str):
            combined_context = retrieved_context
        elif isinstance(retrieved_context, list):
            texts = []
            for item in retrieved_context:
                if hasattr(item, "text"):
                    t = getattr(item, "text", "")
                    if isinstance(t, str):
                        texts.append(t)
                elif isinstance(item, dict):
                    texts.append(str(item.get("text") or item.get("content") or ""))
                elif isinstance(item, str):
                    texts.append(item)
            combined_context = " ".join(texts)

        overlap = compute_content_overlap(clean_query, combined_context) if combined_context else 0.0

        return RelevanceAssessment(
            is_relevant=ans_res.allowed,
            relevance_score=ans_res.confidence,
            overlap_ratio=round(overlap, 4),
            reason=reason_map.get(ans_res.reason, ans_res.reason),
            decision=ans_res.decision,
            signals=ans_res.signals,
        )
