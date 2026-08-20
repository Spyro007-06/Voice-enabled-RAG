"""Adaptive retrieval and confidence-based dynamic reranking decision engine."""

import concurrent.futures
import logging
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

from app.config import get_settings
from app.reranking.base import Reranker
from app.reranking.bge_reranker import get_reranker
from app.reranking.context_selector import ContextSelector
from app.reranking.lightweight_reranker import get_lightweight_reranker
from app.reranking.models import (
    AdaptiveLatencyBreakdown,
    ContextSelectionStats,
    RerankResult,
)
from app.retrieval.cache import get_rag_cache
from app.retrieval.dense import DenseRetriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.filters import get_language_synonyms
from app.retrieval.fusion import HybridFusion
from app.retrieval.models import RetrievalResult
from app.retrieval.service import RetrievalService, get_retrieval_service

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceSignals:
    """Deterministic breakdown of retrieval confidence signals."""

    dense_similarity: float
    dense_margin: float
    bm25_strength: float
    retriever_agreement: float
    rrf_margin: float
    language_confidence: float
    document_diversity: float
    final_confidence: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class TierString(str):
    """Backward-compatible tier string supporting both legacy and Phase 6.4 tier names."""

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, str):
            return False
        s_self = str(self).lower()
        s_other = other.lower()
        if s_self == s_other:
            return True
        mapping = {
            "high": "skip",
            "skip": "high",
            "medium": "lightweight",
            "lightweight": "medium",
            "low": "full",
            "full": "low",
        }
        return mapping.get(s_self) == s_other


@dataclass
class AdaptiveDecision:
    """Decision output detailing confidence scores and reranking routing."""

    confidence_score: float
    dense_confidence: float
    retriever_agreement: float
    score_margin: float
    should_rerank: bool
    reranker_tier: str  # 'high' / 'skip', 'medium' / 'lightweight', 'low' / 'full'
    candidate_k: int
    reason: str
    reranker: str = "none"  # 'none', 'minilm', 'bge', 'fallback_rrf'
    reranking_used: bool = False
    signals: Optional[ConfidenceSignals] = None
    routing_tier: str = "high"


def calculate_retriever_agreement(
    dense_results: List[RetrievalResult],
    bm25_results: List[RetrievalResult],
    k: int = 5,
) -> float:
    """Calculate Jaccard overlap agreement between top-k dense and BM25 retrieval results."""
    if not dense_results or not bm25_results:
        return 0.0

    dense_top = set(r.chunk_id for r in dense_results[:k])
    bm25_top = set(r.chunk_id for r in bm25_results[:k])

    union = dense_top | bm25_top
    if not union:
        return 0.0

    intersection = dense_top & bm25_top
    return round(len(intersection) / len(union), 4)


def compute_confidence_signals(
    dense_results: List[RetrievalResult],
    bm25_results: List[RetrievalResult],
    fusion_results: List[RetrievalResult],
    language: Optional[str] = None,
    agreement_k: int = 5,
) -> ConfidenceSignals:
    """Compute 8 deterministic confidence signals across dense, BM25, and fusion outputs."""
    if not dense_results and not bm25_results and not fusion_results:
        return ConfidenceSignals(
            dense_similarity=0.0,
            dense_margin=0.0,
            bm25_strength=0.0,
            retriever_agreement=0.0,
            rrf_margin=0.0,
            language_confidence=0.0,
            document_diversity=0.0,
            final_confidence=0.0,
        )

    # 1. Dense top-1 similarity signal [0.50, 0.90] -> [0.0, 1.0]
    dense_sim = 0.0
    dense_margin = 0.0
    if dense_results:
        raw_dense_0 = dense_results[0].dense_score or 0.0
        dense_sim = max(0.0, min(1.0, (raw_dense_0 - 0.50) / 0.40))
        if len(dense_results) >= 2:
            raw_dense_1 = dense_results[1].dense_score or 0.0
            dense_margin = max(0.0, min(1.0, (raw_dense_0 - raw_dense_1) / 0.06))
        else:
            dense_margin = 0.5

    # 2. BM25 top-1 strength signal [0.0, 30.0] -> [0.0, 1.0]
    bm25_str = 0.0
    if bm25_results:
        raw_bm25_0 = bm25_results[0].bm25_score or 0.0
        bm25_str = max(0.0, min(1.0, raw_bm25_0 / 30.0))

    # 3. Retriever agreement (Jaccard)
    agreement = calculate_retriever_agreement(dense_results, bm25_results, k=agreement_k)

    # 4. RRF margin signal
    rrf_margin = 0.0
    if fusion_results and len(fusion_results) >= 2:
        f0 = fusion_results[0].fusion_score or 0.0
        f1 = fusion_results[1].fusion_score or 0.0
        rrf_margin = max(0.0, min(1.0, (f0 - f1) / 0.004))
    elif fusion_results and len(fusion_results) == 1:
        rrf_margin = 0.5

    # 5. Language match confidence
    lang_conf = 1.0
    if language and fusion_results:
        synonyms = get_language_synonyms(language)
        matches = sum(
            1 for r in fusion_results[:5]
            if (r.language and r.language.lower() in synonyms) or (r.metadata and r.metadata.get("language", "").lower() in synonyms)
        )
        denom = min(5, len(fusion_results))
        lang_conf = (matches / denom) if denom > 0 else 1.0

    # 6. Document diversity
    doc_div = 1.0
    if fusion_results:
        top_pool = fusion_results[:5]
        unique_docs = set(r.document_id for r in top_pool if r.document_id)
        doc_div = len(unique_docs) / len(top_pool) if top_pool else 1.0

    # 7. Final calibrated confidence calculation
    if agreement == 0.0 and dense_sim >= 0.70:
        # Cross-script or Indic semantic match where lexical token matching is zero
        final_conf = (
            (0.50 * dense_sim)
            + (0.25 * dense_margin)
            + (0.15 * rrf_margin)
            + (0.10 * lang_conf)
        )
    else:
        final_conf = (
            (0.35 * dense_sim)
            + (0.20 * dense_margin)
            + (0.15 * bm25_str)
            + (0.15 * agreement)
            + (0.10 * rrf_margin)
            + (0.05 * lang_conf)
        )

    final_conf = round(max(0.0, min(1.0, float(final_conf))), 4)

    return ConfidenceSignals(
        dense_similarity=round(dense_sim, 4),
        dense_margin=round(dense_margin, 4),
        bm25_strength=round(bm25_str, 4),
        retriever_agreement=round(agreement, 4),
        rrf_margin=round(rrf_margin, 4),
        language_confidence=round(lang_conf, 4),
        document_diversity=round(doc_div, 4),
        final_confidence=final_conf,
    )


def should_rerank(confidence: float, threshold_high: float) -> bool:
    """Determine if neural reranking is required based on confidence threshold."""
    return confidence < threshold_high


def rerank_candidate_count(
    confidence: float,
    threshold_high: float,
    threshold_low: float,
    medium_k: int = 3,
    low_k: int = 5,
    signals: Optional[ConfidenceSignals] = None,
    remaining_budget_ms: Optional[float] = None,
    min_budget_ms: float = 40.0,
    early_exit_margin: float = 0.15,
) -> int:
    """Determine candidate pool size K based on confidence, budget constraints, and early exit checks."""
    if confidence >= threshold_high:
        return 0

    # If remaining latency budget is exhausted, early-exit / skip expensive reranking
    if remaining_budget_ms is not None and remaining_budget_ms < min_budget_ms:
        return 0

    if confidence >= threshold_low:
        return medium_k

    # Candidate dominance early exit check before committing to low_k (5)
    if signals is not None and early_exit_margin > 0.0:
        # Check dense score margin dominance or RRF dominance
        if (
            signals.dense_margin >= early_exit_margin
            or (signals.rrf_margin >= 0.65 and signals.retriever_agreement >= 0.25 and signals.language_confidence >= 0.80)
        ):
            return medium_k

    return low_k


def routing_decision(
    signals: ConfidenceSignals,
    threshold_high: float,
    threshold_low: float,
    medium_k: int = 3,
    low_k: int = 5,
    heavy_enabled: bool = False,
    remaining_budget_ms: Optional[float] = None,
    min_budget_ms: float = 40.0,
    early_exit_margin: float = 0.15,
) -> AdaptiveDecision:
    """Construct deterministic 3-tier routing decision incorporating latency budget and score margins."""
    conf = signals.final_confidence

    # 1. High confidence or budget exhaustion -> Skip neural reranking
    if not should_rerank(conf, threshold_high):
        return AdaptiveDecision(
            confidence_score=conf,
            dense_confidence=signals.dense_similarity,
            retriever_agreement=signals.retriever_agreement,
            score_margin=signals.dense_margin,
            should_rerank=False,
            reranker_tier=TierString("high"),
            candidate_k=0,
            reranker="none",
            reranking_used=False,
            reason=f"High retrieval confidence ({conf:.2f} >= {threshold_high:.2f}). Skipping neural reranking.",
            signals=signals,
            routing_tier="high",
        )

    if remaining_budget_ms is not None and remaining_budget_ms < min_budget_ms:
        return AdaptiveDecision(
            confidence_score=conf,
            dense_confidence=signals.dense_similarity,
            retriever_agreement=signals.retriever_agreement,
            score_margin=signals.dense_margin,
            should_rerank=False,
            reranker_tier=TierString("high"),
            candidate_k=0,
            reranker="none",
            reranking_used=False,
            reason=f"Remaining latency budget ({remaining_budget_ms:.1f}ms < {min_budget_ms:.1f}ms) exhausted. Fallback to RRF ranking.",
            signals=signals,
            routing_tier="high",
        )

    k = rerank_candidate_count(
        confidence=conf,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        medium_k=medium_k,
        low_k=low_k,
        signals=signals,
        remaining_budget_ms=remaining_budget_ms,
        min_budget_ms=min_budget_ms,
        early_exit_margin=early_exit_margin,
    )

    if k == 0:
        return AdaptiveDecision(
            confidence_score=conf,
            dense_confidence=signals.dense_similarity,
            retriever_agreement=signals.retriever_agreement,
            score_margin=signals.dense_margin,
            should_rerank=False,
            reranker_tier=TierString("high"),
            candidate_k=0,
            reranker="none",
            reranking_used=False,
            reason="Early exit triggered by candidate score dominance or budget constraint. Skipping reranking.",
            signals=signals,
            routing_tier="high",
        )

    if conf >= threshold_low or k == medium_k:
        return AdaptiveDecision(
            confidence_score=conf,
            dense_confidence=signals.dense_similarity,
            retriever_agreement=signals.retriever_agreement,
            score_margin=signals.dense_margin,
            should_rerank=True,
            reranker_tier=TierString("medium"),
            candidate_k=medium_k,
            reranker="minilm",
            reranking_used=True,
            reason=f"Medium retrieval confidence ({conf:.2f}). Using lightweight cross-encoder (K={medium_k}).",
            signals=signals,
            routing_tier="medium",
        )
    else:
        chosen_reranker = "bge" if heavy_enabled else "minilm"
        return AdaptiveDecision(
            confidence_score=conf,
            dense_confidence=signals.dense_similarity,
            retriever_agreement=signals.retriever_agreement,
            score_margin=signals.dense_margin,
            should_rerank=True,
            reranker_tier=TierString("low"),
            candidate_k=low_k,
            reranker=chosen_reranker,
            reranking_used=True,
            reason=f"Low retrieval confidence ({conf:.2f} < {threshold_low:.2f}). Using {chosen_reranker} (K={low_k}).",
            signals=signals,
            routing_tier="low",
        )


def calculate_retrieval_confidence(
    dense_results: List[RetrievalResult],
    bm25_results: List[RetrievalResult],
    fusion_results: List[RetrievalResult],
    w_dense: Optional[float] = None,
    w_agreement: Optional[float] = None,
    w_margin: Optional[float] = None,
    threshold_high: Optional[float] = None,
    threshold_low: Optional[float] = None,
    agreement_k: int = 5,
    language: Optional[str] = None,
    medium_k: Optional[int] = None,
    low_k: Optional[int] = None,
    remaining_budget_ms: Optional[float] = None,
    min_budget_ms: Optional[float] = None,
    early_exit_margin: Optional[float] = None,
) -> AdaptiveDecision:
    """Calculate structured retrieval confidence and make budget-aware adaptive 3-tier routing decision."""
    settings = get_settings()
    th_high = threshold_high if threshold_high is not None else settings.ADAPTIVE_THRESHOLD_HIGH
    th_low = threshold_low if threshold_low is not None else settings.ADAPTIVE_THRESHOLD_LOW
    med_k = medium_k if medium_k is not None else getattr(settings, "ADAPTIVE_MEDIUM_K", 3)
    lw_k = low_k if low_k is not None else getattr(settings, "ADAPTIVE_LOW_K", 5)
    heavy_enabled = getattr(settings, "HEAVY_RERANKER_ENABLED", False)
    min_b = min_budget_ms if min_budget_ms is not None else getattr(settings, "RERANKER_MIN_BUDGET_MS", 40.0)
    margin = early_exit_margin if early_exit_margin is not None else getattr(settings, "RERANKER_EARLY_EXIT_MARGIN", 0.15)

    signals = compute_confidence_signals(
        dense_results=dense_results,
        bm25_results=bm25_results,
        fusion_results=fusion_results,
        language=language,
        agreement_k=agreement_k,
    )

    return routing_decision(
        signals=signals,
        threshold_high=th_high,
        threshold_low=th_low,
        medium_k=med_k,
        low_k=lw_k,
        heavy_enabled=heavy_enabled,
        remaining_budget_ms=remaining_budget_ms,
        min_budget_ms=min_b,
        early_exit_margin=margin,
    )


class AdaptiveRetrievalService:
    """Latency-optimized retrieval service using confidence estimation, dynamic routing, and caching."""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        heavy_reranker: Optional[Reranker] = None,
        lightweight_reranker: Optional[Reranker] = None,
        context_selector: Optional[ContextSelector] = None,
    ):
        self.retrieval_service = retrieval_service or get_retrieval_service()
        self._heavy_reranker = heavy_reranker
        self._lightweight_reranker = lightweight_reranker
        self.context_selector = context_selector or ContextSelector()
        self.fusion_engine = getattr(self.retrieval_service, "fusion", HybridFusion())
        self.cache = get_rag_cache()

    @property
    def lightweight_reranker(self) -> Reranker:
        """Lazily load lightweight MiniLM reranker on demand."""
        if self._lightweight_reranker is None:
            self._lightweight_reranker = get_lightweight_reranker()
        return self._lightweight_reranker

    @property
    def heavy_reranker(self) -> Reranker:
        """Lazily load heavy BGE reranker only when explicitly enabled."""
        if self._heavy_reranker is None:
            self._heavy_reranker = get_reranker()
        return self._heavy_reranker

    def adaptive_retrieve(
        self,
        query: str,
        top_k: int = 5,
        strategies: Optional[Union[str, List[str]]] = None,
        language: Optional[str] = None,
        threshold_high: Optional[float] = None,
        threshold_low: Optional[float] = None,
        medium_k: Optional[int] = None,
        low_k: Optional[int] = None,
        use_cache: bool = True,
    ) -> Tuple[List[RerankResult], AdaptiveDecision, AdaptiveLatencyBreakdown]:
        """Execute adaptive confidence-guided retrieval and reranking with budget deadlines and full telemetry."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        t_start = time.perf_counter()
        settings = get_settings()
        try:
            total_deadline_ms = float(getattr(settings, "TOTAL_RETRIEVAL_DEADLINE_MS", 200.0))
        except (TypeError, ValueError):
            total_deadline_ms = 200.0
        deadline = t_start + (total_deadline_ms / 1000.0)
        try:
            reranker_timeout_ms = float(getattr(settings, "RERANKER_TIMEOUT_MS", 133.0))
        except (TypeError, ValueError):
            reranker_timeout_ms = 133.0
        reranker_timeout_s = reranker_timeout_ms / 1000.0
        try:
            min_budget_ms = float(getattr(settings, "RERANKER_MIN_BUDGET_MS", 20.0))
        except (TypeError, ValueError):
            min_budget_ms = 20.0
        try:
            early_exit_margin = float(getattr(settings, "RERANKER_EARLY_EXIT_MARGIN", 0.0))
        except (TypeError, ValueError):
            early_exit_margin = 0.0

        timeout_stage: Optional[str] = None
        fallback_used = False

        # Check Cache
        if use_cache and self.cache.enabled:
            cached_val = self.cache.get_adaptive(query=query, language=language, top_k=top_k, strategies=strategies)
            if cached_val is not None:
                cached_results, cached_decision = cached_val
                t_total_ms = (time.perf_counter() - t_start) * 1000.0
                lat = AdaptiveLatencyBreakdown(
                    embedding=0.0,
                    retrieval=0.0,
                    qdrant=0.0,
                    bm25=0.0,
                    fusion=0.0,
                    reranking=0.0,
                    context=0.0,
                    total=round(t_total_ms, 2),
                    cache_hit=True,
                    timeout_stage=None,
                    fallback_used=False,
                    parallel_execution=False,
                    remaining_budget_ms=round(max(0.0, total_deadline_ms - t_total_ms), 2),
                )
                return cached_results[:top_k], cached_decision, lat

        # 1. Candidate Retrieval (Dense + BM25 with parallel support & timeouts)
        dense_results: List[RetrievalResult] = []
        bm25_results: List[RetrievalResult] = []
        t_embed_ms = 0.0
        t_dense_ms = 0.0
        t_bm25_ms = 0.0
        is_parallel = getattr(settings, "PARALLEL_RETRIEVAL_ENABLED", True)

        if (
            is_parallel
            and isinstance(getattr(self.retrieval_service, "_executor", None), concurrent.futures.Executor)
        ):
            def _fetch_dense():
                return self.retrieval_service.dense_retriever.retrieve(
                    query=query, top_k=10, strategies=strategies, language=language
                )

            def _fetch_bm25():
                return self.retrieval_service.bm25_retriever.retrieve(
                    query=query, top_k=10, strategies=strategies, language=language
                )

            dense_f = self.retrieval_service._executor.submit(_fetch_dense)
            bm25_f = self.retrieval_service._executor.submit(_fetch_bm25)

            remaining_s = max(0.001, deadline - time.perf_counter())
            qdrant_timeout_s = min(getattr(settings, "QDRANT_TIMEOUT_MS", 157.0) / 1000.0, remaining_s)
            bm25_timeout_s = min(getattr(settings, "BM25_TIMEOUT_MS", 136.0) / 1000.0, remaining_s)
            wait_timeout_s = min(max(qdrant_timeout_s, bm25_timeout_s), remaining_s)

            done, not_done = concurrent.futures.wait(
                [dense_f, bm25_f],
                timeout=wait_timeout_s,
                return_when=concurrent.futures.ALL_COMPLETED,
            )

            if dense_f in done:
                try:
                    dense_results, t_embed_ms, t_dense_ms = dense_f.result()
                except Exception as ex:
                    logger.warning("Dense retrieval error (%s); falling back.", ex)
                    dense_results = []
                    fallback_used = True
            else:
                logger.warning("Dense retrieval timeout; falling back.")
                dense_results = []
                timeout_stage = "qdrant"
                fallback_used = True

            if bm25_f in done:
                try:
                    bm25_results, t_bm25_ms = bm25_f.result()
                except Exception as ex:
                    logger.warning("BM25 retrieval error (%s); falling back.", ex)
                    bm25_results = []
                    fallback_used = True
            else:
                logger.warning("BM25 retrieval timeout; falling back.")
                bm25_results = []
                timeout_stage = "bm25" if not timeout_stage else f"{timeout_stage},bm25"
                fallback_used = True
        else:
            try:
                dense_results, t_embed_ms, t_dense_ms = self.retrieval_service.dense_retriever.retrieve(
                    query=query, top_k=10, strategies=strategies, language=language
                )
            except Exception as ex:
                logger.warning("Dense retrieval error in sequential mode: %s", ex)
                dense_results = []
                fallback_used = True

            try:
                bm25_results, t_bm25_ms = self.retrieval_service.bm25_retriever.retrieve(
                    query=query, top_k=10, strategies=strategies, language=language
                )
            except Exception as ex:
                logger.warning("BM25 retrieval error in sequential mode: %s", ex)
                bm25_results = []
                fallback_used = True

        t_fusion_start = time.perf_counter()
        fusion_results = self.fusion_engine.fuse_results(
            dense_results=dense_results,
            bm25_results=bm25_results,
            top_k=10,
            method="rrf",
        )
        t_fusion_ms = (time.perf_counter() - t_fusion_start) * 1000.0
        t_retrieval_ms = t_dense_ms + t_bm25_ms + t_fusion_ms

        # Calculate remaining budget after retrieval stage
        elapsed_after_ret_ms = (time.perf_counter() - t_start) * 1000.0
        remaining_budget_ms = max(0.0, total_deadline_ms - elapsed_after_ret_ms)

        if not fusion_results and not dense_results and not bm25_results:
            empty_signals = ConfidenceSignals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            decision = AdaptiveDecision(
                confidence_score=0.0,
                dense_confidence=0.0,
                retriever_agreement=0.0,
                score_margin=0.0,
                should_rerank=False,
                reranker_tier=TierString("high"),
                candidate_k=0,
                reranker="none",
                reranking_used=False,
                reason="No retrieval results found.",
                signals=empty_signals,
                routing_tier="high",
            )
            lat = AdaptiveLatencyBreakdown(
                embedding=round(t_embed_ms, 2),
                retrieval=round(t_retrieval_ms, 2),
                qdrant=round(t_dense_ms, 2),
                bm25=round(t_bm25_ms, 2),
                fusion=round(t_fusion_ms, 2),
                reranking=0.0,
                context=0.0,
                total=round((time.perf_counter() - t_start) * 1000, 2),
                cache_hit=False,
                timeout_stage=timeout_stage,
                fallback_used=True,
                parallel_execution=is_parallel,
                remaining_budget_ms=round(remaining_budget_ms, 2),
            )
            return [], decision, lat

        # 2. Confidence Estimation & Budget-Aware 3-Tier Decision
        decision = calculate_retrieval_confidence(
            dense_results=dense_results,
            bm25_results=bm25_results,
            fusion_results=fusion_results,
            threshold_high=threshold_high,
            threshold_low=threshold_low,
            language=language,
            medium_k=medium_k,
            low_k=low_k,
            remaining_budget_ms=remaining_budget_ms,
            min_budget_ms=min_budget_ms,
            early_exit_margin=early_exit_margin,
        )

        pool_k = decision.candidate_k if decision.candidate_k > 0 else top_k
        candidates = fusion_results[:pool_k]
        rerank_latency_ms = 0.0
        reranked_pool: List[RerankResult] = []

        # 3. Dynamic Reranking Execution with Timeout Boundary and RRF Fallback
        current_remaining_ms = max(0.0, (deadline - time.perf_counter()) * 1000.0)
        if current_remaining_ms < min_budget_ms and decision.should_rerank:
            logger.info("Insufficient remaining budget (%.1f ms < %.1f ms); skipping neural reranking.", current_remaining_ms, min_budget_ms)
            decision.should_rerank = False
            decision.reranker = "fallback_rrf"
            decision.reranker_tier = TierString("skip")
            decision.reason = f"Budget exhausted ({current_remaining_ms:.1f}ms < {min_budget_ms:.1f}ms)."

        effective_reranker_timeout_s = min(reranker_timeout_s, max(0.001, deadline - time.perf_counter()))

        if not decision.should_rerank or decision.reranker == "fallback_rrf" or decision.reranker_tier in ("high", "skip"):
            # HIGH Confidence or Budget Exhaustion -> Skip neural reranking
            for orig_idx, c in enumerate(candidates, start=1):
                rerank_item = RerankResult(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    text=c.text,
                    chunk_type=c.chunk_type,
                    language=c.language,
                    dense_score=c.dense_score,
                    bm25_score=c.bm25_score,
                    fusion_score=c.fusion_score,
                    reranker_score=c.fusion_score or (c.dense_score or 0.5),
                    original_rank=c.rank if c.rank is not None else orig_idx,
                    rank=orig_idx,
                    metadata=c.metadata,
                )
                reranked_pool.append(rerank_item)
            rerank_latency_ms = 0.0

        elif decision.reranker_tier == "medium":
            # MEDIUM Confidence -> MiniLM K=3 with timeout protection
            try:
                t0_rerank = time.perf_counter()
                reranked_pool, rerank_latency_ms = self.lightweight_reranker.rerank(
                    query=query, candidates=candidates
                )
                if (time.perf_counter() - t0_rerank) > effective_reranker_timeout_s:
                    logger.warning("Reranker execution exceeded timeout %.2fs", effective_reranker_timeout_s)
                    timeout_stage = "reranker" if not timeout_stage else f"{timeout_stage},reranker"
            except Exception as ex:
                logger.warning("Lightweight reranker failed (%s); falling back to RRF rankings.", ex)
                decision.reranker = "fallback_rrf"
                fallback_used = True
                reranked_pool = [
                    RerankResult(
                        chunk_id=c.chunk_id,
                        document_id=c.document_id,
                        text=c.text,
                        chunk_type=c.chunk_type,
                        language=c.language,
                        dense_score=c.dense_score,
                        bm25_score=c.bm25_score,
                        fusion_score=c.fusion_score,
                        reranker_score=c.fusion_score or 0.5,
                        original_rank=c.rank or idx,
                        rank=idx,
                        metadata=c.metadata,
                    )
                    for idx, c in enumerate(candidates, start=1)
                ]

        else:
            # LOW Confidence -> MiniLM K=5 (or BGE if explicitly enabled)
            try:
                t0_rerank = time.perf_counter()
                if getattr(settings, "HEAVY_RERANKER_ENABLED", False):
                    reranked_pool, rerank_latency_ms = self.heavy_reranker.rerank(
                        query=query, candidates=candidates
                    )
                else:
                    reranked_pool, rerank_latency_ms = self.lightweight_reranker.rerank(
                        query=query, candidates=candidates
                    )
                if (time.perf_counter() - t0_rerank) > effective_reranker_timeout_s:
                    logger.warning("Reranker execution exceeded timeout %.2fs", effective_reranker_timeout_s)
                    timeout_stage = "reranker" if not timeout_stage else f"{timeout_stage},reranker"
            except Exception as ex:
                logger.warning("Reranker failed (%s); falling back to RRF rankings.", ex)
                decision.reranker = "fallback_rrf"
                fallback_used = True
                reranked_pool = [
                    RerankResult(
                        chunk_id=c.chunk_id,
                        document_id=c.document_id,
                        text=c.text,
                        chunk_type=c.chunk_type,
                        language=c.language,
                        dense_score=c.dense_score,
                        bm25_score=c.bm25_score,
                        fusion_score=c.fusion_score,
                        reranker_score=c.fusion_score or 0.5,
                        original_rank=c.rank or idx,
                        rank=idx,
                        metadata=c.metadata,
                    )
                    for idx, c in enumerate(candidates, start=1)
                ]

        # 4. Context Selection & Diversity Control
        selected_results, context_stats, t_context_ms = self.context_selector.select_context(
            reranked_candidates=reranked_pool,
            top_k=top_k,
        )

        t_total_ms = (time.perf_counter() - t_start) * 1000
        final_remaining_budget_ms = max(0.0, total_deadline_ms - t_total_ms)

        latency_breakdown = AdaptiveLatencyBreakdown(
            embedding=round(t_embed_ms, 2),
            retrieval=round(t_retrieval_ms, 2),
            qdrant=round(t_dense_ms, 2),
            bm25=round(t_bm25_ms, 2),
            fusion=round(t_fusion_ms, 2),
            reranking=round(rerank_latency_ms, 2),
            context=round(t_context_ms, 2),
            total=round(t_total_ms, 2),
            cache_hit=False,
            timeout_stage=timeout_stage,
            fallback_used=fallback_used,
            parallel_execution=is_parallel,
            remaining_budget_ms=round(final_remaining_budget_ms, 2),
        )

        # Store in cache if enabled
        if use_cache and self.cache.enabled and selected_results:
            self.cache.set_adaptive(
                query=query, language=language, value=(selected_results, decision), top_k=top_k, strategies=strategies
            )

        return selected_results, decision, latency_breakdown


@lru_cache()
def get_adaptive_retrieval_service() -> AdaptiveRetrievalService:
    """Return cached singleton AdaptiveRetrievalService instance."""
    return AdaptiveRetrievalService()
