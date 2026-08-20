"""Context selection and diversity filtering module."""

import logging
import time
from collections import defaultdict
from typing import List, Optional, Tuple

from app.config import get_settings
from app.reranking.models import ContextSelectionStats, RerankResult

logger = logging.getLogger(__name__)


class ContextSelector:
    """Selects and filters reranked candidate chunks enforcing document diversity and token/character limits."""

    def __init__(
        self,
        final_top_k: Optional[int] = None,
        max_chunks_per_doc: Optional[int] = None,
        max_context_chars: Optional[int] = None,
        diversity_enabled: Optional[bool] = None,
    ):
        settings = get_settings()
        self.final_top_k = final_top_k if final_top_k is not None else settings.FINAL_CONTEXT_K
        self.max_chunks_per_doc = max_chunks_per_doc if max_chunks_per_doc is not None else settings.MAX_CHUNKS_PER_DOCUMENT
        self.max_context_chars = max_context_chars if max_context_chars is not None else settings.MAX_CONTEXT_CHARS
        self.diversity_enabled = diversity_enabled if diversity_enabled is not None else settings.CONTEXT_DIVERSITY_ENABLED

    def select_context(
        self,
        reranked_candidates: List[RerankResult],
        top_k: Optional[int] = None,
    ) -> Tuple[List[RerankResult], ContextSelectionStats, float]:
        """Select top diverse candidates within the configured character and count budgets."""
        t0 = time.perf_counter()
        k_target = top_k if top_k is not None else self.final_top_k

        total_candidates = len(reranked_candidates)
        selected_results: List[RerankResult] = []
        seen_chunk_ids = set()
        doc_chunk_counts = defaultdict(int)
        current_chars = 0
        diversity_filtered_count = 0

        for cand in reranked_candidates:
            # 1. Deduplication by chunk_id
            if cand.chunk_id in seen_chunk_ids:
                continue

            # 2. Document-level diversity cap
            if self.diversity_enabled and cand.document_id:
                if doc_chunk_counts[cand.document_id] >= self.max_chunks_per_doc:
                    diversity_filtered_count += 1
                    continue

            # 3. Context character budget check
            cand_len = len(cand.text)
            if current_chars + cand_len > self.max_context_chars:
                if selected_results:
                    # Budget reached, stop adding further chunks
                    break
                else:
                    # If the very first chunk exceeds the budget on its own, include it but log warning
                    logger.warning("First chunk exceeds context character budget: %d chars", cand_len)

            # Accept candidate
            seen_chunk_ids.add(cand.chunk_id)
            if cand.document_id:
                doc_chunk_counts[cand.document_id] += 1
            current_chars += cand_len
            selected_results.append(cand)

            if len(selected_results) >= k_target:
                break

        # Re-assign 1-based ranks
        for idx, item in enumerate(selected_results, start=1):
            item.rank = idx

        t_selection_ms = (time.perf_counter() - t0) * 1000

        stats = ContextSelectionStats(
            total_candidates=total_candidates,
            selected_chunks=len(selected_results),
            total_characters=current_chars,
            avg_chunk_characters=round(current_chars / max(1, len(selected_results)), 1),
            max_context_chars=self.max_context_chars,
            diversity_filtered_count=diversity_filtered_count,
        )

        return selected_results, stats, t_selection_ms
