"""Thread-safe bounded LRU cache with TTL expiration for embeddings, retrieval, and adaptive decisions."""

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

from app.config import get_settings
from app.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)


class RetrievalCache:
    """Thread-safe, bounded LRU cache supporting TTL expiration across RAG stages."""

    def __init__(
        self,
        max_size: Optional[int] = None,
        enabled: Optional[bool] = None,
        ttl_seconds: Optional[int] = None,
    ):
        settings = get_settings()
        self.max_size = max_size if max_size is not None else getattr(settings, "RAG_CACHE_MAX_SIZE", 256)
        self.enabled = enabled if enabled is not None else getattr(settings, "RAG_CACHE_ENABLED", False)
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else getattr(settings, "RAG_CACHE_TTL_SECONDS", 300)
        self._cache: OrderedDict[Tuple, Tuple[float, Any]] = OrderedDict()
        self._embedding_cache: OrderedDict[str, Tuple[float, List[float]]] = OrderedDict()
        self._lock = threading.Lock()

        # Telemetry counters
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.expirations: int = 0

    def stats(self) -> Dict[str, Any]:
        """Return snapshot of cache performance metrics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0
            return {
                "enabled": self.enabled,
                "max_size": self.max_size,
                "current_size": len(self._cache) + len(self._embedding_cache),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "expirations": self.expirations,
                "hit_rate_pct": round(hit_rate * 100, 2),
            }

    def _make_key(
        self,
        query: str,
        top_k: int,
        strategies: Optional[Union[str, List[str]]],
        language: Optional[str],
        fusion_method: str,
    ) -> Tuple:
        """Construct a hashable cache key."""
        clean_q = query.strip().lower()
        if isinstance(strategies, list):
            strat_str = ",".join(sorted(strategies))
        else:
            strat_str = str(strategies or "all").lower()
        lang_str = str(language or "all").lower()
        return (clean_q, top_k, strat_str, lang_str, fusion_method.lower())

    def get(
        self,
        query: str,
        top_k: int,
        strategies: Optional[Union[str, List[str]]] = None,
        language: Optional[str] = None,
        fusion_method: str = "rrf",
    ) -> Optional[List[RetrievalResult]]:
        """Retrieve cached retrieval results if valid, unexpired, and cache is enabled."""
        if not self.enabled:
            return None

        key = self._make_key(query, top_k, strategies, language, fusion_method)
        with self._lock:
            if key in self._cache:
                ts, results = self._cache[key]
                if self.ttl_seconds > 0 and (time.time() - ts) > self.ttl_seconds:
                    del self._cache[key]
                    self.expirations += 1
                    self.misses += 1
                    try:
                        from app.observability.metrics import get_metrics_registry
                        get_metrics_registry().record_cache_expiration()
                        get_metrics_registry().record_cache_miss(language or "en")
                        get_metrics_registry().rag_cache_entries.set(len(self._cache))
                    except Exception:
                        pass
                    return None
                # Move to end to refresh LRU position
                self._cache.move_to_end(key)
                self.hits += 1
                try:
                    from app.observability.metrics import get_metrics_registry
                    get_metrics_registry().record_cache_hit(language or "en")
                except Exception:
                    pass
                return results
            self.misses += 1
            try:
                from app.observability.metrics import get_metrics_registry
                get_metrics_registry().record_cache_miss(language or "en")
            except Exception:
                pass
        return None

    def set(
        self,
        query: str,
        top_k: int,
        strategies: Optional[Union[str, List[str]]] = None,
        language: Optional[str] = None,
        fusion_method: str = "rrf",
        results: Optional[List[RetrievalResult]] = None,
    ) -> None:
        """Store retrieval results in cache, evicting oldest item if size limit exceeded."""
        if not self.enabled or results is None:
            return

        key = self._make_key(query, top_k, strategies, language, fusion_method)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (time.time(), results)

            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
                self.evictions += 1
                try:
                    from app.observability.metrics import get_metrics_registry
                    get_metrics_registry().record_cache_eviction()
                except Exception:
                    pass

            try:
                from app.observability.metrics import get_metrics_registry
                get_metrics_registry().rag_cache_entries.set(len(self._cache))
            except Exception:
                pass

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Retrieve cached query embedding if unexpired."""
        if not self.enabled or not text:
            return None

        clean_text = text.strip()
        with self._lock:
            if clean_text in self._embedding_cache:
                ts, vec = self._embedding_cache[clean_text]
                if self.ttl_seconds > 0 and (time.time() - ts) > self.ttl_seconds:
                    del self._embedding_cache[clean_text]
                    self.expirations += 1
                    self.misses += 1
                    return None
                self._embedding_cache.move_to_end(clean_text)
                self.hits += 1
                return vec
            self.misses += 1
        return None

    def set_embedding(self, text: str, vector: List[float]) -> None:
        """Store query embedding in cache."""
        if not self.enabled or not text or not vector:
            return

        clean_text = text.strip()
        with self._lock:
            if clean_text in self._embedding_cache:
                self._embedding_cache.move_to_end(clean_text)
            self._embedding_cache[clean_text] = (time.time(), vector)

            while len(self._embedding_cache) > self.max_size:
                self._embedding_cache.popitem(last=False)
                self.evictions += 1

    def _make_adaptive_key(
        self,
        query: str,
        language: Optional[str] = None,
        top_k: Optional[int] = None,
        strategies: Optional[Union[str, List[str]]] = None,
    ) -> Tuple:
        """Construct isolated hashable key for adaptive retrieval cache."""
        clean_q = query.strip().lower()
        lang_str = str(language or "all").lower()
        k_str = str(top_k) if top_k is not None else "default"
        if isinstance(strategies, list):
            strat_str = ",".join(sorted(strategies))
        else:
            strat_str = str(strategies or "all").lower()
        return ("adaptive", clean_q, lang_str, k_str, strat_str)

    def get_adaptive(
        self,
        query: str,
        language: Optional[str] = None,
        top_k: Optional[int] = None,
        strategies: Optional[Union[str, List[str]]] = None,
    ) -> Optional[Any]:
        """Retrieve cached adaptive decision/result if present and unexpired."""
        if not self.enabled:
            return None

        key = self._make_adaptive_key(query=query, language=language, top_k=top_k, strategies=strategies)
        with self._lock:
            if key in self._cache:
                ts, val = self._cache[key]
                if self.ttl_seconds > 0 and (time.time() - ts) > self.ttl_seconds:
                    del self._cache[key]
                    self.expirations += 1
                    self.misses += 1
                    return None
                self._cache.move_to_end(key)
                self.hits += 1
                return val
            self.misses += 1
        return None

    def set_adaptive(
        self,
        query: str,
        language: Optional[str] = None,
        value: Any = None,
        top_k: Optional[int] = None,
        strategies: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """Store adaptive decision/result in cache."""
        if not self.enabled or value is None:
            return

        key = self._make_adaptive_key(query=query, language=language, top_k=top_k, strategies=strategies)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (time.time(), value)

            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
                self.evictions += 1

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._embedding_cache.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache) + len(self._embedding_cache)


# Global singleton instance
_GLOBAL_CACHE: Optional[RetrievalCache] = None


def get_rag_cache() -> RetrievalCache:
    """Retrieve global RAG cache instance."""
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        _GLOBAL_CACHE = RetrievalCache()
    return _GLOBAL_CACHE


get_retrieval_cache = get_rag_cache
