"""Strategy registry for discovering and instantiating chunking strategies."""

from typing import Dict, List, Type, Union

from app.chunking.base import ChunkingStrategy
from app.chunking.fixed import FixedChunkingStrategy
from app.chunking.hierarchical import HierarchicalChunkingStrategy
from app.chunking.semantic import SemanticChunkingStrategy
from app.chunking.sentence import SentenceChunkingStrategy
from app.chunking.sliding_window import SlidingWindowChunkingStrategy
from app.config import get_settings


class ChunkingRegistry:
    """Registry maintaining available chunking strategy implementations."""

    _STRATEGIES: Dict[str, Type[ChunkingStrategy]] = {
        "fixed": FixedChunkingStrategy,
        "sentence": SentenceChunkingStrategy,
        "sliding_window": SlidingWindowChunkingStrategy,
        "semantic": SemanticChunkingStrategy,
        "hierarchical": HierarchicalChunkingStrategy,
    }

    @classmethod
    def register(cls, name: str, strategy_cls: Type[ChunkingStrategy]) -> None:
        """Register a new chunking strategy class."""
        cls._STRATEGIES[name.lower().strip()] = strategy_cls

    @classmethod
    def get_strategy(cls, name: str, **kwargs) -> ChunkingStrategy:
        """Instantiate a strategy by name."""
        clean_name = name.lower().strip()
        if clean_name not in cls._STRATEGIES:
            raise ValueError(
                f"Unknown chunking strategy '{name}'. Available: {list(cls._STRATEGIES.keys())}"
            )
        return cls._STRATEGIES[clean_name](**kwargs)

    @classmethod
    def get_configured_strategies(
        cls, strategy_names: Union[str, List[str], None] = None
    ) -> List[ChunkingStrategy]:
        """Instantiate all configured strategies based on argument or environment settings."""
        if strategy_names is None:
            settings = get_settings()
            raw = settings.CHUNKING_STRATEGIES
            if isinstance(raw, str):
                names = [s.strip() for s in raw.split(",") if s.strip()]
            else:
                names = list(raw)
        elif isinstance(strategy_names, str):
            names = [s.strip() for s in strategy_names.split(",") if s.strip()]
        else:
            names = list(strategy_names)

        strategies: List[ChunkingStrategy] = []
        for name in names:
            strategies.append(cls.get_strategy(name))
        return strategies

    @classmethod
    def list_available_strategies(cls) -> List[str]:
        """Return list of all registered strategy names."""
        return list(cls._STRATEGIES.keys())
