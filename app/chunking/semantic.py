"""Semantic chunking strategy based on inter-sentence similarity and topic boundary detection."""

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from app.chunking.base import ChunkingStrategy
from app.chunking.models import Chunk
from app.chunking.sentence import split_into_sentences_with_offsets
from app.config import get_settings
from app.ingestion.models import Document


class SemanticRepresentationProvider(ABC):
    """Abstract interface for generating vector/feature representations for sentences."""

    @abstractmethod
    def encode_sentences(self, sentences: Sequence[str]) -> List[List[float]]:
        """Compute numerical representation vectors for a sequence of sentence strings."""
        pass


class LightweightTfIdfRepresentationProvider(SemanticRepresentationProvider):
    """Fast, dependency-free n-gram term frequency representation provider.
    
    Extracts word and character sub-word n-grams to calculate sparse term vectors
    supporting multi-lingual, Indic, and English texts for local boundary detection.
    """

    def _tokenize_ngrams(self, text: str) -> List[str]:
        words = re.findall(r"\w+", text.lower(), re.UNICODE)
        tokens = list(words)
        # Add character tri-grams for sub-word morphology matching
        for word in words:
            if len(word) >= 3:
                tokens.extend([word[i:i+3] for i in range(len(word) - 2)])
        return tokens

    def encode_sentences(self, sentences: Sequence[str]) -> List[List[float]]:
        if not sentences:
            return []

        tokenized_sentences = [self._tokenize_ngrams(s) for s in sentences]
        
        # Calculate document frequency (DF) across the document's sentences
        df: Counter[str] = Counter()
        for tokens in tokenized_sentences:
            unique_tokens = set(tokens)
            df.update(unique_tokens)

        num_sentences = len(sentences)
        vocabulary = sorted(df.keys())
        vocab_index = {token: idx for idx, token in enumerate(vocabulary)}

        vectors: List[List[float]] = []
        for tokens in tokenized_sentences:
            vec = [0.0] * len(vocabulary)
            tf = Counter(tokens)
            for token, count in tf.items():
                if token in vocab_index:
                    idx = vocab_index[token]
                    idf = math.log((num_sentences + 1) / (df[token] + 1)) + 1.0
                    vec[idx] = (count / max(1, len(tokens))) * idf

            # L2 normalize vector
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)

        return vectors


def compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two normalized vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    return max(0.0, min(1.0, dot_product))


class SemanticChunkingStrategy(ChunkingStrategy):
    """Chunks documents dynamically by detecting semantic topic boundaries between sentences.
    
    1. Splits document into sentences.
    2. Encodes each sentence via SemanticRepresentationProvider.
    3. Calculates cosine similarity between consecutive sentences: S_i and S_{i+1}.
    4. Identifies boundary cut-points where similarity falls below similarity_threshold.
    5. Assembles sentence groups into cohesive semantic chunks.
    """

    strategy_name: str = "semantic"

    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        max_chunk_size: Optional[int] = None,
        representation_provider: Optional[SemanticRepresentationProvider] = None,
    ):
        settings = get_settings()
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.SEMANTIC_SIMILARITY_THRESHOLD
        )
        self.max_chunk_size = (
            max_chunk_size if max_chunk_size is not None else settings.SENTENCE_MAX_CHUNK_SIZE
        )
        self.representation_provider = (
            representation_provider or LightweightTfIdfRepresentationProvider()
        )

    def chunk(self, document: Document) -> List[Chunk]:
        """Detect semantic boundaries and split document into semantic chunks."""
        text = document.text
        if not text:
            return []

        sentences = split_into_sentences_with_offsets(text)
        if not sentences:
            return []

        if len(sentences) == 1:
            return [
                self.create_chunk(
                    document=document,
                    text=text,
                    chunk_index=0,
                    start_position=sentences[0][1],
                    end_position=sentences[0][2],
                    extra_metadata={"sentence_count": 1, "strategy": "semantic"},
                )
            ]

        # Generate sentence embeddings/representations
        raw_sentence_texts = [s[0] for s in sentences]
        vectors = self.representation_provider.encode_sentences(raw_sentence_texts)

        # Compute similarities between consecutive sentences
        similarities: List[float] = []
        for i in range(len(vectors) - 1):
            sim = compute_cosine_similarity(vectors[i], vectors[i + 1])
            similarities.append(sim)

        # Detect boundaries where similarity drops below threshold
        chunks: List[Chunk] = []
        current_sentences: List[Tuple[str, int, int]] = [sentences[0]]
        current_len = len(sentences[0][0])
        chunk_index = 0

        for i, sim in enumerate(similarities):
            next_sentence = sentences[i + 1]
            next_len = len(next_sentence[0])

            is_boundary = sim < self.similarity_threshold
            is_oversized = (current_len + next_len + 1) > self.max_chunk_size

            if is_boundary or is_oversized:
                # Flush current semantic cluster
                chunk_text = " ".join(s[0] for s in current_sentences)
                chunks.append(
                    self.create_chunk(
                        document=document,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        start_position=current_sentences[0][1],
                        end_position=current_sentences[-1][2],
                        extra_metadata={
                            "sentence_count": len(current_sentences),
                            "boundary_similarity": sim,
                            "strategy": "semantic",
                        },
                    )
                )
                chunk_index += 1
                current_sentences = [next_sentence]
                current_len = next_len
            else:
                current_sentences.append(next_sentence)
                current_len += next_len + 1

        # Flush final cluster
        if current_sentences:
            chunk_text = " ".join(s[0] for s in current_sentences)
            chunks.append(
                self.create_chunk(
                    document=document,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_position=current_sentences[0][1],
                    end_position=current_sentences[-1][2],
                    extra_metadata={
                        "sentence_count": len(current_sentences),
                        "strategy": "semantic",
                    },
                )
            )

        return chunks
