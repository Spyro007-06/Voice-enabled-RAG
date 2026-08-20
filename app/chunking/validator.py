"""Chunk validation rules and integrity verification."""

from typing import List, Optional, Set, Tuple

from app.chunking.models import Chunk, generate_chunk_id


class ChunkValidationError(ValueError):
    """Raised when one or more chunks fail structural or semantic validation."""
    pass


class ChunkValidator:
    """Validator ensuring chunk integrity, deterministic IDs, non-empty content, and valid bounds."""

    def __init__(
        self,
        min_char_length: int = 1,
        max_char_length: Optional[int] = None,
    ):
        self.min_char_length = min_char_length
        self.max_char_length = max_char_length

    def validate_chunk(self, chunk: Chunk) -> List[str]:
        """Validate a single chunk and return a list of issue descriptions."""
        errors: List[str] = []

        # 1. Non-empty text check
        if not chunk.text or not chunk.text.strip():
            errors.append(f"Chunk '{chunk.chunk_id}' has empty text content.")

        # 2. Valid document ID
        if not chunk.document_id or not chunk.document_id.strip():
            errors.append(f"Chunk '{chunk.chunk_id}' has missing or empty document_id.")

        # 3. Deterministic chunk ID check
        expected_chunk_id = generate_chunk_id(
            document_id=chunk.document_id,
            chunk_type=chunk.chunk_type,
            chunk_index=chunk.chunk_index,
        )
        if chunk.chunk_id != expected_chunk_id:
            errors.append(
                f"Chunk ID '{chunk.chunk_id}' is non-deterministic. Expected: '{expected_chunk_id}'"
            )

        # 4. Valid chunk index
        if chunk.chunk_index < 0:
            errors.append(f"Chunk '{chunk.chunk_id}' has negative chunk_index: {chunk.chunk_index}")

        # 5. Valid offsets
        if chunk.start_position < 0:
            errors.append(f"Chunk '{chunk.chunk_id}' has negative start_position: {chunk.start_position}")
        if chunk.end_position < chunk.start_position:
            errors.append(
                f"Chunk '{chunk.chunk_id}' has end_position ({chunk.end_position}) < start_position ({chunk.start_position})"
            )

        # 6. Size constraints
        if len(chunk.text) < self.min_char_length:
            errors.append(
                f"Chunk '{chunk.chunk_id}' length ({len(chunk.text)}) below minimum ({self.min_char_length})"
            )
        if self.max_char_length and len(chunk.text) > self.max_char_length:
            errors.append(
                f"Chunk '{chunk.chunk_id}' length ({len(chunk.text)}) exceeds maximum ({self.max_char_length})"
            )

        # 7. Metadata validation
        if not isinstance(chunk.metadata, dict):
            errors.append(f"Chunk '{chunk.chunk_id}' metadata is not a dictionary.")

        return errors

    def validate_chunks(self, chunks: List[Chunk], raise_on_error: bool = False) -> Tuple[bool, List[str]]:
        """Validate a collection of chunks including uniqueness of IDs."""
        all_errors: List[str] = []
        seen_ids: Set[str] = set()

        for chunk in chunks:
            # Check for duplicate chunk IDs
            if chunk.chunk_id in seen_ids:
                all_errors.append(f"Duplicate chunk_id detected: '{chunk.chunk_id}'")
            else:
                seen_ids.add(chunk.chunk_id)

            item_errors = self.validate_chunk(chunk)
            all_errors.extend(item_errors)

        is_valid = len(all_errors) == 0
        if not is_valid and raise_on_error:
            raise ChunkValidationError(
                f"Chunk validation failed with {len(all_errors)} errors:\n" + "\n".join(all_errors[:10])
            )

        return is_valid, all_errors
