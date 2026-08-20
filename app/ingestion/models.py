"""Internal Document representations for dataset ingestion."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Normalized document representation extracted from raw dataset records.
    
    Fields are derived strictly from the discovered MSMARCO-XI dataset schema.
    No unverified or invented metadata is added.
    """

    document_id: str = Field(
        ...,
        description="Unique identifier for the document, derived deterministically from query_id and passage index.",
    )
    text: str = Field(
        ...,
        description="Clean, normalized passage content in the target language (or original English if specified).",
    )
    title: Optional[str] = Field(
        default=None,
        description="Document title if available in the dataset (None if not present).",
    )
    language: Optional[str] = Field(
        default=None,
        description="Target language code from the dataset (e.g. 'hin_Deva', 'hi', 'asm_Beng').",
    )
    source: str = Field(
        default="ai4bharat/MSMARCO-XI",
        description="Dataset source identifier.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Preserved raw metadata including query_id, query, is_selected relevance, source_lang, and translation meta.",
    )
