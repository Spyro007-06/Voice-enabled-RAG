"""Validation and provenance verification for LLM generation outputs and citations."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from app.generation.models import CitationProvenance, ValidationResult
from app.reranking.models import RerankResult

logger = logging.getLogger(__name__)


class CitationValidator:
    """Validates citations against actual retrieved context, removing fabricated or ungrounded references."""

    @staticmethod
    def extract_valid_chunk_map(
        retrieved_context: Union[str, List[Any]],
    ) -> Dict[str, CitationProvenance]:
        """Build an authoritative lookup map of valid chunk IDs from the retrieved evidence.

        Args:
            retrieved_context: String context or list of chunk objects (e.g. RerankResult, dicts).

        Returns:
            Dict mapping normalized chunk_id/doc_id to CitationProvenance.
        """
        valid_map: Dict[str, CitationProvenance] = {}

        if not retrieved_context:
            return valid_map

        if isinstance(retrieved_context, str):
            if retrieved_context.strip():
                prov = CitationProvenance(
                    chunk_id="direct_context",
                    document_id="doc_0",
                    rank=1,
                    chunk_type="text",
                    language=None,
                    score=1.0,
                    snippet=retrieved_context.strip()[:150],
                )
                valid_map["direct_context"] = prov
                valid_map["doc_0"] = prov
            return valid_map

        if isinstance(retrieved_context, list):
            for idx, item in enumerate(retrieved_context):
                if isinstance(item, RerankResult):
                    prov = CitationProvenance(
                        chunk_id=item.chunk_id,
                        document_id=item.document_id,
                        rank=item.rank if item.rank else idx + 1,
                        chunk_type=item.chunk_type,
                        language=item.language,
                        score=item.reranker_score if item.reranker_score is not None else item.fusion_score,
                        snippet=item.text[:150] if item.text else None,
                    )
                    valid_map[item.chunk_id] = prov
                    valid_map[item.document_id] = prov
                    valid_map[f"doc_{idx+1}"] = prov
                    valid_map[f"chunk_{idx+1}"] = prov
                elif isinstance(item, dict):
                    cid = str(item.get("chunk_id") or f"chunk_{idx+1}")
                    did = str(item.get("document_id") or f"doc_{idx+1}")
                    text = str(item.get("text") or item.get("content") or "")
                    prov = CitationProvenance(
                        chunk_id=cid,
                        document_id=did,
                        rank=int(item.get("rank") or idx + 1),
                        chunk_type=str(item.get("chunk_type") or "dict"),
                        language=item.get("language"),
                        score=float(item["score"]) if "score" in item and item["score"] is not None else None,
                        snippet=text[:150] if text else None,
                    )
                    valid_map[cid] = prov
                    valid_map[did] = prov
                    valid_map[f"doc_{idx+1}"] = prov
                    valid_map[f"chunk_{idx+1}"] = prov
                elif hasattr(item, "text"):
                    cid = str(getattr(item, "chunk_id", f"chunk_{idx+1}"))
                    did = str(getattr(item, "document_id", f"doc_{idx+1}"))
                    text = str(getattr(item, "text", ""))
                    prov = CitationProvenance(
                        chunk_id=cid,
                        document_id=did,
                        rank=int(getattr(item, "rank", idx + 1)),
                        chunk_type=str(getattr(item, "chunk_type", "object")),
                        language=getattr(item, "language", None),
                        score=getattr(item, "score", None),
                        snippet=text[:150] if text else None,
                    )
                    valid_map[cid] = prov
                    valid_map[did] = prov
                    valid_map[f"doc_{idx+1}"] = prov
                    valid_map[f"chunk_{idx+1}"] = prov
                elif isinstance(item, str) and item.strip():
                    cid = f"chunk_{idx+1}"
                    did = f"doc_{idx+1}"
                    prov = CitationProvenance(
                        chunk_id=cid,
                        document_id=did,
                        rank=idx + 1,
                        chunk_type="text",
                        snippet=item.strip()[:150],
                    )
                    valid_map[cid] = prov
                    valid_map[did] = prov

        return valid_map

    @classmethod
    def validate_citations(
        cls,
        raw_citations: List[Any],
        retrieved_context: Union[str, List[Any]],
    ) -> Tuple[List[str], List[CitationProvenance], List[str]]:
        """Validate candidate citation IDs against retrieved evidence.

        Args:
            raw_citations: List of citation strings or objects returned by generation.
            retrieved_context: Authoritative retrieved context.

        Returns:
            Tuple of (validated_citation_ids, citation_provenance_objects, fabricated_citations).
        """
        valid_map = cls.extract_valid_chunk_map(retrieved_context)
        validated_ids: List[str] = []
        provenance_list: List[CitationProvenance] = []
        fabricated_ids: List[str] = []
        seen_ids = set()

        for cite in raw_citations or []:
            cite_str = str(cite).strip() if cite is not None else ""
            if not cite_str:
                continue

            if cite_str in valid_map:
                prov = valid_map[cite_str]
                actual_chunk_id = prov.chunk_id
                if actual_chunk_id not in seen_ids:
                    seen_ids.add(actual_chunk_id)
                    validated_ids.append(actual_chunk_id)
                    provenance_list.append(prov)
            else:
                fabricated_ids.append(cite_str)
                logger.warning(
                    "CitationValidator: Rejected fabricated/unverified citation ID '%s'",
                    cite_str,
                )

        return validated_ids, provenance_list, fabricated_ids


class AnswerValidator:
    """Validates generation answer quality, bounds, confidence, and output formatting."""

    MAX_ANSWER_CHARS: int = 10000

    @classmethod
    def validate_answer(
        cls,
        answer: Optional[str],
        grounded: bool = True,
        confidence: float = 1.0,
        citations: Optional[List[Any]] = None,
        retrieved_context: Optional[Union[str, List[Any]]] = None,
        max_length: Optional[int] = None,
    ) -> ValidationResult:
        """Validate and sanitize answer output from LLM generation.

        Checks:
        - Non-empty answer
        - Maximum length boundary
        - Confidence calibration within [0.0, 1.0]
        - Citation verification against context
        - Grounding consistency
        """
        issues: List[str] = []
        max_len = max_length or cls.MAX_ANSWER_CHARS

        # 1. Validate Answer Text Presence
        clean_answer = answer.strip() if answer else ""
        if not clean_answer:
            return ValidationResult(
                is_valid=False,
                issues=["Answer cannot be empty or whitespace only."],
                cleaned_answer="",
                validated_citations=[],
                citation_provenance=[],
                fabricated_citations=[],
                confidence=0.0,
            )

        # 2. Validate Length
        if len(clean_answer) > max_len:
            issues.append(f"Answer exceeded maximum length limit ({len(clean_answer)} > {max_len} chars). Truncated.")
            clean_answer = clean_answer[:max_len].rstrip() + "..."

        # 3. Validate Confidence Bounds
        calibrated_confidence = max(0.0, min(1.0, float(confidence)))
        if confidence < 0.0 or confidence > 1.0:
            issues.append(f"Confidence score {confidence} outside [0.0, 1.0] range. Clamped to {calibrated_confidence}.")

        # 4. Validate Citations
        validated_ids, provenance_list, fabricated_ids = CitationValidator.validate_citations(
            raw_citations=citations or [],
            retrieved_context=retrieved_context or "",
        )

        if fabricated_ids:
            issues.append(f"Removed {len(fabricated_ids)} unverified/fabricated citation(s): {fabricated_ids}")
            # Penalize confidence slightly if hallucinations/fabricated citations were detected
            calibrated_confidence = max(0.0, round(calibrated_confidence * 0.8, 2))

        # 5. Check Grounding Consistency
        has_context = bool(retrieved_context and (len(retrieved_context) > 0 if isinstance(retrieved_context, (list, str)) else True))
        is_refusal = (
            "insufficient information" in clean_answer.lower()
            or "context does not contain" in clean_answer.lower()
            or "no context available" in clean_answer.lower()
        )

        final_grounded = grounded
        if is_refusal or not has_context:
            final_grounded = False

        return ValidationResult(
            is_valid=len(issues) == 0 or (len(issues) == 1 and "Removed" in issues[0]),
            issues=issues,
            cleaned_answer=clean_answer,
            validated_citations=validated_ids,
            citation_provenance=provenance_list,
            fabricated_citations=fabricated_ids,
            confidence=calibrated_confidence,
        )
