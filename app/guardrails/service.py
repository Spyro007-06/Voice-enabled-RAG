"""Guardrail service coordinating pre- and post-generation safety, grounding, and policy enforcement."""

import logging
import re
import time
from functools import lru_cache
from typing import Any, List, Optional, Tuple, Union

from app.config import get_settings
from app.generation.validators import CitationValidator
from app.guardrails.grounding import GroundingGuard
from app.guardrails.models import (
    GroundingAssessment,
    GuardrailAction,
    GuardrailResult,
    RelevanceAssessment,
)
from app.guardrails.relevance import RelevanceGuard

logger = logging.getLogger(__name__)


# Prompt injection and adversarial attack signature patterns
INJECTION_PATTERNS = [
    r"(?i)\b(ignore|disregard|forget)\b.*\b(all|previous|prior|above|system)\b.*\b(instructions|rules|prompts|directives)\b",
    r"(?i)\b(system\s*override|admin\s*override|developer\s*mode|jailbreak|dan\s*mode|godmode)\b",
    r"(?i)\byou\s+are\s+now\s+.*?\b(unconstrained|dan|unfiltered|godmode|jailbroken)\b",
    r"(?i)\b(reveal|show|print|leak)\b.*\b(system\s*prompt|secret\s*key|hidden\s*instructions)\b",
    r"(?i)\[TAG:(SYSTEM|CONTEXT|RULES)\]",
]

# Obvious unsafe harm/weapon synthesis signatures (non-educational)
HARMFUL_PATTERNS = [
    r"(?i)\b(how\s+to\s+(build|make|synthesize)\s+(a\s+bomb|biological\s+weapon|chemical\s+weapon|explosive))\b",
    r"(?i)\b(instructions\s+for\s+(ddos|ransomware\s+attack|credit\s+card\s+fraud))\b",
]


class GuardrailService:
    """Orchestrates pre-generation and post-generation verification and grounding guardrails."""

    def __init__(
        self,
        min_confidence_threshold: Optional[float] = None,
        min_relevance_threshold: Optional[float] = None,
        min_grounding_threshold: Optional[float] = None,
        replace_ungrounded: Optional[bool] = None,
        fallback_message: Optional[str] = None,
    ):
        settings = get_settings()
        self.min_confidence = (
            min_confidence_threshold
            if min_confidence_threshold is not None
            else settings.GUARDRAIL_MIN_CONFIDENCE
        )
        self.min_relevance = (
            min_relevance_threshold
            if min_relevance_threshold is not None
            else settings.GUARDRAIL_MIN_RELEVANCE
        )
        self.min_grounding = (
            min_grounding_threshold
            if min_grounding_threshold is not None
            else 0.40
        )
        self.replace_ungrounded = (
            replace_ungrounded
            if replace_ungrounded is not None
            else settings.GUARDRAIL_REPLACE_UNGROUNDED
        )
        self.fallback_message = (
            fallback_message
            if fallback_message is not None
            else settings.GUARDRAIL_FALLBACK_MESSAGE
        )

        self.relevance_guard = RelevanceGuard(min_relevance_threshold=self.min_relevance)
        self.grounding_guard = GroundingGuard(min_grounding_threshold=self.min_grounding)

    def validate_input(self, query: str) -> GuardrailResult:
        """Validate input query against prompt injection and malicious exploit attempts."""
        t_start = time.perf_counter()
        clean_query = query.strip() if query else ""

        if not clean_query:
            lat_ms = (time.perf_counter() - t_start) * 1000.0
            return GuardrailResult(
                allowed=False,
                reason="empty_query",
                grounded=False,
                confidence=0.0,
                action=GuardrailAction.BLOCK_INPUT.value,
                issues=["Query cannot be empty or whitespace only."],
                latency_ms=round(lat_ms, 3),
                safe_fallback_text="Please provide a valid non-empty question.",
            )

        # 1. Check Prompt Injection
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, clean_query):
                lat_ms = (time.perf_counter() - t_start) * 1000.0
                logger.warning("GuardrailService: Blocked prompt injection pattern in query '%s'", clean_query[:60])
                return GuardrailResult(
                    allowed=False,
                    reason="prompt_injection_detected",
                    grounded=False,
                    confidence=0.0,
                    action=GuardrailAction.BLOCK_INPUT.value,
                    issues=["Input contains disallowed prompt injection or system override directives."],
                    latency_ms=round(lat_ms, 3),
                    safe_fallback_text=self.fallback_message,
                )

        # 2. Check Harmful / Dangerous Requests
        for pattern in HARMFUL_PATTERNS:
            if re.search(pattern, clean_query):
                lat_ms = (time.perf_counter() - t_start) * 1000.0
                logger.warning("GuardrailService: Blocked harmful query '%s'", clean_query[:60])
                return GuardrailResult(
                    allowed=False,
                    reason="unsafe_input_detected",
                    grounded=False,
                    confidence=0.0,
                    action=GuardrailAction.BLOCK_INPUT.value,
                    issues=["Input violates safety policy."],
                    latency_ms=round(lat_ms, 3),
                    safe_fallback_text="I cannot assist with requests involving harmful or unsafe materials.",
                )

        lat_ms = (time.perf_counter() - t_start) * 1000.0
        return GuardrailResult(
            allowed=True,
            reason="input_safe",
            grounded=True,
            confidence=1.0,
            action=GuardrailAction.ALLOW.value,
            issues=[],
            latency_ms=round(lat_ms, 3),
        )

    def validate_pre_generation(
        self,
        query: str,
        retrieved_context: Union[str, List[Any]],
        retrieval_confidence: float = 1.0,
    ) -> GuardrailResult:
        """Execute pre-generation guardrails: input safety, empty context, low confidence, and off-topic checks."""
        t_start = time.perf_counter()

        # 1. Input Safety Check
        input_result = self.validate_input(query)
        if not input_result.allowed:
            input_result.latency_ms = round((time.perf_counter() - t_start) * 1000.0, 3)
            return input_result

        # 2. Empty-Context Guard
        has_context = bool(
            retrieved_context
            and (len(retrieved_context) > 0 if isinstance(retrieved_context, (list, str)) else True)
        )
        if not has_context:
            lat_ms = (time.perf_counter() - t_start) * 1000.0
            return GuardrailResult(
                allowed=False,
                reason="empty_retrieval_context",
                grounded=False,
                confidence=0.0,
                action=GuardrailAction.REFUSE_GENERATION.value,
                issues=["No usable context retrieved for query."],
                latency_ms=round(lat_ms, 3),
                safe_fallback_text=self.fallback_message,
            )

        # 3. Low-Confidence Retrieval Guard
        if retrieval_confidence < self.min_confidence:
            lat_ms = (time.perf_counter() - t_start) * 1000.0
            logger.info(
                "GuardrailService: Low retrieval confidence (%.2f < %.2f) for query '%s'",
                retrieval_confidence,
                self.min_confidence,
                query[:50],
            )
            return GuardrailResult(
                allowed=False,
                reason="low_retrieval_confidence",
                grounded=False,
                confidence=round(retrieval_confidence, 4),
                action=GuardrailAction.REFUSE_GENERATION.value,
                issues=[f"Adaptive retrieval confidence {retrieval_confidence:.2f} below threshold {self.min_confidence:.2f}."],
                latency_ms=round(lat_ms, 3),
                safe_fallback_text=self.fallback_message,
            )

        # 4. Off-Topic / Evidence Relevance Guard
        relevance_eval = self.relevance_guard.evaluate_relevance(
            query=query,
            retrieved_context=retrieved_context,
        )
        rel_data = relevance_eval.model_dump()
        if not relevance_eval.is_relevant:
            lat_ms = (time.perf_counter() - t_start) * 1000.0
            return GuardrailResult(
                allowed=False,
                reason="off_topic_query",
                grounded=False,
                confidence=round(relevance_eval.relevance_score, 4),
                action=GuardrailAction.REFUSE_GENERATION.value,
                issues=[relevance_eval.reason],
                latency_ms=round(lat_ms, 3),
                safe_fallback_text=self.fallback_message,
                relevance=rel_data,
            )

        lat_ms = (time.perf_counter() - t_start) * 1000.0
        return GuardrailResult(
            allowed=True,
            reason="pre_generation_passed",
            grounded=True,
            confidence=round(max(retrieval_confidence, relevance_eval.relevance_score), 4),
            action=GuardrailAction.ALLOW.value,
            issues=[],
            latency_ms=round(lat_ms, 3),
            relevance=rel_data,
        )

    def validate_post_generation(
        self,
        query: str,
        answer: str,
        retrieved_context: Union[str, List[Any]],
        citations: Optional[List[str]] = None,
        raw_confidence: float = 1.0,
    ) -> GuardrailResult:
        """Execute post-generation guardrails: grounding validation, hallucination detection, citation verification."""
        t_start = time.perf_counter()
        issues: List[str] = []

        # 1. Grounding & Hallucination Assessment
        grounding_eval = self.grounding_guard.evaluate_grounding(
            answer=answer,
            retrieved_context=retrieved_context,
        )

        # 2. Citation Verification
        validated_cites, prov_list, fabricated_cites = CitationValidator.validate_citations(
            raw_citations=citations or [],
            retrieved_context=retrieved_context,
        )

        if fabricated_cites:
            issues.append(f"Detected {len(fabricated_cites)} unverified/fabricated citation(s): {fabricated_cites}")

        # 3. Decision Logic
        if grounding_eval.is_refusal:
            lat_ms = (time.perf_counter() - t_start) * 1000.0
            return GuardrailResult(
                allowed=True,
                reason="honest_refusal",
                grounded=False,
                confidence=1.0,
                action=GuardrailAction.ALLOW.value,
                issues=[],
                latency_ms=round(lat_ms, 3),
            )

        if not grounding_eval.is_grounded:
            issues.append(grounding_eval.reason)
            action = (
                GuardrailAction.REPLACE_WITH_FALLBACK.value
                if self.replace_ungrounded
                else GuardrailAction.FLAG_HALLUCINATION.value
            )
            calibrated_conf = round(raw_confidence * 0.3, 2)
            lat_ms = (time.perf_counter() - t_start) * 1000.0

            return GuardrailResult(
                allowed=not self.replace_ungrounded,
                reason="unsupported_claims_detected",
                grounded=False,
                confidence=calibrated_conf,
                action=action,
                issues=issues,
                latency_ms=round(lat_ms, 3),
                safe_fallback_text=self.fallback_message,
            )

        # Grounded & Valid
        calibrated_conf = round(raw_confidence * (0.9 if fabricated_cites else 1.0), 2)
        lat_ms = (time.perf_counter() - t_start) * 1000.0

        return GuardrailResult(
            allowed=True,
            reason="grounded_answer_verified",
            grounded=True,
            confidence=max(0.0, min(1.0, calibrated_conf)),
            action=GuardrailAction.ALLOW.value,
            issues=issues,
            latency_ms=round(lat_ms, 3),
        )


@lru_cache(maxsize=1)
def get_guardrail_service() -> GuardrailService:
    """Provide a thread-safe singleton instance of GuardrailService."""
    return GuardrailService()
