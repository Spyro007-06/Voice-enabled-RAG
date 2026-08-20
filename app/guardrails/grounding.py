"""Grounding validation and hallucination detection for generated answers."""

import re
from typing import Any, List, Union

from app.guardrails.models import GroundingAssessment
from app.retrieval.bm25 import IndicUnicodeTokenizer
from app.reranking.models import RerankResult


class GroundingGuard:
    """Verifies that claims in the generated answer are supported by the retrieved context."""

    def __init__(self, min_grounding_threshold: float = 0.40):
        self.min_grounding_threshold = min_grounding_threshold
        self.tokenizer = IndicUnicodeTokenizer()

    def evaluate_grounding(
        self,
        answer: str,
        retrieved_context: Union[str, List[Any]],
    ) -> GroundingAssessment:
        """Validate answer against retrieved context text for factual grounding and hallucination.

        Args:
            answer: Generated answer text.
            retrieved_context: Retrieved chunk list or raw context text.

        Returns:
            GroundingAssessment with grounding score, unsupported claims list, and decision reason.
        """
        clean_answer = answer.strip() if answer else ""
        if not clean_answer:
            return GroundingAssessment(
                is_grounded=False,
                grounding_score=0.0,
                unsupported_claims=["Empty answer"],
                is_refusal=False,
                reason="Answer is empty.",
            )

        # 1. Check for standard honest refusal statements
        lower_ans = clean_answer.lower()
        refusal_patterns = [
            "don't have enough information",
            "do not have enough information",
            "insufficient information",
            "context does not contain",
            "no context available",
            "not enough information in the retrieved context",
            "available context is insufficient",
        ]
        if any(p in lower_ans for p in refusal_patterns):
            return GroundingAssessment(
                is_grounded=True,
                grounding_score=1.0,
                unsupported_claims=[],
                is_refusal=True,
                reason="Answer is an honest statement of insufficient context.",
            )

        # 2. Extract context text
        context_texts: List[str] = []
        if isinstance(retrieved_context, str):
            if retrieved_context.strip():
                context_texts.append(retrieved_context.strip())
        elif isinstance(retrieved_context, list):
            for item in retrieved_context:
                if isinstance(item, RerankResult):
                    context_texts.append(item.text)
                elif isinstance(item, dict):
                    txt = str(item.get("text") or item.get("content") or "")
                    if txt:
                        context_texts.append(txt)
                elif hasattr(item, "text"):
                    txt = str(getattr(item, "text", ""))
                    if txt:
                        context_texts.append(txt)
                elif isinstance(item, str) and item.strip():
                    context_texts.append(item.strip())

        if not context_texts:
            # If no context was provided but model generated factual content, it is ungrounded
            return GroundingAssessment(
                is_grounded=False,
                grounding_score=0.0,
                unsupported_claims=[clean_answer[:200]],
                is_refusal=False,
                reason="Answer produced factual claims without any supporting context.",
            )

        combined_context = " ".join(context_texts).lower()
        context_tokens = set(self.tokenizer.tokenize(combined_context))

        # 3. Strip mock or system template preambles before sentence splitting
        eval_answer = clean_answer
        if "[MOCK]" in eval_answer:
            eval_answer = re.sub(r"\[MOCK\].*?source\(s\):\s*", "", eval_answer, flags=re.DOTALL).strip()

        # Fast path: if the substantive answer is directly contained in the combined context
        if eval_answer and (eval_answer.lower() in combined_context or any(eval_answer.lower() in ctx.lower() for ctx in context_texts)):
            return GroundingAssessment(
                is_grounded=True,
                grounding_score=1.0,
                unsupported_claims=[],
                is_refusal=False,
                reason="Answer is directly supported by retrieved evidence text.",
            )

        # Sentence-level grounding check
        sentences = [s.strip() for s in re.split(r"[।\.\?\!\n]+", eval_answer) if len(s.strip()) > 2]
        if not sentences:
            sentences = [eval_answer.strip() or clean_answer]

        supported_sentences_count = 0
        unsupported_claims: List[str] = []

        for sent in sentences:
            sent_tokens = [t for t in self.tokenizer.tokenize(sent.lower()) if len(t) > 1]
            if not sent_tokens:
                supported_sentences_count += 1
                continue

            matching_tokens = [t for t in sent_tokens if (t in context_tokens or t in combined_context)]
            support_ratio = len(matching_tokens) / len(sent_tokens)

            # Check if sufficient portion of content words are found in context or direct substring
            if support_ratio >= 0.20 or (len(sent) > 5 and sent.lower() in combined_context):
                supported_sentences_count += 1
            else:
                unsupported_claims.append(sent)

        grounding_score = supported_sentences_count / len(sentences)

        if grounding_score >= self.min_grounding_threshold:
            is_grounded = True
            reason = (
                f"Answer is supported by retrieved evidence "
                f"({supported_sentences_count}/{len(sentences)} claims verified, score: {grounding_score:.2f})."
            )
        else:
            is_grounded = False
            reason = (
                f"Answer contains unsupported claims "
                f"({len(unsupported_claims)}/{len(sentences)} claims ungrounded, score: {grounding_score:.2f})."
            )

        return GroundingAssessment(
            is_grounded=is_grounded,
            grounding_score=round(grounding_score, 4),
            unsupported_claims=unsupported_claims,
            is_refusal=False,
            reason=reason,
        )
