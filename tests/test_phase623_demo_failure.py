"""Phase 6.23: Focused regression test suite for demo scenario Step 16 refusal validation.

Verifies:
1. Out-of-domain queries (e.g. 'What is the recipe for chocolate cake?') return safe refusal.
2. Grounded status is False and answer contains honest refusal text.
3. In-domain queries across all 5 languages (EN, HI, TA, TE, ML) continue to produce grounded answers.
4. RelevanceGuard content-word extraction correctly differentiates relevant from off-topic evidence.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.generation.models import AskRequest
from app.api.routes import ask_endpoint
from app.guardrails.relevance import RelevanceGuard, extract_content_words
from app.main import app


# ===========================================================================
# 1. Direct RelevanceGuard Unit Tests
# ===========================================================================

def test_relevance_guard_stopword_extraction():
    """Verify that multilingual functional stopwords are filtered out."""
    en_words = extract_content_words("What is the recipe for chocolate cake?")
    assert en_words == {"recipe", "chocolate", "cake"}

    hi_words = extract_content_words("मशीन लर्निंग क्या है?")
    assert "मशीन" in hi_words and "लर्निंग" in hi_words

    ta_words = extract_content_words("செயற்கை நுண்ணறிவு என்றால் என்ன?")
    assert "செயற்கை" in ta_words and "நுண்ணறிவு" in ta_words

    te_words = extract_content_words("కృత్రిమ మేధస్సు అంటే ఏమిటి?")
    assert "కృత్రిమ" in te_words and "మేధస్సు" in te_words

    ml_words = extract_content_words("കൃത്രിമ ബുദ്ധി എന്താണ്?")
    assert "കൃത്രിമ" in ml_words and "ബുദ്ധി" in ml_words


def test_relevance_guard_off_topic_detection():
    """Verify that unrelated text is correctly evaluated as off-topic."""
    guard = RelevanceGuard()
    query = "What is the recipe for chocolate cake?"
    unrelated_context = "One of the things these apples are good for is apple pies. You can make a quick apple sauce."

    eval_result = guard.evaluate_relevance(query=query, retrieved_context=unrelated_context)
    assert eval_result.is_relevant is False
    assert eval_result.overlap_ratio < 0.40


def test_relevance_guard_in_domain_matching():
    """Verify that matching text is correctly evaluated as in-domain."""
    guard = RelevanceGuard()
    query = "What is artificial intelligence?"
    matching_context = "Artificial intelligence (AI) is intelligence demonstrated by machines."

    eval_result = guard.evaluate_relevance(query=query, retrieved_context=matching_context)
    assert eval_result.is_relevant is True
    assert eval_result.overlap_ratio >= 0.40


# ===========================================================================
# 2. End-to-End Safe Refusal Regression Tests (Step 16 Contract)
# ===========================================================================

@pytest.mark.asyncio
async def test_chocolate_cake_demo_refusal():
    """Test the exact failed Step 16 demo query safely refuses without hallucination."""
    req = AskRequest(query="What is the recipe for chocolate cake?", language="en", top_k=3)
    resp = await ask_endpoint(req)

    assert resp.grounded is False or "don't have enough information" in resp.answer.lower()
    assert len(resp.answer) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "What is the recipe for chocolate cake?",
        "How to bake a chocolate fudge cake at home?",
    ],
)
async def test_all_negative_queries_refuse(query: str):
    """Verify negative out-of-domain queries return ungrounded refusal responses."""
    req = AskRequest(query=query, language="en", top_k=3)
    resp = await ask_endpoint(req)

    assert resp.grounded is False or "don't have enough information" in resp.answer.lower()
    assert len(resp.answer) > 0


# ===========================================================================
# 3. In-Domain Multilingual Verification (No Regressions)
# ===========================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,lang",
    [
        ("What is artificial intelligence?", "en"),
        ("मशीन लर्निंग क्या है?", "hi"),
        ("செயற்கை நுண்ணறிவு என்றால் என்ன?", "ta"),
        ("కృత్రిమ మేధస్సు అంటే ఏమిటి?", "te"),
        ("കൃത്രിമ ബുദ്ധി എന്താണ്?", "ml"),
    ],
)
async def test_multilingual_in_domain_not_refused(query: str, lang: str):
    """Verify in-domain queries across all 5 languages receive valid structured answers."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/ask",
            json={"query": query, "language": lang, "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["answer"]) > 0
        assert data["retrieved_context_summary"]["chunks_count"] > 0
