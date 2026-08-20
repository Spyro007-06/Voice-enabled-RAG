"""Unit tests for Phase 6.6 generation benchmarking utilities."""

import pytest
from scripts.benchmark_generation import (
    EVALUATION_QUERIES,
    calculate_percentiles,
    execute_rag_pipeline_step,
)


def test_calculate_percentiles_empty():
    """Verify calculate_percentiles handles empty list without exception."""
    res = calculate_percentiles([])
    assert res["mean_ms"] == 0.0
    assert res["p50_ms"] == 0.0
    assert res["p95_ms"] == 0.0
    assert res["p100_ms"] == 0.0


def test_calculate_percentiles_accurate_distribution():
    """Verify calculate_percentiles calculates accurate percentile statistics."""
    # 1 to 100
    data = [float(i) for i in range(1, 101)]
    res = calculate_percentiles(data)

    assert res["mean_ms"] == 50.5
    assert res["p50_ms"] == 50.5
    assert res["p70_ms"] == 70.3
    assert res["p95_ms"] == 95.05
    assert res["p100_ms"] == 100.0
    assert res["min_ms"] == 1.0


def test_evaluation_queries_multilingual_coverage():
    """Verify the benchmark query suite covers all required languages (EN, HI, TA, TE, ML)."""
    assert len(EVALUATION_QUERIES) >= 30

    languages = {q["language"] for q in EVALUATION_QUERIES}
    assert "en" in languages
    assert "hi" in languages
    assert "ta" in languages
    assert "te" in languages
    assert "ml" in languages


@pytest.mark.asyncio
async def test_execute_rag_pipeline_step_structure():
    """Verify execute_rag_pipeline_step returns all required stage latencies and metrics."""
    result = await execute_rag_pipeline_step(
        query="What is the capital of Goa?",
        language="en",
        top_k=2,
    )

    # Check stage measurements
    assert "embedding_ms" in result
    assert "retrieval_ms" in result
    assert "reranking_ms" in result
    assert "context_selection_ms" in result
    assert "prompt_ms" in result
    assert "generation_ms" in result
    assert "grounding_ms" in result
    assert "total_ms" in result

    # Check throughput metrics
    assert "tokens_est" in result
    assert "context_chars" in result
    assert "output_chars" in result
    assert "is_success" in result
    assert "is_refusal" in result
    assert result["total_ms"] > 0
