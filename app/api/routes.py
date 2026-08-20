import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from app.observability.metrics import get_metrics_registry
from app.observability.security import sanitize_output_text
from app.observability.tracing import get_current_request_id

from app.config import get_settings
from app.generation.models import (
    AskLatencyBreakdown,
    AskRequest,
    AskResponse,
    GenerationConfig,
    RetrievedContextSummary,
)
from app.generation.prompt import get_prompt_builder
from app.generation.service import get_generation_service
from app.orchestration.exceptions import (
    AudioPayloadTooLargeError,
    AudioValidationError,
    STTError,
    TTSError,
    UnsupportedAudioFormatError,
    VoiceOrchestrationError,
)
from app.orchestration.models import VoiceAskResponse
from app.orchestration.voice_rag import get_voice_rag_orchestrator
from app.reranking.adaptive import get_adaptive_retrieval_service
from app.reranking.models import (
    AdaptiveRetrieveRequest,
    AdaptiveRetrieveResponse,
    RerankRequest,
    RerankResponse,
)
from app.reranking.service import get_reranking_service
from app.retrieval.models import RetrievalRequest, RetrievalResponse
from app.retrieval.service import get_retrieval_service

logger = logging.getLogger(__name__)
router = APIRouter()


def validate_request_security_limits(query: Optional[str] = None, language: Optional[str] = None) -> None:
    """Validate query and language parameter string lengths against safe limits."""
    settings = get_settings()
    max_q_len = getattr(settings, "MAX_QUERY_LENGTH", 2000)
    max_l_len = getattr(settings, "MAX_LANGUAGE_LENGTH", 32)
    if query is not None and len(query) > max_q_len:
        raise HTTPException(
            status_code=422,
            detail=f"Query length ({len(query)} characters) exceeds maximum allowed limit of {max_q_len} characters.",
        )
    if language is not None and len(language) > max_l_len:
        raise HTTPException(
            status_code=422,
            detail=f"Language code length ({len(language)} characters) exceeds maximum allowed limit of {max_l_len} characters.",
        )


@router.get("/", response_model=Dict[str, str], tags=["Root"])
async def root() -> Dict[str, str]:
    """Root endpoint returning system status, project name, and version."""
    return {
        "status": "online",
        "project": "HH Goa 2026 Voice RAG",
        "version": "1.0.0",
    }


@router.get("/health", response_model=Dict[str, Any], tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint returning system health status and active provider names."""
    from app.providers.factory import get_active_providers_info
    return {
        "status": "healthy",
        "providers": get_active_providers_info(),
    }


@router.get(
    "/api/debug/last-request",
    tags=["Debug"],
    summary="Development-only pipeline inspector for the last executed request",
)
async def debug_last_request() -> Dict[str, Any]:
    """Inspect the last processed request for debugging. Only accessible when DEBUG=true."""
    settings = get_settings()
    if not settings.DEBUG:
        raise HTTPException(status_code=404, detail="Debug endpoint disabled in production.")
    from app.orchestration.voice_rag import get_last_request_trace
    trace = get_last_request_trace()
    if not trace:
        return {"message": "No requests processed yet."}
    return trace


@router.get(
    "/api/debug/retrieval",
    tags=["Debug"],
    summary="Safe development retrieval diagnostic function",
)
async def debug_retrieval(
    query: str = Query(..., description="Search query string"),
    language: Optional[str] = Query(None, description="Optional language filter"),
    top_k: int = Query(5, ge=1, le=20, description="Top-k candidates"),
) -> Dict[str, Any]:
    """Inspect the full retrieval pipeline: dense, BM25, RRF, reranking, and guardrail decision."""
    settings = get_settings()
    if not settings.DEBUG:
        raise HTTPException(status_code=404, detail="Debug endpoint disabled in production.")

    validate_request_security_limits(query=query, language=language)
    from app.retrieval.language import get_language_filter_synonyms, normalize_language_code
    from app.retrieval.service import get_retrieval_service
    from app.reranking.adaptive import get_adaptive_retrieval_service
    from app.guardrails.service import get_guardrail_service

    ret_svc = get_retrieval_service()
    adaptive_svc = get_adaptive_retrieval_service()
    guard_svc = get_guardrail_service()

    # 1. Fetch dense & bm25
    dense_res, _, _ = ret_svc.dense_retriever.retrieve(query=query, top_k=top_k * 2, language=language)
    bm25_res, _ = ret_svc.bm25_retriever.retrieve(query=query, top_k=top_k * 2, language=language)
    rrf_res = ret_svc.fusion.fuse_results(dense_results=dense_res, bm25_results=bm25_res, top_k=top_k * 2, method="rrf")

    # 2. Adaptive retrieval
    final_res, decision, _ = adaptive_svc.adaptive_retrieve(query=query, top_k=top_k, language=language, use_cache=False)

    # 3. Guardrail validation
    pre_guard = guard_svc.validate_pre_generation(query=query, retrieved_context=final_res, retrieval_confidence=decision.confidence_score)

    def _fmt_chunk(item: Any, rank: int) -> Dict[str, Any]:
        cid = getattr(item, "chunk_id", None) or (item.get("chunk_id") if isinstance(item, dict) else str(getattr(item, "id", "")))
        doc_id = getattr(item, "document_id", None) or (item.get("document_id") if isinstance(item, dict) else "")
        txt = getattr(item, "text", "") or (item.get("text") if isinstance(item, dict) else "")
        snippet = (txt[:120] + "...") if len(txt) > 120 else txt
        score = getattr(item, "reranker_score", None) or getattr(item, "fusion_score", None) or getattr(item, "dense_score", None) or getattr(item, "bm25_score", 0.0)
        return {
            "rank": rank,
            "chunk_id": cid,
            "document_id": doc_id,
            "language": getattr(item, "language", None) or (item.get("language") if isinstance(item, dict) else "unknown"),
            "chunking_strategy": getattr(item, "chunk_type", None) or (item.get("chunk_type") if isinstance(item, dict) else "fixed"),
            "score": round(float(score or 0.0), 4),
            "snippet": snippet,
        }

    return {
        "query": query,
        "requested_language": language,
        "resolved_languages": get_language_filter_synonyms(language) if language else ["all"],
        "dense_results": [_fmt_chunk(c, idx) for idx, c in enumerate(dense_res[:top_k], start=1)],
        "bm25_results": [_fmt_chunk(c, idx) for idx, c in enumerate(bm25_res[:top_k], start=1)],
        "rrf_results": [_fmt_chunk(c, idx) for idx, c in enumerate(rrf_res[:top_k], start=1)],
        "reranked_results": [_fmt_chunk(c, idx) for idx, c in enumerate(final_res, start=1)],
        "final_context": [_fmt_chunk(c, idx) for idx, c in enumerate(final_res, start=1)],
        "retrieval_confidence": decision.confidence_score,
        "grounding_decision": {
            "allowed": pre_guard.allowed,
            "reason": pre_guard.reason,
            "safe_fallback": pre_guard.safe_fallback_text,
        },
    }


@router.get(
    "/metrics",
    tags=["Observability"],
    summary="Prometheus-compatible telemetry metrics endpoint",
)
async def metrics_endpoint() -> Response:
    """Prometheus-compatible metrics endpoint exporting system counters, gauges, and latency histograms."""
    registry = get_metrics_registry()
    content = registry.generate_prometheus_text()
    return Response(
        content=content,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# ---------------------------------------------------------------------------
# SSE Helper
# ---------------------------------------------------------------------------

def _sse_event(event: str, data: object) -> str:
    """Format a single Server-Sent Event frame."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def _ask_stream_generator(
    query: str,
    language: Optional[str],
    top_k: int,
) -> AsyncGenerator[str, None]:
    """Core SSE generator for /api/ask-stream.

    Emits ordered stage events then LLM token chunks then a final `done` event.
    Falls back to a single buffered `done` event if streaming is unavailable.

    STREAMING STATUS: LLM token streaming is supported via Sarvam SSE.
    STREAMING STATUS: TTS DOES NOT SUPPORT STREAMING (single base64 payload — voice-ask only).
    """
    settings = get_settings()
    t_start = time.perf_counter()

    # --- Validate ---
    validate_request_security_limits(query=query, language=language)
    query = query.strip()
    if not query:
        yield _sse_event("error", {"message": "Query must not be empty.", "code": 422})
        return

    # --- Stage 1: Retrieval ---
    yield _sse_event("stage", {"stage": "retrieving", "message": "Retrieving sources..."})

    try:
        adaptive_service = get_adaptive_retrieval_service()
        results, decision, ret_latency = adaptive_service.adaptive_retrieve(
            query=query,
            top_k=top_k,
            language=language,
        )
    except Exception as rex:
        logger.error("SSE retrieval error: %s", rex, exc_info=True)
        yield _sse_event("error", {"message": "Retrieval failed.", "code": 502})
        return

    # --- Stage 2: Ranking / Guardrails ---
    yield _sse_event("stage", {"stage": "ranking", "message": "Ranking context..."})

    from app.guardrails.service import get_guardrail_service
    guardrail_service = get_guardrail_service()
    pre_guard = guardrail_service.validate_pre_generation(
        query=query,
        retrieved_context=results,
        retrieval_confidence=decision.confidence_score,
    )

    if not pre_guard.allowed:
        fallback = pre_guard.safe_fallback_text or "I don't have enough information to answer that."
        elapsed = round((time.perf_counter() - t_start) * 1000.0, 2)
        yield _sse_event("done", {
            "answer": fallback,
            "grounded": False,
            "citations": [],
            "citation_provenance": [],
            "language": language or "en",
            "guardrail_decision": "refuse",
            "latency_ms": elapsed,
            "streaming": False,
        })
        return

    # Build prompt
    prompt_builder = get_prompt_builder()
    built_prompt = prompt_builder.build(query=query, retrieved_context=results, language=language)

    # --- Stage 3: Token streaming ---
    yield _sse_event("stage", {"stage": "generating", "message": "Generating answer..."})

    full_answer = ""
    streaming_succeeded = False

    try:
        # Attempt LLM token streaming
        gen_service = get_generation_service()
        llm_provider = getattr(gen_service, "_provider", None) or getattr(gen_service, "provider", None)

        if llm_provider is not None and hasattr(llm_provider, "generate_stream"):
            gen_config = GenerationConfig(max_tokens=192)
            async for token in llm_provider.generate_stream(
                query=query,
                context=results,
                config=gen_config,
            ):
                full_answer += token
                yield _sse_event("token", {"token": token})
            streaming_succeeded = True
        else:
            raise AttributeError("Provider does not support generate_stream.")

    except Exception as stream_err:
        logger.warning("SSE token streaming failed (%s); falling back to buffered generation.", stream_err)
        streaming_succeeded = False
        full_answer = ""

        try:
            gen_service = get_generation_service()
            gen_result = await gen_service.generate(
                query=query,
                context=results,
                config=GenerationConfig(max_tokens=192),
                language=language,
            )
            full_answer = gen_result.answer
            # Emit full answer as a single synthetic token for UI consistency
            if full_answer:
                yield _sse_event("token", {"token": full_answer})
        except Exception as gen_err:
            logger.error("SSE buffered generation failed: %s", gen_err)
            full_answer = "I don't have enough information in the retrieved context to answer that."
            yield _sse_event("token", {"token": full_answer})

    # --- Post-generation guardrails ---
    post_guard = guardrail_service.validate_post_generation(
        query=query,
        answer=full_answer,
        retrieved_context=results,
        citations=[],
        raw_confidence=0.0,
    )

    if post_guard.action == "replace_with_fallback" and post_guard.safe_fallback_text:
        full_answer = post_guard.safe_fallback_text

    citations = [p.chunk_id for p in built_prompt.provenance] if built_prompt.provenance else []

    elapsed = round((time.perf_counter() - t_start) * 1000.0, 2)
    yield _sse_event("done", {
        "answer": full_answer,
        "grounded": post_guard.grounded,
        "citations": citations,
        "citation_provenance": [],
        "language": language or "en",
        "retrieval_confidence": round(decision.confidence_score, 4),
        "reranking_used": decision.should_rerank,
        "guardrail_decision": "allow" if post_guard.action == "allow" else "replace_fallback",
        "latency_ms": elapsed,
        "streaming": streaming_succeeded,
    })


@router.get(
    "/api/ask-stream",
    tags=["Streaming RAG"],
    summary="Server-Sent Events streaming endpoint (Retrieval + Ranked LLM token stream)",
    response_class=StreamingResponse,
)
async def ask_stream_endpoint(
    request: Request,
    query: str = Query(..., description="Text query to answer."),
    language: Optional[str] = Query(None, description="Optional BCP-47 language code."),
    top_k: int = Query(5, ge=1, le=20, description="Top-K context chunks to retrieve."),
) -> StreamingResponse:
    """SSE streaming endpoint: emits stage events then LLM token chunks then a final done payload.

    Event types emitted:
    - `stage`  — pipeline stage change: {stage, message}
    - `token`  — LLM response text delta: {token}
    - `done`   — final complete payload with citations and latency
    - `error`  — error with HTTP-style code and message

    STREAMING STATUS: LLM token streaming supported via Sarvam SSE.
    STREAMING STATUS: TTS DOES NOT SUPPORT STREAMING (use /api/voice-ask for synthesized audio).
    """
    return StreamingResponse(
        _ask_stream_generator(query=query, language=language, top_k=top_k),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post(
    "/api/retrieve",
    response_model=RetrievalResponse,
    tags=["Retrieval"],
    summary="Execute hybrid dense and lexical vector retrieval (Development Endpoint)",
)
async def retrieve_endpoint(request: RetrievalRequest) -> RetrievalResponse:
    """Execute hybrid retrieval over indexed MSMARCO-XI chunks with latency telemetry."""
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=422,
            detail="Query string cannot be empty or whitespace.",
        )
    validate_request_security_limits(query=query, language=request.language)

    try:
        service = get_retrieval_service()
        results, latency = service.retrieve(
            query=query,
            top_k=request.top_k,
            strategies=request.strategies,
            language=request.language,
            fusion_method=request.fusion_method,
            use_cache=True,
        )

        return RetrievalResponse(
            query=query,
            fusion_method=request.fusion_method,
            total_results=len(results),
            results=results,
            latency_ms=latency,
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=422,
            detail=sanitize_output_text(str(ve)),
        )
    except Exception as e:
        logger.error("Error executing retrieval endpoint: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval execution failed: {sanitize_output_text(str(e))}",
        )


@router.post(
    "/api/rerank",
    response_model=RerankResponse,
    tags=["Reranking"],
    summary="Execute candidate retrieval, cross-encoder reranking, and context selection (Development Endpoint)",
)
async def rerank_endpoint(request: RerankRequest) -> RerankResponse:
    """Execute cross-encoder reranking and diverse context selection over retrieved candidate chunks."""
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=422,
            detail="Query string cannot be empty or whitespace.",
        )
    validate_request_security_limits(query=query, language=request.language)

    try:
        service = get_reranking_service()
        results, context_stats, latency = service.rerank_and_select(
            query=query,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            strategies=request.strategies,
            language=request.language,
            retrieval_mode=request.retrieval_mode,
        )

        return RerankResponse(
            query=query,
            total_results=len(results),
            results=results,
            context_stats=context_stats,
            latency_ms=latency,
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=422,
            detail=sanitize_output_text(str(ve)),
        )
    except Exception as e:
        logger.error("Error executing reranking endpoint: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Reranking execution failed: {sanitize_output_text(str(e))}",
        )


@router.post(
    "/api/adaptive-retrieve",
    response_model=AdaptiveRetrieveResponse,
    tags=["Adaptive Retrieval"],
    summary="Execute confidence-guided latency-optimized adaptive retrieval (Phase 5.5 Endpoint)",
)
async def adaptive_retrieve_endpoint(request: AdaptiveRetrieveRequest) -> AdaptiveRetrieveResponse:
    """Execute confidence-based dynamic routing to skip or invoke lightweight/full reranking."""
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=422,
            detail="Query string cannot be empty or whitespace.",
        )
    validate_request_security_limits(query=query, language=request.language)

    try:
        service = get_adaptive_retrieval_service()
        results, decision, latency = service.adaptive_retrieve(
            query=query,
            top_k=request.top_k,
            strategies=request.strategies,
            language=request.language,
        )

        reranker_name = (
            "none (skipped)" if not decision.should_rerank or decision.reranker_tier in ("high", "skip") else (
                "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1" if decision.reranker == "minilm" or decision.reranker_tier in ("medium", "lightweight", "low")
                else ("fallback_rrf" if decision.reranker == "fallback_rrf" else "BAAI/bge-reranker-v2-m3")
            )
        )

        return AdaptiveRetrieveResponse(
            query=query,
            results=results,
            retrieval_confidence=decision.confidence_score,
            reranking_used=decision.should_rerank,
            reranker=reranker_name,
            routing_tier=decision.reranker_tier,
            candidate_k=decision.candidate_k,
            confidence_breakdown=decision.signals.to_dict() if decision.signals else None,
            latency_ms=latency,
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=422,
            detail=sanitize_output_text(str(ve)),
        )
    except Exception as e:
        logger.error("Error executing adaptive retrieval endpoint: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Adaptive retrieval execution failed: {sanitize_output_text(str(e))}",
        )


@router.post(
    "/api/ask",
    response_model=AskResponse,
    tags=["Generation"],
    summary="Execute end-to-end adaptive retrieval, grounded prompt construction, and LLM response generation",
)
async def ask_endpoint(request: AskRequest) -> AskResponse:
    """Execute complete end-to-end voice/text RAG query pipeline.

    Flow:
    1. Validate query.
    2. Adaptive hybrid retrieval & confidence-guided reranking.
    3. Diverse context selection & budgeting.
    4. Grounded, injection-resistant prompt construction with provenance.
    5. Provider-independent generation execution.
    6. Return structured answer with stage-by-stage latency telemetry.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=422,
            detail="Query string cannot be empty or whitespace.",
        )
    validate_request_security_limits(query=query, language=request.language)

    start_time = time.perf_counter()
    settings = get_settings()

    # 1. Execute Adaptive Retrieval Pipeline
    try:
        adaptive_service = get_adaptive_retrieval_service()
        results, decision, ret_latency = adaptive_service.adaptive_retrieve(
            query=query,
            top_k=request.top_k,
            strategies=request.strategies,
            language=request.language,
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=422,
            detail=sanitize_output_text(str(ve)),
        )
    except Exception as exc:
        logger.error("Adaptive retrieval failure in /api/ask: %s", exc, exc_info=True)
        elapsed_total_ms = (time.perf_counter() - start_time) * 1000.0
        return AskResponse(
            query=query,
            answer="",
            grounded=False,
            citations=[],
            retrieval_confidence=0.0,
            reranking_used=False,
            model=settings.LLM_MODEL_NAME,
            latency_ms=AskLatencyBreakdown(
                embedding=0.0,
                retrieval=0.0,
                reranking=0.0,
                context_selection=0.0,
                prompt_construction=0.0,
                generation=0.0,
                total=round(elapsed_total_ms, 2),
            ),
            retrieved_context_summary=RetrievedContextSummary(
                chunks_count=0,
                total_characters=0,
                languages=[],
                chunk_ids=[],
            ),
            error=f"Retrieval failure: {sanitize_output_text(str(exc))}",
        )

    # 2. Pre-Generation Guardrails (Input safety, empty context, low confidence, off-topic detection)
    from app.guardrails.service import get_guardrail_service
    guardrail_service = get_guardrail_service()

    pre_guard = guardrail_service.validate_pre_generation(
        query=query,
        retrieved_context=results,
        retrieval_confidence=decision.confidence_score,
    )

    if not pre_guard.allowed:
        elapsed_total_ms = (time.perf_counter() - start_time) * 1000.0
        fallback_msg = (
            pre_guard.safe_fallback_text
            or "I don't have enough information in the retrieved context to answer that."
        )
        return AskResponse(
            query=query,
            answer=fallback_msg,
            grounded=False,
            confidence=pre_guard.confidence,
            citations=[],
            citation_provenance=[],
            retrieval={
                "confidence": decision.confidence_score,
                "reranking_used": decision.should_rerank,
                "chunks_count": len(results),
                "total_characters": sum(len(r.text) for r in results),
                "languages": sorted(list({r.language for r in results if r.language})),
            },
            guardrail_decision=pre_guard.model_dump(),
            retrieval_confidence=decision.confidence_score,
            reranking_used=decision.should_rerank,
            model=settings.LLM_MODEL_NAME,
            latency_ms=AskLatencyBreakdown(
                embedding=round(ret_latency.embedding, 2),
                retrieval=round(ret_latency.retrieval, 2),
                reranking=round(ret_latency.reranking, 2),
                context_selection=round(ret_latency.context, 2),
                guardrails=round(pre_guard.latency_ms, 2),
                prompt_construction=0.0,
                generation=0.0,
                total=round(elapsed_total_ms, 2),
            ),
            retrieved_context_summary=RetrievedContextSummary(
                chunks_count=len(results),
                total_characters=sum(len(r.text) for r in results),
                languages=sorted(list({r.language for r in results if r.language})),
                chunk_ids=[r.chunk_id for r in results],
            ),
            error=None,
        )

    # 3. Grounded Prompt Construction & Sanitization
    t_prompt_start = time.perf_counter()
    prompt_builder = get_prompt_builder()
    built_prompt = prompt_builder.build(
        query=query,
        retrieved_context=results,
        language=request.language,
    )
    t_prompt_ms = (time.perf_counter() - t_prompt_start) * 1000.0

    # 4. Context Telemetry Summary
    languages = sorted(list({r.language for r in results if r.language}))
    chunk_ids = [r.chunk_id for r in results]
    total_chars = sum(len(r.text) for r in results)
    context_summary = RetrievedContextSummary(
        chunks_count=len(results),
        total_characters=total_chars,
        languages=languages,
        chunk_ids=chunk_ids,
    )

    # 5. LLM Generation
    t_gen_start = time.perf_counter()
    gen_service = get_generation_service()
    gen_result = await gen_service.generate(
        query=query,
        context=results,
        config=request.config,
        language=request.language,
    )
    t_gen_ms = (time.perf_counter() - t_gen_start) * 1000.0

    # 6. Post-Generation Guardrails (Grounding verification, hallucination check, citation validation)
    post_guard = guardrail_service.validate_post_generation(
        query=query,
        answer=gen_result.answer,
        retrieved_context=results,
        citations=gen_result.citations,
        raw_confidence=gen_result.confidence,
    )

    total_guardrail_ms = pre_guard.latency_ms + post_guard.latency_ms
    elapsed_total_ms = (time.perf_counter() - start_time) * 1000.0

    final_answer = gen_result.answer
    final_grounded = post_guard.grounded and gen_result.grounded and (gen_result.error is None)
    if gen_result.error:
        final_answer = gen_result.answer or "I don't have enough information in the retrieved context to answer that."
    elif post_guard.action == "replace_with_fallback" and post_guard.safe_fallback_text:
        final_answer = post_guard.safe_fallback_text

    ans_lower = final_answer.lower()
    refusal_patterns = [
        "don't have enough information",
        "not enough information",
        "not available",
        "not present in the provided context",
        "not contain information",
        "does not contain information",
        "cannot be found in the provided context",
        "cannot be found in the context",
        "no information",
        "not mentioned",
        "not provided",
    ]
    is_refusal = (
        not final_grounded
        or any(pattern in ans_lower for pattern in refusal_patterns)
    )
    if is_refusal:
        final_grounded = False
        final_citations = []
        final_provenance = []
    else:
        final_citations = gen_result.citations if gen_result.citations else [p.chunk_id for p in built_prompt.provenance]
        final_provenance = gen_result.citation_provenance if gen_result.citation_provenance else []

    retrieval_meta = {
        "confidence": decision.confidence_score,
        "reranking_used": decision.should_rerank,
        "chunks_count": len(results),
        "total_characters": total_chars,
        "languages": languages,
        "chunk_ids": chunk_ids,
    }

    from app.orchestration.voice_rag import record_request_trace
    record_request_trace({
        "request_language": request.language or "en",
        "resolved_language": request.language or "en",
        "transcript": query,
        "retrieval_count": len(results),
        "retrieved_languages": languages,
        "reranker_used": decision.should_rerank,
        "retrieval_confidence": decision.confidence_score,
        "grounded": final_grounded,
        "generation_provider": getattr(gen_service.provider, "provider_name", settings.LLM_PROVIDER),
        "tts_provider": "none",
        "latency": {"total": round(elapsed_total_ms, 2)},
    })

    return AskResponse(

        query=query,
        answer=final_answer,
        grounded=final_grounded,
        confidence=post_guard.confidence if not gen_result.error else 0.0,
        citations=final_citations,
        citation_provenance=final_provenance,
        retrieval=retrieval_meta,
        guardrail_decision=post_guard.model_dump(),
        retrieval_confidence=decision.confidence_score,
        reranking_used=decision.should_rerank,
        model=gen_result.model or settings.LLM_MODEL_NAME,
        latency_ms=AskLatencyBreakdown(
            embedding=round(ret_latency.embedding, 2),
            retrieval=round(ret_latency.retrieval, 2),
            reranking=round(ret_latency.reranking, 2),
            context_selection=round(ret_latency.context, 2),
            guardrails=round(total_guardrail_ms, 2),
            prompt_construction=round(t_prompt_ms, 2),
            generation=round(t_gen_ms, 2),
            total=round(elapsed_total_ms, 2),
        ),
        retrieved_context_summary=context_summary,
        language=request.language or "en",
        relevance=getattr(pre_guard, "relevance", None),
        error=gen_result.error,
    )


@router.post(
    "/api/voice-ask",
    response_model=VoiceAskResponse,
    tags=["Voice RAG"],
    summary="Execute end-to-end multilingual Voice RAG pipeline (Audio -> STT -> Adaptive Retrieval -> Grounded LLM -> TTS -> Audio)",
)
async def voice_ask_endpoint(
    audio: UploadFile = File(..., description="Uploaded audio query file (WAV, MP3, OGG, WebM, M4A)."),
    language: Optional[str] = Form(None, description="Optional BCP-47 language code (e.g. 'hi', 'en', 'ta', 'te', 'ml')."),
    top_k: int = Form(5, ge=1, le=20, description="Top-K context chunks to retrieve."),
    synthesize_speech: bool = Form(True, description="Whether to synthesize response speech audio via TTS."),
    speaker: Optional[str] = Form(None, description="Optional TTS speaker voice identifier."),
    pace: Optional[float] = Form(None, description="Optional TTS speech rate pace multiplier."),
) -> VoiceAskResponse:
    """End-to-end production Multilingual Voice RAG endpoint.

    Workflow:
    1. Validate audio upload format, size, and integrity.
    2. Transcribe speech audio to text via configured STT provider (Mock or Sarvam).
    3. Normalize query Unicode text and resolve canonical language.
    4. Execute calibrated Phase 6.5.1 Multilingual Hybrid Retrieval & Adaptive Reranking.
    5. Evaluate Pre-generation guardrails & construct grounded, injection-resistant prompt.
    6. Generate grounded response using interchangeable LLM provider.
    7. Validate grounding and citations against retrieved chunk provenance.
    8. Synthesize speech audio using configured TTS provider.
    9. Return structured JSON response with audio payload and monotonic latency telemetry.
    """
    if audio is None:
        raise HTTPException(
            status_code=422,
            detail="Audio file payload is required.",
        )

    validate_request_security_limits(query=None, language=language)

    try:
        audio_bytes = await audio.read()
    except Exception as read_ex:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to read uploaded audio stream: {sanitize_output_text(str(read_ex))}",
        )

    orchestrator = get_voice_rag_orchestrator()
    req_id = get_current_request_id()

    try:
        return await orchestrator.execute_voice_rag(
            audio_bytes=audio_bytes,
            language=language,
            content_type=audio.content_type,
            filename=audio.filename,
            top_k=top_k,
            request_id=req_id,
            synthesize_speech=synthesize_speech,
            speaker=speaker,
            pace=pace,
        )
    except AudioPayloadTooLargeError as apl:
        raise HTTPException(status_code=413, detail=sanitize_output_text(str(apl.message)))
    except UnsupportedAudioFormatError as uaf:
        raise HTTPException(status_code=415, detail=sanitize_output_text(str(uaf.message)))
    except AudioValidationError as ave:
        raise HTTPException(status_code=422, detail=sanitize_output_text(str(ave.message)))
    except STTError as se:
        raise HTTPException(status_code=502, detail=f"Speech-to-Text Error: {sanitize_output_text(se.message)}")
    except TTSError as te:
        raise HTTPException(status_code=502, detail=f"Text-to-Speech Error: {sanitize_output_text(te.message)}")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unhandled error in /api/voice-ask: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Voice RAG execution failed: {sanitize_output_text(str(exc))}",
        )


