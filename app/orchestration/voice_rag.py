"""Voice RAG Orchestrator coordinating STT, Adaptive Retrieval, Grounded LLM, and TTS."""

import asyncio
import base64
import logging
import time
import unicodedata
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from app.config import get_settings
from app.generation.models import GenerationConfig
from app.generation.prompt import PromptBuilder, get_prompt_builder
from app.generation.service import GenerationService, get_generation_service
from app.guardrails.service import GuardrailService, get_guardrail_service
from app.orchestration.exceptions import (
    AudioPayloadTooLargeError,
    AudioValidationError,
    STTError,
    TTSError,
    UnsupportedAudioFormatError,
    VoiceOrchestrationError,
)
from app.orchestration.models import (
    AudioOutputMetadata,
    VoiceAskResponse,
    VoiceLatencyBreakdown,
)
from app.providers.stt.base import STTProvider
from app.providers.tts.base import TTSProvider
from app.speech.validation import sanitize_filename, validate_audio_security
from app.reranking.adaptive import AdaptiveRetrievalService, get_adaptive_retrieval_service
from app.reranking.context_selector import ContextSelector
from app.reranking.models import RerankResult

from app.retrieval.language import detect_script_language, to_bcp47

logger = logging.getLogger(__name__)

# Module-level trace store for GET /api/debug/last-request
_LAST_REQUEST_TRACE: Dict[str, Any] = {}


def get_last_request_trace() -> Dict[str, Any]:
    """Return sanitized debug trace of the most recent request."""
    return dict(_LAST_REQUEST_TRACE)


def record_request_trace(trace_data: Dict[str, Any]) -> None:
    """Record a sanitized trace for development debugging."""
    global _LAST_REQUEST_TRACE
    _LAST_REQUEST_TRACE = dict(trace_data)


# Canonical language normalization dictionary
_LANGUAGE_MAP: Dict[str, str] = {
    "en": "en",
    "eng": "en",
    "en-in": "en",
    "en-us": "en",
    "english": "en",
    "eng_latn": "en",
    "hi": "hi",
    "hin": "hi",
    "hi-in": "hi",
    "hindi": "hi",
    "hin_deva": "hi",
    "ta": "ta",
    "tam": "ta",
    "ta-in": "ta",
    "tamil": "ta",
    "tam_taml": "ta",
    "te": "te",
    "tel": "te",
    "te-in": "te",
    "telugu": "te",
    "tel_telu": "te",
    "ml": "ml",
    "mal": "ml",
    "ml-in": "ml",
    "malayalam": "ml",
    "mal_mlym": "ml",
}



def normalize_language_code(lang: Optional[str], default: str = "en") -> str:
    """Normalize input language strings (BCP-47, FLORES-200, names) into canonical 2-letter codes."""
    if not lang or not isinstance(lang, str):
        return default
    clean = lang.strip().lower().replace("_", "-")
    return _LANGUAGE_MAP.get(clean, _LANGUAGE_MAP.get(clean.split("-")[0], default))


def normalize_query_text(text: str) -> str:
    """Safely normalize query text, preserving Indic Unicode scripts and stripping control characters."""
    if not text:
        return ""
    # NFC Unicode normalization preserves Indic character combining marks
    normalized = unicodedata.normalize("NFC", text)
    # Remove non-printable control characters except standard whitespace
    cleaned = "".join(ch for ch in normalized if ch == " " or not unicodedata.category(ch).startswith("C"))
    return cleaned.strip()


class VoiceRAGOrchestrator:
    """Production orchestrator executing end-to-end Multilingual Voice RAG pipeline."""

    def __init__(
        self,
        stt_provider: Optional[STTProvider] = None,
        tts_provider: Optional[TTSProvider] = None,
        adaptive_retrieval_service: Optional[AdaptiveRetrievalService] = None,
        generation_service: Optional[GenerationService] = None,
        guardrail_service: Optional[GuardrailService] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        context_selector: Optional[ContextSelector] = None,
    ):
        """Initialize VoiceRAGOrchestrator with lazy dependency resolution."""
        self._stt_provider = stt_provider
        self._tts_provider = tts_provider
        self._adaptive_service = adaptive_retrieval_service
        self._generation_service = generation_service
        self._guardrail_service = guardrail_service
        self._prompt_builder = prompt_builder
        self._context_selector = context_selector

    @property
    def stt_provider(self) -> STTProvider:
        if self._stt_provider is not None:
            return self._stt_provider
        from app.providers.factory import get_stt_provider
        return get_stt_provider()

    @property
    def tts_provider(self) -> TTSProvider:
        if self._tts_provider is not None:
            return self._tts_provider
        from app.providers.factory import get_tts_provider
        return get_tts_provider()

    @property
    def adaptive_service(self) -> AdaptiveRetrievalService:
        if self._adaptive_service is None:
            self._adaptive_service = get_adaptive_retrieval_service()
        return self._adaptive_service

    @property
    def generation_service(self) -> GenerationService:
        if self._generation_service is None:
            self._generation_service = get_generation_service()
        return self._generation_service

    @property
    def guardrail_service(self) -> GuardrailService:
        if self._guardrail_service is None:
            self._guardrail_service = get_guardrail_service()
        return self._guardrail_service

    @property
    def prompt_builder(self) -> PromptBuilder:
        if self._prompt_builder is None:
            self._prompt_builder = get_prompt_builder()
        return self._prompt_builder

    @property
    def context_selector(self) -> ContextSelector:
        if self._context_selector is None:
            self._context_selector = ContextSelector()
        return self._context_selector

    def validate_audio_payload(
        self,
        audio_bytes: bytes,
        content_type: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Validate input audio payload size, emptiness, and content format security."""
        return validate_audio_security(
            audio_bytes=audio_bytes,
            content_type=content_type,
            filename=filename,
        )

    async def execute_voice_rag(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        content_type: Optional[str] = None,
        filename: Optional[str] = None,
        top_k: int = 5,
        strategies: Optional[Union[str, List[str]]] = None,
        request_id: Optional[str] = None,
        synthesize_speech: bool = True,
        speaker: Optional[str] = None,
        pace: Optional[float] = None,
    ) -> VoiceAskResponse:
        """Execute complete end-to-end Voice RAG pipeline with stage timing and graceful fallbacks."""
        t_pipeline_start = time.perf_counter()
        req_id = request_id or str(uuid.uuid4())
        settings = get_settings()

        # Telemetry accumulator
        t_stt_ms = 0.0
        t_norm_ms = 0.0
        t_embed_ms = 0.0
        t_qdrant_ms = 0.0
        t_bm25_ms = 0.0
        t_fusion_ms = 0.0
        t_rerank_ms = 0.0
        t_context_ms = 0.0
        t_guardrails_ms = 0.0
        t_prompt_ms = 0.0
        t_llm_ms = 0.0
        t_grounding_ms = 0.0
        t_tts_ms = 0.0

        stage_error: Optional[str] = None
        target_lang = normalize_language_code(language, default="en") if language else "en"
        detected_lang: Optional[str] = None

        # ---------------------------------------------------------------------
        # 1. Audio Validation
        # ---------------------------------------------------------------------
        t_val_start = time.perf_counter()
        self.validate_audio_payload(audio_bytes=audio_bytes, content_type=content_type, filename=filename)
        t_val_ms = (time.perf_counter() - t_val_start) * 1000.0

        # ---------------------------------------------------------------------
        # 2. Speech-to-Text Transcription
        # ---------------------------------------------------------------------
        t_stt_start = time.perf_counter()
        stt_timeout_s = getattr(settings, "STT_TIMEOUT_MS", 5000.0) / 1000.0
        transcript = ""
        stt_confidence = 1.0

        try:
            stt_coro = self.stt_provider.transcribe(
                audio_bytes=audio_bytes,
                language=language,
            )
            stt_res = await asyncio.wait_for(stt_coro, timeout=stt_timeout_s)
            transcript = stt_res.text
            detected_lang = stt_res.language
            stt_confidence = stt_res.confidence
        except asyncio.TimeoutError:
            logger.warning("STT transcription timed out after %.2fs", stt_timeout_s)
            raise STTError("STT transcription timed out.", provider=getattr(self.stt_provider, "provider_name", "stt"))
        except Exception as ex:
            logger.error("STT transcription error: %s", ex, exc_info=True)
            raise STTError(str(ex), provider=getattr(self.stt_provider, "provider_name", "stt"))
        finally:
            t_stt_ms = (time.perf_counter() - t_stt_start) * 1000.0

        # ---------------------------------------------------------------------
        # 3. Query Normalization & Language Resolution
        # ---------------------------------------------------------------------
        t_norm_start = time.perf_counter()
        normalized_query = normalize_query_text(transcript)
        if not normalized_query:
            raise AudioValidationError("Transcribed audio yielded an empty query transcript.")

        # Script-aware canonical language resolution
        script_lang = detect_script_language(normalized_query)
        if script_lang and script_lang != "en":
            # If the user spoke in an Indic language (e.g. Hindi, Tamil, Telugu, Malayalam), use detected script
            target_lang = script_lang
        elif language:
            target_lang = normalize_language_code(language, default="en")
        elif detected_lang:
            target_lang = normalize_language_code(detected_lang, default="en")
        else:
            target_lang = "en"
        t_norm_ms = (time.perf_counter() - t_norm_start) * 1000.0


        # ---------------------------------------------------------------------
        # 4. Multilingual Hybrid Retrieval & Adaptive Reranking
        # ---------------------------------------------------------------------
        retrieved_results: List[RerankResult] = []
        is_cache_hit = False
        timeout_stage: Optional[str] = None
        fallback_used = False
        try:
            retrieved_results, decision, ret_lat = self.adaptive_service.adaptive_retrieve(
                query=normalized_query,
                top_k=top_k,
                strategies=strategies,
                language=target_lang,
            )
            t_embed_ms = ret_lat.embedding
            t_qdrant_ms = ret_lat.qdrant
            t_bm25_ms = ret_lat.bm25
            t_fusion_ms = ret_lat.fusion
            t_rerank_ms = ret_lat.reranking
            t_context_ms = ret_lat.context
            is_cache_hit = ret_lat.cache_hit
            timeout_stage = ret_lat.timeout_stage
            fallback_used = ret_lat.fallback_used
        except Exception as rex:
            logger.error("Adaptive retrieval failure in Voice RAG: %s", rex, exc_info=True)
            retrieved_results = []
            stage_error = f"Retrieval degraded: {str(rex)}"
            fallback_used = True
            from app.reranking.adaptive import ConfidenceSignals, AdaptiveDecision, TierString
            decision = AdaptiveDecision(
                confidence_score=0.0,
                dense_confidence=0.0,
                retriever_agreement=0.0,
                score_margin=0.0,
                should_rerank=False,
                reranker_tier=TierString("high"),
                candidate_k=0,
                reranker="none",
                reranking_used=False,
                reason="Retrieval execution error.",
                signals=ConfidenceSignals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                routing_tier="high",
            )

        # ---------------------------------------------------------------------
        # 5. Pre-Generation Guardrails Check
        # ---------------------------------------------------------------------
        t_guard_pre_start = time.perf_counter()
        pre_guard = self.guardrail_service.validate_pre_generation(
            query=normalized_query,
            retrieved_context=retrieved_results,
            retrieval_confidence=decision.confidence_score,
        )
        t_guard_pre_ms = (time.perf_counter() - t_guard_pre_start) * 1000.0
        t_guardrails_ms = t_guard_pre_ms

        if not pre_guard.allowed:
            fallback_text = (
                pre_guard.safe_fallback_text
                or "I don't have enough information in the retrieved context to answer that."
            )
            # Synthesize fallback speech if requested
            audio_meta = AudioOutputMetadata(available=False, format="wav")
            if synthesize_speech:
                t_tts_start = time.perf_counter()
                try:
                    tts_res = await self.tts_provider.synthesize(
                        text=fallback_text, language=target_lang, speaker=speaker, pace=pace
                    )
                    audio_meta = AudioOutputMetadata(
                        available=True,
                        format=tts_res.audio_format,
                        sample_rate=tts_res.sample_rate,
                        duration_s=tts_res.duration_s,
                        provider=tts_res.provider,
                        audio_base64=base64.b64encode(tts_res.audio_bytes).decode("ascii"),
                    )
                except Exception as tex:
                    logger.warning("TTS failed on guardrail fallback (%s); returning text only.", tex)
                    audio_meta = AudioOutputMetadata(available=False, format="wav")
                finally:
                    t_tts_ms = (time.perf_counter() - t_tts_start) * 1000.0

            t_total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
            t_retrieval_total = t_embed_ms + t_qdrant_ms + t_bm25_ms + t_fusion_ms + t_rerank_ms + t_context_ms
            t_gen_total = t_guardrails_ms
            t_voice_total = t_stt_ms + t_tts_ms

            latency_obj = VoiceLatencyBreakdown(
                audio_validation=round(t_val_ms, 2),
                stt=round(t_stt_ms, 2),
                query_normalization=round(t_norm_ms, 2),
                embedding=round(t_embed_ms, 2),
                qdrant=round(t_qdrant_ms, 2),
                bm25=round(t_bm25_ms, 2),
                fusion=round(t_fusion_ms, 2),
                reranking=round(t_rerank_ms, 2),
                context=round(t_context_ms, 2),
                guardrails=round(t_guardrails_ms, 2),
                guardrails_pre=round(t_guard_pre_ms, 2),
                prompt=0.0,
                llm=0.0,
                grounding=0.0,
                guardrails_post=0.0,
                tts=round(t_tts_ms, 2),
                serialization=0.0,
                total=round(t_total_ms, 2),
                retrieval_total_ms=round(t_retrieval_total, 2),
                generation_total_ms=round(t_gen_total, 2),
                voice_total_ms=round(t_voice_total, 2),
                end_to_end_total_ms=round(t_total_ms, 2),
                cache_hit=is_cache_hit,
                execution_type="warm" if not is_cache_hit else "cache_hit",
                timeout_stage=timeout_stage,
                fallback_used=fallback_used or not pre_guard.allowed,
            )

            record_request_trace({
                "request_language": language or "en",
                "resolved_language": target_lang,
                "transcript": transcript,
                "retrieval_count": len(retrieved_results),
                "retrieved_languages": list({r.language for r in retrieved_results if r.language}),
                "reranker_used": decision.should_rerank,
                "retrieval_confidence": decision.confidence_score,
                "grounded": False,
                "generation_provider": getattr(self.generation_service.provider, "provider_name", settings.LLM_PROVIDER),
                "tts_provider": getattr(self.tts_provider, "provider_name", settings.TTS_PROVIDER),
                "latency": latency_obj.model_dump(),
            })

            return VoiceAskResponse(
                status="partial_success" if not stage_error else "error",
                request_id=req_id,
                transcript=transcript,
                detected_language=detected_lang,
                language=target_lang,
                answer=fallback_text,
                grounded=False,
                confidence=pre_guard.confidence,
                citations=[],
                citation_provenance=[],
                retrieval_confidence=decision.confidence_score,
                reranking_used=decision.should_rerank,
                reranker_tier=decision.reranker_tier,
                guardrail_decision="refuse" if pre_guard.action == "refuse" else "block",
                fallback_status=True,
                audio=audio_meta,
                latency_ms=latency_obj,
                error=stage_error or pre_guard.reason,
            )


        # ---------------------------------------------------------------------
        # 6. Context Selection & Grounded Prompt Construction
        # ---------------------------------------------------------------------
        t_prompt_start = time.perf_counter()
        selected_context, _, t_sel = self.context_selector.select_context(
            reranked_candidates=retrieved_results,
            top_k=top_k,
        )
        built_prompt = self.prompt_builder.build(
            query=normalized_query,
            retrieved_context=selected_context,
            language=target_lang,
        )
        t_prompt_ms = (time.perf_counter() - t_prompt_start) * 1000.0

        # ---------------------------------------------------------------------
        # 7. LLM Response Generation
        # ---------------------------------------------------------------------
        t_llm_start = time.perf_counter()
        llm_timeout_s = getattr(settings, "LLM_TIMEOUT", 30.0)
        gen_answer = ""
        gen_grounded = False
        gen_citations: List[str] = []
        gen_provenance: List[Dict[str, Any]] = []
        gen_confidence = 0.0

        try:
            gen_res = await asyncio.wait_for(
                self.generation_service.generate(
                    query=normalized_query,
                    context=selected_context,
                    language=target_lang,
                    config=GenerationConfig(max_tokens=192),  # reduced 256→192 for lower tail latency
                ),
                timeout=llm_timeout_s,
            )
            gen_answer = gen_res.answer
            gen_grounded = gen_res.grounded
            gen_citations = gen_res.citations or [p.chunk_id for p in built_prompt.provenance]
            gen_provenance = gen_res.citation_provenance or []
            gen_confidence = gen_res.confidence
        except asyncio.TimeoutError:
            logger.warning("LLM generation timed out after %.2fs", llm_timeout_s)
            gen_answer = "I don't have enough information in the retrieved context to answer that."
            stage_error = "LLM generation timed out; safe fallback provided."
            fallback_used = True
        except Exception as lex:
            logger.error("LLM generation error: %s", lex, exc_info=True)
            gen_answer = "I don't have enough information in the retrieved context to answer that."
            stage_error = f"LLM generation failed: {str(lex)}"
            fallback_used = True
        finally:
            t_llm_ms = (time.perf_counter() - t_llm_start) * 1000.0

        # ---------------------------------------------------------------------
        # 8. Post-Generation Grounding Validation
        # ---------------------------------------------------------------------
        t_ground_start = time.perf_counter()
        post_guard = self.guardrail_service.validate_post_generation(
            query=normalized_query,
            answer=gen_answer,
            retrieved_context=selected_context,
            citations=gen_citations,
            raw_confidence=gen_confidence,
        )
        t_ground_post_ms = (time.perf_counter() - t_ground_start) * 1000.0
        t_grounding_ms = t_ground_post_ms
        t_guardrails_ms += t_ground_post_ms

        final_answer = gen_answer
        final_grounded = gen_grounded and post_guard.grounded and not stage_error

        if post_guard.action == "replace_with_fallback" and post_guard.safe_fallback_text:
            final_answer = post_guard.safe_fallback_text
            final_grounded = False

        # ---------------------------------------------------------------------
        # 9. Text-to-Speech Audio Synthesis
        # ---------------------------------------------------------------------
        audio_output = AudioOutputMetadata(available=False, format="wav")
        if synthesize_speech and final_answer:
            t_tts_start = time.perf_counter()
            tts_timeout_s = getattr(settings, "TTS_TIMEOUT_MS", 5000.0) / 1000.0
            try:
                tts_coro = self.tts_provider.synthesize(
                    text=final_answer,
                    language=target_lang,
                    speaker=speaker,
                    pace=pace,
                )
                # asyncio.shield() prevents TTS synthesis from being aborted if the
                # outer request is cancelled mid-flight (e.g. client disconnect)
                tts_res = await asyncio.wait_for(
                    asyncio.shield(tts_coro), timeout=tts_timeout_s
                )
                audio_output = AudioOutputMetadata(
                    available=True,
                    format=tts_res.audio_format,
                    sample_rate=tts_res.sample_rate,
                    duration_s=tts_res.duration_s,
                    provider=tts_res.provider,
                    audio_base64=base64.b64encode(tts_res.audio_bytes).decode("ascii"),
                )
            except asyncio.TimeoutError:
                logger.warning("TTS synthesis timed out after %.2fs; returning text only.", tts_timeout_s)
                audio_output = AudioOutputMetadata(available=False, format="wav")
                if not stage_error:
                    stage_error = "TTS audio synthesis timed out."
                fallback_used = True
            except Exception as tex:
                logger.warning("TTS synthesis failed (%s); returning text only.", tex)
                audio_output = AudioOutputMetadata(available=False, format="wav")
                if not stage_error:
                    stage_error = f"TTS synthesis failed: {str(tex)}"
                fallback_used = True
            finally:
                t_tts_ms = (time.perf_counter() - t_tts_start) * 1000.0

        # ---------------------------------------------------------------------
        # 10. Monotonic Latency Telemetry Assembly & Observability
        # ---------------------------------------------------------------------
        t_total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
        t_retrieval_total = t_embed_ms + t_qdrant_ms + t_bm25_ms + t_fusion_ms + t_rerank_ms + t_context_ms
        t_gen_total = t_prompt_ms + t_llm_ms + t_grounding_ms + t_guardrails_ms
        t_voice_total = t_stt_ms + t_tts_ms

        # Record metrics in centralized registry
        try:
            from app.observability.metrics import get_metrics_registry
            m = get_metrics_registry()
            m.record_voice("stt", target_lang, t_stt_ms / 1000.0)
            m.record_retrieval(
                language=target_lang,
                tier=str(decision.reranker_tier),
                cache_hit=is_cache_hit,
                duration_s=t_retrieval_total / 1000.0,
                dense_s=t_embed_ms / 1000.0,
                bm25_s=t_bm25_ms / 1000.0,
                rerank_s=t_rerank_ms / 1000.0,
                confidence=decision.confidence_score,
            )
            if not decision.should_rerank:
                m.record_rerank_skip(target_lang)
            m.record_generation(target_lang, t_llm_ms / 1000.0, "success" if not stage_error else "fallback")
            if synthesize_speech and final_answer:
                m.record_voice("tts", target_lang, t_tts_ms / 1000.0)
            if fallback_used:
                m.record_fallback("voice_pipeline", "partial_success_fallback")
            if post_guard.action == "replace_with_fallback":
                m.record_unsupported_answer(target_lang)
            m.record_end_to_end(t_total_ms / 1000.0, target_lang, "success" if not stage_error else "partial_success")
        except Exception as mex:
            logger.debug("Metrics recording telemetry warning: %s", mex)

        latency_breakdown = VoiceLatencyBreakdown(
            audio_validation=round(t_val_ms, 2),
            stt=round(t_stt_ms, 2),
            query_normalization=round(t_norm_ms, 2),
            embedding=round(t_embed_ms, 2),
            qdrant=round(t_qdrant_ms, 2),
            bm25=round(t_bm25_ms, 2),
            fusion=round(t_fusion_ms, 2),
            reranking=round(t_rerank_ms, 2),
            context=round(t_context_ms, 2),
            guardrails=round(t_guardrails_ms, 2),
            guardrails_pre=round(t_guard_pre_ms, 2),
            prompt=round(t_prompt_ms, 2),
            llm=round(t_llm_ms, 2),
            grounding=round(t_grounding_ms, 2),
            guardrails_post=round(t_ground_post_ms, 2),
            tts=round(t_tts_ms, 2),
            serialization=0.0,
            total=round(t_total_ms, 2),
            retrieval_total_ms=round(t_retrieval_total, 2),
            generation_total_ms=round(t_gen_total, 2),
            voice_total_ms=round(t_voice_total, 2),
            end_to_end_total_ms=round(t_total_ms, 2),
            cache_hit=is_cache_hit,
            execution_type="warm" if not is_cache_hit else "cache_hit",
            timeout_stage=timeout_stage,
            fallback_used=fallback_used,
        )

        final_status = "success" if not stage_error else "partial_success"

        record_request_trace({
            "request_language": language or "en",
            "resolved_language": target_lang,
            "transcript": transcript,
            "retrieval_count": len(retrieved_results),
            "retrieved_languages": list({r.language for r in retrieved_results if r.language}),
            "reranker_used": decision.should_rerank,
            "retrieval_confidence": decision.confidence_score,
            "grounded": final_grounded,
            "generation_provider": getattr(self.generation_service.provider, "provider_name", settings.LLM_PROVIDER),
            "tts_provider": getattr(self.tts_provider, "provider_name", settings.TTS_PROVIDER),
            "latency": latency_breakdown.model_dump(),
        })

        return VoiceAskResponse(
            status=final_status,
            request_id=req_id,
            transcript=transcript,
            detected_language=detected_lang,
            language=target_lang,
            answer=final_answer,
            grounded=final_grounded,
            confidence=round(post_guard.confidence, 4) if final_grounded else 0.0,
            citations=gen_citations,
            citation_provenance=[p.model_dump() if hasattr(p, "model_dump") else p for p in gen_provenance],
            retrieval_confidence=round(decision.confidence_score, 4),
            reranking_used=decision.should_rerank,
            reranker_tier=decision.reranker_tier,
            guardrail_decision="allow" if post_guard.action == "allow" else "replace_fallback",
            fallback_status=fallback_used or (final_status == "partial_success"),
            audio=audio_output,
            latency_ms=latency_breakdown,
            error=stage_error,
        )



_ORCHESTRATOR_INSTANCE: Optional[VoiceRAGOrchestrator] = None


def get_voice_rag_orchestrator() -> VoiceRAGOrchestrator:
    """Return singleton cached VoiceRAGOrchestrator instance."""
    global _ORCHESTRATOR_INSTANCE
    if _ORCHESTRATOR_INSTANCE is None:
        _ORCHESTRATOR_INSTANCE = VoiceRAGOrchestrator()
    return _ORCHESTRATOR_INSTANCE
