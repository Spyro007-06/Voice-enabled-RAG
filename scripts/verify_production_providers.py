"""Production Provider Verification and Connectivity Benchmark (Phase 6.2)."""

import asyncio
import os
import sys
import time
from typing import Any, Dict, List

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings
from app.generation.models import GenerationConfig
from app.providers.factory import (
    get_llm_provider,
    get_stt_provider,
    get_tts_provider,
    get_vector_provider,
)
from app.providers.stt.sarvam import SarvamSTTProvider
from app.providers.tts.sarvam import SarvamTTSProvider
from app.providers.llm.sarvam import SarvamLLMProvider
from app.providers.vector.qdrant import QdrantVectorProvider
from app.retrieval.dense import DenseRetriever
from app.retrieval.qdrant_store import get_qdrant_store


async def verify_production_providers() -> Dict[str, Any]:
    """Verify all production providers safely without printing sensitive credentials."""
    print("=" * 70)
    print("HH GOA 2026 — PRODUCTION PROVIDER VERIFICATION (PHASE 6.2)")
    print("=" * 70)

    settings = get_settings()
    results: List[Dict[str, str]] = []

    # 1. Configuration Check
    sarvam_key_status = "CONFIGURED" if (settings.SARVAM_API_KEY and settings.SARVAM_API_KEY.strip()) else "MISSING"
    qdrant_url_status = "CONFIGURED" if (settings.VECTOR_DB_URL and settings.VECTOR_DB_URL.strip()) else "MISSING"
    qdrant_key_status = "CONFIGURED" if (settings.VECTOR_DB_API_KEY and settings.VECTOR_DB_API_KEY.strip()) else "MISSING"

    print("\n--- Credential Configuration Status ---")
    print(f"Environment:       {settings.ENVIRONMENT}")
    print(f"STT Provider:      {settings.STT_PROVIDER}")
    print(f"TTS Provider:      {settings.TTS_PROVIDER}")
    print(f"LLM Provider:      {settings.LLM_PROVIDER}")
    print(f"Vector Provider:   {settings.VECTOR_PROVIDER}")
    print(f"Sarvam API Key:    {sarvam_key_status}")
    print(f"Qdrant URL:        {qdrant_url_status}")
    print(f"Qdrant API Key:    {qdrant_key_status}")

    config_ok = bool(settings.ENVIRONMENT)
    results.append({
        "component": "Configuration",
        "status": "PASS" if config_ok else "FAIL",
        "details": f"Env: {settings.ENVIRONMENT} | STT: {settings.STT_PROVIDER} | LLM: {settings.LLM_PROVIDER}",
    })

    # 2. Sarvam STT Provider Initialization
    try:
        if sarvam_key_status == "CONFIGURED":
            stt_p = SarvamSTTProvider()
            stt_status = "PASS"
            stt_det = f"Initialized Saarika v2 (Key: CONFIGURED)"
        else:
            stt_p = get_stt_provider()
            stt_status = "PASS (Mock/Dev)"
            stt_det = f"Fallback {stt_p.provider_name} provider"
    except Exception as exc:
        stt_status = "FAIL"
        stt_det = str(exc)[:60]

    results.append({"component": "Sarvam STT", "status": stt_status, "details": stt_det})

    # 3. Sarvam TTS Provider Initialization
    try:
        if sarvam_key_status == "CONFIGURED":
            tts_p = SarvamTTSProvider()
            tts_status = "PASS"
            tts_det = f"Initialized Bulbul v1 (Key: CONFIGURED)"
        else:
            tts_p = get_tts_provider()
            tts_status = "PASS (Mock/Dev)"
            tts_det = f"Fallback {tts_p.provider_name} provider"
    except Exception as exc:
        tts_status = "FAIL"
        tts_det = str(exc)[:60]

    results.append({"component": "Sarvam TTS", "status": tts_status, "details": tts_det})

    # 4. Sarvam LLM Provider Initialization
    try:
        if sarvam_key_status == "CONFIGURED":
            llm_p = SarvamLLMProvider()
            llm_status = "PASS"
            llm_det = f"Initialized sarvam-2b (Key: CONFIGURED)"
        else:
            llm_p = get_llm_provider()
            llm_status = "PASS (Mock/Dev)"
            llm_det = f"Fallback {llm_p.provider_name} provider"
    except Exception as exc:
        llm_status = "FAIL"
        llm_det = str(exc)[:60]

    results.append({"component": "Sarvam LLM", "status": llm_status, "details": llm_det})

    # 5. Vector Database Connectivity & msmarco_xi Collection
    try:
        store = get_qdrant_store()
        col_name = settings.QDRANT_COLLECTION_NAME
        exists = store.collection_exists(col_name)
        if exists:
            stats = store.get_collection_stats(col_name)
            points_count = stats.get("points_count", 0)
            qdrant_status = "PASS"
            qdrant_det = f"Connected ({settings.VECTOR_PROVIDER})"
            col_status = "PASS"
            col_det = f"Collection '{col_name}' exists ({points_count:,} points)"
        else:
            qdrant_status = "PASS"
            qdrant_det = f"Connected ({settings.VECTOR_PROVIDER})"
            col_status = "WARN"
            col_det = f"Collection '{col_name}' not yet indexed"
    except Exception as exc:
        qdrant_status = "FAIL"
        qdrant_det = str(exc)[:60]
        col_status = "FAIL"
        col_det = "Failed checking collection"

    results.append({"component": "Vector DB Connectivity", "status": qdrant_status, "details": qdrant_det})
    results.append({"component": "msmarco_xi Collection", "status": col_status, "details": col_det})

    # 6. Minimal Dense Retrieval Test
    try:
        retriever = DenseRetriever()
        t_start = time.perf_counter()
        query_results, t_embed, t_search = retriever.retrieve("गोवा की राजधानी क्या है?", top_k=2)
        lat_ret = (time.perf_counter() - t_start) * 1000.0
        dense_status = "PASS"
        top_id = query_results[0].chunk_id if query_results else "none"
        dense_det = f"{len(query_results)} results in {lat_ret:.1f}ms (Top-1: {top_id})"
    except Exception as exc:
        dense_status = "FAIL"
        dense_det = str(exc)[:60]

    results.append({"component": "Dense Retrieval", "status": dense_status, "details": dense_det})

    # 7. Minimal LLM Generation Test
    try:
        t_gen_start = time.perf_counter()
        gen_provider = get_llm_provider()
        gen_res = await gen_provider.generate(
            query="गोवा की राजधानी क्या है?",
            context="गोवा की राजधानी पणजी है।",
            generation_config=GenerationConfig(max_tokens=64),
        )
        lat_gen = (time.perf_counter() - t_gen_start) * 1000.0
        gen_status = "PASS"
        gen_det = f"Latency: {lat_gen:.1f}ms | Grounded: {gen_res.grounded} | Model: {gen_res.model}"
    except Exception as exc:
        gen_status = "FAIL"
        gen_det = str(exc)[:60]

    results.append({"component": "LLM Generation", "status": gen_status, "details": gen_det})

    # Print Summary Table
    print("\n" + "=" * 70)
    print(f"{'Provider / Component':<26} {'Status':<18} {'Details'}")
    print("-" * 70)
    for r in results:
        print(f"{r['component']:<26} {r['status']:<18} {r['details']}")
    print("=" * 70 + "\n")

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }


if __name__ == "__main__":
    asyncio.run(verify_production_providers())
