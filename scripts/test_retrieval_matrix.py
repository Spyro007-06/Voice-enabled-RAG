"""Execute 25-query factual retrieval test matrix across English, Hindi, Tamil, Telugu, and Malayalam."""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retrieval.service import get_retrieval_service
from app.retrieval.language import get_language_filter_synonyms, normalize_language_code

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s")
logger = logging.getLogger("test_retrieval_matrix")

TEST_MATRIX = {
    "English": {
        "lang_code": "en",
        "canonical": "eng_Latn",
        "queries": [
            "What is a computer?",
            "What is machine learning?",
            "What is artificial intelligence?",
            "What is the internet?",
            "What is a database?",
        ],
    },
    "Hindi": {
        "lang_code": "hi",
        "canonical": "hin_Deva",
        "queries": [
            "कंप्यूटर क्या है?",
            "मशीन लर्निंग क्या है?",
            "कृत्रिम बुद्धिमत्ता क्या है?",
            "इंटरनेट क्या है?",
            "डेटाबेस क्या है?",
        ],
    },
    "Tamil": {
        "lang_code": "ta",
        "canonical": "tam_Taml",
        "queries": [
            "கணினி என்றால் என்ன?",
            "இயந்திர கற்றல் என்றால் என்ன?",
            "செயற்கை நுண்ணறிவு என்றால் என்ன?",
            "இணையம் என்றால் என்ன?",
            "தரவுத்தளம் என்றால் என்ன?",
        ],
    },
    "Telugu": {
        "lang_code": "te",
        "canonical": "tel_Telu",
        "queries": [
            "కంప్యూటర్ అంటే ఏమిటి?",
            "మెషిన్ లెర్నింగ్ అంటే ఏమిటి?",
            "కృత్రిమ మేధస్సు అంటే ఏమిటి?",
            "ఇంటర్నెట్ అంటే ఏమిటి?",
            "డేటాబేస్ అంటే ఏమిటి?",
        ],
    },
    "Malayalam": {
        "lang_code": "ml",
        "canonical": "mal_Mlym",
        "queries": [
            "കമ്പ്യൂട്ടർ എന്താണ്?",
            "മെഷീൻ ലേണിംഗ് എന്താണ്?",
            "കൃത്രിമ ബുദ്ധി എന്താണ്?",
            "ഇന്റർനെറ്റ് എന്താണ്?",
            "ഡാറ്റാബേസ് എന്താണ്?",
        ],
    },
}


def run_matrix():
    service = get_retrieval_service()
    
    # Warmup retrieval pipeline
    print("Warming up hybrid retrieval service...")
    try:
        service.retrieve("warmup query", top_k=2, language=None, use_cache=False)
    except Exception as e:
        logger.warning("Warmup notice: %s", e)

    report_rows = []
    print("=" * 80)
    print("PHASE 6.19 — 25-QUERY MULTILINGUAL RETRIEVAL TEST MATRIX")
    print("=" * 80)

    total_queries = 0
    passed_queries = 0
    latencies = []

    for lang_name, cfg in TEST_MATRIX.items():
        lang_code = cfg["lang_code"]
        canonical = cfg["canonical"]
        synonyms = [s.lower() for s in get_language_filter_synonyms(lang_code)]
        print(f"\n--- Testing {lang_name} (Filter: {lang_code} / {canonical}) ---")

        for q in cfg["queries"]:
            total_queries += 1
            t0 = time.perf_counter()
            results, lat_breakdown = service.retrieve(
                query=q,
                top_k=5,
                language=lang_code,
                use_cache=False,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            num_results = len(results)
            valid_lang_count = 0
            for r in results:
                r_lang = (r.language or r.metadata.get("language") or "").strip().lower()
                if r_lang in synonyms or any(r_lang.startswith(s) for s in synonyms):
                    valid_lang_count += 1

            is_pass = num_results > 0 and valid_lang_count == num_results
            if is_pass:
                passed_queries += 1

            status_str = "PASS" if is_pass else "WARN"
            top_score = results[0].score if results else 0.0
            top_snippet = results[0].text[:70].replace("\n", " ") if results else "No matches"

            print(f"  [{status_str}] Query: '{q}'")
            print(f"         Hits: {num_results} | Valid Lang: {valid_lang_count}/{num_results} | Latency: {lat_ms:.1f}ms | Top Score: {top_score:.3f}")
            print(f"         Top Snippet: {top_snippet}...")

            report_rows.append({
                "language": lang_name,
                "lang_code": lang_code,
                "canonical": canonical,
                "query": q,
                "hits": num_results,
                "valid_language_hits": valid_lang_count,
                "latency_ms": lat_ms,
                "top_score": top_score,
                "passed": is_pass,
            })

    # Cross-lingual Test
    print("\n--- Cross-Lingual Retrieval (language=None) ---")
    cross_queries = [
        ("What is machine learning?", None),
        ("கணினி என்றால் என்ன?", None),
    ]
    for q, l in cross_queries:
        t0 = time.perf_counter()
        results, _ = service.retrieve(query=q, top_k=5, language=l, use_cache=False)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  [CROSS-LINGUAL] Query: '{q}' -> Hits: {len(results)} | Latency: {lat_ms:.1f}ms")
        for idx, r in enumerate(results[:3]):
            print(f"    #{idx+1} [Lang: {r.language or r.metadata.get('language')}] Score: {r.score:.3f} | {r.text[:60]}...")

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0

    print("\n" + "=" * 80)
    print(f"TEST MATRIX SUMMARY: {passed_queries} / {total_queries} queries retrieved matching language chunks.")
    print(f"Latency P50: {sorted(latencies)[len(latencies)//2]:.1f}ms | P95: {p95_lat:.1f}ms | Avg: {avg_lat:.1f}ms")
    print("=" * 80)

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/phase619_multilingual_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_queries": total_queries,
            "passed_queries": passed_queries,
            "latency_p50_ms": sorted(latencies)[len(latencies)//2],
            "latency_p95_ms": p95_lat,
            "latency_avg_ms": avg_lat,
            "results": report_rows,
        }, f, indent=2, ensure_ascii=False)
    print("Saved results to benchmarks/phase619_multilingual_results.json")


if __name__ == "__main__":
    run_matrix()
