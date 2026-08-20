"""Test mMiniLM and Fast Hybrid Rescorer."""

import time
import torch
from sentence_transformers import CrossEncoder

pairs = [
    ("भारत की राजधानी क्या है?", "नई दिल्ली भारत की राजधानी है और यह देश का प्रशासनिक केंद्र है।"),
    ("भारत की राजधानी क्या है?", "मुंबई भारत की आर्थिक और वित्तीय राजधानी के रूप में जानी जाती है।"),
    ("भारत की राजधानी क्या है?", "कोलकाता पश्चिम बंगाल राज्य की राजधानी है और सांस्कृतिक केंद्र है।"),
    ("भारत की राजधानी क्या है?", "चेन्नई तमिलनाडु की राजधानी है और दक्षिण भारत का प्रमुख शहर है।"),
    ("भारत की राजधानी क्या है?", "बेंगलुरु कर्नाटक की राजधानी है और भारत का सिलिकॉन वैली कहलाता है।"),
]

print("\n--- 4. Testing Compact Multilingual Cross-Encoder (mmarco-mMiniLMv2-L12-H384-v1) ---")
try:
    minilm = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device="cpu", max_length=128)
    minilm.predict(pairs[:2])
    
    # Measure 5 pairs
    t0 = time.perf_counter()
    scores_minilm = minilm.predict(pairs)
    t_minilm = (time.perf_counter() - t0) * 1000
    print(f"mMiniLM (max_len=128) 5 pairs latency: {t_minilm:.2f} ms ({t_minilm/5:.2f} ms/pair)")
    print(f"mMiniLM scores: {scores_minilm}")

    # Measure 3 pairs (K=3)
    t0 = time.perf_counter()
    _ = minilm.predict(pairs[:3])
    t_minilm_k3 = (time.perf_counter() - t0) * 1000
    print(f"mMiniLM (max_len=128) 3 pairs (K=3) latency: {t_minilm_k3:.2f} ms")

except Exception as e:
    print("MiniLM test failed:", e)

print("\n--- 5. Testing Fast Lexical-Semantic Rescorer (Sub-millisecond) ---")
def fast_hybrid_rescore(query, doc_text, dense_score=0.8, bm25_score=10.0):
    q_words = set(query.lower().split())
    d_words = set(doc_text.lower().split())
    overlap = len(q_words & d_words) / max(len(q_words), 1)
    return (0.7 * dense_score) + (0.3 * overlap)

t0 = time.perf_counter()
for q, d in pairs:
    _ = fast_hybrid_rescore(q, d)
t_fast = (time.perf_counter() - t0) * 1000
print(f"Fast Hybrid Rescorer 5 pairs latency: {t_fast:.4f} ms ({t_fast/5:.4f} ms/pair)")
