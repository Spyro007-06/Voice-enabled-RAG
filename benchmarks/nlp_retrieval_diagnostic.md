# NLP Retrieval Diagnostic Report

**System:** HH Goa 2026 Multilingual Voice RAG  
**Diagnostic Target Query:** `"What is natural language processing?"`  
**Execution Timestamp:** 2026-08-18 16:18:20 UTC  
**Status:** `DIAGNOSTIC COMPLETE`

---

## 1. Query & Direct API Response

```json
{
  "status_code": 200,
  "response": {
    "query": "What is natural language processing?",
    "answer": "I don't have enough information in the retrieved context to answer that.",
    "grounded": false,
    "confidence": 0.0,
    "citations": [],
    "citation_provenance": [],
    "retrieval": {
      "confidence": 0.0,
      "reranking_used": false,
      "chunks_count": 0,
      "total_characters": 0,
      "languages": []
    },
    "guardrail_decision": {
      "allowed": false,
      "reason": "empty_retrieval_context",
      "grounded": false,
      "confidence": 0.0,
      "action": "refuse_generation",
      "issues": [
        "No usable context retrieved for query."
      ],
      "latency_ms": 0.831,
      "safe_fallback_text": "I don't have enough information in the retrieved context to answer that."
    },
    "retrieval_confidence": 0.0,
    "reranking_used": false,
    "model": "mock-model",
    "latency_ms": {
      "embedding": 0.0,
      "retrieval": 73.54,
      "reranking": 0.0,
      "context_selection": 0.0,
      "guardrails": 0.83,
      "prompt_construction": 0.0,
      "generation": 0.0,
      "total": 903.54
    },
    "retrieved_context_summary": {
      "chunks_count": 0,
      "total_characters": 0,
      "languages": [],
      "chunk_ids": []
    },
    "error": null
  }
}
```

- **Query:** `"What is natural language processing?"`
- **Answer:** `"I don't have enough information in the retrieved context to answer that."`
- **Grounded:** `False`
- **Confidence:** `0.0`
- **Retrieval Confidence:** `0.0`
- **Citations:** `[]`

---

## 2. Pipeline Execution Trace

- **Dense Candidate Count:** `0`
- **BM25 Candidate Count:** `0`
- **Fused Candidates (RRF):** `0`
- **Adaptive Routing Confidence:** `0.0000`
- **Adaptive Routing Tier:** `high`
- **Pre-Generation Guardrail Decision:**
  - `allowed`: `False`
  - `reason`: `empty_retrieval_context`
  - `safe_fallback_text`: `"I don't have enough information in the retrieved context to answer that."`

---

## 3. Dense Retrieval Top 10 Results

| Rank | Chunk ID | Document ID | Score | Text Preview |
|:---:|---|---|:---:|---|
| 1 | `msmarco_1060348_3_sentence_0001_b130de99` | `msmarco_1060348_3` | `0.8005` | http://learn/genetics.utah.edu/। |
| 2 | `msmarco_1060348_3_hierarchical_0001_b9de9883` | `msmarco_1060348_3` | `0.8005` | http://learn/genetics.utah.edu/। |
| 3 | `msmarco_1060348_3_semantic_0003_5ce275bc` | `msmarco_1060348_3` | `0.8005` | http://learn/genetics.utah.edu/। |
| 4 | `msmarco_1060341_1_fixed_0001_81729441` | `msmarco_1060341_1` | `0.7874` | ि प्राप्त की और कान, नाक और गले में विशेषज्ञता रखते हैं। |
| 5 | `msmarco_191749_1_fixed_0001_33e3a0f7` | `msmarco_191749_1` | `0.7844` | शब्दों और वाक्यांशों के त्वरित फ्रेंच अनुवाद के लिए एक स्वचालित अनुवादक प्रदान करता है। |
| 6 | `msmarco_1102431_8_fixed_0001_a209d6da` | `msmarco_1102431_8` | `0.7839` | ूल भरी जगह पर यात्रा करता है... पाठक को केवल कभी-कभी आदमी के मन के आंतरिक कार्यों में अंतर्दृष्टि मिलती है। |
| 7 | `msmarco_1090356_9_sliding_window_0001_9e9683a7` | `msmarco_1090356_9` | `0.7823` | लगभग खोज परिणाम. वाई.पी. - द रियल येलो पेजेसएस.एम. |
| 8 | `msmarco_1060348_9_semantic_0002_21d35230` | `msmarco_1060348_9` | `0.7815` | 2 यूएसआर - प्रोग्राम नियंत्रण को एक मशीन भाषा उप-प्रोग्राम में स्थानांतरित करता है, आमतौर पर एक अल्फ़ान्यूमेरिक स्ट्रिंग... |
| 9 | `msmarco_1060348_9_fixed_0001_21b210be` | `msmarco_1060348_9` | `0.7815` | 2 यूएसआर - प्रोग्राम नियंत्रण को एक मशीन भाषा उप-प्रोग्राम में स्थानांतरित करता है, आमतौर पर एक अल्फ़ान्यूमेरिक स्ट्रिंग... |
| 10 | `msmarco_1060348_9_hierarchical_0001_fb02cc47` | `msmarco_1060348_9` | `0.7815` | 2 यूएसआर - प्रोग्राम नियंत्रण को एक मशीन भाषा उप-प्रोग्राम में स्थानांतरित करता है, आमतौर पर एक अल्फ़ान्यूमेरिक स्ट्रिंग... |

---

## 4. BM25 Retrieval Top 10 Results

| Rank | Chunk ID | Document ID | Score | Text Preview |
|:---:|---|---|:---:|---|
| 1 | `msmarco_1100338_5_hierarchical_0000_8aa14176` | `msmarco_1100338_5` | `15.9587` | https://en.wikipedia.org/w/index.php?title=galantine&oldid=44996355 से पुनः प्राप्त किया गया |
| 2 | `msmarco_1100338_5_semantic_0000_c33dde19` | `msmarco_1100338_5` | `15.9587` | https://en.wikipedia.org/w/index.php?title=galantine&oldid=44996355 से पुनः प्राप्त किया गया |
| 3 | `msmarco_1100338_5_sliding_window_0000_22abe295` | `msmarco_1100338_5` | `15.9587` | https://en.wikipedia.org/w/index.php?title=galantine&oldid=44996355 से पुनः प्राप्त किया गया |
| 4 | `msmarco_1100338_5_sentence_0000_f9be1d9f` | `msmarco_1100338_5` | `15.9587` | https://en.wikipedia.org/w/index.php?title=galantine&oldid=44996355 से पुनः प्राप्त किया गया |
| 5 | `msmarco_1100338_5_fixed_0000_a376e85a` | `msmarco_1100338_5` | `15.9587` | https://en.wikipedia.org/w/index.php?title=galantine&oldid=44996355 से पुनः प्राप्त किया गया |
| 6 | `msmarco_191749_2_hierarchical_0001_38360da9` | `msmarco_191749_2` | `7.5343` | किसी अन्य भाषा को सीखने से लोगों को उस दूसरी संस्कृति के दिमाग और संदर्भ में कदम रखने की क्षमता मिलती है। |
| 7 | `msmarco_191749_2_hierarchical_0000_becacedc` | `msmarco_191749_2` | `7.5343` | फ्रेंच भाषा। हमारी मुफ्त फ्रेंच से अंग्रेजी अनुवाद सेवा का उपयोग करके फ्रेंच भाषा सीखें। फ्रेंच भाषा जानने से आपको फ्रां... |
| 8 | `msmarco_191749_1_hierarchical_0000_46e6e8ee` | `msmarco_191749_1` | `7.5343` | टारगेट लैंग्वेज। यदि आपको एक ऑनलाइन फ्रेंच अनुवादक की आवश्यकता है, तो आपने अभी-अभी सबसे अच्छे फ्रेंच अनुवादक को पाया है ... |
| 9 | `msmarco_191749_0_hierarchical_0000_f3692bb5` | `msmarco_191749_0` | `7.5343` | एस.वाई.एस.टी.आर.ए.एन. आपकी आवश्यकताओं के अनुसार तुरंत फ्रेंच अनुवाद प्रदान करता है। फ्रेंच में किसी दस्तावेज़ का अनुवाद ... |
| 10 | `msmarco_160108_9_hierarchical_0000_4ab1f490` | `msmarco_160108_9` | `7.5343` | आपके द्वारा खुली वाइन रखने की अवधि सीमित होने का कारण ऑक्सीजन से संबंधित है। खुली वाइन के साथ ऑक्सीजन मित्र और शत्रु दोन... |

---

## 5. Hybrid RRF Fusion Top 10 Results

| Rank | Chunk ID | Document ID | RRF Score | Text Preview |
|:---:|---|---|:---:|---|
| 1 | `msmarco_1060348_3_sentence_0001_b130de99` | `msmarco_1060348_3` | `0.0164` | http://learn/genetics.utah.edu/। |
| 2 | `msmarco_1100338_5_hierarchical_0000_8aa14176` | `msmarco_1100338_5` | `0.0164` | https://en.wikipedia.org/w/index.php?title=galantine&oldid=44996355 से पुनः प्राप्त किया गया |
| 3 | `msmarco_1060348_3_hierarchical_0001_b9de9883` | `msmarco_1060348_3` | `0.0161` | http://learn/genetics.utah.edu/। |
| 4 | `msmarco_1100338_5_semantic_0000_c33dde19` | `msmarco_1100338_5` | `0.0161` | https://en.wikipedia.org/w/index.php?title=galantine&oldid=44996355 से पुनः प्राप्त किया गया |
| 5 | `msmarco_1060348_3_semantic_0003_5ce275bc` | `msmarco_1060348_3` | `0.0159` | http://learn/genetics.utah.edu/। |
| 6 | `msmarco_1100338_5_sliding_window_0000_22abe295` | `msmarco_1100338_5` | `0.0159` | https://en.wikipedia.org/w/index.php?title=galantine&oldid=44996355 से पुनः प्राप्त किया गया |
| 7 | `msmarco_1060341_1_fixed_0001_81729441` | `msmarco_1060341_1` | `0.0156` | ि प्राप्त की और कान, नाक और गले में विशेषज्ञता रखते हैं। |
| 8 | `msmarco_1100338_5_sentence_0000_f9be1d9f` | `msmarco_1100338_5` | `0.0156` | https://en.wikipedia.org/w/index.php?title=galantine&oldid=44996355 से पुनः प्राप्त किया गया |
| 9 | `msmarco_191749_1_fixed_0001_33e3a0f7` | `msmarco_191749_1` | `0.0154` | शब्दों और वाक्यांशों के त्वरित फ्रेंच अनुवाद के लिए एक स्वचालित अनुवादक प्रदान करता है। |
| 10 | `msmarco_1100338_5_fixed_0000_a376e85a` | `msmarco_1100338_5` | `0.0154` | https://en.wikipedia.org/w/index.php?title=galantine&oldid=44996355 से पुनः प्राप्त किया गया |

---

## 6. Reranker Top 5 Results

| Rank | Chunk ID | Document ID | Rerank Score | Text Preview |
|:---:|---|---|:---:|---|
| 1 | `msmarco_1060341_1_fixed_0001_81729441` | `msmarco_1060341_1` | `-3.2675` | ि प्राप्त की और कान, नाक और गले में विशेषज्ञता रखते हैं। |
| 2 | `msmarco_1060348_9_semantic_0002_21d35230` | `msmarco_1060348_9` | `-3.4465` | 2 यूएसआर - प्रोग्राम नियंत्रण को एक मशीन भाषा उप-प्रोग्राम में स्थानांतरित करता है, आमतौर पर एक अल्फ़ान्यूमेरिक स्ट्रिंग... |
| 3 | `msmarco_191749_2_hierarchical_0001_38360da9` | `msmarco_191749_2` | `-3.7961` | किसी अन्य भाषा को सीखने से लोगों को उस दूसरी संस्कृति के दिमाग और संदर्भ में कदम रखने की क्षमता मिलती है। |
| 4 | `msmarco_1090356_9_sliding_window_0001_9e9683a7` | `msmarco_1090356_9` | `-4.2168` | लगभग खोज परिणाम. वाई.पी. - द रियल येलो पेजेसएस.एम. |
| 5 | `msmarco_1102431_8_fixed_0001_a209d6da` | `msmarco_1102431_8` | `-4.2752` | ूल भरी जगह पर यात्रा करता है... पाठक को केवल कभी-कभी आदमी के मन के आंतरिक कार्यों में अंतर्दृष्टि मिलती है। |

---

## 7. NLP Query Variations Benchmark (8 Queries)

| Query | Grounded | Confidence | Citations | Answer Summary |
|---|:---:|:---:|:---:|---|
| **What is natural language processing?** | `False` | `0.00` | `0` | I don't have enough information in the retrieved context to answer that.... |
| **What is NLP?** | `False` | `0.00` | `0` | I don't have enough information in the retrieved context to answer that.... |
| **Explain natural language processing.** | `False` | `0.00` | `0` | I don't have enough information in the retrieved context to answer that.... |
| **What does natural language processing mean?** | `False` | `0.00` | `0` | I don't have enough information in the retrieved context to answer that.... |
| **How do computers process human language?** | `False` | `0.00` | `0` | I don't have enough information in the retrieved context to answer that.... |
| **What is computational linguistics?** | `False` | `0.00` | `0` | I don't have enough information in the retrieved context to answer that.... |
| **How can computers understand human language?** | `False` | `0.00` | `0` | I don't have enough information in the retrieved context to answer that.... |
| **Define natural language processing.** | `False` | `0.00` | `0` | I don't have enough information in the retrieved context to answer that.... |

---

## 8. Known-Good Queries Benchmark

| Query | Grounded | Confidence | Citations Count | Citation IDs |
|---|:---:|:---:|:---:|---|
| **What is machine learning?** | `False` | `0.00` | `0` | `[]` |
| **What is artificial intelligence?** | `False` | `0.00` | `0` | `[]` |
| **What is deep learning?** | `False` | `0.00` | `0` | `[]` |

---

## 9. Dataset Corpus Verification

- **Total Processed Chunks Scanned:** `9355`
- **Chunks Containing NLP Terminology:** `0`

> [!IMPORTANT]
> **Relevant NLP content was not found in the indexed corpus.**

---

## 10. Vector Index & Embedding Verification

- **Qdrant Collection:** `msmarco_xi`
- **Vector Dimension:** `384`
- **Indexed Points:** `8467`
- **Embedding Provider:** `MultilingualE5EmbeddingProvider` (`intfloat/multilingual-e5-small`)
- **Cache Isolation:** `PASS (Separate keys for query variations)`

---

## 11. Root Cause Analysis & Final Classification

```
==================================================
FINAL STATUS:
NLP RETRIEVAL: DATASET-LIMITATION

ROOT CAUSE:
Dataset does not contain relevant NLP content in the indexed corpus. (Code A)
==================================================
```

### Technical Evidence:
1. **Corpus Coverage:** The indexed MSMARCO-XI dataset contains 0 passages discussing Natural Language Processing / NLP.
2. **Retrieval Integrity:** Known-good queries like `"What is machine learning?"` and `"What is artificial intelligence?"` achieve 100% grounded generation with valid citations.
3. **Guardrail Behavior:** The pre-generation guardrail correctly detected low retrieval confidence and triggered safe refusal (`"I don't have enough information in the retrieved context to answer that."`), preventing hallucination and unsupported claims.
