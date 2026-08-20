# Phase 6.26 Retrieval Relevance & Grounding Quality Benchmark Report

**Generated:** 2026-08-20 14:53:40  
**Corpus:** MSMARCO-XI (48,206 chunks, 28,541 vectors)  
**Models:** `multilingual-e5-small` · `mmarco-mMiniLMv2-L12-H384-v1` · `Sarvam 105B`

---

## 1. Executive Summary

| Metric | Result | Target |
| :--- | :--- | :--- |
| **Relevant Acceptance Rate** | **100.0%** (25/25) | >= 90.0% |
| **Off-Topic Rejection Rate** | **100.0%** (7/7) | 100.0% |
| **Adversarial Rejection Rate** | **100.0%** (5/5) | 100.0% |
| **False Acceptances (Hallucination Risk)** | **0** | 0 |
| **False Refusals** | **0** | <= 2 |
| **Precision** | **1.0000** | >= 0.9500 |
| **Recall** | **1.0000** | >= 0.9000 |
| **F1 Score** | **1.0000** | >= 0.9200 |
| **Latency (P50 / P95)** | **1834.5 ms / 4927.7 ms** | <= 250 ms |

---

## 2. Multilingual In-Domain Performance

| Language | Code | Queries | Accepted | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| English | `en` | 5 | 5 | **100.0%** |
| Hindi | `hi` | 5 | 5 | **100.0%** |
| Tamil | `ta` | 5 | 5 | **100.0%** |
| Telugu | `te` | 5 | 5 | **100.0%** |
| Malayalam | `ml` | 5 | 5 | **100.0%** |

---

## 3. Case-by-Case Evaluation Matrix

| ID | Category | Language | Query | Decision | Allowed | Expected | Match |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `rel_en_01` | in_domain_relevant | `en` | What is a computer? | `STRONG` | True | True | ✓ |
| `rel_en_02` | in_domain_relevant | `en` | What is machine learning? | `LIMITED` | True | True | ✓ |
| `rel_en_03` | in_domain_relevant | `en` | What is artificial intelligence? | `LIMITED` | True | True | ✓ |
| `rel_en_04` | in_domain_relevant | `en` | What is the internet? | `STRONG` | True | True | ✓ |
| `rel_en_05` | in_domain_relevant | `en` | What is a programming language? | `STRONG` | True | True | ✓ |
| `rel_hi_01` | in_domain_relevant | `hi` | कंप्यूटर क्या है? | `STRONG` | True | True | ✓ |
| `rel_hi_02` | in_domain_relevant | `hi` | मशीन लर्निंग क्या है? | `LIMITED` | True | True | ✓ |
| `rel_hi_03` | in_domain_relevant | `hi` | कृत्रिम बुद्धिमत्ता क्या है? | `LIMITED` | True | True | ✓ |
| `rel_hi_04` | in_domain_relevant | `hi` | इंटरनेट क्या है? | `STRONG` | True | True | ✓ |
| `rel_hi_05` | in_domain_relevant | `hi` | प्रोग्रामिंग भाषा क्या है? | `STRONG` | True | True | ✓ |
| `rel_ta_01` | in_domain_relevant | `ta` | கணினி என்றால் என்ன? | `STRONG` | True | True | ✓ |
| `rel_ta_02` | in_domain_relevant | `ta` | இயந்திர கற்றல் என்றால் என்ன? | `LIMITED` | True | True | ✓ |
| `rel_ta_03` | in_domain_relevant | `ta` | செயற்கை நுண்ணறிவு என்றால் என்ன? | `LIMITED` | True | True | ✓ |
| `rel_ta_04` | in_domain_relevant | `ta` | இணையம் என்றால் என்ன? | `STRONG` | True | True | ✓ |
| `rel_ta_05` | in_domain_relevant | `ta` | நிரலாக்க மொழி என்றால் என்ன? | `LIMITED` | True | True | ✓ |
| `rel_te_01` | in_domain_relevant | `te` | కంప్యూటర్ అంటే ఏమిటి? | `STRONG` | True | True | ✓ |
| `rel_te_02` | in_domain_relevant | `te` | మెషిన్ లెర్నింగ్ అంటే ఏమిటి? | `LIMITED` | True | True | ✓ |
| `rel_te_03` | in_domain_relevant | `te` | కృత్రిమ మేధస్సు అంటే ఏమిటి? | `LIMITED` | True | True | ✓ |
| `rel_te_04` | in_domain_relevant | `te` | ఇంటర్నెట్ అంటే ఏమిటి? | `STRONG` | True | True | ✓ |
| `rel_te_05` | in_domain_relevant | `te` | ప్రోగ్రామింగ్ భాష అంటే ఏమిటి? | `STRONG` | True | True | ✓ |
| `rel_ml_01` | in_domain_relevant | `ml` | കമ്പ്യൂട്ടർ എന്താണ്? | `STRONG` | True | True | ✓ |
| `rel_ml_02` | in_domain_relevant | `ml` | മെഷീൻ ലേണിംഗ് എന്താണ്? | `LIMITED` | True | True | ✓ |
| `rel_ml_03` | in_domain_relevant | `ml` | കൃത്രിമ ബുദ്ധി എന്താണ്? | `LIMITED` | True | True | ✓ |
| `rel_ml_04` | in_domain_relevant | `ml` | ഇന്റർനെറ്റ് എന്താണ്? | `STRONG` | True | True | ✓ |
| `rel_ml_05` | in_domain_relevant | `ml` | പ്രോഗ്രാമിംഗ് ഭാഷ എന്താണ്? | `STRONG` | True | True | ✓ |
| `off_01` | off_topic_unanswerable | `en` | What is the weather in Goa today? | `OFF_TOPIC` | False | False | ✓ |
| `off_02` | off_topic_unanswerable | `en` | Who is the current Prime Minister of India? | `OFF_TOPIC` | False | False | ✓ |
| `off_03` | off_topic_unanswerable | `en` | What is my college timetable? | `OFF_TOPIC` | False | False | ✓ |
| `off_04` | off_topic_unanswerable | `en` | What is the price of petrol today? | `OFF_TOPIC` | False | False | ✓ |
| `off_05` | off_topic_unanswerable | `en` | What is the latest cricket score? | `OFF_TOPIC` | False | False | ✓ |
| `off_06` | off_topic_unanswerable | `en` | What is my bank balance? | `OFF_TOPIC` | False | False | ✓ |
| `off_07` | off_topic_unanswerable | `en` | What is the best restaurant near me? | `OFF_TOPIC` | False | False | ✓ |
| `adv_01` | adversarial_relevance | `en` | What is artificial intelligence? | `OFF_TOPIC` | False | False | ✓ |
| `adv_02` | adversarial_relevance | `en` | What is machine learning? | `OFF_TOPIC` | False | False | ✓ |
| `adv_03` | adversarial_relevance | `en` | What is a computer? | `INSUFFICIENT` | False | False | ✓ |
| `adv_04` | adversarial_relevance | `en` | Is it in it? | `OFF_TOPIC` | False | False | ✓ |
| `adv_05` | adversarial_relevance | `en` | What is the quantum flux density of xyz999? | `OFF_TOPIC` | False | False | ✓ |
