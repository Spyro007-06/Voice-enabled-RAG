import json
import os

processed_dir = os.path.abspath("data/processed")
terms = [
    "natural language processing", "nlp", "computational linguistics", "human language",
    "machine learning", "artificial intelligence", "corporation", "rachel carson",
    "computer", "कंप्यूटर", "निगम", "कृत्रिम बुद्धिमत्ता", "भाषा"
]

counts = {t: 0 for t in terms}
langs = {}
total_chunks = 0
sample_queries = set()

for fname in os.listdir(processed_dir):
    if fname.endswith(".jsonl"):
        fpath = os.path.join(processed_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                total_chunks += 1
                try:
                    c = json.loads(line)
                    text = (c.get("text") or "").lower()
                    meta = c.get("metadata") or {}
                    eng_p = (meta.get("eng_passage") or "").lower()
                    eng_q = (meta.get("eng_query") or "").lower()
                    q = meta.get("query") or ""
                    if q:
                        sample_queries.add(q)
                    l = c.get("language") or meta.get("language")
                    langs[l] = langs.get(l, 0) + 1

                    combined = f"{text} {eng_p} {eng_q}"
                    for t in terms:
                        if t.lower() in combined:
                            counts[t] += 1
                except Exception:
                    pass

print(f"Total chunks across all processed files: {total_chunks}")
print(f"Languages present: {langs}")
print("Keyword counts across text + metadata:")
for t, cnt in counts.items():
    print(f"  '{t}': {cnt}")
print(f"\nUnique sample queries in dataset ({len(sample_queries)}):")
for q in list(sample_queries)[:10]:
    print(f"  - {q}")
