import json
from pathlib import Path

eval_v2 = json.loads(Path("data/evaluation/eval_v2.json").read_text(encoding="utf-8"))
chunks = [json.loads(line) for line in open("data/eval_corpus_19/chunks_eval_62.jsonl", encoding="utf-8")]

print("=== MATCHING QUESTIONS FOR FIRST 8 CHUNKS ===")
for i in range(8):
    c = chunks[i]
    c_text_lower = c["text"].lower()
    matching_qs = []
    for q in eval_v2:
        spans = q.get("gold_spans", [])
        if spans and all(s.lower() in c_text_lower for s in spans):
            matching_qs.append(q)
    print(f"Chunk {i} ({c['doc']}, order={c['chunk_order_index']}): {len(matching_qs)} matching questions")
    if matching_qs:
        best_q = matching_qs[0]
        print(f"   Selected Q{best_q['id']}: {best_q['question']}")
        print(f"   Spans: {best_q['gold_spans']}")
