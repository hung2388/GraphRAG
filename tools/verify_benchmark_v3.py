import json
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

path = Path("data/evaluation/eval_v3_distributed_65.json")
data = json.loads(path.read_text(encoding="utf-8"))
print(f"Total questions: {len(data)}")

by_doc = {}
total_spans = 0
for q in data:
    by_doc[q["doc"]] = by_doc.get(q["doc"], 0) + 1
    total_spans += len(q.get("gold_spans", []))

print(f"Total gold spans: {total_spans} (avg {total_spans/len(data):.2f}/q)")
print("Distribution across 19 documents:")
for doc, cnt in by_doc.items():
    print(f"  {doc:35s}: {cnt} questions")

random.seed(42)
samples = random.sample(data, 3)
print("\nSample questions:")
for s in samples:
    print(f"  [Q{s['id']}] doc={s['doc']}: {s['question']}")
    print(f"       Spans: {s['gold_spans']}")
    print(f"       Short Answer: {s['short_answer']}")
