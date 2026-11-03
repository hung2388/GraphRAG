import json
import urllib.request
import numpy as np
from pathlib import Path

ROOT = Path("d:/Lab")
chunks = [json.loads(line) for line in open(ROOT / "data/eval_corpus_19/chunks_eval_62.jsonl", encoding="utf-8") if line.strip()]
questions = json.load(open(ROOT / "data/evaluation/eval_v3_distributed_65.json", encoding="utf-8"))
chunk_ids = [c["chunk_id"] for c in chunks]
chunk_texts = ["search_document: " + c["text"].strip() for c in chunks]
query_texts = ["search_query: " + q["question"].strip() for q in questions]

def get_embeddings(texts, model_name):
    all_embs = []
    for i in range(0, len(texts), 16):
        batch = texts[i:i+16]
        req = urllib.request.Request(
            "http://127.0.0.1:11435/v1/embeddings",
            data=json.dumps({"input": batch, "model": model_name}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            all_embs.extend([item["embedding"] for item in data["data"]])
    arr = np.array(all_embs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms[norms == 0] = 1e-12
    return arr / norms

print("Evaluating via port 11435...")
results = {}
for m_label, m_name in [("Base (Nomic v1.5)", "nomic-embed-text"), ("Adapted (LoRA Merged)", "nomic-embed-text-adapted")]:
    c_embs = get_embeddings(chunk_texts, m_name)
    q_embs = get_embeddings(query_texts, m_name)
    sims = np.dot(q_embs, c_embs.T)
    h1, h3, h5, mrrs = 0, 0, 0, []
    for i, q in enumerate(questions):
        gold = q.get("chunk_id") or q.get("target_chunk_id")
        ranked = [chunk_ids[idx] for idx in np.argsort(-sims[i])]
        try:
            r = ranked.index(gold) + 1
            mrrs.append(1.0 / r)
            if r <= 1: h1 += 1
            if r <= 3: h3 += 1
            if r <= 5: h5 += 1
        except ValueError:
            mrrs.append(0.0)
    n = len(questions)
    res = {
        "hit@1": h1 / n,
        "hit@3": h3 / n,
        "hit@5": h5 / n,
        "mrr": float(np.mean(mrrs))
    }
    results[m_label] = res
    print(f"{m_label:25s} | Hit@1: {res['hit@1']:.4f} ({h1}/{n}) | Hit@3: {res['hit@3']:.4f} ({h3}/{n}) | Hit@5: {res['hit@5']:.4f} ({h5}/{n}) | MRR: {res['mrr']:.4f}")

h1_base, h1_ad = results["Base (Nomic v1.5)"]["hit@1"], results["Adapted (LoRA Merged)"]["hit@1"]
mrr_base, mrr_ad = results["Base (Nomic v1.5)"]["mrr"], results["Adapted (LoRA Merged)"]["mrr"]
print("-" * 75)
print(f"DELTA Hit@1: {h1_ad - h1_base:+.4f} ({(h1_ad - h1_base)*100:+.2f}%)")
print(f"DELTA MRR  : {mrr_ad - mrr_base:+.4f} ({(mrr_ad - mrr_base)*100:+.2f}%)")
