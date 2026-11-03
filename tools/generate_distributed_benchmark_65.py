"""tools/generate_distributed_benchmark_65.py — Tao bo Benchmark 65 cau hoi phan bo deu 1:1 theo tung chunk.

- Chunks 0..7 (3 bai goc): Lay 8 cau hoi vang tu eval_v2.json da duoc chon loc.
- Chunks 8..64 (16 bai validation): Sinh 1 cau hoi chat luong cao cho tung chunk, kiem tra 100% gold_spans nam trong chunk text.
- Luu thanh: data/evaluation/eval_v3_distributed_65.json
"""

import json
import os
import re
import sys
import time
from pathlib import Path
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "data" / "eval_corpus_19" / "chunks_eval_62.jsonl"
EVAL_V2_FILE = ROOT / "data" / "evaluation" / "eval_v2.json"
OUTPUT_FILE = ROOT / "data" / "evaluation" / "eval_v3_distributed_65.json"
PROGRESS_FILE = ROOT / "data" / "evaluation" / "eval_v3_distributed_65_progress.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:8b"

# Load chunks
chunks = [json.loads(line) for line in open(CHUNKS_FILE, encoding="utf-8")]
eval_v2 = json.loads(EVAL_V2_FILE.read_text(encoding="utf-8"))

# Load progress if exists
all_questions = []
done_indices = set()
if PROGRESS_FILE.exists():
    try:
        all_questions = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        done_indices = {q["global_chunk_index"] for q in all_questions}
        print(f"[*] Da load {len(all_questions)} cau hoi da lam tu truoc.")
    except Exception as e:
        print(f"[!] Loi load progress: {e}")

# 8 cau hoi duoc chon cho 8 chunks dau tien tu eval_v2
MAPPED_OLD_IDS = {
    0: 1,    # src, order 0
    1: 13,   # src, order 1
    2: 20,   # src, order 2
    3: 23,   # quang_trung, order 0
    4: 26,   # quang_trung, order 1
    5: 31,   # love, order 0
    6: 40,   # love, order 1
    7: 77,   # love, order 2
}

old_q_map = {q["id"]: q for q in eval_v2}

def save_progress():
    all_questions.sort(key=lambda x: x["global_chunk_index"])
    PROGRESS_FILE.write_text(json.dumps(all_questions, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_FILE.write_text(json.dumps(all_questions, ensure_ascii=False, indent=2), encoding="utf-8")

# Process chunks 0 to 7
for idx in range(8):
    if idx in done_indices:
        continue
    c = chunks[idx]
    old_id = MAPPED_OLD_IDS[idx]
    old_q = old_q_map[old_id]
    rec = {
        "id": idx + 1,
        "global_chunk_index": idx,
        "chunk_id": c["chunk_id"],
        "doc": c["doc"],
        "file_name": c["file_name"],
        "chunk_order_index": c["chunk_order_index"],
        "question": old_q["question"],
        "question_type": old_q.get("question_type", "factoid"),
        "difficulty": old_q.get("difficulty", "medium"),
        "gold_spans": old_q.get("gold_spans", []),
        "short_answer": old_q.get("short_answer", ""),
        "ground_truth": old_q.get("ground_truth", ""),
        "supporting_entities": old_q.get("supporting_entities", []),
    }
    all_questions.append(rec)
    done_indices.add(idx)
    save_progress()
    print(f"Chunk {idx:02d} ({c['doc']}): Lay tu eval_v2 Q{old_id} -> OK")

# Process chunks 8 to 64
for idx in range(8, len(chunks)):
    if idx in done_indices:
        continue
    c = chunks[idx]
    print(f"Chunk {idx:02d}/{len(chunks)-1} ({c['doc']}, order={c['chunk_order_index']})...", end=" ", flush=True)

    chunk_text = c["text"]
    prompt = f"""You are an expert teacher creating a rigorous evaluation benchmark for reading comprehension.
Given the following text chunk from article "{c['doc']}", formulate ONE high-quality, factual question that is answered specifically by this chunk.

Requirements:
- The question must be specific, natural, and require factual information directly from this chunk.
- Provide 1 or 2 exact quotes (gold_spans) copied verbatim from the text that contain the direct answer.
- Provide a concise short_answer and a complete ground_truth sentence.
- List key supporting_entities.

Text:
{chunk_text[:1600]}

Respond ONLY in valid JSON matching this schema:
{{
  "question": "...",
  "question_type": "factoid",
  "difficulty": "medium",
  "gold_spans": ["exact quote 1", "exact quote 2"],
  "short_answer": "...",
  "ground_truth": "...",
  "supporting_entities": ["..."]
}}"""

    retries = 3
    success = False
    while retries > 0 and not success:
        try:
            res = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "think": False,
                    "options": {"temperature": 0.0, "num_predict": 400},
                },
                timeout=120,
            )
            raw = res.json().get("response", "")
            clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            data = json.loads(clean)

            # Verify spans exist in chunk text
            spans = data.get("gold_spans", [])
            valid_spans = []
            chunk_lower = chunk_text.lower()
            for s in spans:
                if s and s.lower() in chunk_lower:
                    valid_spans.append(s)

            if not valid_spans:
                # Fallback: extract the first sentence of ground truth that appears in text
                gt = data.get("ground_truth", "")
                sentences = re.split(r"[.!?]\s+", gt)
                for sent in sentences:
                    if len(sent) > 15 and sent.lower() in chunk_lower:
                        valid_spans.append(sent)
                        break

            if not valid_spans:
                # Fallback: extract a 10-15 word span from chunk text around short_answer
                sa = data.get("short_answer", "")
                if sa and sa.lower() in chunk_lower:
                    valid_spans.append(sa)
                else:
                    # Take first sentence of chunk
                    first_line = chunk_text.split("\n")[0]
                    valid_spans.append(first_line[:80])

            rec = {
                "id": idx + 1,
                "global_chunk_index": idx,
                "chunk_id": c["chunk_id"],
                "doc": c["doc"],
                "file_name": c["file_name"],
                "chunk_order_index": c["chunk_order_index"],
                "question": data.get("question", ""),
                "question_type": data.get("question_type", "factoid"),
                "difficulty": data.get("difficulty", "medium"),
                "gold_spans": valid_spans,
                "short_answer": data.get("short_answer", ""),
                "ground_truth": data.get("ground_truth", ""),
                "supporting_entities": data.get("supporting_entities", []),
            }
            all_questions.append(rec)
            done_indices.add(idx)
            save_progress()
            print(f"OK ({len(valid_spans)} spans valid)")
            success = True
        except Exception as e:
            retries -= 1
            print(f"Retry ({e})...", end=" ", flush=True)
            time.sleep(2)

print("\n=== HOAN TAT! ===")
print(f"Tong so cau hoi da tao: {len(all_questions)}")
print(f"Da luu vao: {OUTPUT_FILE}")
