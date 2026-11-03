"""step2_gen_embedding_pairs.py — Sinh cap (cau hoi, chunk dung) cho adapter embedding.

Cac yeu cau theo chi dao:
- Moi chunk sinh 2-3 cau hoi voi nhieu kieu:
    1. 'single_event': tra cuu mot su kien, moc thoi gian, quyet dinh, nhan vat cu the.
    2. 'multi_attribute': ket hop tu 2 thuoc tinh/su kien duoc de cap cung luc trong chunk.
    3. 'paraphrase': dien dat lai y chinh/khai niem, khong sao chep nguyen van cau chu.
- Dung model qwen3:8b (think: False, format: json) qua Ollama API.
- Loc cau hoi:
    1. Qua chung chung (khong xac dinh duoc chunk, thieu thuc the/tu khoa cu the).
    2. Cau trung / gan trung.
    3. Cau sao chep nguyen van cum dai tu chunk (max n-gram lien tiep >= 6 tu).
- Chi chay tren chunks_train.jsonl (tap validation TUYET DOI khong dung de train).
- Luu thanh: data/adapter_data/embedding_train_pairs_raw.jsonl
"""

import json
import os
import re
import string
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "adapter_data"
INPUT_CHUNKS = DATA_DIR / "chunks_train.jsonl"
OUTPUT_FILE = DATA_DIR / "embedding_train_pairs_raw.jsonl"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:8b"

SYSTEM_PROMPT = """You are an expert NLP researcher generating high-quality retrieval training data.
Given a text passage from a Wikipedia article, create exactly 3 distinct search questions that can be specifically answered using ONLY this passage.

Question types required:
1. "single_event": A specific lookup query about an event, date, battle, release, decision, or fact in the passage.
2. "multi_attribute": A query linking at least two distinct attributes, people, dates, or concepts mentioned in the passage.
3. "paraphrase": A natural search query asking about the core topic/outcome using synonymous or abstract phrasing, strictly avoiding verbatim copied clauses from the passage.

Strict Rules:
- Every question MUST explicitly name the key entity/subject (person, song, place, dynasty). Never ask vague questions like 'What happened in the text?' or 'Who was mentioned?'.
- Do NOT copy verbatim long sentences or clauses from the text.
- Questions must be natural queries that a user would type into a search engine.
- Output ONLY a JSON object in this exact schema:
{
  "questions": [
    {"type": "single_event", "question": "string"},
    {"type": "multi_attribute", "question": "string"},
    {"type": "paraphrase", "question": "string"}
  ]
}"""

USER_TEMPLATE = """Document Title: {title} ({group})

Passage:
{text}
"""


def normalize_for_dedup(text: str) -> str:
    """Chuan hoa chuoi de kiem tra duplicate."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def tokenize_words(text: str) -> List[str]:
    """Tach tu don gian bo dau cau."""
    clean_t = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in clean_t.split() if w]


def max_contiguous_word_overlap(q_words: List[str], doc_words: List[str]) -> int:
    """Tim do dai n-gram tu lien tiep dai nhat cua cau hoi xuat hien nguyen van trong doc."""
    if not q_words or not doc_words:
        return 0
    doc_text_joined = " " + " ".join(doc_words) + " "
    max_len = 0
    # Kiem tra n-gram tu do dai lon xuong
    for window in range(len(q_words), 2, -1):
        for i in range(len(q_words) - window + 1):
            ngram = " " + " ".join(q_words[i : i + window]) + " "
            if ngram in doc_text_joined:
                return window
    return max_len


def is_too_generic(q: str, title: str) -> Tuple[bool, str]:
    """Kiem tra xem cau hoi co qua chung chung khong."""
    q_clean = q.strip()
    words = q_clean.split()
    if len(words) < 5 or len(q_clean) < 22:
        return True, "too_short"

    # Cac mau cau hoi chung chung vo nghia
    vague_patterns = [
        r"^what (is|are) (this|the) (text|passage|article|paragraph)",
        r"^what (happened|happens) in (this|the) (text|passage|article)",
        r"who (is|was) mentioned in (this|the) (text|passage)",
        r"^can you tell me about (this|the) (text|passage|article)",
        r"what is described in the passage",
        r"what does the author say",
    ]
    for pat in vague_patterns:
        if re.search(pat, q_clean, re.IGNORECASE):
            return True, "vague_pattern"

    # Phai co it nhat 1 tu viet hoa (ten rieng) hoac chua tu khoa cua title
    capitalized_tokens = [w for w in words[1:] if w and w[0].isupper() and w.isalpha()]
    title_words = set(tokenize_words(title))
    q_words_set = set(tokenize_words(q))
    has_title_keyword = bool(q_words_set.intersection(title_words))

    if not capitalized_tokens and not has_title_keyword:
        return True, "lacks_named_entity_or_topic"

    return False, "ok"


def call_llm_for_questions(chunk: Dict, max_retries: int = 3) -> List[Dict]:
    """Goi qwen3:8b qua Ollama de sinh 3 cau hoi."""
    user_prompt = USER_TEMPLATE.format(
        title=chunk.get("title", ""),
        group=chunk.get("group", ""),
        text=chunk["text"],
    )
    payload = {
        "model": MODEL_NAME,
        "prompt": f"{SYSTEM_PROMPT}\n\n{user_prompt}",
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096,
        },
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=90)
            if resp.status_code != 200:
                time.sleep(2)
                continue
            raw = resp.json().get("response", "")
            # Don dep neu con think tag
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            data = json.loads(raw)
            q_list = data.get("questions", [])
            if isinstance(q_list, list) and len(q_list) > 0:
                return q_list
        except Exception as e:
            time.sleep(2)
    return []


def main():
    if not INPUT_CHUNKS.exists():
        print(f"[!] Khong tim thay {INPUT_CHUNKS}")
        sys.exit(1)

    # 1. Load train chunks
    chunks = []
    with open(INPUT_CHUNKS, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"[*] Tong so train chunks can sinh: {len(chunks)} chunks")

    # 2. Checkpoint resume
    processed_chunk_ids: Set[str] = set()
    global_seen_queries: Set[str] = set()
    existing_records = []

    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing_records.append(rec)
                    processed_chunk_ids.add(rec["chunk_id"])
                    global_seen_queries.add(normalize_for_dedup(rec["question"]))
        print(f"[*] Resume tu checkpoint: Da co {len(existing_records)} queries tu {len(processed_chunk_ids)} chunks.")

    out_fp = open(OUTPUT_FILE, "a", encoding="utf-8")

    stats = {
        "total_generated": 0,
        "filtered_generic": 0,
        "filtered_duplicate": 0,
        "filtered_verbatim": 0,
        "kept": 0,
    }

    t_start = time.time()
    try:
        for idx, chunk in enumerate(chunks, 1):
            c_id = chunk["chunk_id"]
            if c_id in processed_chunk_ids:
                continue

            doc_slug = chunk["doc"]
            doc_title = chunk["title"]
            chunk_text = chunk["text"]
            doc_words = tokenize_words(chunk_text)

            candidates = call_llm_for_questions(chunk)
            stats["total_generated"] += len(candidates)

            kept_for_chunk = []
            for item in candidates:
                q_text = item.get("question", "").strip()
                q_type = item.get("type", "unknown")
                if not q_text:
                    continue

                # Loc 1: Qua chung chung
                is_generic, reason = is_too_generic(q_text, doc_title)
                if is_generic:
                    stats["filtered_generic"] += 1
                    continue

                # Loc 2: Duplicate
                q_norm = normalize_for_dedup(q_text)
                if q_norm in global_seen_queries:
                    stats["filtered_duplicate"] += 1
                    continue

                # Loc 3: Sao chep nguyen van cum dai (>= 6 tu)
                q_words = tokenize_words(q_text)
                max_overlap = max_contiguous_word_overlap(q_words, doc_words)
                if max_overlap >= 6:
                    stats["filtered_verbatim"] += 1
                    continue

                # Hop le -> giu lai
                global_seen_queries.add(q_norm)
                record = {
                    "query_id": f"q-{c_id[:12]}-{len(kept_for_chunk)+1}",
                    "question": q_text,
                    "question_type": q_type,
                    "pos_chunk_id": c_id,
                    "pos_doc": doc_slug,
                    "pos_title": doc_title,
                    "pos_group": chunk.get("group", ""),
                    "pos_chunk_order_index": chunk.get("chunk_order_index", 0),
                    "pos_text": chunk_text,
                    "max_verbatim_overlap": max_overlap,
                }
                kept_for_chunk.append(record)
                stats["kept"] += 1

                # Ghi ngay xuong file
                out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_fp.flush()

            elapsed = time.time() - t_start
            speed = idx / elapsed if elapsed > 0 else 0
            if idx % 5 == 0 or idx == len(chunks):
                print(
                    f"[{idx:3d}/{len(chunks)}] chunk={c_id[:8]}.. | "
                    f"kept={len(kept_for_chunk)} | "
                    f"total_kept={len(existing_records) + stats['kept']} | "
                    f"filt(gen/dup/verb)={stats['filtered_generic']}/{stats['filtered_duplicate']}/{stats['filtered_verbatim']} | "
                    f"{speed:.2f} chunks/s"
                )

    finally:
        out_fp.close()

    total_all_kept = len(existing_records) + stats["kept"]
    print("\n" + "=" * 60)
    print("HOAN THANH SINH CAP CAU HOI - CHUNK DUNG")
    print("=" * 60)
    print(f"Tong so chunks da xu ly : {len(chunks)}")
    print(f"Tong cau hoi da sinh ra : {stats['total_generated']}")
    print(f"Loc vi qua chung chung  : {stats['filtered_generic']}")
    print(f"Loc vi bi trung lap      : {stats['filtered_duplicate']}")
    print(f"Loc vi sao chep nguyen van: {stats['filtered_verbatim']}")
    print(f"Tong cau hoi giu lai    : {total_all_kept}")
    print(f"Luu tai                  : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
