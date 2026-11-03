"""step3_mine_hard_negatives.py — Dao hard negative bang nomic goc cho adapter embedding.

Cac yeu cau theo chi dao:
- Dung nomic-embed-text goc de tinh embedding cho toan bo chunk tap train va cau hoi.
- Tinh cosine similarity matrix, xep hang toan bo train chunks cho tung cau hoi.
- Xac dinh rank cua positive chunk (1-indexed).
- Loc false negative: loai cac chunk cung bai (dac biet cac chunk lien ke |idx - pos_idx| <= 1).
- Lay chunk hard negative xep hang khoang 10-30 cho cau hoi do.
- Chinh sach giu lai:
    + Uu tien 100% cap ma nomic goc xep sai (rank > 1), vi day la cho adapter co gi de hoc.
    + Giu lai khoang 35% cap ma nomic goc da xep dung hang 1 de model duy tri on dinh.
- Luu thanh: data/adapter_data/embedding_train_triplets.jsonl
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "adapter_data"
CHUNKS_TRAIN_FILE = DATA_DIR / "chunks_train.jsonl"
QUESTIONS_FILE = DATA_DIR / "embedding_train_pairs_raw.jsonl"
OUTPUT_FILE = DATA_DIR / "embedding_train_triplets.jsonl"
EMBED_CACHE_FILE = DATA_DIR / "cache_nomic_train_embeddings.npz"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
MODEL_NAME = "nomic-embed-text"
BATCH_SIZE = 16


def get_embeddings_batch(texts: List[str], max_retries: int = 3) -> np.ndarray:
    """Goi Ollama API lay embedding batched."""
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                OLLAMA_EMBED_URL,
                json={"model": MODEL_NAME, "input": texts},
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                embs = data.get("embeddings", [])
                if len(embs) == len(texts):
                    return np.array(embs, dtype=np.float32)
        except Exception as e:
            time.sleep(1.5)
    raise RuntimeError(f"Loi goi embed API sau {max_retries} lan thu.")


def embed_all_texts(texts: List[str], desc: str = "Embedding") -> np.ndarray:
    """Embed danh sach van ban theo batch co thanh tien trinh."""
    all_embs = []
    total = len(texts)
    print(f"[*] Bat dau embed {total} texts ({desc})...")
    t0 = time.time()
    for i in range(0, total, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        embs = get_embeddings_batch(batch)
        all_embs.append(embs)
        done = min(i + BATCH_SIZE, total)
        if done % 64 == 0 or done == total:
            elapsed = time.time() - t0
            speed = done / elapsed if elapsed > 0 else 0
            print(f"    [{done:4d}/{total}] {desc} | {speed:.1f} texts/s")
    return np.vstack(all_embs)


def main():
    random.seed(42)
    np.random.seed(42)

    if not CHUNKS_TRAIN_FILE.exists():
        print(f"[!] Khong tim thay {CHUNKS_TRAIN_FILE}")
        sys.exit(1)
    if not QUESTIONS_FILE.exists():
        print(f"[!] Khong tim thay {QUESTIONS_FILE}")
        sys.exit(1)

    # 1. Load train chunks
    chunks = []
    chunk_id_to_idx = {}
    with open(CHUNKS_TRAIN_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                c = json.loads(line)
                chunks.append(c)
                chunk_id_to_idx[c["chunk_id"]] = idx

    print(f"[*] Loaded {len(chunks)} train chunks.")

    # 2. Load questions
    queries = []
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    print(f"[*] Loaded {len(queries)} generated questions.")

    # 3. Tinh / load chunk embeddings
    if EMBED_CACHE_FILE.exists():
        print(f"[*] Loading cached chunk embeddings tu {EMBED_CACHE_FILE}...")
        chunk_embeddings = np.load(EMBED_CACHE_FILE)["chunk_embeddings"]
    else:
        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = embed_all_texts(chunk_texts, desc="Train Chunks")
        np.savez_compressed(EMBED_CACHE_FILE, chunk_embeddings=chunk_embeddings)
        print(f"[+] Saved chunk embeddings cache: {EMBED_CACHE_FILE}")

    # Chuan hoa L2 de tich vo huong chinh la cosine similarity
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    chunk_embeddings_norm = chunk_embeddings / np.maximum(chunk_norms, 1e-12)

    # 4. Embed queries
    query_texts = [q["question"] for q in queries]
    query_embeddings = embed_all_texts(query_texts, desc="Queries")
    q_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_embeddings_norm = query_embeddings / np.maximum(q_norms, 1e-12)

    # 5. Tinh ma tran cosine similarity: (N_queries, N_chunks)
    print("\n[*] Tinh toan similarity matrix...")
    sim_matrix = np.dot(query_embeddings_norm, chunk_embeddings_norm.T)
    print(f"[*] Sim matrix shape: {sim_matrix.shape}")

    # 6. Phan tich rank va dao hard negatives
    output_triplets = []
    rank_1_count = 0
    rank_failed_count = 0
    rank_distribution = {
        "rank_1": 0,
        "rank_2_5": 0,
        "rank_6_10": 0,
        "rank_11_30": 0,
        "rank_30_plus": 0,
    }

    print("\n[*] Bat dau dao hard negatives theo quy tac...")
    for q_idx, q_item in enumerate(queries):
        pos_id = q_item["pos_chunk_id"]
        if pos_id not in chunk_id_to_idx:
            continue
        pos_idx = chunk_id_to_idx[pos_id]
        pos_doc = q_item["pos_doc"]
        pos_order = q_item.get("pos_chunk_order_index", 0)

        # Lay diem va sort rank tat ca chunks
        scores = sim_matrix[q_idx]
        sorted_indices = np.argsort(-scores)  # giam dan

        # Tim rank cua pos_chunk (1-indexed)
        pos_rank = int(np.where(sorted_indices == pos_idx)[0][0]) + 1
        pos_score = float(scores[pos_idx])

        # Thong ke phan bo rank
        if pos_rank == 1:
            rank_distribution["rank_1"] += 1
            rank_1_count += 1
        elif 2 <= pos_rank <= 5:
            rank_distribution["rank_2_5"] += 1
            rank_failed_count += 1
        elif 6 <= pos_rank <= 10:
            rank_distribution["rank_6_10"] += 1
            rank_failed_count += 1
        elif 11 <= pos_rank <= 30:
            rank_distribution["rank_11_30"] += 1
            rank_failed_count += 1
        else:
            rank_distribution["rank_30_plus"] += 1
            rank_failed_count += 1

        # Chinh sach loc giu lai:
        # Neu nomic da xep dung hang 1: giu lai 35% ngau nhien
        # Neu nomic xep sai (rank > 1): giu lai 100% vi day la cho adapter can hoc!
        if pos_rank == 1:
            if random.random() > 0.35:
                continue

        # Tim hard negatives trong khoang rank 10-30 (sorted_indices[9:30])
        # Loai cac chunk false negative:
        # 1. Cung bai va lien ke (|order - pos_order| <= 1)
        # 2. Hoac bat ky chunk nao cung doc
        candidate_neg_indices = []
        for r_idx in range(5, min(40, len(sorted_indices))):
            cand_idx = sorted_indices[r_idx]
            if cand_idx == pos_idx:
                continue
            cand_chunk = chunks[cand_idx]
            # Loai neu cung doc de tranh false negative
            if cand_chunk["doc"] == pos_doc:
                continue
            # Uu tien nhung chunk trong rank 10-30
            candidate_neg_indices.append((r_idx + 1, cand_idx))

        # Lay 3-5 hard negatives
        rank_10_30_cands = [item for item in candidate_neg_indices if 10 <= item[0] <= 30]
        if len(rank_10_30_cands) >= 3:
            sampled_neg_tuples = random.sample(rank_10_30_cands, min(4, len(rank_10_30_cands)))
        else:
            # Mo rong ra khoang 8-35 neu khong du
            sampled_neg_tuples = candidate_neg_indices[:4]

        if not sampled_neg_tuples:
            continue

        hard_negatives = []
        for r, neg_idx in sampled_neg_tuples:
            c = chunks[neg_idx]
            hard_negatives.append({
                "chunk_id": c["chunk_id"],
                "doc": c["doc"],
                "title": c["title"],
                "rank": r,
                "score": round(float(scores[neg_idx]), 4),
                "text": c["text"],
            })

        triplet_record = {
            "query_id": q_item["query_id"],
            "question": q_item["question"],
            "question_type": q_item["question_type"],
            "pos_chunk_id": pos_id,
            "pos_doc": pos_doc,
            "pos_title": q_item["pos_title"],
            "pos_group": q_item["pos_group"],
            "pos_text": q_item["pos_text"],
            "nomic_pos_rank": pos_rank,
            "nomic_pos_score": round(pos_score, 4),
            "hard_negatives": hard_negatives,
        }
        output_triplets.append(triplet_record)

    # Ghi file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for t in output_triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("HOAN THANH DAO HARD NEGATIVES CHO ADAPTER EMBEDDING")
    print("=" * 60)
    print(f"Tong so cau hoi dau vao     : {len(queries)}")
    print(f"Phan bo Rank cua base Nomic:")
    for k, v in rank_distribution.items():
        pct = (v / len(queries) * 100) if len(queries) > 0 else 0
        print(f"  - {k:<12}: {v:4d} ({pct:5.1f}%)")
    print(f"Tong queries nomic dung (rank=1): {rank_1_count}")
    print(f"Tong queries nomic sai (rank>1) : {rank_failed_count}")
    print(f"Tong triplets duoc chon de train: {len(output_triplets)}")
    print(f"Luu tai                         : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
