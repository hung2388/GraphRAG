"""tools/eval_retrieval_delta.py — Comprehensive Retrieval Evaluation

Measures Hit@1, Hit@3, Hit@5, Hit@10 and MRR across the 65 evaluation queries
against the 65 corpus chunks in data/eval_corpus_19/chunks_eval_62.jsonl.

Compares:
1. Base nomic-embed-text-v1.5
2. Adapted nomic_adapted_merged (LoRA trained on contrastive triplets)
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Redirect HF cache to Drive D
os.environ["HF_HOME"] = "D:/Lab/.cache/huggingface"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvalRetrievalDelta")

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "data" / "eval_corpus_19" / "chunks_eval_62.jsonl"
EVAL_FILE = ROOT / "data" / "evaluation" / "eval_v3_distributed_65.json"
BASE_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
ADAPTED_MODEL_PATH = ROOT / "models" / "nomic_adapted_merged"

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def load_chunks():
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    logger.info(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")
    return chunks


def load_benchmark():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data if isinstance(data, list) else data.get("questions", data)
    logger.info(f"Loaded {len(questions)} evaluation questions from {EVAL_FILE}")
    return questions



def evaluate_model(model_name_or_path, chunks, questions, model_label="Model"):
    logger.info(f"\n==========================================")
    logger.info(f"Evaluating {model_label}: {model_name_or_path}")
    logger.info(f"==========================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(model_name_or_path), trust_remote_code=True, device=device)

    # Encode all corpus chunks
    chunk_ids = [c["chunk_id"] for c in chunks]
    chunk_texts = [f"search_document: {c['text'].strip()}" for c in chunks]
    logger.info(f"Encoding {len(chunk_texts)} corpus chunks...")
    t0 = time.perf_counter()
    chunk_embeddings = model.encode(chunk_texts, batch_size=8, show_progress_bar=False, normalize_embeddings=True)
    logger.info(f"Chunk encoding completed in {time.perf_counter() - t0:.2f}s")

    # Encode all questions
    query_texts = [f"search_query: {q['question'].strip()}" for q in questions]
    t0 = time.perf_counter()
    query_embeddings = model.encode(query_texts, batch_size=8, show_progress_bar=False, normalize_embeddings=True)
    logger.info(f"Query encoding completed in {time.perf_counter() - t0:.2f}s")

    # Compute cosine similarity matrix: (num_queries, num_chunks)
    sim_matrix = np.dot(query_embeddings, chunk_embeddings.T)

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks = []

    for i, q in enumerate(questions):
        gold_chunk_id = q.get("chunk_id") or q.get("target_chunk_id")
        sims = sim_matrix[i]

        # Rank chunk indices descending
        ranked_indices = np.argsort(-sims)
        ranked_chunk_ids = [chunk_ids[idx] for idx in ranked_indices]

        try:
            rank = ranked_chunk_ids.index(gold_chunk_id) + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / rank)
            if rank <= 1:
                hits_at_1 += 1
            if rank <= 3:
                hits_at_3 += 1
            if rank <= 5:
                hits_at_5 += 1
            if rank <= 10:
                hits_at_10 += 1
        except ValueError:
            reciprocal_ranks.append(0.0)

    n = len(questions)
    hit1 = hits_at_1 / n
    hit3 = hits_at_3 / n
    hit5 = hits_at_5 / n
    hit10 = hits_at_10 / n
    mrr = float(np.mean(reciprocal_ranks))

    logger.info(f"[{model_label} Results]")
    logger.info(f"  Hit@1 : {hit1:.4f} ({hits_at_1}/{n})")
    logger.info(f"  Hit@3 : {hit3:.4f} ({hits_at_3}/{n})")
    logger.info(f"  Hit@5 : {hit5:.4f} ({hits_at_5}/{n})")
    logger.info(f"  Hit@10: {hit10:.4f} ({hits_at_10}/{n})")
    logger.info(f"  MRR   : {mrr:.4f}")

    del model
    del query_embeddings
    del chunk_embeddings
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "hit@1": hit1,
        "hit@3": hit3,
        "hit@5": hit5,
        "hit@10": hit10,
        "mrr": mrr,
    }



def main():
    chunks = load_chunks()
    questions = load_benchmark()

    base_cache_file = ROOT / "data" / "evaluation" / "base_retrieval_metrics.json"
    if base_cache_file.exists():
        logger.info(f"Loading cached base model metrics from {base_cache_file}...")
        with open(base_cache_file, "r", encoding="utf-8") as f:
            base_res = json.load(f)
    else:
        # 1. Base Model
        base_res = evaluate_model(BASE_MODEL_NAME, chunks, questions, model_label="Base Nomic v1.5")
        with open(base_cache_file, "w", encoding="utf-8") as f:
            json.dump(base_res, f, indent=2)

    # 2. Adapted Merged Model
    adapted_res = evaluate_model(ADAPTED_MODEL_PATH, chunks, questions, model_label="Adapted Nomic (LoRA Merged)")

    # 3. Delta
    logger.info("\n" + "=" * 50)
    logger.info("       DELTA IMPROVEMENT COMPARISON")
    logger.info("=" * 50)
    logger.info(f"{'Metric':<10} | {'Base Nomic':<12} | {'Adapted':<12} | {'Delta':<10}")
    logger.info("-" * 50)
    for m in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr"]:
        b = base_res[m]
        a = adapted_res[m]
        d = a - b
        sign = "+" if d >= 0 else ""
        logger.info(f"{m.upper():<10} | {b:<12.4f} | {a:<12.4f} | {sign}{d:.4f}")
    logger.info("=" * 50)



if __name__ == "__main__":
    main()
