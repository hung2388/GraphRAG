"""tools/index_hirag_v2.py — Index 19 tai lieu eval vao HiRAG tao cache moi (hirag_cache_v2).
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
HIRAG_DIR = ROOT / "HiRAG"
sys.path.insert(0, str(HIRAG_DIR))

from hirag import HiRAG
from hi_Search_ollama import (
    OLLAMA_embedding,
    OLLAMA_model_if_cache,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_URL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HiRAG_Index_v2")

EVAL_DIR = ROOT / "data" / "eval_corpus_19"
TARGET_CACHE_DIR = HIRAG_DIR / "hirag_cache_v2"


async def main():
    TARGET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Target cache directory: {TARGET_CACHE_DIR}")

    files = sorted([f for f in EVAL_DIR.glob("*.txt") if f.is_file()])
    logger.info(f"Tim thay {len(files)} files trong {EVAL_DIR}")

    docs = []
    for fp in files:
        content = fp.read_text(encoding="utf-8").strip()
        logger.info(f"  Loaded {fp.name:35s}: {len(content):6d} chars")
        docs.append(content)

    rag = HiRAG(
        working_dir=str(TARGET_CACHE_DIR),
        enable_llm_cache=True,
        embedding_func=OLLAMA_embedding,
        best_model_func=OLLAMA_model_if_cache,
        cheap_model_func=OLLAMA_model_if_cache,
        enable_hierachical_mode=True,
        enable_naive_rag=True,
        best_model_max_async=1,
        cheap_model_max_async=1,
        embedding_batch_num=4,
        embedding_func_max_async=2,
    )

    logger.info(f"Bat dau insert {len(docs)} documents vao HiRAG cache v2...")
    start_t = time.perf_counter()
    await rag.ainsert(docs)
    elapsed = time.perf_counter() - start_t
    logger.info(f"=== HOAN TAT INDEXING HIRAG V2 TRONG {elapsed:.2f}s ({elapsed/60:.2f} phut) ===")


if __name__ == "__main__":
    asyncio.run(main())
