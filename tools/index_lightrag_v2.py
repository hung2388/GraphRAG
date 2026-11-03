"""tools/index_lightrag_v2.py — Index 19 tai lieu eval vao LightRAG tao storage moi (lightrag_storage_v2).
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
sys.path.insert(0, str(ROOT))

import config
from light.rag_light import build_rag

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LightRAG_Index_v2")

EVAL_DIR = ROOT / "data" / "eval_corpus_19"
TARGET_STORAGE_DIR = ROOT / "light" / "lightrag_storage_v2"


async def main():
    TARGET_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Target LightRAG storage: {TARGET_STORAGE_DIR}")

    files = sorted([f for f in EVAL_DIR.glob("*.txt") if f.is_file()])
    logger.info(f"Tim thay {len(files)} files trong {EVAL_DIR}")

    # Build rag pointing to TARGET_STORAGE_DIR (think=False for indexing to avoid long CoT during extraction)
    rag = await build_rag(working_dir=TARGET_STORAGE_DIR, think=False)

    logger.info(f"Bat dau insert {len(files)} documents vao LightRAG storage v2...")
    start_t = time.perf_counter()

    for idx, fp in enumerate(files, 1):
        content = fp.read_text(encoding="utf-8").strip()
        t0 = time.perf_counter()
        logger.info(f"[{idx}/{len(files)}] Inserting {fp.name:35s} ({len(content)} chars)...")
        await rag.ainsert(content)
        t_doc = time.perf_counter() - t0
        logger.info(f"  --> Xong {fp.name} trong {t_doc:.2f}s")

    elapsed = time.perf_counter() - start_t
    logger.info(f"=== HOAN TAT INDEXING LIGHTRAG V2 TRONG {elapsed:.2f}s ({elapsed/60:.2f} phut) ===")


if __name__ == "__main__":
    asyncio.run(main())
