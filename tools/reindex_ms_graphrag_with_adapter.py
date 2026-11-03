"""tools/reindex_ms_graphrag_with_adapter.py — Re-embed MS GraphRAG v2 LanceDB tables
using the fine-tuned LoRA merged embedding model.

Preserves 100% of the extracted graph, communities, reports, and text units,
updating only the vector embeddings in LanceDB and pointing settings.yaml to the adapter server.
"""

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

os.environ["HF_HOME"] = "D:/Lab/.cache/huggingface"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ReindexMSGraphRAG")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "nomic_adapted_merged"


def normalize(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    norms[norms == 0] = 1e-12
    return a / norms


def reindex_ms_graphrag(src_dir: Path, dst_dir: Path, model_path: Path, device: str = "cuda"):
    logger.info("=== RE-INDEXING MS GRAPHRAG V2 WITH ADAPTED EMBEDDINGS ===")
    logger.info(f"Source: {src_dir}")
    logger.info(f"Destination: {dst_dir}")
    logger.info(f"Model: {model_path} on {device}")

    if dst_dir.exists():
        logger.info(f"Dich {dst_dir} da ton tai, update tables...")
    else:
        logger.info(f"Sao chep toan bo {src_dir} sang {dst_dir}...")
        shutil.copytree(src_dir, dst_dir)

    out_dir = dst_dir / "output"
    lancedb_dir = out_dir / "lancedb"
    assert lancedb_dir.exists(), f"Khong tim thay lancedb tai {lancedb_dir}"

    from sentence_transformers import SentenceTransformer
    import lancedb

    logger.info("Dang load adapted SentenceTransformer...")
    model = SentenceTransformer(str(model_path), trust_remote_code=True, device=device)

    db = lancedb.connect(str(lancedb_dir))

    # 1. entity_description
    entities_pq = out_dir / "entities.parquet"
    if entities_pq.exists():
        logger.info("Dang re-embed entity_description.lance...")
        df_ent = pd.read_parquet(entities_pq)
        t_ent = db.open_table("entity_description")
        df_lance_ent = t_ent.to_pandas()

        # Build mapping id -> text
        id_to_text = {}
        for _, row in df_ent.iterrows():
            eid = str(row["id"])
            title = str(row.get("title", ""))
            desc = str(row.get("description", ""))
            text = f"{title}: {desc}".strip() if desc else title
            id_to_text[eid] = f"search_document: {text}"

        texts = [id_to_text.get(str(eid), f"search_document: {eid}") for eid in df_lance_ent["id"]]
        t0 = time.perf_counter()
        embs = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        embs = normalize(np.array(embs, dtype=np.float32))
        logger.info(f"Re-embedded {len(texts)} entities in {time.perf_counter() - t0:.2f}s")

        df_lance_ent["vector"] = [v.tolist() for v in embs]
        db.create_table("entity_description", df_lance_ent, mode="overwrite")
        logger.info("Da ghi de table entity_description thanh cong!")

    # 2. community_full_content
    comm_pq = out_dir / "community_reports.parquet"
    if comm_pq.exists():
        logger.info("Dang re-embed community_full_content.lance...")
        df_comm = pd.read_parquet(comm_pq)
        t_comm = db.open_table("community_full_content")
        df_lance_comm = t_comm.to_pandas()

        id_to_text = {}
        for _, row in df_comm.iterrows():
            cid = str(row["id"])
            content = str(row.get("full_content", row.get("summary", "")))
            id_to_text[cid] = f"search_document: {content.strip()}"

        texts = [id_to_text.get(str(cid), f"search_document: {cid}") for cid in df_lance_comm["id"]]
        t0 = time.perf_counter()
        embs = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        embs = normalize(np.array(embs, dtype=np.float32))
        logger.info(f"Re-embedded {len(texts)} communities in {time.perf_counter() - t0:.2f}s")

        df_lance_comm["vector"] = [v.tolist() for v in embs]
        db.create_table("community_full_content", df_lance_comm, mode="overwrite")
        logger.info("Da ghi de table community_full_content thanh cong!")

    # 3. text_unit_text
    units_pq = out_dir / "text_units.parquet"
    if units_pq.exists():
        logger.info("Dang re-embed text_unit_text.lance...")
        df_units = pd.read_parquet(units_pq)
        t_units = db.open_table("text_unit_text")
        df_lance_units = t_units.to_pandas()

        id_to_text = {}
        for _, row in df_units.iterrows():
            uid = str(row["id"])
            text = str(row.get("text", ""))
            id_to_text[uid] = f"search_document: {text.strip()}"

        texts = [id_to_text.get(str(uid), f"search_document: {uid}") for uid in df_lance_units["id"]]
        t0 = time.perf_counter()
        embs = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        embs = normalize(np.array(embs, dtype=np.float32))
        logger.info(f"Re-embedded {len(texts)} text units in {time.perf_counter() - t0:.2f}s")

        df_lance_units["vector"] = [v.tolist() for v in embs]
        db.create_table("text_unit_text", df_lance_units, mode="overwrite")
        logger.info("Da ghi de table text_unit_text thanh cong!")

    # 4. Patch settings.yaml
    settings_file = dst_dir / "settings.yaml"
    if settings_file.exists():
        with open(settings_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        emb_models = cfg.get("embedding_models", {})
        for m in emb_models.values():
            m["model_provider"] = "openai"
            m["model"] = "nomic-embed-text-adapted"
            m["api_base"] = "http://127.0.0.1:11435/v1"
            m["api_key"] = "adapted"

        with open(settings_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        logger.info("Da cap nhat settings.yaml chi sang embedding server port 11435!")

    logger.info("=== RE-INDEXING MS GRAPHRAG V2 HOAN TAT THANH CONG! ===")


def main():
    parser = argparse.ArgumentParser(description="Re-index MS GraphRAG LanceDB with adapted model")
    parser.add_argument("--src-dir", type=Path, default=ROOT / "microsoft" / "ragtest_ms_v2")
    parser.add_argument("--dst-dir", type=Path, default=ROOT / "microsoft" / "ragtest_ms_v2_adapted")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    reindex_ms_graphrag(args.src_dir, args.dst_dir, args.model_path, args.device)


if __name__ == "__main__":
    main()
