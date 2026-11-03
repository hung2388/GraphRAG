"""tools/reindex_cache_with_adapter.py — Re-embed vector stores (chunks, entities, relations)
in HiRAG / LightRAG caches using the fine-tuned LoRA merged embedding model.

Preserves 100% of the graph structure, community reports, and LLM extractions,
updating only the vector representation space to match the adapted model.
"""

import argparse
import base64
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np

# Set Hugging Face cache to Drive D
os.environ["HF_HOME"] = "D:/Lab/.cache/huggingface"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ReindexCache")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "nomic_adapted_merged"


def array_to_buffer_string(array: np.ndarray) -> str:
    return base64.b64encode(array.astype(np.float32).tobytes()).decode("ascii")


def normalize(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    norms[norms == 0] = 1e-12
    return a / norms


def reindex_chunks(dst_dir: Path, model, batch_size: int = 8):
    vdb_file = dst_dir / "vdb_chunks.json"
    kv_file = dst_dir / "kv_store_text_chunks.json"
    if not vdb_file.exists() or not kv_file.exists():
        logger.warning(f"Bo qua chunks: khong tim thay {vdb_file} hoac {kv_file}")
        return

    logger.info("Dang re-embed vdb_chunks...")
    with open(vdb_file, "r", encoding="utf-8") as f:
        vdb = json.load(f)
    with open(kv_file, "r", encoding="utf-8") as f:
        kv = json.load(f)

    data_items = vdb.get("data", [])
    logger.info(f"So luong chunks can re-embed: {len(data_items)}")

    texts = []
    for item in data_items:
        chunk_id = item.get("__id__")
        chunk_info = kv.get(chunk_id, {})
        content = chunk_info.get("content", "")
        # Add Nomic search document prefix
        texts.append(f"search_document: {content.strip()}")

    t0 = time.perf_counter()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    embeddings = normalize(np.array(embeddings, dtype=np.float32))
    logger.info(f"Re-embedded {len(texts)} chunks in {time.perf_counter() - t0:.2f}s")

    vdb["matrix"] = array_to_buffer_string(embeddings)
    vdb["embedding_dim"] = embeddings.shape[1]

    with open(vdb_file, "w", encoding="utf-8") as f:
        json.dump(vdb, f)
    logger.info(f"Da luu {vdb_file} voi shape {embeddings.shape}")


def reindex_entities(dst_dir: Path, model, batch_size: int = 16):
    vdb_file = dst_dir / "vdb_entities.json"
    if not vdb_file.exists():
        logger.warning(f"Bo qua entities: khong tim thay {vdb_file}")
        return

    logger.info("Dang re-embed vdb_entities...")
    with open(vdb_file, "r", encoding="utf-8") as f:
        vdb = json.load(f)

    data_items = vdb.get("data", [])
    logger.info(f"So luong entities can re-embed: {len(data_items)}")

    # Load graphml if available to get descriptions
    descriptions = {}
    graphml_file = dst_dir / "graph_chunk_entity_relation.graphml"
    if graphml_file.exists():
        try:
            import networkx as nx
            G = nx.read_graphml(graphml_file)
            for node, attrs in G.nodes(data=True):
                desc = attrs.get("description", "")
                descriptions[node] = desc
                # also map cleaned / unquoted node name
                descriptions[node.strip('"').strip("'")] = desc
        except Exception as e:
            logger.warning(f"Khong the doc graphml de lay entity descriptions: {e}")

    texts = []
    for item in data_items:
        entity_name = item.get("entity_name", item.get("__id__", ""))
        clean_name = entity_name.strip('"').strip("'")
        desc = descriptions.get(clean_name, descriptions.get(entity_name, ""))
        text = f"{clean_name}: {desc}" if desc else clean_name
        texts.append(f"search_document: {text.strip()}")

    t0 = time.perf_counter()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    embeddings = normalize(np.array(embeddings, dtype=np.float32))
    logger.info(f"Re-embedded {len(texts)} entities in {time.perf_counter() - t0:.2f}s")

    vdb["matrix"] = array_to_buffer_string(embeddings)
    vdb["embedding_dim"] = embeddings.shape[1]

    with open(vdb_file, "w", encoding="utf-8") as f:
        json.dump(vdb, f)
    logger.info(f"Da luu {vdb_file} voi shape {embeddings.shape}")


def reindex_relationships(dst_dir: Path, model, batch_size: int = 16):
    vdb_file = dst_dir / "vdb_relationships.json"
    if not vdb_file.exists():
        return

    logger.info("Dang re-embed vdb_relationships...")
    with open(vdb_file, "r", encoding="utf-8") as f:
        vdb = json.load(f)

    data_items = vdb.get("data", [])
    logger.info(f"So luong relationships can re-embed: {len(data_items)}")

    graphml_file = dst_dir / "graph_chunk_entity_relation.graphml"
    edge_descriptions = {}
    if graphml_file.exists():
        try:
            import networkx as nx
            G = nx.read_graphml(graphml_file)
            for u, v, attrs in G.edges(data=True):
                desc = attrs.get("description", "")
                edge_descriptions[(u, v)] = desc
                edge_descriptions[(u.strip('"'), v.strip('"'))] = desc
        except Exception as e:
            logger.warning(f"Khong the doc graphml de lay edge descriptions: {e}")

    texts = []
    for item in data_items:
        src = item.get("src_id", "")
        tgt = item.get("tgt_id", "")
        desc = item.get("description", "")
        if not desc:
            desc = edge_descriptions.get((src, tgt), edge_descriptions.get((src.strip('"'), tgt.strip('"')), ""))
        text = f"{src} -> {tgt}: {desc}" if desc else f"{src} -> {tgt}"
        texts.append(f"search_document: {text.strip()}")

    t0 = time.perf_counter()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    embeddings = normalize(np.array(embeddings, dtype=np.float32))
    logger.info(f"Re-embedded {len(texts)} relationships in {time.perf_counter() - t0:.2f}s")

    vdb["matrix"] = array_to_buffer_string(embeddings)
    vdb["embedding_dim"] = embeddings.shape[1]

    with open(vdb_file, "w", encoding="utf-8") as f:
        json.dump(vdb, f)
    logger.info(f"Da luu {vdb_file} voi shape {embeddings.shape}")


def main():
    parser = argparse.ArgumentParser(description="Re-index vector stores with adapted embedding model")
    parser.add_argument("--src-dir", type=Path, required=True, help="Nguon cache cu (pre-adapter)")
    parser.add_argument("--dst-dir", type=Path, required=True, help="Dich cache moi (post-adapter)")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path toi adapted merged model")
    parser.add_argument("--device", type=str, default="cpu", help="Device cho embedding model (cpu hoac cuda)")
    args = parser.parse_args()

    assert args.src_dir.exists(), f"Khong tim thay src-dir: {args.src_dir}"
    assert args.model_path.exists(), f"Khong tim thay model-path: {args.model_path}"

    logger.info(f"=== BAT DAU RE-INDEX CACHE VOI ADAPTED MODEL ===")
    logger.info(f"Source: {args.src_dir}")
    logger.info(f"Destination: {args.dst_dir}")
    logger.info(f"Model: {args.model_path} on {args.device}")

    # Copy src to dst if dst does not exist or overwrite
    if args.dst_dir.exists():
        logger.info(f"Thu muc {args.dst_dir} da ton tai, update vector stores...")
    else:
        logger.info(f"Sao chep toan bo {args.src_dir} sang {args.dst_dir}...")
        shutil.copytree(args.src_dir, args.dst_dir)

    from sentence_transformers import SentenceTransformer
    logger.info("Dang load adapted SentenceTransformer...")
    model = SentenceTransformer(str(args.model_path), trust_remote_code=True, device=args.device)

    reindex_chunks(args.dst_dir, model)
    reindex_entities(args.dst_dir, model)
    reindex_relationships(args.dst_dir, model)

    logger.info("=== HOAN TAT RE-INDEX CACHE THANH CONG! ===")


if __name__ == "__main__":
    main()
