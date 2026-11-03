"""tools/serve_adapted_embeddings.py — High-speed local embedding server on port 11435.
Serves the fine-tuned LoRA merged embedding model (models/nomic_adapted_merged).
Supports both OpenAI API (/v1/embeddings) and Ollama API (/api/embeddings, /api/embed).
"""

import argparse
import json
import logging
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
import numpy as np

os.environ["HF_HOME"] = "D:/Lab/.cache/huggingface"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AdaptedEmbedServer")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "nomic_adapted_merged"
BASE_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

BASE_MODEL = None
ADAPTED_MODEL = None


def normalize(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    norms[norms == 0] = 1e-12
    return a / norms


def get_model(model_name: str = ""):
    global BASE_MODEL, ADAPTED_MODEL
    if "adapted" in str(model_name).lower():
        return ADAPTED_MODEL or BASE_MODEL
    return BASE_MODEL or ADAPTED_MODEL


def encode_texts(texts: list[str], is_search_query: bool = True, model_name: str = "") -> list[list[float]]:
    model = get_model(model_name)
    prefix = "search_query: " if is_search_query else "search_document: "
    prefixed_texts = []
    for t in texts:
        t_str = str(t).strip()
        if not t_str.startswith("search_query:") and not t_str.startswith("search_document:"):
            t_str = prefix + t_str
        prefixed_texts.append(t_str)

    embs = model.encode(prefixed_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    embs = normalize(np.array(embs, dtype=np.float32))
    return embs.tolist()


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class EmbeddingHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quiet standard HTTP logging
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok","model":"nomic-embed-text-adapted"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            data = json.loads(body.decode("utf-8")) if body else {}

            path = self.path.split("?")[0].rstrip("/")

            # 1. OpenAI format: /v1/embeddings
            if path == "/v1/embeddings":
                raw_input = data.get("input", "")
                if isinstance(raw_input, str):
                    texts = [raw_input]
                elif isinstance(raw_input, list):
                    texts = [str(x) for x in raw_input]
                else:
                    texts = [str(raw_input)]

                req_model = data.get("model", "")
                embeddings = encode_texts(texts, is_search_query=True, model_name=req_model)
                response = {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "embedding": emb, "index": i}
                        for i, emb in enumerate(embeddings)
                    ],
                    "model": req_model or "nomic-embed-text-adapted",
                    "usage": {"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": sum(len(t.split()) for t in texts)},
                }

            # 2. Ollama legacy: /api/embeddings
            elif path == "/api/embeddings":
                prompt = data.get("prompt", "")
                req_model = data.get("model", "")
                embeddings = encode_texts([prompt], is_search_query=True, model_name=req_model)
                response = {
                    "embedding": embeddings[0] if embeddings else []
                }

            # 3. Ollama 0.5: /api/embed
            elif path == "/api/embed":
                raw_input = data.get("input", "")
                if isinstance(raw_input, str):
                    texts = [raw_input]
                elif isinstance(raw_input, list):
                    texts = [str(x) for x in raw_input]
                else:
                    texts = [str(raw_input)]

                req_model = data.get("model", "")
                embeddings = encode_texts(texts, is_search_query=True, model_name=req_model)
                response = {
                    "model": req_model or "nomic-embed-text-adapted",
                    "embeddings": embeddings
                }

            else:
                self.send_response(404)
                self.end_headers()
                return

            resp_bytes = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)

        except Exception as e:
            logger.error(f"Loi xu ly request {self.path}: {e}")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))


def run_server(port: int = 11435, model_path: Path = DEFAULT_MODEL_PATH, device: str = "cuda"):
    global BASE_MODEL, ADAPTED_MODEL
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading adapted SentenceTransformer from {model_path} onto {device}...")
    ADAPTED_MODEL = SentenceTransformer(str(model_path), trust_remote_code=True, device=device)
    logger.info("Adapted SentenceTransformer loaded successfully!")

    logger.info(f"Loading base SentenceTransformer from {BASE_MODEL_NAME} onto {device}...")
    BASE_MODEL = SentenceTransformer(BASE_MODEL_NAME, trust_remote_code=True, device=device)
    logger.info("Base SentenceTransformer loaded successfully!")

    server = ThreadingHTTPServer(("0.0.0.0", port), EmbeddingHandler)
    logger.info(f"Dual Embedding Server listening on http://127.0.0.1:{port}")
    logger.info("Endpoints: /v1/embeddings (OpenAI), /api/embeddings (Ollama), /api/embed (Ollama 0.5)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping server...")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve adapted embeddings")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_server(port=args.port, model_path=args.model_path, device=args.device)
