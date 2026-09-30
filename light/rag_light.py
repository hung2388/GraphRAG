"""
rag_light.py — LightRAG chạy với Ollama local, độc lập hoàn toàn
với pipeline Neo4j cũ (không dùng chung dữ liệu).
"""

import os
import asyncio
import requests

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./lightrag_storage"
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "qwen3:8b"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768

os.makedirs(WORKING_DIR, exist_ok=True)


# ============================================================
# TOKENIZER TỰ VIẾT (thay thế tiktoken vì cần tương thích tiếng Việt
# và không phụ thuộc mạng ngoài để tải encoding)
# ============================================================

class TokenSpan:
    """LightRAG nội bộ đôi khi cần .token_count trên phần tử trả về
    từ split_by_token_limit, không chỉ chuỗi thô."""
    __slots__ = ("content", "token_count")

    def __init__(self, content: str, token_count: int):
        self.content = content
        self.token_count = token_count

    def __str__(self):
        return self.content

    def __len__(self):
        return self.token_count


class SimpleTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(c) for c in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) if 0 <= t <= 1114111 else "?" for t in tokens)

    def split_by_token_limit(self, content: str, max_tokens: int, overlap_tokens: int = 0):
        tokens = self.encode(content)
        if max_tokens <= 0 or not tokens:
            return [TokenSpan(content, len(tokens))]

        step = max(1, max_tokens - overlap_tokens)
        spans = []
        i = 0
        while i < len(tokens):
            piece = tokens[i:i + max_tokens]
            spans.append(TokenSpan(self.decode(piece), len(piece)))
            if i + max_tokens >= len(tokens):
                break
            i += step
        return spans

    def truncate_by_token_limit(self, content: str, max_tokens: int) -> str:
        """Cắt content về tối đa max_tokens 'token', trả về 1 chuỗi duy nhất
        (không chia nhiều đoạn như split_by_token_limit)."""
        tokens = self.encode(content)
        if len(tokens) <= max_tokens:
            return content
        return self.decode(tokens[:max_tokens])


# ============================================================
# EMBEDDING TỰ VIẾT (bypass ollama_embed gốc — tránh bug reshape
# cứng chiều 1024 dù model thật trả về 768)
# ============================================================

def custom_ollama_embed(texts):
    embeddings = []
    for text in texts:
        r = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        r.raise_for_status()
        vec = r.json()["embedding"]
        if len(vec) != EMBED_DIM:
            raise ValueError(
                f"Model {EMBED_MODEL} trả về {len(vec)} chiều, "
                f"nhưng EMBED_DIM đang khai là {EMBED_DIM}. Sửa EMBED_DIM cho khớp."
            )
        embeddings.append(vec)
    return embeddings


# ============================================================
# BUILD RAG
# ============================================================

async def build_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={
            "host": OLLAMA_HOST,
            "options": {"num_ctx": 8192},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            max_token_size=8192,
            func=custom_ollama_embed,
        ),
        tokenizer=SimpleTokenizer(),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def lightrag_answer(question, rag, mode="hybrid"):
    result = await rag.aquery(question, param=QueryParam(mode=mode))
    return result


# ============================================================
# TEST NHANH
# ============================================================

async def query_test(rag):
    modes = ["naive", "local", "global", "hybrid"]
    question = "Quang Trung sinh năm nào và tên khai sinh là gì?"
    for mode in modes:
        result = await rag.aquery(question, param=QueryParam(mode=mode))
        print(f"\n=== Mode: {mode} ===")
        print(result)


async def main():
    rag = await build_rag()

    for path in ["data/input/quang_trung.txt", "data/input/src.txt"]:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        await rag.ainsert(content)
        print(f"Đã nạp: {path}")

    print("Xây KG xong.")

    print("\n--- BẮT ĐẦU TEST QUERY ---")
    await query_test(rag)


if __name__ == "__main__":
    asyncio.run(main())