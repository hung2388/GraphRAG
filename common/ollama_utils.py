import asyncio
import logging
import os
import numpy as np
import requests

from config import OLLAMA_HOST, EMBED_MODEL, EMBED_DIM

logger = logging.getLogger("OllamaUtils")


class TokenSpan:
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

    def split_by_token_limit(self, content, max_tokens, overlap_tokens=0):
        tokens = self.encode(content)
        if max_tokens <= 0 or not tokens:
            return [TokenSpan(content, len(tokens))]
        step = max(1, max_tokens - overlap_tokens)
        spans, i = [], 0
        while i < len(tokens):
            piece = tokens[i:i + max_tokens]
            spans.append(TokenSpan(self.decode(piece), len(piece)))
            if i + max_tokens >= len(tokens):
                break
            i += step
        return spans

    def truncate_by_token_limit(self, content, max_tokens):
        tokens = self.encode(content)
        return content if len(tokens) <= max_tokens else self.decode(tokens[:max_tokens])


def _embed_one(text: str) -> list[float]:
    import time
    # Truncate safe text if overly long (Nomic max 2048 tokens ~ 6000 chars)
    safe_text = text[:5000] if len(text) > 5000 else text
    last_err = None
    embed_host = os.getenv("LIGHTRAG_EMBED_HOST", OLLAMA_HOST)
    embed_model = os.getenv("LIGHTRAG_EMBED_MODEL", EMBED_MODEL)
    for attempt in range(4):
        try:
            r = requests.post(
                f"{embed_host}/api/embeddings",
                json={"model": embed_model, "prompt": safe_text},
                timeout=60,
            )
            r.raise_for_status()
            vec = r.json().get("embedding", [])
            if len(vec) == EMBED_DIM:
                return vec
            elif len(vec) > 0:
                return vec[:EMBED_DIM] if len(vec) > EMBED_DIM else vec + [0.0] * (EMBED_DIM - len(vec))
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))

    logger.warning(f"Embedding failed after 4 retries ({last_err}), returning zero vector")
    return [0.0] * EMBED_DIM


async def custom_ollama_embed(texts: list[str]) -> np.ndarray:
    sem = asyncio.Semaphore(8)

    async def _safe_embed(t: str):
        async with sem:
            return await asyncio.to_thread(_embed_one, t)

    vecs = await asyncio.gather(*(_safe_embed(t) for t in texts))
    return np.array(vecs, dtype=np.float32)

