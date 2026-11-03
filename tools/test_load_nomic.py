import os
import sys
from pathlib import Path

# Set HF_HOME to Drive D
os.environ["HF_HOME"] = "D:/Lab/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "D:/Lab/.cache/huggingface/hub"

print(f"Python: {sys.executable}")
import torch
print(f"PyTorch CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

from sentence_transformers import SentenceTransformer

model_name = "nomic-ai/nomic-embed-text-v1.5"
print(f"Loading {model_name}...")
model = SentenceTransformer(model_name, trust_remote_code=True, device="cuda" if torch.cuda.is_available() else "cpu")
print("Model successfully loaded!")
print("Model modules:", [type(m).__name__ for m in model])

# Quick embedding test
queries = ["search_query: Ai là hoàng đế thứ 11 của nhà Trần?"]
docs = ["search_document: Trần Thuận Tông là hoàng đế thứ 11 của nhà Trần."]
q_emb = model.encode(queries)
d_emb = model.encode(docs)
import numpy as np
sim = np.dot(q_emb[0], d_emb[0]) / (np.linalg.norm(q_emb[0]) * np.linalg.norm(d_emb[0]))
print(f"Cosine similarity: {sim:.4f}")
