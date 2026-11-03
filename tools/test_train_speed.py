import os
import sys
import time

os.environ["HF_HOME"] = "D:/Lab/.cache/huggingface"

import torch
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model

device = "cpu"
print(f"Device: {device}")

st_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device=device)
st_model.max_seq_length = 256  # 256 tokens is plenty for contrastive anchor and documents

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["Wqkv", "out_proj"],
    lora_dropout=0.05,
    bias="none",
)
st_model[0].auto_model = get_peft_model(st_model[0].auto_model, peft_config)

optimizer = torch.optim.AdamW(st_model[0].auto_model.parameters(), lr=2e-4)

# 4 samples
queries = ["search_query: Ai là hoàng đế đầu tiên của nhà Hậu Lê?", "search_query: Chiến thắng Bạch Đằng năm nào?"]
positives = ["search_document: Lê Lợi là vị hoàng đế đầu tiên của nhà Hậu Lê.", "search_document: Chiến thắng Bạch Đằng diễn ra vào năm 1288."]
negatives = ["search_document: Trần Cảnh là hoàng đế đầu tiên nhà Trần.", "search_document: Trận Chi Lăng Xương Giang diễn ra năm 1427."]

t0 = time.perf_counter()
feat_q = st_model.tokenize(queries)
feat_pos = st_model.tokenize(positives)
feat_neg = st_model.tokenize(negatives)

emb_q = st_model(feat_q)["sentence_embedding"]
emb_pos = st_model(feat_pos)["sentence_embedding"]
emb_neg = st_model(feat_neg)["sentence_embedding"]

emb_q = torch.nn.functional.normalize(emb_q, p=2, dim=1)
emb_pos = torch.nn.functional.normalize(emb_pos, p=2, dim=1)
emb_neg = torch.nn.functional.normalize(emb_neg, p=2, dim=1)

sim_in_batch = torch.matmul(emb_q, emb_pos.T) / 0.05
sim_hard_neg = torch.sum(emb_q * emb_neg, dim=1, keepdim=True) / 0.05
logits = torch.cat([sim_in_batch, sim_hard_neg], dim=1)
labels = torch.arange(len(queries), dtype=torch.long)

loss = torch.nn.functional.cross_entropy(logits, labels)
loss.backward()
optimizer.step()
step_time = time.perf_counter() - t0

print(f"Loss: {loss.item():.4f}")
print(f"Step completed in {step_time:.3f}s on CPU!")
