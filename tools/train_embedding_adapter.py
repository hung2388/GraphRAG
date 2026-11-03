"""tools/train_embedding_adapter.py — Native PyTorch LoRA Training for nomic-embed-text-v1.5

Trains a LoRA adapter (r=16, alpha=32) on 539 contrastive triplets
from data/adapter_data/embedding_train_triplets.jsonl using InfoNCE contrastive loss
with hard negatives and in-batch negatives.

Exports:
- models/nomic_lora_v1 (LoRA weights)
- models/nomic_adapted_merged (fully merged standalone SentenceTransformer model)
"""

import json
import logging
import math
import os
import sys
import time
from pathlib import Path

# Redirect HuggingFace cache to Drive D
os.environ["HF_HOME"] = "D:/Lab/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "D:/Lab/.cache/huggingface/hub"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainEmbeddingAdapter")

ROOT = Path(__file__).resolve().parent.parent
TRIPLETS_FILE = ROOT / "data" / "adapter_data" / "embedding_train_triplets.jsonl"
OUTPUT_LORA_DIR = ROOT / "models" / "nomic_lora_v1"
OUTPUT_MERGED_DIR = ROOT / "models" / "nomic_adapted_merged"

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model


class ContrastiveTripletDataset(Dataset):
    def __init__(self, jsonl_path: Path):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                q = f"search_query: {item['question'].strip()}"
                pos = f"search_document: {item['pos_text'].strip()}"
                hard_negs = item.get("hard_negatives", [])
                neg = f"search_document: {hard_negs[0]['text'].strip()}" if hard_negs else pos
                self.data.append((q, pos, neg))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    queries = [item[0] for item in batch]
    positives = [item[1] for item in batch]
    negatives = [item[2] for item in batch]
    return queries, positives, negatives


def compute_infonce_loss(st_model, queries, positives, negatives, device, temperature=0.05):
    # Tokenize each set
    feat_q = st_model.tokenize(queries)
    feat_pos = st_model.tokenize(positives)
    feat_neg = st_model.tokenize(negatives)

    for feat in [feat_q, feat_pos, feat_neg]:
        for k in feat:
            feat[k] = feat[k].to(device)

    autocast_device = "cuda" if device == "cuda" else "cpu"
    autocast_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    with torch.amp.autocast(device_type=autocast_device, dtype=autocast_dtype):
        emb_q = st_model(feat_q)["sentence_embedding"]
        emb_pos = st_model(feat_pos)["sentence_embedding"]
        emb_neg = st_model(feat_neg)["sentence_embedding"]

        # Normalize to unit sphere
        emb_q = F.normalize(emb_q, p=2, dim=1)
        emb_pos = F.normalize(emb_pos, p=2, dim=1)
        emb_neg = F.normalize(emb_neg, p=2, dim=1)

        batch_size = emb_q.size(0)

        # Similarity matrix between queries and all positive documents in batch: (B, B)
        sim_in_batch = torch.matmul(emb_q, emb_pos.T) / temperature

        # Similarity between queries and their hard negatives: (B, 1)
        sim_hard_neg = torch.sum(emb_q * emb_neg, dim=1, keepdim=True) / temperature

        # Combine in-batch candidates with hard negative: (B, B + 1)
        logits = torch.cat([sim_in_batch, sim_hard_neg], dim=1)

        # True positive for query i is at index i
        labels = torch.arange(batch_size, dtype=torch.long, device=device)

        loss = F.cross_entropy(logits, labels)
    return loss



def evaluate_sample(st_model, test_queries, test_docs, device):
    st_model.eval()
    with torch.no_grad():
        feat_q = st_model.tokenize(test_queries)
        feat_d = st_model.tokenize(test_docs)
        for feat in [feat_q, feat_d]:
            for k in feat:
                feat[k] = feat[k].to(device)
        eq = F.normalize(st_model(feat_q)["sentence_embedding"], p=2, dim=1)
        ed = F.normalize(st_model(feat_d)["sentence_embedding"], p=2, dim=1)
        sims = torch.matmul(eq, ed.T).cpu().numpy()
    return sims


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Target device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load dataset
    dataset = ContrastiveTripletDataset(TRIPLETS_FILE)
    logger.info(f"Loaded {len(dataset)} training triplets from {TRIPLETS_FILE}")
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 2. Load base model
    model_name = "nomic-ai/nomic-embed-text-v1.5"
    logger.info(f"Loading base model: {model_name}...")
    st_model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    st_model.max_seq_length = 512
    logger.info("Configured st_model.max_seq_length = 512 for optimal VRAM efficiency")

    # Baseline similarity check on sample test
    test_q = ["search_query: Ai là hoàng đế thứ 11 của triều đại nhà Trần?"]
    test_pos = ["search_document: Trần Thuận Tông (1378 – 1399) là hoàng đế thứ 11 của triều đại nhà Trần trong lịch sử Việt Nam."]
    test_neg = ["search_document: Lý Thái Tổ là vị hoàng đế sáng lập ra triều đại nhà Lý trong lịch sử Việt Nam."]
    base_pos_sim = evaluate_sample(st_model, test_q, test_pos, device)[0][0]
    base_neg_sim = evaluate_sample(st_model, test_q, test_neg, device)[0][0]
    logger.info(f"Pre-adapter Baseline Similarity: Pos={base_pos_sim:.4f}, Neg={base_neg_sim:.4f}, Margin={base_pos_sim - base_neg_sim:.4f}")

    # 3. Apply LoRA
    logger.info("Applying LoRA (r=16, alpha=32, target_modules=['Wqkv', 'out_proj'])...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["Wqkv", "out_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    st_model[0].auto_model = get_peft_model(st_model[0].auto_model, peft_config)
    st_model[0].auto_model.print_trainable_parameters()

    # 4. Optimizer & Scheduler
    epochs = 3
    total_steps = len(dataloader) * epochs
    warmup_steps = int(total_steps * 0.1)
    lr = 2e-4
    optimizer = torch.optim.AdamW(st_model[0].auto_model.parameters(), lr=lr, weight_decay=0.01)

    def get_lr(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

    # 5. Training loop
    logger.info(f"Starting LoRA training: {epochs} epochs ({total_steps} total steps, warmup={warmup_steps})...")
    st_model[0].auto_model.train()
    global_step = 0
    t0 = time.perf_counter()

    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for step, (queries, positives, negatives) in enumerate(dataloader, 1):
            global_step += 1
            optimizer.zero_grad()
            loss = compute_infonce_loss(st_model, queries, positives, negatives, device)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(st_model[0].auto_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(st_model[0].auto_model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            epoch_loss += loss.item()
            if step % 20 == 0 or step == len(dataloader):
                cur_lr = scheduler.get_last_lr()[0] * lr
                logger.info(f"Epoch [{epoch}/{epochs}] Step [{step}/{len(dataloader)}] - Loss: {loss.item():.4f} - LR: {cur_lr:.6f}")


        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"--> Epoch {epoch} Finished | Avg Loss: {avg_loss:.4f}")

    elapsed = time.perf_counter() - t0
    logger.info(f"=== LoRA Training Completed in {elapsed:.2f}s ({elapsed/60:.2f} min) ===")

    # 6. Save LoRA Adapter Weights
    OUTPUT_LORA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving LoRA adapter to {OUTPUT_LORA_DIR}...")
    st_model[0].auto_model.save_pretrained(str(OUTPUT_LORA_DIR))

    # 7. Merge LoRA weights into base model
    OUTPUT_MERGED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Merging LoRA weights into base model and saving to {OUTPUT_MERGED_DIR}...")
    merged_model = st_model[0].auto_model.merge_and_unload()
    st_model[0].auto_model = merged_model
    st_model.save(str(OUTPUT_MERGED_DIR))
    logger.info(f"Merged model saved successfully to {OUTPUT_MERGED_DIR}")

    # 8. Post-adapter verification on test sample
    post_pos_sim = evaluate_sample(st_model, test_q, test_pos, device)[0][0]
    post_neg_sim = evaluate_sample(st_model, test_q, test_neg, device)[0][0]
    logger.info(f"Post-adapter Similarity: Pos={post_pos_sim:.4f}, Neg={post_neg_sim:.4f}, Margin={post_pos_sim - post_neg_sim:.4f}")
    delta_margin = (post_pos_sim - post_neg_sim) - (base_pos_sim - base_neg_sim)
    logger.info(f"Delta Margin Improvement: +{delta_margin:.4f}")
    logger.info("=== HOAN TAT TRAINING & EXPORT EMBEDDING ADAPTER! ===")


if __name__ == "__main__":
    main()
