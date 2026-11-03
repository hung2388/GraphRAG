"""step6_manual_inspection.py — Kiem tra thu cong 30-50 mau ngau nhien truoc khi train.

Doc va danh gia chat luong:
1. Embedding training pairs (cau hoi, chunk dung, kieu cau hoi, do trung lap).
2. Embedding hard negative triplets (rank nomic goc, score, cac hard negatives).
3. Extraction training triples (entities, types, descriptions, relationships, strengths).
4. Extraction gold validation set (danh sach thuc the va quan he chuan dung de do).

Xuat bao cao Markdown vao data/adapter_data/inspection_report.md
"""

import json
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "adapter_data"

PAIRS_FILE = DATA_DIR / "embedding_train_pairs_raw.jsonl"
TRIPLETS_FILE = DATA_DIR / "embedding_train_triplets.jsonl"
EXTRACT_TRAIN_FILE = DATA_DIR / "extraction_train_triples.jsonl"
EXTRACT_GOLD_FILE = DATA_DIR / "extraction_val_gold.jsonl"
REPORT_MD = DATA_DIR / "inspection_report.md"


def load_jsonl(path: Path):
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    random.seed(42)
    md_lines = ["# Báo cáo Kiểm Tra Thủ Công Dữ Liệu Train & Eval Adapter\n"]
    md_lines.append(f"> Thời gian tạo: 2026-10-22. Seed: 42. Quy tắc: Lấy mẫu ngẫu nhiên 30-50 mẫu.\n\n")

    # 1. Inspection: Embedding Query Pairs
    pairs = load_jsonl(PAIRS_FILE)
    md_lines.append(f"## 1. Mẫu Cặp (Câu hỏi - Chunk đúng) cho Adapter Embedding\n")
    md_lines.append(f"- **Tổng số câu hỏi hiện có**: {len(pairs)}\n\n")
    if pairs:
        sample_pairs = random.sample(pairs, min(35, len(pairs)))
        for i, p in enumerate(sample_pairs, 1):
            md_lines.append(f"### Mẫu #{i} — [{p.get('question_type', 'unknown')}] {p.get('pos_title', '')}\n")
            md_lines.append(f"- **Query ID**: `{p.get('query_id')}`\n")
            md_lines.append(f"- **Câu hỏi**: **{p.get('question')}**\n")
            md_lines.append(f"- **Loại**: `{p.get('question_type')}` | **Max verbatim overlap**: `{p.get('max_verbatim_overlap')}` từ\n")
            snippet = p.get('pos_text', '').replace('\n', ' ')[:250]
            md_lines.append(f"- **Chunk đúng (trích)**: *\"{snippet}...\"*\n\n")
    else:
        md_lines.append("*Chưa có dữ liệu sinh câu hỏi.*\n\n")

    # 2. Inspection: Hard Negative Triplets
    triplets = load_jsonl(TRIPLETS_FILE)
    md_lines.append(f"## 2. Mẫu Hard Negative Triplets cho Adapter Embedding\n")
    md_lines.append(f"- **Tổng số triplets**: {len(triplets)}\n\n")
    if triplets:
        sample_triplets = random.sample(triplets, min(30, len(triplets)))
        for i, t in enumerate(sample_triplets, 1):
            md_lines.append(f"### Triplet #{i} — {t.get('pos_title', '')}\n")
            md_lines.append(f"- **Câu hỏi**: **{t.get('question')}**\n")
            md_lines.append(f"- **Nomic Gốc**: Rank **#{t.get('nomic_pos_rank')}** (Score: `{t.get('nomic_pos_score')}`)\n")
            md_lines.append(f"- **Hard Negatives được đào**:\n")
            for neg in t.get("hard_negatives", []):
                md_lines.append(f"  - Rank #{neg.get('rank')} (Score: `{neg.get('score')}`) | Bài: *{neg.get('title')}* | Trích: \"{neg.get('text', '')[:120]}...\"\n")
            md_lines.append("\n")
    else:
        md_lines.append("*Chưa có dữ liệu đào hard negative.*\n\n")

    # 3. Inspection: Extraction Training Triples
    extractions = load_jsonl(EXTRACT_TRAIN_FILE)
    md_lines.append(f"## 3. Mẫu Nhãn Triples cho Adapter Extraction (Tập Train)\n")
    md_lines.append(f"- **Tổng số chunks đã extract**: {len(extractions)}\n\n")
    if extractions:
        sample_exts = random.sample(extractions, min(30, len(extractions)))
        for i, ex in enumerate(sample_exts, 1):
            md_lines.append(f"### Extraction #{i} — Chunk `{ex.get('chunk_id')[:8]}` ({ex.get('title')})\n")
            md_lines.append(f"- **Thực thể ({len(ex.get('entities', []))})**:\n")
            for ent in ex.get("entities", [])[:6]:
                md_lines.append(f"  - `({ent.get('type')})` **{ent.get('name')}**: {ent.get('description')[:100]}...\n")
            md_lines.append(f"- **Quan hệ ({len(ex.get('relationships', []))})**:\n")
            for rel in ex.get("relationships", [])[:5]:
                md_lines.append(f"  - **{rel.get('src')}** -> **{rel.get('tgt')}** (Strength: `{rel.get('strength')}`): {rel.get('description')[:100]}...\n")
            md_lines.append("\n")
    else:
        md_lines.append("*Chưa có dữ liệu extraction train.*\n\n")

    # 4. Inspection: Extraction Gold Validation
    golds = load_jsonl(EXTRACT_GOLD_FILE)
    md_lines.append(f"## 4. Mẫu Gold Triples (Tập Validation — Chỉ Dùng Để Đo)\n")
    md_lines.append(f"- **Tổng số gold validation chunks**: {len(golds)}\n\n")
    if golds:
        sample_golds = random.sample(golds, min(20, len(golds)))
        for i, g in enumerate(sample_golds, 1):
            md_lines.append(f"### Gold #{i} — Chunk `{g.get('chunk_id')[:8]}` ({g.get('title')})\n")
            md_lines.append(f"- **Thực thể chuẩn ({len(g.get('gold_entities', []))})**:\n")
            for ent in g.get("gold_entities", [])[:5]:
                md_lines.append(f"  - `({ent.get('type')})` **{ent.get('name')}**: {ent.get('description')[:100]}...\n")
            md_lines.append(f"- **Quan hệ chuẩn ({len(g.get('gold_relationships', []))})**:\n")
            for rel in g.get("gold_relationships", [])[:4]:
                md_lines.append(f"  - **{rel.get('src')}** -> **{rel.get('tgt')}** (Strength: `{rel.get('strength')}`): {rel.get('description')[:100]}...\n")
            md_lines.append("\n")
    else:
        md_lines.append("*Chưa có dữ liệu gold validation.*\n\n")

    REPORT_MD.write_text("".join(md_lines), encoding="utf-8")
    print(f"[+] Da tao bao cao kiem tra thu cong: {REPORT_MD}")


if __name__ == "__main__":
    main()
