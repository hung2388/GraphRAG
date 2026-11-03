"""tools/build_eval_corpus_62.py — Tao tap kho 19 tai lieu va cat thanh dung 62 chunks theo chuan HiRAG.
"""

import json
import shutil
import sys
from hashlib import md5
from pathlib import Path
import tiktoken

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from step1_split_and_chunk import SeparatorSplitter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVAL_DIR = ROOT / "data" / "eval_corpus_19"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# 1. 19 tap tin theo thu tu co dinh
ORDERED_FILES = [
    # 3 bai eval goc
    "src.txt",
    "quang_trung.txt",
    "love.txt",
    # 7 bai lich su validation
    "tran_thu_do.txt",
    "ly_thuong_kiet.txt",
    "tran_quang_khai.txt",
    "binh_ngo_dai_cao.txt",
    "ho_dynasty.txt",
    "lady_trieu.txt",
    "nguyen_phuc_nguyen.txt",
    # 9 bai hat validation
    "thriller_song.txt",
    "a_sky_full_of_stars.txt",
    "love_story_taylor_swift_song.txt",
    "single_ladies_put_a_ring_on_it.txt",
    "just_the_way_you_are_bruno_mars_song.txt",
    "new_rules.txt",
    "water_under_the_bridge.txt",
    "when_we_were_young_adele_song.txt",
    "mine_taylor_swift_song.txt",
]

enc = tiktoken.encoding_for_model("gpt-4o")
splitter = SeparatorSplitter(
    separators=[enc.encode("\n\n"), enc.encode("\n"), enc.encode(" ")],
    chunk_size=1200,
    chunk_overlap=100,
)

all_chunks = []
chunk_idx = 0

for fname in ORDERED_FILES:
    fpath = EVAL_DIR / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Khong tim thay file: {fpath}")
    text = fpath.read_text(encoding="utf-8").strip()
    tokens = enc.encode(text)
    splits = splitter.split_tokens(tokens)
    doc_slug = fname.replace(".txt", "")

    for order_idx, split_tokens in enumerate(splits):
        chunk_text = enc.decode(split_tokens).strip()
        h = md5(chunk_text.encode("utf-8")).hexdigest()[:8]
        cid = f"eval-chunk-{chunk_idx:02d}-{doc_slug}-{order_idx}-{h}"
        chunk_rec = {
            "global_chunk_index": chunk_idx,
            "chunk_id": cid,
            "doc": doc_slug,
            "file_name": fname,
            "chunk_order_index": order_idx,
            "tokens": len(split_tokens),
            "text": chunk_text,
        }
        all_chunks.append(chunk_rec)
        chunk_idx += 1

out_path = EVAL_DIR / "chunks_eval_62.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for c in all_chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print(f"Tong so chunks tao duoc: {len(all_chunks)}")
print(f"Da ghi vao: {out_path}")

by_doc = {}
for c in all_chunks:
    by_doc[c["doc"]] = by_doc.get(c["doc"], 0) + 1

for doc, count in by_doc.items():
    print(f"  {doc:35s}: {count} chunks")
