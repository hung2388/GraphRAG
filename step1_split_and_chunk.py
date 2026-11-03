"""step1_split_and_chunk.py — Chia train/val 85/15 theo bai va cat chunk dung chuan HiRAG.

Quy tac:
- Chia theo bai voi seed co dinh (42), stratified theo nhom (history, song).
- Luu danh sach bai train / val rieng biet.
- Cat chunk bang SeparatorSplitter dung logic HiRAG (chunk_size=1200, overlap=100, tiktoken gpt-4o).
- Luu thanh file JSONL: chunk_id, doc, title, group, split, chunk_order_index, tokens, text.
"""

import csv
import json
import os
import random
import re
import sys
import unicodedata
from hashlib import md5
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal

import tiktoken

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "data" / "train_docs"
OUT_DIR = ROOT / "data" / "adapter_data"
MANIFEST_FILE = DOCS_DIR / "manifest.csv"

# SeparatorSplitter tu HiRAG
class SeparatorSplitter:
    def __init__(
        self,
        separators: Optional[List[List[int]]] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = "end",
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        length_function: callable = len,
    ):
        self._separators = separators or []
        self._keep_separator = keep_separator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function

    def split_tokens(self, tokens: List[int]) -> List[List[int]]:
        splits = self._split_tokens_with_separators(tokens)
        return self._merge_splits(splits)

    def _split_tokens_with_separators(self, tokens: List[int]) -> List[List[int]]:
        splits = []
        current_split = []
        i = 0
        while i < len(tokens):
            separator_found = False
            for separator in self._separators:
                if tokens[i:i+len(separator)] == separator:
                    if self._keep_separator in [True, "end"]:
                        current_split.extend(separator)
                    if current_split:
                        splits.append(current_split)
                        current_split = []
                    if self._keep_separator == "start":
                        current_split.extend(separator)
                    i += len(separator)
                    separator_found = True
                    break
            if not separator_found:
                current_split.append(tokens[i])
                i += 1
        if current_split:
            splits.append(current_split)
        return [s for s in splits if s]

    def _merge_splits(self, splits: List[List[int]]) -> List[List[int]]:
        if not splits:
            return []
        merged_splits = []
        current_chunk = []
        for split in splits:
            if not current_chunk:
                current_chunk = split
            elif self._length_function(current_chunk) + self._length_function(split) <= self._chunk_size:
                current_chunk.extend(split)
            else:
                merged_splits.append(current_chunk)
                current_chunk = split
        if current_chunk:
            merged_splits.append(current_chunk)
        if len(merged_splits) == 1 and self._length_function(merged_splits[0]) > self._chunk_size:
            return self._split_chunk(merged_splits[0])
        if self._chunk_overlap > 0:
            return self._enforce_overlap(merged_splits)
        return merged_splits

    def _split_chunk(self, chunk: List[int]) -> List[List[int]]:
        result = []
        for i in range(0, len(chunk), self._chunk_size - self._chunk_overlap):
            new_chunk = chunk[i:i + self._chunk_size]
            if len(new_chunk) > self._chunk_overlap:
                result.append(new_chunk)
        return result

    def _enforce_overlap(self, chunks: List[List[int]]) -> List[List[int]]:
        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
            else:
                prev_overlap = chunks[i - 1][-self._chunk_overlap:]
                combined = prev_overlap + chunk
                if len(combined) > self._chunk_size:
                    combined = combined[:self._chunk_size]
                result.append(combined)
        return result


DEFAULT_TEXT_SEPARATORS = [
    "\n\n", "\r\n\r\n", "\n", "\r\n", "。", "．", ".", "！", "!", "？", "?", " ", "\t"
]


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", fold(title)).strip("_")


def compute_mdhash_id(content: str, prefix: str = "chunk-") -> str:
    return prefix + md5(content.encode("utf-8")).hexdigest()


def chunk_document_hirag(
    doc_id: str,
    text: str,
    encoder: tiktoken.Encoding,
    max_token_size: int = 1200,
    overlap_token_size: int = 100,
) -> List[Dict]:
    separators = [encoder.encode(s) for s in DEFAULT_TEXT_SEPARATORS]
    splitter = SeparatorSplitter(
        separators=separators,
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    tokens = encoder.encode(text)
    chunk_tokens_list = splitter.split_tokens(tokens)
    chunks = []
    for i, c_tokens in enumerate(chunk_tokens_list):
        content = encoder.decode(c_tokens).strip()
        if not content:
            continue
        c_id = compute_mdhash_id(content, prefix="chunk-")
        chunks.append({
            "chunk_id": c_id,
            "chunk_order_index": i,
            "tokens": len(c_tokens),
            "text": content,
        })
    return chunks


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    # 1. Doc manifest.csv
    if not MANIFEST_FILE.exists():
        print(f"[!] Khong tim thay manifest: {MANIFEST_FILE}")
        sys.exit(1)

    history_docs = []
    song_docs = []
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            
            slug = slugify(row["title"])
            txt_path = DOCS_DIR / f"{slug}.txt"
            if not txt_path.exists():
                slug2 = slugify(row["requested"])
                txt_path2 = DOCS_DIR / f"{slug2}.txt"
                if txt_path2.exists():
                    txt_path = txt_path2
                    slug = slug2
                else:
                    candidates = list(DOCS_DIR.glob(f"*{slug[:8]}*.txt"))
                    if candidates:
                        txt_path = candidates[0]
                        slug = txt_path.stem
                    else:
                        print(f"[WARN] Khong tim thay txt cho: {row['title']} (slug={slug})")
                        continue

            doc_entry = {
                "group": row["group"],
                "requested": row["requested"],
                "title": row["title"],
                "url": row["url"],
                "words": int(row["words"]),
                "file_path": str(txt_path),
                "slug": slug,
            }
            if row["group"] == "history":
                history_docs.append(doc_entry)
            else:
                song_docs.append(doc_entry)

    print(f"[*] Tong so bai hop le da doc: History = {len(history_docs)}, Songs = {len(song_docs)}")

    # 2. Split stratified 85 / 15 theo bai
    # History: 33 bai -> 28 train (84.8%), 5 val (15.2%)
    # Song: 30 bai -> 25 train (83.3%), 5 val (16.7%)
    random.shuffle(history_docs)
    random.shuffle(song_docs)

    n_hist_val = max(1, round(len(history_docs) * 0.15))
    n_song_val = max(1, round(len(song_docs) * 0.15))

    val_history = history_docs[:n_hist_val]
    train_history = history_docs[n_hist_val:]

    val_song = song_docs[:n_song_val]
    train_song = song_docs[n_song_val:]

    for d in train_history + train_song:
        d["split"] = "train"
    for d in val_history + val_song:
        d["split"] = "val"

    train_all = train_history + train_song
    val_all = val_history + val_song

    print(f"[*] Split ket qua (Seed=42):")
    print(f"    - History: Train = {len(train_history)} bai, Val = {len(val_history)} bai")
    print(f"    - Songs  : Train = {len(train_song)} bai, Val = {len(val_song)} bai")
    print(f"    - TONG   : Train = {len(train_all)} bai, Val = {len(val_all)} bai")

    # Luu danh sach bai
    (OUT_DIR / "train_articles.json").write_text(
        json.dumps(train_all, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "val_articles.json").write_text(
        json.dumps(val_all, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "split_manifest.json").write_text(
        json.dumps({"train": train_all, "val": val_all}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[+] Da ghi train_articles.json ({len(train_all)} bai) va val_articles.json ({len(val_all)} bai)")

    # 3. Cat chunk bang SeparatorSplitter cua HiRAG
    encoder = tiktoken.encoding_for_model("gpt-4o")
    train_chunks = []
    val_chunks = []

    for doc_list, target_list in [(train_all, train_chunks), (val_all, val_chunks)]:
        for d in doc_list:
            text = Path(d["file_path"]).read_text(encoding="utf-8")
            doc_chunks = chunk_document_hirag(
                doc_id=d["slug"],
                text=text,
                encoder=encoder,
                max_token_size=1200,
                overlap_token_size=100,
            )
            for c in doc_chunks:
                record = {
                    "chunk_id": c["chunk_id"],
                    "doc": d["slug"],
                    "title": d["title"],
                    "group": d["group"],
                    "split": d["split"],
                    "chunk_order_index": c["chunk_order_index"],
                    "tokens": c["tokens"],
                    "text": c["text"],
                }
                target_list.append(record)

    all_chunks = train_chunks + val_chunks
    print(f"\n[*] Tong so chunk sinh ra:")
    print(f"    - Train Chunks: {len(train_chunks)} chunks")
    print(f"    - Val Chunks  : {len(val_chunks)} chunks")
    print(f"    - Tong Chunks : {len(all_chunks)} chunks")

    # Ghi ra JSONL
    def write_jsonl(path: Path, data: List[Dict]):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_jsonl(OUT_DIR / "chunks_train.jsonl", train_chunks)
    write_jsonl(OUT_DIR / "chunks_val.jsonl", val_chunks)
    write_jsonl(OUT_DIR / "chunks_all.jsonl", all_chunks)

    print(f"[+] Da luu:")
    print(f"    - {OUT_DIR / 'chunks_train.jsonl'}")
    print(f"    - {OUT_DIR / 'chunks_val.jsonl'}")
    print(f"    - {OUT_DIR / 'chunks_all.jsonl'}")


if __name__ == "__main__":
    main()
