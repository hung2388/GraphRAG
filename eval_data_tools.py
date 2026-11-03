"""Kiem tra bo eval va tinh Hit@k / Recall@k theo span.

Chay kiem tra (doi DOC_FILES cho khop may anh):
    python eval_data_tools.py eval_v2.json

Dung trong code danh gia:
    from eval_data_tools import hit_at_k, recall_at_k
    chunks = [...]  # text cac chunk goc da retrieve, theo thu tu xep hang
    hit_at_k(chunks, item, k=5); recall_at_k(chunks, item, k=5)
"""
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# Doi duong dan neu khac
DOC_FILES = {
    "hong_bang": "data/input/src.txt",
    "quang_trung": "data/input/quang_trung.txt",
    "love_the_way_you_lie": "data/input/love.txt",
}


def norm(s: str) -> str:
    """Chuan hoa de so khop: bo marker [12]/[2a], thong nhat dau nhay/gach, gop khoang trang, lowercase."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\[\d+[ab]?\]", " ", s)
    s = (s.replace("\u201c", '"').replace("\u201d", '"')
          .replace("\u2018", "'").replace("\u2019", "'")
          .replace("\u2013", "-").replace("\u2014", "-"))
    return re.sub(r"\s+", " ", s).strip().lower()


def _covered(chunks, spans, k=None):
    top = [norm(c) for c in (chunks[:k] if k else chunks)]
    return sum(any(norm(sp) in c for c in top) for sp in spans)


def hit_at_k(chunks, item, k):
    """1.0 neu it nhat mot gold_span nam trong top-k chunk. None neu cau khong co gold_spans."""
    spans = item.get("gold_spans", [])
    return None if not spans else float(_covered(chunks, spans, k) > 0)


def recall_at_k(chunks, item, k):
    """Ti le gold_spans duoc phu boi top-k chunk. None neu cau khong co gold_spans."""
    spans = item.get("gold_spans", [])
    return None if not spans else _covered(chunks, spans, k) / len(spans)


def renumber_file(path: str, backup: bool = True) -> int:
    """Danh so lai id tu 1 den len(items) cho bo cau hoi eval de tranh lech diem."""
    p = Path(path)
    if not p.exists():
        p = Path(__file__).resolve().parent / path
    if not p.exists():
        print(f"[ERROR] Khong tim thay file: {path}")
        return 0

    items = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        print(f"[ERROR] File khong phai danh sach JSON: {path}")
        return 0

    old_ids = [it.get("id") for it in items]
    int_ids = [i for i in old_ids if isinstance(i, int)]
    max_id = max(int_ids, default=0)
    total = len(items)

    print(f"[*] File co {total} cau hoi. Max ID hien tai la: {max_id}")

    if backup:
        backup_path = p.with_name(p.stem + "_backup" + p.suffix)
        backup_path.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[*] Da luu ban sao luu: {backup_path}")

    for new_id, it in enumerate(items, 1):
        if "id" in it and it["id"] != new_id:
            it["old_id"] = it["id"]
        it["id"] = new_id

    p.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Da danh so lai thanh cong {total} cau hoi (ID: 1 -> {total}) vao {p}")
    return total


def validate(paths):
    items = []
    for p in paths:
        target_path = Path(p)
        if not target_path.exists():
            target_path = Path(__file__).resolve().parent / p
        if target_path.exists():
            items += json.loads(target_path.read_text(encoding="utf-8"))
        else:
            print(f"[WARN] Khong tim thay file eval: {p}")

    if not items:
        print("Khong co du lieu cau hoi de kiem tra.")
        return

    ids = [i.get("id") for i in items]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        print("[WARN] ID trung lap:", dup)

    int_ids = [i for i in ids if isinstance(i, int)]
    if int_ids and max(int_ids) != len(items):
        print(f"[WARN] Max ID ({max(int_ids)}) khong khop voi tong so cau ({len(items)}).")
        print(f"       Goi y: Chay 'python eval_data_tools.py renumber <file>' de danh so lai tu 1 den {len(items)}.")

    docs = {}
    for name, path in DOC_FILES.items():
        doc_path = Path(path)
        if not doc_path.exists():
            doc_path = Path(__file__).resolve().parent / path
        if doc_path.exists():
            docs[name] = norm(doc_path.read_text(encoding="utf-8"))
        else:
            print(f"[WARN] khong thay file tai lieu '{name}': {path}")

    bad = 0
    checked = 0
    for it in items:
        text = docs.get(it.get("doc"))
        if text is None:
            continue
        for field in ("gold_spans", "related_spans"):
            for sp in it.get(field, []):
                checked += 1
                if norm(sp) not in text:
                    bad += 1
                    print(f"[MISS] id={it['id']} {field}: {sp}")
        if not it.get("answerable") and it.get("gold_spans"):
            print(f"[SCHEMA] id={it['id']} unanswerable nhung co gold_spans")

    print(f"\n{len(items)} cau | answerable={sum(bool(i.get('answerable')) for i in items)} "
          f"| unanswerable={sum(not i.get('answerable') for i in items)} "
          f"| partial={sum(i.get('partial', False) for i in items)}")
    print("theo tai lieu:", dict(Counter(i.get("doc") for i in items)))
    print("theo loai cau:", dict(Counter(i.get("question_type") for i in items)))
    if not checked:
        print("Chua kiem tra duoc span nao (khong doc duoc file tai lieu). Sua DOC_FILES.")
    else:
        print(f"Da kiem tra {checked} span, khong khop: {bad}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "renumber":
        target = args[1] if len(args) > 1 else "eval_v2.json"
        renumber_file(target)
    else:
        validate(args or ["eval_v2.json"])
