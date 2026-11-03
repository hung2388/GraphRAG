"""step5_curate_gold_triples.py — Xay dung bo Gold Triples tu tap validation.

Cac yeu cau theo chi dao:
- Lay 50-100 chunk tu tap validation (dung toan bo 54 chunk cua tap validation).
- Chay prompt extraction chuan HiRAG.
- Tu kiem tra, lam sach va chuan hoa nhan (loai bo hallucinations, kiem tra thuc the co trong text).
- Dung bo nay DO EXTRACTION, TUYET DOI KHONG DUA VAO TRAIN.
- Luu thanh: data/adapter_data/extraction_val_gold.jsonl
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "adapter_data"
INPUT_CHUNKS = DATA_DIR / "chunks_val.jsonl"
OUTPUT_FILE = DATA_DIR / "extraction_val_gold.jsonl"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:8b"

TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]
ENTITY_TYPES_STR = ",".join(DEFAULT_ENTITY_TYPES)

HIRAG_EXTRACTION_PROMPT = f"""-Goal-
Given a text document and a list of entity types, identify the core entities and the key relationships among them.

-Steps-
1. Identify 5 to 10 main named entities. For each entity, extract:
- entity_name: Name of the entity, capitalized
- entity_type: One of [{ENTITY_TYPES_STR}]
- entity_description: Concise 1-2 sentence description of attributes and role in the text
Format each entity as ("entity"{TUPLE_DELIMITER}<entity_name>{TUPLE_DELIMITER}<entity_type>{TUPLE_DELIMITER}<entity_description>)

2. For pairs of related entities from step 1, extract 5 to 12 direct relationships:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: concise explanation of their direct connection
- relationship_strength: numeric score (1 to 10)
Format each relationship as ("relationship"{TUPLE_DELIMITER}<source_entity>{TUPLE_DELIMITER}<target_entity>{TUPLE_DELIMITER}<relationship_description>{TUPLE_DELIMITER}<relationship_strength>)

3. Return output in English as a single list containing BOTH entities and relationships separated by **{RECORD_DELIMITER}**.
4. When finished, output {COMPLETION_DELIMITER}

######################
-Examples-
######################
Example 1:
Text:
In 1285, the Yuan prince Toghon led a major invasion into Dai Viet. Grand Commander Tran Hung Dao retreated from Thang Long to preserve troops. Later that year, Vietnamese forces launched counter-offensives at Tay Ket and Chuong Duong, forcing Toghon to retreat into hiding inside a bronze cylinder.
Output:
("entity"{TUPLE_DELIMITER}"TOGHON"{TUPLE_DELIMITER}"person"{TUPLE_DELIMITER}"Yuan prince who led the 1285 Mongol invasion into Dai Viet and was forced to retreat."){RECORD_DELIMITER}
("entity"{TUPLE_DELIMITER}"TRAN HUNG DAO"{TUPLE_DELIMITER}"person"{TUPLE_DELIMITER}"Grand Commander of Dai Viet who orchestrated strategic retreats and counter-offensives against Mongol forces."){RECORD_DELIMITER}
("entity"{TUPLE_DELIMITER}"DAI VIET"{TUPLE_DELIMITER}"geo"{TUPLE_DELIMITER}"Historical Vietnamese kingdom targeted by the Yuan invasion in 1285."){RECORD_DELIMITER}
("entity"{TUPLE_DELIMITER}"1285 YUAN INVASION"{TUPLE_DELIMITER}"event"{TUPLE_DELIMITER}"Major military campaign led by Toghon against Dai Viet."){RECORD_DELIMITER}
("relationship"{TUPLE_DELIMITER}"TOGHON"{TUPLE_DELIMITER}"DAI VIET"{TUPLE_DELIMITER}"Toghon led an invasion force into Dai Viet in 1285."{TUPLE_DELIMITER}8){RECORD_DELIMITER}
("relationship"{TUPLE_DELIMITER}"TRAN HUNG DAO"{TUPLE_DELIMITER}"TOGHON"{TUPLE_DELIMITER}"Tran Hung Dao commanded the defenses that defeated Toghon's invasion."{TUPLE_DELIMITER}9){RECORD_DELIMITER}
{COMPLETION_DELIMITER}

######################
-Real Data-
######################
Text:
{{input_text}}
Output:
"""


def clean_str(s: str) -> str:
    s = s.strip()
    return re.sub(r'^["\']+|["\']+$', '', s).strip()


def verify_and_clean_extraction(raw_text: str, chunk_text: str) -> Tuple[List[Dict], List[Dict], Dict]:
    """Parse, kiem tra groundedness va lam sach bo nhan vang (Gold standard)."""
    clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    records = re.split(r"##|\n(?=\()", clean_text)
    chunk_lower = chunk_text.lower()

    entities = []
    relationships = []
    valid_entity_names = set()
    discarded = {"unsupported_entities": 0, "invalid_relations": 0}

    # Pass 1: Parse & verify entities
    for rec in records:
        rec = rec.replace(COMPLETION_DELIMITER, "").strip()
        m = re.search(r"\((.*)\)", rec, re.DOTALL)
        if not m:
            continue
        parts = [clean_str(p) for p in m.group(1).split(TUPLE_DELIMITER)]
        if not parts:
            continue
        if clean_str(parts[0]).lower() == "entity" and len(parts) >= 4:
            name = clean_str(parts[1]).upper()
            etype = clean_str(parts[2]).lower()
            desc = parts[3].strip()

            if not name or not desc or etype not in DEFAULT_ENTITY_TYPES:
                continue

            # Groundedness check: Ten thuc the (hoac phan lon tu khoa) phai xuat hien trong chunk text
            name_words = [w.lower() for w in re.findall(r"\w+", name)]
            name_in_text = name.lower() in chunk_lower or any(
                w in chunk_lower for w in name_words if len(w) > 3
            )
            if not name_in_text:
                discarded["unsupported_entities"] += 1
                continue

            if name not in valid_entity_names:
                valid_entity_names.add(name)
                entities.append({
                    "name": name,
                    "type": etype,
                    "description": desc,
                })

    # Pass 2: Parse & verify relationships
    seen_rel_pairs = set()
    for rec in records:
        rec = rec.replace(COMPLETION_DELIMITER, "").strip()
        m = re.search(r"\((.*)\)", rec, re.DOTALL)
        if not m:
            continue
        parts = [clean_str(p) for p in m.group(1).split(TUPLE_DELIMITER)]
        if not parts:
            continue
        if clean_str(parts[0]).lower() == "relationship" and len(parts) >= 5:
            src = clean_str(parts[1]).upper()
            tgt = clean_str(parts[2]).upper()
            desc = parts[3].strip()
            try:
                w = float(re.sub(r"[^\d.]", "", parts[4]))
                w = max(1.0, min(10.0, w))
            except (ValueError, IndexError):
                w = 5.0

            if not src or not tgt or src == tgt or not desc:
                continue

            # Ca src va tgt phai la thuc the hop le da duoc xac nhan
            if src not in valid_entity_names or tgt not in valid_entity_names:
                discarded["invalid_relations"] += 1
                continue

            pair = tuple(sorted([src, tgt]))
            if pair not in seen_rel_pairs:
                seen_rel_pairs.add(pair)
                relationships.append({
                    "src": src,
                    "tgt": tgt,
                    "description": desc,
                    "strength": w,
                })

    return entities, relationships, discarded


def call_extraction_llm(chunk_text: str, max_retries: int = 3) -> str:
    prompt = HIRAG_EXTRACTION_PROMPT.format(input_text=chunk_text)
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1500,
            "num_ctx": 4096,
        },
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=90)
            if resp.status_code == 200:
                raw = resp.json().get("response", "")
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                if raw:
                    return raw
        except Exception:
            time.sleep(2)
    return ""


def main():
    if not INPUT_CHUNKS.exists():
        print(f"[!] Khong tim thay {INPUT_CHUNKS}")
        sys.exit(1)

    val_chunks = []
    with open(INPUT_CHUNKS, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_chunks.append(json.loads(line))

    print(f"[*] Tong so Validation Chunks de lam bo Gold Triple: {len(val_chunks)} chunks")
    print(f"[*] LUU Y: Bo nay chi dung de danh gia (EVAL), TUYET DOI KHONG DUA VAO TRAIN!")

    processed_ids = set()
    existing_records = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing_records.append(r)
                    processed_ids.add(r["chunk_id"])
        print(f"[*] Resume tu checkpoint: Da co {len(existing_records)} gold chunks.")

    out_fp = open(OUTPUT_FILE, "a", encoding="utf-8")
    stats = {
        "processed": len(existing_records),
        "total_gold_entities": sum(len(r.get("gold_entities", [])) for r in existing_records),
        "total_gold_relations": sum(len(r.get("gold_relationships", [])) for r in existing_records),
    }

    t0 = time.time()
    try:
        for idx, chunk in enumerate(val_chunks, 1):
            c_id = chunk["chunk_id"]
            if c_id in processed_ids:
                continue

            chunk_text = chunk["text"]
            raw_out = call_extraction_llm(chunk_text)
            entities, relations, disc = verify_and_clean_extraction(raw_out, chunk_text)

            record = {
                "chunk_id": c_id,
                "doc": chunk["doc"],
                "title": chunk["title"],
                "group": chunk.get("group", ""),
                "split": "val_gold_eval_only",
                "text": chunk_text,
                "gold_entities": entities,
                "gold_relationships": relations,
                "gold_entity_count": len(entities),
                "gold_relationship_count": len(relations),
                "discarded_unsupported": disc,
            }

            out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fp.flush()

            stats["processed"] += 1
            stats["total_gold_entities"] += len(entities)
            stats["total_gold_relations"] += len(relations)

            elapsed = time.time() - t0
            speed = (idx - len(existing_records)) / elapsed if elapsed > 0 else 0
            print(
                f"[{stats['processed']:2d}/{len(val_chunks)}] val_chunk={c_id[:8]}.. ({chunk['title'][:20]}) | "
                f"ents={len(entities)} rels={len(relations)} | "
                f"tot_ents={stats['total_gold_entities']} tot_rels={stats['total_gold_relations']} | "
                f"{speed:.2f} chunks/s",
                flush=True,
            )

    finally:
        out_fp.close()

    print("\n" + "=" * 60)
    print("HOAN THANH XAY DUNG BO GOLD TRIPLES (VAL ONLY)")
    print("=" * 60)
    print(f"Tong so val chunks da xu ly : {stats['processed']}/{len(val_chunks)}")
    print(f"Tong gold entities hop le    : {stats['total_gold_entities']}")
    print(f"Tong gold relations hop le   : {stats['total_gold_relations']}")
    print(f"Luu tai                      : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
