"""step4_gen_extraction_triples.py — Sinh nhan triple cho adapter extraction bang chuan HiRAG.

Cac yeu cau theo chi dao:
- Chay dung prompt extraction cua HiRAG tren chunk cua tap train bang model qwen3:8b (think: False).
- Luu output dung dinh dang repo parse (tuple_delimiter=<|>, record_delimiter=##, completion_delimiter=<|COMPLETE|>).
- Loc mau parse loi hoac thieu truong.
- Chi chay tren chunks_train.jsonl (tap validation TUYET DOI khong dung de train).
- Luu thanh: data/adapter_data/extraction_train_triples.jsonl
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
INPUT_CHUNKS = DATA_DIR / "chunks_train.jsonl"
OUTPUT_FILE = DATA_DIR / "extraction_train_triples.jsonl"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:8b"

TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]
ENTITY_TYPES_STR = ",".join(DEFAULT_ENTITY_TYPES)

# Prompt extraction chuan cua HiRAG (tương thích Microsoft GraphRAG & LightRAG)
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
    """Xoa bo khoang trang thua, dau ngoac kep, nhay kep quanh chuoi."""
    s = s.strip()
    # Loai bo tat ca dau ngoac kep/nhay don o dau va cuoi
    s = re.sub(r'^["\']+|["\']+$', '', s).strip()
    return s


def parse_hirag_extraction(raw_text: str) -> Tuple[List[Dict], List[Dict], List[str]]:
    """Parse output theo dung logic _op.py cua HiRAG."""
    clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    # Ho tro ca ## va xuong dong lam delimiter
    records = re.split(r"##|\n(?=\()", clean_text)

    entities = []
    relationships = []
    errors = []
    seen_entities = set()
    seen_relations = set()

    for rec in records:
        rec = rec.replace(COMPLETION_DELIMITER, "").strip()
        if not rec:
            continue
        m = re.search(r"\((.*)\)", rec, re.DOTALL)
        if not m:
            continue
        inner = m.group(1).strip()
        parts = [clean_str(p) for p in inner.split(TUPLE_DELIMITER)]
        if not parts:
            continue

        tag = clean_str(parts[0]).lower()
        if tag == "entity":
            if len(parts) >= 4:
                name = clean_str(parts[1]).upper()
                etype = clean_str(parts[2]).lower()
                desc = parts[3].strip()
                if name and desc and etype in DEFAULT_ENTITY_TYPES:
                    if name not in seen_entities:
                        seen_entities.add(name)
                        entities.append({
                            "name": name,
                            "type": etype,
                            "description": desc,
                        })
            else:
                errors.append(f"entity_missing_fields: {parts}")
        elif tag == "relationship":
            if len(parts) >= 5:
                src = clean_str(parts[1]).upper()
                tgt = clean_str(parts[2]).upper()
                desc = parts[3].strip()
                try:
                    w = float(re.sub(r"[^\d.]", "", parts[4]))
                    w = max(1.0, min(10.0, w))
                except (ValueError, IndexError):
                    w = 5.0

                if src and tgt and src != tgt and desc:
                    pair_key = tuple(sorted([src, tgt]))
                    if pair_key not in seen_relations:
                        seen_relations.add(pair_key)
                        relationships.append({
                            "src": src,
                            "tgt": tgt,
                            "description": desc,
                            "strength": w,
                        })
            else:
                errors.append(f"rel_missing_fields: {parts}")

    return entities, relationships, errors


def call_extraction_llm(chunk_text: str, max_retries: int = 3) -> str:
    """Goi qwen3:8b qua Ollama de extract entities va relationships."""
    prompt = HIRAG_EXTRACTION_PROMPT.format(input_text=chunk_text)
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1024,
            "num_ctx": 4096,
        },
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
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

    # 1. Load train chunks
    chunks = []
    with open(INPUT_CHUNKS, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    print(f"[*] Tong so train chunks can extract triples: {len(chunks)}")

    # 2. Checkpoint resume
    processed_chunk_ids: Set[str] = set()
    existing_records = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing_records.append(rec)
                    processed_chunk_ids.add(rec["chunk_id"])
        print(f"[*] Resume tu checkpoint: Da co {len(existing_records)} chunks extracted.")

    out_fp = open(OUTPUT_FILE, "a", encoding="utf-8")

    stats = {
        "processed": len(existing_records),
        "total_entities": sum(len(r.get("entities", [])) for r in existing_records),
        "total_relations": sum(len(r.get("relationships", [])) for r in existing_records),
        "total_errors": 0,
        "empty_chunks": 0,
    }

    t0 = time.time()
    try:
        for idx, chunk in enumerate(chunks, 1):
            c_id = chunk["chunk_id"]
            if c_id in processed_chunk_ids:
                continue

            doc_slug = chunk["doc"]
            doc_title = chunk["title"]
            chunk_text = chunk["text"]

            raw_out = call_extraction_llm(chunk_text)
            entities, relations, errs = parse_hirag_extraction(raw_out)
            stats["total_errors"] += len(errs)

            if not entities:
                stats["empty_chunks"] += 1

            record = {
                "chunk_id": c_id,
                "doc": doc_slug,
                "title": doc_title,
                "group": chunk.get("group", ""),
                "chunk_order_index": chunk.get("chunk_order_index", 0),
                "text": chunk_text,
                "raw_extraction": raw_out,
                "entities": entities,
                "relationships": relations,
                "entity_count": len(entities),
                "relationship_count": len(relations),
                "parse_errors": errs,
            }

            out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fp.flush()

            stats["processed"] += 1
            stats["total_entities"] += len(entities)
            stats["total_relations"] += len(relations)

            elapsed = time.time() - t0
            speed = (idx - len(existing_records)) / elapsed if elapsed > 0 else 0
            print(
                f"[{stats['processed']:3d}/{len(chunks)}] chunk={c_id[:8]}.. ({doc_title[:20]}) | "
                f"ents={len(entities)} rels={len(relations)} | "
                f"tot_ents={stats['total_entities']} tot_rels={stats['total_relations']} | "
                f"errs={stats['total_errors']} | {speed:.2f} chunks/s",
                flush=True,
            )

    finally:
        out_fp.close()

    print("\n" + "=" * 60)
    print("HOAN THANH SINH NHAN TRIPLES CHO ADAPTER EXTRACTION")
    print("=" * 60)
    print(f"Tong so chunks da xu ly : {stats['processed']}/{len(chunks)}")
    print(f"Tong so entities trich xuat: {stats['total_entities']}")
    print(f"Tong so relations trich xuat: {stats['total_relations']}")
    print(f"So loi parse bo qua     : {stats['total_errors']}")
    print(f"Luu tai                  : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
