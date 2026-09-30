import os
import glob
import json
import hashlib
import unicodedata
import re
import requests
import ollama
import difflib
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ============================================================
# LOAD CONFIG
# ============================================================

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# ============================================================
# OLLAMA CONFIG
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"
REASONING_MODELS = {"qwen3:8b", "deepseek-r1"}
extraction_stats = {"success": 0, "failed": 0}
DEBUG_DIR = os.path.join("data", "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)
_debug_call_count = 0
TOKEN_LOG = []
current_phase = "indexing"  # Sửa thành "query" nếu bạn đang chạy query script

def chat_ollama(messages, format_json=False, temperature=0.0, debug_tag="", num_predict=2048, timeout=180, repeat_penalty=1.15):
    global _debug_call_count
    _debug_call_count += 1
    call_id = _debug_call_count

    tag = f"_{debug_tag}" if debug_tag else ""
    log_path = os.path.join(DEBUG_DIR, f"call_{call_id:04d}{tag}.txt")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"=== CHAT MESSAGES (call #{call_id}) ===\n")
        for m in messages:
            f.write(f"[{m['role'].upper()}]\n{m['content']}\n\n")
        f.write("=== WAITING FOR RESPONSE... ===\n")

    print(f"  [Ollama chat #{call_id}{tag}] Đang gọi model... (xem {log_path})")

    kwargs = {
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "repeat_penalty": repeat_penalty,
        }
    }
    # tắt reasoning mode của qwen3, tránh treo/loop dài
    if any(OLLAMA_MODEL.startswith(rm) for rm in REASONING_MODELS):
        kwargs["think"] = False

    if format_json:
        kwargs["format"] = "json"

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        **kwargs
    )
    response_text = response["message"]["content"].strip()
    
    # Tracking tokens cho chat_ollama
    usage = {
        "prompt_tokens": response.get("prompt_eval_count", 0),
        "output_tokens": response.get("eval_count", 0),
        "prompt_duration": response.get("prompt_eval_duration", 0),
        "eval_duration": response.get("eval_duration", 0),
    }
    TOKEN_LOG.append({
        "tag": debug_tag, 
        "phase": current_phase,
        "model": OLLAMA_MODEL, 
        **usage
    })

    # phòng trường hợp think vẫn lọt vào output
    response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"=== RESPONSE ===\n")
        f.write(response_text)
        f.write(f"\n\n=== END (length: {len(response_text)} chars) ===\n")

    print(f"  [Ollama chat #{call_id}] Xong, {len(response_text)} chars")
    return response_text



def call_ollama(prompt, format_json=False, temperature=0.0, debug_tag="",
                 model=None, repeat_penalty=1.15, num_predict=2048):
    global _debug_call_count
    _debug_call_count += 1
    call_id = _debug_call_count

    m = model or OLLAMA_MODEL
    payload = {
        "model": m,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "repeat_penalty": repeat_penalty,
        },
    }
    if any(m.startswith(rm) for rm in REASONING_MODELS):
        payload["think"] = False
        
    if format_json:
        payload["format"] = "json"

    tag = f"_{debug_tag}" if debug_tag else ""
    log_path = os.path.join(DEBUG_DIR, f"call_{call_id:04d}{tag}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"=== PROMPT (call #{call_id}, model={payload['model']}) ===\n")
        f.write(prompt)
        f.write("\n\n=== WAITING FOR RESPONSE... ===\n")

    print(f"  [Ollama call #{call_id}{tag} | {payload['model']}] Đang gọi model... (xem {log_path})")

    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    response_text = data.get("response", "").strip()

    # Tracking tokens cho call_ollama
    usage = {
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "output_tokens": data.get("eval_count", 0),
        "prompt_duration": data.get("prompt_eval_duration", 0),
        "eval_duration": data.get("eval_duration", 0),
    }
    TOKEN_LOG.append({
        "tag": debug_tag, 
        "phase": current_phase,
        "model": payload["model"], 
        **usage
    })

    response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"=== RESPONSE ===\n")
        f.write(response_text)
        f.write(f"\n\n=== END (length: {len(response_text)} chars) ===\n")

    print(f"  [Ollama call #{call_id}] Xong, {len(response_text)} chars")
    return response_text


def save_token_log(filename=None):
    if filename is None:
        filename = f"token_log_{current_phase}.json"
    path = os.path.join("data", "cache", filename)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(TOKEN_LOG, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu TOKEN_LOG vào {path}")


# ============================================================
# READ SOURCE FILES FROM FOLDER
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_texts_from_folder(folder_path=None):
    if folder_path is None:
        folder_path = os.path.join(BASE_DIR, "data", "input")
    combined_texts = []
    txt_files = glob.glob(os.path.join(folder_path, "**", "*.txt"), recursive=True)

    if not txt_files:
        print(f"Không tìm thấy file .txt nào trong thư mục '{folder_path}'!")
        return ""

    for file_path in txt_files:
        print(f"Đang đọc: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    combined_texts.append(content)
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read().strip()
                if content:
                    combined_texts.append(content)

    return "\n\n".join(combined_texts)


# ============================================================
# NORMALIZE ENTITY NAME
# ============================================================

def normalize_name(name):
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip().lower()


def make_entity_id(name):
    normalized = normalize_name(name)
    return "entity_" + hashlib.md5(normalized.encode("utf-8")).hexdigest()[:16]


# ============================================================
# CHUNKING (Pure Python thay thế Langchain để tránh lỗi DLL)
# ============================================================

def split_into_chunks(text, chunk_size=1500, chunk_overlap=200):
    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]
    
    def _split_text(text, separators):
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                new_separators = separators[i + 1:]
                break
                
        splits = _split_text_with_regex(text, separator)
        
        _good_splits = []
        _separator = separator if separator is not None else ""
        for s in splits:
            if len(s) < chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged = _merge_splits(_good_splits, _separator, chunk_size, chunk_overlap)
                    final_chunks.extend(merged)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = _split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged = _merge_splits(_good_splits, _separator, chunk_size, chunk_overlap)
            final_chunks.extend(merged)
        return final_chunks
        
    def _split_text_with_regex(text, separator):
        if separator:
            splits = text.split(separator)
            return [s + separator for s in splits[:-1]] + ([splits[-1]] if splits[-1] else [])
        return list(text)
        
    def _merge_splits(splits, separator, chunk_size, chunk_overlap):
        docs = []
        current_doc = []
        total = 0
        for d in splits:
            _len = len(d)
            if total + _len > chunk_size and current_doc:
                docs.append("".join(current_doc).strip())
                while total > chunk_overlap or (total + _len > chunk_size and total > 0):
                    total -= len(current_doc[0])
                    current_doc.pop(0)
            current_doc.append(d)
            total += _len
        if current_doc:
            docs.append("".join(current_doc).strip())
        return docs

    return _split_text(text, separators)


# ============================================================
# CACHE & LOGGING CONFIG
# ============================================================

CACHE_DIR = os.path.join("data", "cache", "kg_cache") if os.path.exists(os.path.join("data", "cache", "kg_cache")) or not os.path.exists("kg_cache") else "kg_cache"
GLOBAL_SUMMARIES_CACHE = os.path.join("data", "cache", "community_summaries.json") if os.path.exists(os.path.join("data", "cache", "community_summaries.json")) or not os.path.exists("community_summaries.json") else "community_summaries.json"
FAILED_CHUNKS_LOG = os.path.join("data", "cache", "failed_chunks.txt") if os.path.exists(os.path.join("data", "cache", "failed_chunks.txt")) or not os.path.exists("failed_chunks.txt") else "failed_chunks.txt"
VERBOSE = False  # Đặt True khi cần xem chi tiết từng chunk

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(GLOBAL_SUMMARIES_CACHE) or ".", exist_ok=True)
os.makedirs(os.path.dirname(FAILED_CHUNKS_LOG) or ".", exist_ok=True)


# ============================================================
# OLLAMA EXTRACTION
# ============================================================



def extract_kg(text, current_depth=0, max_depth=2):
    """
    Trích xuất KG có cơ chế tự khắc phục lỗi JSON:
    1. Thử lại với temperature tăng dần (0.0 -> 0.3 -> 0.6)
    2. Nếu vẫn thất bại, chia đôi văn bản để trích xuất rồi gộp lại.
    """
    cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    prompt = f"""
Đọc đoạn text sau và trích xuất entity + relation dưới dạng JSON.

TEXT:
{text}

Chỉ trích xuất entity thực sự xuất hiện trong đoạn text.

Các loại entity:
- Person
- Location
- Organization
- Date
- Event
- Ideology
- Document
- Concept

Mỗi entity có ID cục bộ:
e1, e2, e3...

Relation phải sử dụng đúng ID của entity.

Không được tạo entity không có trong text.

Trả về ĐÚNG định dạng JSON sau:

{{
  "entities": [
    {{
      "id": "e1",
      "name": "tên entity",
      "type": "Person"
    }}
  ],
  "relations": [
    {{
      "source": "e1",
      "target": "e2",
      "relation": "quan_hệ"
    }}
  ]
}}

Không thêm bất kỳ giải thích nào.
"""

    kg_data = None
    # Lớp 1: Tăng dần temperature để thoát vòng lặp token
    temperatures = [0.0, 0.3, 0.6]
    
    for attempt, temp in enumerate(temperatures, start=1):
        raw = call_ollama(prompt, format_json=True, temperature=temp)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].removeprefix("json").strip()
            
        try:
            kg_data = json.loads(raw)
            break  # Parse thành công thì thoát vòng lặp
        except json.JSONDecodeError as e:
            print(f"  -> [Attempt {attempt} | Temp {temp}] Lỗi parse JSON: {e}")

    # Lớp 2 & 3: Nếu vẫn hỏng, chia đôi chunk (đệ quy)
    if kg_data is None:
        if current_depth < max_depth and len(text) > 300:
            print(f"  -> [SPLIT] Chunk quá phức tạp hoặc bị cắt cụt. Chia đôi (Depth: {current_depth+1})...")
            
            # Cố gắng cắt ở dấu chấm câu hoặc xuống dòng để không làm đứt thực thể
            mid = len(text) // 2
            split_point = text.rfind('\n', 0, mid + 200)
            if split_point == -1 or split_point < len(text) * 0.3:
                split_point = text.rfind('. ', 0, mid + 200)
            if split_point == -1 or split_point < len(text) * 0.3:
                split_point = mid # fallback cắt cứng giữa chunk
                
            part1 = text[:split_point]
            part2 = text[split_point:]

            kg1 = extract_kg(part1, current_depth=current_depth+1, max_depth=max_depth)
            kg2 = extract_kg(part2, current_depth=current_depth+1, max_depth=max_depth)

            # Gộp kết quả
            kg_data = {
                "entities": kg1.get("entities", []) + kg2.get("entities", []),
                "relations": kg1.get("relations", []) + kg2.get("relations", [])
            }
            print(f"  -> [SPLIT SUCCESS] Đã gộp kết quả từ 2 nửa (Depth: {current_depth+1}).")
            
        else:
            # Hết đường cứu
            print(f"  -> [FAILED] Bỏ cuộc sau {len(temperatures)} lần thử. Lưu log lỗi.")
            if current_depth == 0:
                extraction_stats["failed"] += 1
                with open(FAILED_CHUNKS_LOG, "a", encoding="utf-8") as f:
                    f.write(f"{cache_key}\n")
            return {"entities": [], "relations": []}  # KHÔNG cache

    # Ghi cache và tăng biến đếm (chỉ đếm cho chunk gốc)
    if current_depth == 0:
        extraction_stats["success"] += 1
        
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(kg_data, f, ensure_ascii=False, indent=2)
        
    return kg_data

# ============================================================
# NEO4J INSERT
# ============================================================

def insert_chunk(tx, chunk_id, text):
    tx.run(
        """
        MERGE (c:Chunk {id: $id})
        SET c.text = $text
        """,
        id=chunk_id,
        text=text
    )


def insert_entity(tx, entity, chunk_id):
    global_id = make_entity_id(entity["name"])

    tx.run(
        """
        MERGE (n:Entity {id: $id})
        ON CREATE SET
            n.cluster_id = $id,
            n.is_canonical = true
        SET
            n.name = $name,
            n.type = $type
        """,
        id=global_id,
        name=entity["name"],
        type=entity["type"]
    )

    tx.run(
        """
        MATCH (e:Entity {id: $eid}), (c:Chunk {id: $cid})
        MERGE (e)-[:MENTIONED_IN]->(c)
        """,
        eid=global_id,
        cid=chunk_id
    )
    return global_id


def sanitize_relation_type(relation):
    relation_type = relation.upper().strip()
    relation_type = re.sub(r"[\s\-]+", "_", relation_type)
    relation_type = re.sub(r"[^A-Z0-9_ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]", "", relation_type)
    relation_type = re.sub(r"_+", "_", relation_type).strip("_")

    # Cypher không cho phép identifier bắt đầu bằng số
    if relation_type and relation_type[0].isdigit():
        relation_type = "R_" + relation_type

    return relation_type if relation_type else "RELATED_TO"


def insert_relation(tx, source_id, target_id, relation):
    relation_type = sanitize_relation_type(relation)

    query = f"""
        MATCH
            (a:Entity {{id: $source}}),
            (b:Entity {{id: $target}})
        MERGE
            (a)-[:{relation_type}]->(b)
    """

    tx.run(query, source=source_id, target=target_id)


def link_same_as(tx, id1, id2):
    tx.run(
        """
        MATCH (a:Entity {id: $id1}), (b:Entity {id: $id2})
        WHERE a <> b
        MERGE (a)-[:SAME_AS]-(b)
        """,
        id1=id1,
        id2=id2
    )


# ============================================================
# LLM-BASED ENTITY LINKING / DEDUPLICATION
# ============================================================

SKIP_DEDUP_TYPES = {"Document", "Date"}
MAX_GROUP_SIZE = 4
DEDUP_BATCH_SIZE = 10


def _parse_json_response(raw_text):
    raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    if raw_text.startswith("```"):
        parts = raw_text.split("```")
        if len(parts) >= 2:
            raw_text = parts[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()
    return json.loads(raw_text)


# ------------------------------------------------------------
# 1. MIGRATION: chạy 1 LẦN DUY NHẤT để backfill entity cũ
#    (đã tồn tại trước khi có field cluster_id) và gom SAME_AS
#    cũ (dạng mesh lộn xộn) thành cluster sạch.
# ------------------------------------------------------------

def migrate_to_cluster_schema():
    print("\n=== MIGRATION: gán cluster_id cho entity cũ ===")
    with driver.session() as session:
        # bước 1: entity thiếu cluster_id -> tự làm canonical riêng
        session.run("""
            MATCH (e:Entity)
            WHERE e.cluster_id IS NULL
            SET e.cluster_id = e.id, e.is_canonical = true
        """)

        # bước 2: gom các entity đang nối SAME_AS (mesh cũ) vào chung 1 cluster_id
        result = session.run("""
            MATCH (a:Entity)-[:SAME_AS]-(b:Entity)
            RETURN DISTINCT a.id AS a, b.id AS b
        """)
        pairs = [(r["a"], r["b"]) for r in result]

    if not pairs:
        print("Không có SAME_AS cũ cần gom.")
        return

    uf = UnionFind()
    for a, b in pairs:
        uf.union(a, b)

    with driver.session() as session:
        for group in uf.groups():
            records = session.run(
                "MATCH (e:Entity) WHERE e.id IN $ids RETURN e.id AS id, e.name AS name, "
                "size([(e)-[:MENTIONED_IN]->() | 1]) AS n_chunks",
                ids=group
            )
            members = [dict(r) for r in records]
            canonical = choose_canonical(members)
            cid = canonical["id"]
            session.run("""
                MATCH (e:Entity) WHERE e.id IN $ids
                SET e.cluster_id = $cid, e.is_canonical = false
            """, ids=group, cid=cid)
            session.run("""
                MATCH (e:Entity {id: $cid}) SET e.is_canonical = true
            """, cid=cid)
            print(f"[MIGRATED] cluster='{canonical['name']}' <- {[m['name'] for m in members]}")

    print("Migration xong.")


# ------------------------------------------------------------
# 2. UNION-FIND (chỉ dùng tạm trong bộ nhớ cho 1 lần chạy)
# ------------------------------------------------------------

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry

    def groups(self):
        out = {}
        for x in self.parent:
            out.setdefault(self.find(x), []).append(x)
        return list(out.values())


# ------------------------------------------------------------
# 3. RELATION SEMANTICS (positive / negative / neutral)
# ------------------------------------------------------------

def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

POSITIVE_KEYWORDS = [
    "ten_khac", "co_ten", "cung_nhan", "cung_mot", "tuc_la", "hieu_la",
    "duoc_goi", "biet_danh", "alias", "tuoc_hieu", "con_goi", "khai_sinh"
]
NEGATIVE_KEYWORDS = [
    "cha_cua", "con_cua", "anh_em", "vo_cua", "chong_cua", "me_cua",
    "thay_cua", "hoc_tro", "mon_khach", "ke_vi", "chau_cua", "chu_cua",
    "bac_cua", "vua_toi", "em_cua", "anh_cua", "chi_cua"
]

def classify_relation(rel_type):
    norm = _strip_accents(rel_type).lower()
    if any(kw in norm for kw in POSITIVE_KEYWORDS):
        return "POSITIVE"
    if any(kw in norm for kw in NEGATIVE_KEYWORDS):
        return "NEGATIVE"
    return "NEUTRAL"


def get_relation_between(tx, id_a, id_b):
    result = tx.run("""
        MATCH (a:Entity {id:$a})-[r]-(b:Entity {id:$b})
        WHERE type(r) <> 'SAME_AS' AND type(r) <> 'MENTIONED_IN'
        RETURN type(r) AS rel_type
    """, a=id_a, b=id_b)
    return [rec["rel_type"] for rec in result]


# ------------------------------------------------------------
# 4. REGEX ALIAS DETECTION
# ------------------------------------------------------------

ALIAS_TRIGGER_WORDS = [
    "còn gọi là", "còn có tên là", "tên khác là", "tục gọi là",
    "tục danh là", "hiệu là", "tước hiệu là", "húy là", "tên húy là",
    "tức là", "tức", "bí danh"
]

def find_alias_evidence(name_a, name_b, chunk_texts):
    na, nb = re.escape(name_a), re.escape(name_b)
    for text in chunk_texts:
        for trigger in ALIAS_TRIGGER_WORDS:
            for pat in (rf"{na}[,\s]*{trigger}\s+{nb}", rf"{nb}[,\s]*{trigger}\s+{na}"):
                m = re.search(pat, text, flags=re.IGNORECASE)
                if m:
                    return m.group(0)
        pat_paren = rf"({na}\s*\([^)]*{nb}[^)]*\))|({nb}\s*\([^)]*{na}[^)]*\))"
        m = re.search(pat_paren, text, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return None


# ------------------------------------------------------------
# 5. CANDIDATE RETRIEVAL (blocking, rẻ, không cần LLM)
# ------------------------------------------------------------

def _name_similarity(a, b):
    return difflib.SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()

def get_candidates(new_entity, canonical_pool, max_candidates=8):
    """canonical_pool: list dict {id, name, type, chunks}"""
    scored = []
    na = normalize_name(new_entity["name"])
    tokens_a = set(na.split())

    for canon in canonical_pool:
        if canon["type"] != new_entity["type"]:
            continue
        if canon["id"] == new_entity["id"]:
            continue

        nb = normalize_name(canon["name"])
        score = 0.0

        if na in nb or nb in na:
            score += 3
        tokens_b = set(nb.split())
        if tokens_a & tokens_b:
            score += 2
        if set(new_entity.get("chunks", [])) & set(canon.get("chunks", [])):
            score += 2
        sim = _name_similarity(new_entity["name"], canon["name"])
        if sim > 0.85:
            score += 1.5 * sim
        elif sim > 0.6:
            score += 0.5 * sim

        if score > 0:
            scored.append((canon, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:max_candidates]]


# ------------------------------------------------------------
# 6. LLM: 1 lần gọi so khớp với NHIỀU candidate cùng lúc
# ------------------------------------------------------------

def llm_match_against_candidates(new_entity, candidates, new_chunks_text):
    if not candidates:
        return None, 0.0

    cand_block = "\n".join(
        f"{i+1}. \"{c['name']}\" | Ngữ cảnh: {' '.join(c.get('chunks', []))[:250]}"
        for i, c in enumerate(candidates)
    )

    prompt = f"""
Thực thể mới: "{new_entity['name']}"
Ngữ cảnh của thực thể mới: {new_chunks_text[:400]}

Danh sách ứng viên có thể là CÙNG một đối tượng ngoài đời thật với thực thể mới:
{cand_block}

QUY TẮC:
1. Chỉ chọn khi có bằng chứng TRỰC TIẾP (tên khác, tên húy, tên hiệu, tức là...).
2. Có liên quan/quan hệ với nhau (anh em, cha con, vợ chồng, thầy trò, vua tôi...)
   là bằng chứng CHỐNG LẠI việc là cùng 1 đối tượng, không phải ủng hộ.
3. Nếu không ứng viên nào chắc chắn khớp, trả về match = null.
4. Chỉ trả JSON, không giải thích.

{{"match_index": <số thứ tự ứng viên hoặc null>, "confidence": 0.0, "evidence": "..."}}
"""
    raw = call_ollama(prompt, format_json=True, temperature=0, debug_tag="resolve_entity")
    try:
        result = json.loads(raw)
        idx = result.get("match_index")
        conf = float(result.get("confidence", 0))
        evidence = result.get("evidence", "").strip()
        if idx is None or not evidence:
            return None, 0.0
        idx = int(idx) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx], conf
        return None, 0.0
    except Exception:
        return None, 0.0


# ------------------------------------------------------------
# 7. NGƯỠNG CONFIDENCE THEO KÍCH THƯỚC CLUSTER
# ------------------------------------------------------------

MAX_CLUSTER_SIZE = 8

def threshold_for(cluster_size):
    if cluster_size <= 2:
        return 0.70
    if cluster_size <= 4:
        return 0.85
    return 0.95


# ------------------------------------------------------------
# 8. CHỌN CANONICAL ỔN ĐỊNH
# ------------------------------------------------------------

def choose_canonical(members):
    """members: list dict có ít nhất {id, name}, tốt nhất có n_chunks/chunks"""
    def sort_key(m):
        n_chunks = m.get("n_chunks") or len(m.get("chunks", []))
        return (-n_chunks, -len(m["name"]), m["id"])
    return sorted(members, key=sort_key)[0]


# ------------------------------------------------------------
# 9. RESOLVE 1 ENTITY MỚI: regex -> relation -> LLM (1 call)
# ------------------------------------------------------------

def resolve_entity(tx_session, new_entity, candidates):
    # Lớp 1: regex
    for canon in candidates:
        evidence = find_alias_evidence(
            new_entity["name"], canon["name"],
            new_entity.get("chunks", []) + canon.get("chunks", [])
        )
        if evidence:
            return canon, "regex", 1.0

    # Lớp 2: relation semantics
    remaining = []
    for canon in candidates:
        rels = tx_session.execute_read(get_relation_between, new_entity["id"], canon["id"])
        classes = {classify_relation(r) for r in rels}
        if "NEGATIVE" in classes:
            continue  # loại thẳng, không đưa xuống LLM
        if "POSITIVE" in classes:
            return canon, "relation_positive", 1.0
        remaining.append(canon)

    if not remaining:
        return None, "no_candidate", 0.0

    # Lớp 3: LLM, 1 lần gọi cho tất cả candidate còn lại
    new_chunks_text = " ".join(new_entity.get("chunks", []))
    match, conf = llm_match_against_candidates(new_entity, remaining, new_chunks_text)
    if match is None:
        return None, "no_match", conf

    cluster_size = match.get("cluster_size", 1)
    if conf >= threshold_for(cluster_size) and cluster_size < MAX_CLUSTER_SIZE:
        return match, "llm", conf

    return None, "below_threshold_or_cluster_full", conf


# ------------------------------------------------------------
# 10. PIPELINE CHÍNH: chỉ resolve entity MỚI (cluster_id = chính id)
#     so với các CANONICAL đã tồn tại. Không đụng cluster cũ.
# ------------------------------------------------------------

def run_entity_resolution():
    """Chỉ merge khi có bằng chứng chắc chắn (regex hoặc relation tường minh).
    KHÔNG dùng LLM để tự quyết merge nữa — quá rủi ro, đẩy việc này sang query time."""
    print("\n" + "=" * 70)
    print("ENTITY RESOLUTION (SAFE-ONLY: regex + relation_positive)")
    print("=" * 70)

    with driver.session() as session:
        pending = session.run("""
            MATCH (e:Entity)
            WHERE e.is_canonical = true AND e.resolved IS NULL
              AND NOT e.type IN $skip
            OPTIONAL MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
            RETURN e.id AS id, e.name AS name, e.type AS type, collect(c.text) AS chunks
        """, skip=list(SKIP_DEDUP_TYPES))
        pending = [dict(r) for r in pending]

        canonical_pool = session.run("""
            MATCH (e:Entity {is_canonical: true})
            OPTIONAL MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
            RETURN e.id AS id, e.name AS name, e.type AS type, collect(c.text) AS chunks
        """)
        canonical_pool = [dict(r) for r in canonical_pool]

    for entity in pending:
        candidates = get_candidates(entity, canonical_pool)
        matched = None

        for canon in candidates:
            evidence = find_alias_evidence(
                entity["name"], canon["name"], entity["chunks"] + canon["chunks"]
            )
            if evidence:
                matched = (canon, "regex")
                break

        if not matched:
            with driver.session() as session:
                for canon in candidates:
                    rels = session.execute_read(get_relation_between, entity["id"], canon["id"])
                    classes = {classify_relation(r) for r in rels}
                    if "NEGATIVE" in classes:
                        continue
                    if "POSITIVE" in classes:
                        matched = (canon, "relation_positive")
                        break

        with driver.session() as session:
            if matched:
                canon, method = matched
                session.run("""
                    MATCH (e:Entity {id:$eid}), (c:Entity {id:$cid})
                    SET e.cluster_id = $cid, e.is_canonical = false, e.resolved = true
                    MERGE (e)-[:SAME_AS]->(c)
                """, eid=entity["id"], cid=canon["id"])
                print(f"[MERGE:{method}] '{entity['name']}' -> '{canon['name']}'")
            else:
                session.run("MATCH (e:Entity {id:$eid}) SET e.resolved = true", eid=entity["id"])
                canonical_pool.append(entity)
                print(f"[KEEP SEPARATE] '{entity['name']}' — không đủ bằng chứng, để nguyên")

    print("\nXong. Case chưa merge sẽ được xử lý lúc trả lời câu hỏi (query time).")
    

# ============================================================
# PROCESS ONE CHUNK
# ============================================================

def process_chunk(text, chunk_number):
    chunk_id = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

    if is_chunk_already_processed(chunk_id):
        print(f"[SKIP] Chunk {chunk_number} đã xử lý trước đó ({chunk_id})")
        return chunk_id

    print(f"\n\n{'=' * 70}")
    print(f"PROCESS CHUNK {chunk_number}")
    print(f"Chunk ID: {chunk_id}")
    print("=" * 70)

    kg_data = extract_kg(text)

    # Validate entity: chỉ giữ entity có đủ "id" và "name"
    valid_entities = []
    for entity in kg_data.get("entities", []):
        if "id" in entity and "name" in entity and entity["name"]:
            valid_entities.append(entity)
        else:
            print(f"[SKIP ENTITY] Entity thiếu field hợp lệ, bỏ qua: {entity}")

    id_map = {}
    for entity in valid_entities:
        global_id = make_entity_id(entity["name"])
        id_map[entity["id"]] = global_id

    try:
        with driver.session() as session:
            session.execute_write(insert_chunk, chunk_id, text)

            for entity in valid_entities:
                session.execute_write(insert_entity, entity, chunk_id)

            for relation in kg_data.get("relations", []):
                source = relation.get("source")
                target = relation.get("target")
                if source in id_map and target in id_map:
                    session.execute_write(
                        insert_relation,
                        id_map[source],
                        id_map[target],
                        relation.get("relation", "RELATED_TO")
                    )

        mark_chunk_as_processed(chunk_id)

    except Exception as e:
        print(f"[ERROR chunk {chunk_number}] {e}")
        print(f"Bỏ qua chunk này, tiếp tục chunk tiếp theo...")

    return chunk_id

def is_chunk_already_processed(chunk_id):
    def _q(tx):
        r = tx.run("MATCH (c:Chunk {id:$id}) RETURN coalesce(c.processed,false) AS p", id=chunk_id).single()
        return bool(r and r["p"])
    with driver.session() as s:
        return s.execute_read(_q)

def mark_chunk_as_processed(chunk_id):
    with driver.session() as s:
        s.run("MATCH (c:Chunk {id:$id}) SET c.processed = true", id=chunk_id)


# ============================================================
# LOCAL SEARCH (TRAVERSE QUA SAME_AS)
# ============================================================

def local_search(entity_name, hops=2, limit_chunks=5):
    def _query(tx):
        query = f"""
        MATCH (start:Entity)
        WHERE toLower(start.name) = toLower($name)
        WITH start,
             CASE WHEN start.is_canonical THEN start.id ELSE start.cluster_id END AS root_id
        MATCH (member:Entity {{cluster_id: root_id}})
        WITH collect(DISTINCT member) AS cluster_members
        UNWIND cluster_members AS r
        OPTIONAL MATCH (r)-[*1..{hops}]-(neighbor:Entity)
        WITH cluster_members, collect(DISTINCT neighbor) AS neighbors
        WITH cluster_members + neighbors AS all_nodes
        UNWIND all_nodes AS n
        MATCH (n)-[:MENTIONED_IN]->(c:Chunk)
        RETURN DISTINCT
            n.name AS entity_name,
            n.type AS entity_type,
            c.id AS chunk_id,
            c.text AS chunk_text
        LIMIT $limit
        """
        result = tx.run(query, name=entity_name, limit=limit_chunks)
        return [dict(record) for record in result]

    with driver.session() as session:
        return session.execute_read(_query)


def get_same_as_groups(entity_names):
    def _query(tx):
        query = """
        UNWIND $names AS name
        MATCH (e:Entity)
        WHERE toLower(e.name) = toLower(name)
        OPTIONAL MATCH (e)-[:SAME_AS*1..2]-(alias:Entity)
        WITH e, collect(DISTINCT alias.name) AS aliases
        RETURN e.name AS name, aliases
        """
        result = tx.run(query, names=entity_names)
        return [dict(r) for r in result]

    with driver.session() as session:
        rows = session.execute_read(_query)

    seen_groups = set()
    groups = []
    for row in rows:
        if row["aliases"]:
            group = frozenset([row["name"]] + row["aliases"])
            if len(group) > 1 and group not in seen_groups:
                seen_groups.add(group)
                groups.append(group)
    return groups


def find_similar_unmerged_entities(entity_name, entity_type=None, threshold=0.6):
    def _query(tx):
        query = """
            MATCH (e:Entity)
            WHERE ($type IS NULL OR e.type = $type)
            RETURN e.name AS name, e.id AS id, e.cluster_id AS cluster_id
        """
        return [dict(r) for r in tx.run(query, type=entity_type)]

    with driver.session() as session:
        all_entities = session.execute_read(_query)

    na_norm = normalize_name(entity_name)
    na_tokens = set(na_norm.split())

    matches = []
    for e in all_entities:
        nb_norm = normalize_name(e["name"])
        if nb_norm == na_norm:
            continue
        nb_tokens = set(nb_norm.split())

        # CHỈ coi là candidate nếu: 1 tên là substring của tên kia,
        # HOẶC share >= 2 token chung (không phải chỉ 1 âm tiết trùng ngẫu nhiên)
        is_substring = na_norm in nb_norm or nb_norm in na_norm
        shared_tokens = na_tokens & nb_tokens

        if is_substring or len(shared_tokens) >= 2:
            matches.append(e["name"])

    return matches


def build_context(entity_names, hops=2):
    if isinstance(entity_names, str):
        entity_names = [entity_names]

    all_entities_found = set()
    chunk_map = {}
    uncertain_aliases = {}   # entity_name -> list tên tương tự chưa chắc cùng người

    for name in entity_names:
        rows = local_search(name, hops=hops)
        for row in rows:
            all_entities_found.add(row["entity_name"])
            chunk_map[row["chunk_id"]] = row["chunk_text"]

        # Bù thêm: tìm entity tên giống nhưng KHÔNG cùng cluster
        similar = find_similar_unmerged_entities(name)
        if similar:
            uncertain_aliases[name] = similar
            for alt_name in similar:
                alt_rows = local_search(alt_name, hops=hops)
                for row in alt_rows:
                    all_entities_found.add(row["entity_name"])
                    chunk_map[row["chunk_id"]] = row["chunk_text"]

    if not chunk_map:
        return None

    context = ""

    same_as_groups = get_same_as_groups(list(all_entities_found))
    if same_as_groups:
        context += "GHI CHÚ (đã xác nhận CHẮC CHẮN là cùng một đối tượng):\n"
        for group in same_as_groups:
            context += f"- {' và '.join(sorted(group))} là CÙNG MỘT đối tượng.\n"
        context += "\n"

    if uncertain_aliases:
        context += "GHI CHÚ (CHƯA CHẮC CHẮN, cần bạn tự xét ngữ cảnh bên dưới để quyết định):\n"
        for name, alts in uncertain_aliases.items():
            context += f"- \"{name}\" và {alts} CÓ THỂ là cùng một đối tượng với tên gọi khác nhau, HOẶC là các đối tượng khác nhau (ví dụ: quan hệ họ hàng, đồng nghiệp). Hãy đọc kỹ đoạn văn bên dưới để tự phân biệt, đừng mặc định là giống nhau.\n"
        context += "\n"

    context += "Các entity liên quan:\n" + ", ".join(sorted(all_entities_found)) + "\n\n"
    context += "Các chunk liên quan:\n\n"
    for chunk_id, text in chunk_map.items():
        context += f"[Chunk {chunk_id}]\n{text}\n\n---\n\n"

    return context


def build_context_all_chunks():
    def _query(tx):
        result = tx.run("MATCH (c:Chunk) RETURN c.text AS chunk_text")
        return [r["chunk_text"] for r in result]
    with driver.session() as session:
        chunks = session.execute_read(_query)
    
    if not chunks:
        return ""
    
    context = "Toàn bộ tài liệu (dùng cho câu hỏi liệt kê):\n\n"
    for i, text in enumerate(chunks, 1):
        context += f"[Phần {i}]\n{text}\n\n---\n\n"
    return context


def get_all_entity_names():
    def _query(tx):
        result = tx.run("MATCH (e:Entity) RETURN DISTINCT e.name AS name")
        return [r["name"] for r in result]
    with driver.session() as session:
        return session.execute_read(_query)


def fix_entity_name(raw_name, known_names, threshold=0.8):
    """Tìm tên gần đúng nhất trong graph, dùng khi model trả về tên lẫn ký tự thừa."""
    import difflib
    matches = difflib.get_close_matches(raw_name, known_names, n=1, cutoff=threshold)
    return matches[0] if matches else raw_name


def extract_query_entity(question):
    prompt = f"""
Câu hỏi:
"{question}"

Xác định TẤT CẢ tên riêng (người, bài hát, album, địa điểm...) THỰC SỰ XUẤT HIỆN 
TRONG CÂU HỎI này. 

TUYỆT ĐỐI KHÔNG được thêm tên người/thực thể nào KHÔNG xuất hiện trong câu hỏi, 
kể cả khi bạn biết họ có liên quan trong đời thực (ví dụ: không tự thêm tên 
nghệ sĩ khác chỉ vì cùng thể loại nhạc).

CHỈ ghi tên riêng cụ thể, KHÔNG ghi cụm mô tả chung chung như "bài hát", 
"nhạc sĩ", "tempo", "thể loại"...

Mỗi tên entity trên 1 dòng riêng. Không giải thích, không đánh số.
Nếu không có tên riêng nào trong câu hỏi: UNKNOWN
"""

    raw = chat_ollama(
    [{"role": "user", "content": prompt}],
    temperature=0,
    debug_tag="extract_entity"
    )

    if raw.strip(". \n").upper() == "UNKNOWN" or not raw:
        return []

    QUESTION_WORDS = {"ai", "gì", "nào", "đâu", "sao", "thế nào", "bao nhiêu", "khi nào"}

    entities = [line.strip() for line in raw.split("\n") if line.strip()]
    entities = [e for e in entities if e.strip(". \n").upper() != "UNKNOWN"]
    entities = [e for e in entities if normalize_name(e) not in QUESTION_WORDS]

    # Sửa lại tên gần đúng dựa trên tên thật có trong graph
    known_names = get_all_entity_names()
    entities = [fix_entity_name(e, known_names) for e in entities]

    return entities


def generate_answer(question, context):
    prompt = f"""
Bạn là chatbot Graph RAG.

Dựa CHỈ trên context được cung cấp, hãy trả lời câu hỏi bằng tiếng Việt.

Nếu context có phần "GHI CHÚ (CHƯA CHẮC CHẮN)", đó là gợi ý hệ thống retrieval 
đưa ra do 2 tên giống nhau về mặt chữ viết — KHÔNG có nghĩa chúng chắc chắn là 
cùng một đối tượng. Bạn phải tự đọc các đoạn văn (chunk) bên dưới để xác định 
thật sự có phải cùng một người/vật hay không (dựa vào câu văn có nói rõ 
"còn gọi là", "tức là"... hay không), TRƯỚC KHI dùng thông tin đó để trả lời.

Không được tự bịa thông tin ngoài context.
Nếu context không đủ thông tin, hãy nói rõ rằng không tìm thấy đủ thông tin.

CONTEXT:
{context}

CÂU HỎI:
{question}

TRẢ LỜI (chỉ dùng tiếng Việt, không dùng bất kỳ ngôn ngữ nào khác kể cả tiếng Anh hay tiếng Trung):
"""
    return chat_ollama([{"role": "user", "content": prompt}], debug_tag="generate_answer")


def chatbot(question, hops=2):
    entities = extract_query_entity(question)

    print("\n[Query Entity]:", entities)

    if not entities:
        LISTING_KEYWORDS = ["sắp xếp", "trình tự", "liệt kê", "những người nào", "những ai"]
        if any(kw in question.lower() for kw in LISTING_KEYWORDS):
            print("  [FALLBACK] Không có entity, câu hỏi dạng liệt kê -> dùng toàn bộ chunk")
            context = build_context_all_chunks()
            return generate_answer(question, context)
        else:
            print("  [FALLBACK] Không có entity -> global_search")
            return global_search(question)

    context = build_context(entities, hops=hops)

    if not context:
        return f"Không tìm thấy thông tin liên quan đến {', '.join(entities)}."

    print("\n========== RETRIEVED CONTEXT (LOCAL) ==========")
    print(context)
    print("===============================================")

    return generate_answer(question, context)


# ============================================================
# COMMUNITY DETECTION (GDS LEIDEN)
# ============================================================

def run_community_detection():
    with driver.session() as session:
        session.run("""
            CALL gds.graph.drop('entityGraph', false)
        """)

        session.run("""
            CALL gds.graph.project(
                'entityGraph',
                'Entity',
                {
                    ALL: {
                        type: '*',
                        orientation: 'UNDIRECTED'
                    }
                }
            )
        """)

        result = session.run("""
            CALL gds.leiden.write('entityGraph', {
                writeProperty: 'communityId'
            })
            YIELD communityCount, modularity
            RETURN communityCount, modularity
        """)

        record = result.single()
        print(f"Số community: {record['communityCount']}, modularity: {record['modularity']}")

        session.run("CALL gds.graph.drop('entityGraph', false)")


def get_entities_by_community():
    def _query(tx):
        result = tx.run("""
            MATCH (e:Entity)
            WHERE e.communityId IS NOT NULL
            OPTIONAL MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
            RETURN
                e.communityId AS community_id,
                e.name AS entity_name,
                collect(DISTINCT c.text) AS chunk_texts
        """)
        return [dict(r) for r in result]

    with driver.session() as session:
        rows = session.execute_read(_query)

    communities = {}
    for row in rows:
        cid = row["community_id"]
        if cid not in communities:
            communities[cid] = {"entities": set(), "chunks": set()}
        communities[cid]["entities"].add(row["entity_name"])
        for chunk in row["chunk_texts"]:
            if chunk:
                communities[cid]["chunks"].add(chunk)
    return communities


def summarize_community(entities, chunks):
    entities_str = ", ".join(sorted(entities))
    chunks_str = "\n---\n".join(sorted(chunks))
    prompt = f"""
Dưới đây là các thực thể và đoạn văn liên quan thuộc cùng 1 nhóm chủ đề:

Các thực thể: {entities_str}

Đoạn văn liên quan:
{chunks_str}

Hãy tóm tắt ngắn gọn (3-5 câu) chủ đề chính mà nhóm này đề cập
(chỉ dùng tiếng Việt, không dùng bất kỳ ngôn ngữ nào khác).
"""

    return chat_ollama(
    [{"role": "user", "content": prompt}],
    debug_tag="summarize_community"
)


def build_community_summaries():
    communities = get_entities_by_community()
    summaries = {}
    for cid, data in communities.items():
        if len(data["entities"]) < 2:
            continue
        summary = summarize_community(data["entities"], data["chunks"])
        summaries[str(cid)] = {   # Ép str(cid) thay vì để int cid
            "entities": data["entities"],
            "summary": summary
        }
    return summaries


# ============================================================
# MERGE SIMILAR SUMMARIES
# ============================================================

def merge_similar_summaries(summaries):
    if len(summaries) <= 1:
        return summaries

    summaries_list = list(summaries.items())
    formatted = "\n\n".join(
        f"[ID: {cid}]\nEntities: {', '.join(v['entities'])}\nSummary: {v['summary']}"
        for cid, v in summaries_list
    )

    prompt = f"""
Dưới đây là danh sách các nhóm chủ đề (mỗi nhóm có ID riêng), được tạo tự động
từ thuật toán phân cụm. Một số nhóm có thể trùng lặp nội dung (cùng nói về
1 chủ đề), do thuật toán chia quá vụn.

{formatted}

Hãy xác định các nhóm ID nào nên được GỘP LẠI vì chúng cùng nói về 1 chủ đề.

Trả về ĐÚNG định dạng JSON, không giải thích gì thêm.
QUAN TRỌNG: chỉ ghi số ID trần, KHÔNG thêm chữ "ID:" hay bất kỳ tiền tố nào khác.

Ví dụ đúng: {{"merged_groups": [["2", "8", "9"], ["4"]]}}
Ví dụ SAI (không được làm): {{"merged_groups": [["ID: 2", "ID: 8"]]}}
"""

    raw = chat_ollama(
    [{"role": "user", "content": prompt}],
    format_json=True,
    temperature=0,
    debug_tag="merge_summaries"
)
    print(f"\n===== MERGE RAW OUTPUT =====\n{raw}\n=============================\n")

    try:
        result = json.loads(raw)
        merged_groups = result["merged_groups"]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Lỗi parse merge groups: {e}\nRaw: {raw}")
        return summaries

    def clean_id(raw_id):
        raw_id = str(raw_id).strip()
        raw_id = re.sub(r"^(ID\s*[:_]?\s*)", "", raw_id, flags=re.IGNORECASE)
        return raw_id.strip()

    merged_groups = [[clean_id(cid) for cid in group] for group in merged_groups]

    print(f"[MERGE DEBUG] merged_groups (đã clean) = {merged_groups}")

    merged_summaries = {}

    for group in merged_groups:
        group_entities = set()
        group_summaries = []

        for cid in group:
            if cid in summaries:
                group_entities.update(summaries[cid]["entities"])
                group_summaries.append(summaries[cid]["summary"])
            else:
                print(f"[MERGE WARNING] ID '{cid}' không khớp key nào trong summaries: {list(summaries.keys())}")

        if not group_entities:
            continue

        new_id = "_".join(group)

        if len(group_summaries) == 1:
            merged_summaries[new_id] = {
                "entities": list(group_entities),
                "summary": group_summaries[0]
            }
        else:
            combined_text = "\n".join(group_summaries)
            merge_prompt = f"""
Dưới đây là nhiều đoạn tóm tắt cùng nói về 1 chủ đề (do bị chia vụn từ thuật toán):

{combined_text}

Hãy viết lại thành 1 đoạn tóm tắt DUY NHẤT, ngắn gọn (3-5 câu), không lặp ý,
chỉ dùng tiếng Việt.
"""
            merge_response = chat_ollama(
        [{"role": "user", "content": merge_prompt}],
        debug_tag="merge_summary_content"
        )

        merged_summaries[new_id] = {
        "entities": list(group_entities),
        "summary": merge_response
        }

    return merged_summaries


# ============================================================
# GLOBAL SEARCH
# ============================================================

def get_or_build_summaries():
    if os.path.exists(GLOBAL_SUMMARIES_CACHE):
        with open(GLOBAL_SUMMARIES_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)

    run_community_detection()
    summaries = build_community_summaries()

    summaries = merge_similar_summaries(summaries)

    serializable = {
        str(cid): {"entities": list(v["entities"]), "summary": v["summary"]}
        for cid, v in summaries.items()
    }
    with open(GLOBAL_SUMMARIES_CACHE, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    return serializable


def global_search(question):
    summaries = get_or_build_summaries()

    all_summaries_text = "\n\n".join(
        f"[Nhóm {cid}] (gồm: {', '.join(v['entities'])})\n{v['summary']}"
        for cid, v in summaries.items()
    )

    prompt = f"""
Dưới đây là tóm tắt các nhóm chủ đề trong toàn bộ tài liệu:

{all_summaries_text}

Dựa trên các tóm tắt trên, hãy trả lời câu hỏi sau
(chỉ dùng tiếng Việt, không dùng bất kỳ ngôn ngữ nào khác):

{question}
"""

    return chat_ollama(
    [{"role": "user", "content": prompt}],
    debug_tag="global_search"
)


# ============================================================
# ROUTER
# ============================================================

GLOBAL_KEYWORDS = [
    "tổng quan", "tóm tắt", "nói về gì", "nói về những gì", 
    "chủ đề chính", "toàn bộ", "khái quát", "tổng hợp", "nội dung chính"
]

def route_question(question):
    q_lower = question.lower()
    if any(kw in q_lower for kw in GLOBAL_KEYWORDS):
        return "global"
    return "local"


# ============================================================
# MAIN
# ============================================================

driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

if __name__ == "__main__":
    

    try:
        source_text = load_texts_from_folder()

        if not source_text:
            print("Không có nội dung để xử lý. Vui lòng kiểm tra thư mục 'data/input/'.")
            exit()

        chunks = split_into_chunks(source_text, chunk_size=1500, chunk_overlap=200)

        print("\n======================================")
        print(f"Đã chia thành {len(chunks)} chunks")
        print("======================================")

        for i, chunk in enumerate(chunks, start=1):
            print(f"\nChunk {i}: {len(chunk)} characters")

        print("\n\n======================================")
        print("BUILDING KNOWLEDGE GRAPH")
        print("======================================")

        for i, chunk in enumerate(chunks, start=1):
            process_chunk(chunk, i)
            
        migrate_to_cluster_schema()
        run_entity_resolution()

        total = extraction_stats["success"] + extraction_stats["failed"]
        if total:
            print(f"\nExtraction: {extraction_stats['success']} ok, {extraction_stats['failed']} lỗi ({extraction_stats['failed']/total*100:.1f}%)")

    finally:
        save_token_log()
        driver.close()