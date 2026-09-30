import os
import json
import re
import string
import collections
import concurrent.futures
import requests

import asyncio
from rag_light import build_rag, lightrag_answer

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"
JUDGE_MODEL = "llama3.1:8b"

def call_ollama(prompt, format_json=False, temperature=0.0, model=None, debug_tag=""):
    payload = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }
    if format_json:
        payload["format"] = "json"
        
    print(f"  [Ollama call | {payload['model']}] Đang gọi model...")
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    response_text = r.json().get("response", "").strip()
    response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
    return response_text

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_FILE = os.path.join(BASE_DIR, "data", "evaluation", "test_ques_2.json")
REPORT_FILE = os.path.join(BASE_DIR, "reports", "evaluation_report_light.json")
os.makedirs(os.path.dirname(REPORT_FILE) or ".", exist_ok=True)

# ============================================================
# CÁC HÀM TÍNH TOÁN EXACT MATCH VÀ F1 TRUYỀN THỐNG
# ============================================================

def normalize_text(s):
    """Chuẩn hóa chuỗi văn bản phục vụ tính F1 và EM."""
    if s is None:
        return ""
    s = str(s).lower()
    # Loại bỏ dấu câu
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # Chuẩn hóa khoảng trắng
    s = re.sub(r"\s+", " ", s).strip()
    return s

def compute_exact_match(prediction, ground_truth):
    """Đo lường Exact Match (0 hoặc 1)."""
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0

def compute_f1(prediction, ground_truth):
    """Đo lường Token-level F1 Score (0.0 đến 1.0)."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)

# ============================================================
# LLM JUDGE (ĐÁNH GIÁ NGỮ CẢNH VÀ TRUNG THỰC)
# ============================================================

def llm_judge(question, ground_truth, context, answer):
    """Dùng LLM chấm điểm theo thang 1-5 cho các tiêu chí ngữ nghĩa."""
    prompt = f"""
Bạn là giám khảo chấm điểm hệ thống RAG dựa trên thông tin đối chiếu sau:

Câu hỏi: {question}
Đáp án chuẩn (Ground Truth): {ground_truth}
Ngữ cảnh tìm được (Context): {context}
Câu trả lời của Bot: {answer}

Chấm điểm từ 1 đến 5 cho 4 tiêu chí sau:
1. context_relevance: Ngữ cảnh có chứa thông tin liên quan đến câu hỏi không?
2. context_sufficiency: Ngữ cảnh có cung cấp đủ cơ sở để trả lời câu hỏi không?
3. faithfulness: Câu trả lời có dựa hoàn toàn vào Ngữ cảnh không (không bịa đặt)?
4. answer_relevance: Câu trả lời có bám sát câu hỏi và khớp ý với Đáp án chuẩn không?

TRẢ VỀ ĐÚNG ĐỊNH DẠNG JSON SAU, KHÔNG GIẢI THÍCH:
{{
  "context_relevance": 5,
  "context_sufficiency": 5,
  "faithfulness": 5,
  "answer_relevance": 5
}}
"""
    try:
        raw = call_ollama(
            prompt,
            format_json=True,
            temperature=0,
            model=JUDGE_MODEL,
            debug_tag="llm_judge",
        )
        if raw.startswith("```"):
            raw = raw.split("```")[1].removeprefix("json").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"Lỗi chấm điểm LLM: {e}")
        return {
            "context_relevance": 0,
            "context_sufficiency": 0,
            "faithfulness": 0,
            "answer_relevance": 0
        }

# ============================================================
# CHƯƠNG TRÌNH THỰC THI CHÍNH
# ============================================================

def _print_summary(results, total_q):
    valid_results = [r for r in results if r.get("status") == "success"]
    answerable = [r for r in valid_results if r.get("ground_truth") is not None]
    unanswerable = [r for r in valid_results if r.get("ground_truth") is None]

    if answerable:
        avg_f1 = sum(r["metrics"]["f1_score"] for r in answerable) / len(answerable)
        avg_em = sum(r["metrics"]["exact_match"] for r in answerable) / len(answerable)
        avg_cr = sum(r["metrics"]["context_relevance"] for r in answerable) / len(answerable)
        avg_cs = sum(r["metrics"]["context_sufficiency"] for r in answerable) / len(answerable)
        avg_faith = sum(r["metrics"]["faithfulness"] for r in answerable) / len(answerable)
        avg_ar = sum(r["metrics"]["answer_relevance"] for r in answerable) / len(answerable)

        print("\n" + "=" * 50)
        print("BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (SUMMARY)")
        print(f"Tổng số câu hoàn thành: {len(valid_results)}/{total_q}")
        print(f"- Câu hỏi có đáp án (Answerable)   : {len(answerable)}")
        print(f"- Câu hỏi không có đáp án (Unanswerable): {len(unanswerable)}")
        print("-" * 50)
        print(f"Metrics cho câu hỏi Answerable:")
        print(f"Exact Match (EM)         : {avg_em:.4f}")
        print(f"F1 Score                 : {avg_f1:.4f}")
        print(f"Context Relevance (1-5)  : {avg_cr:.2f}")
        print(f"Context Sufficiency (1-5): {avg_cs:.2f}")
        print(f"Faithfulness (1-5)       : {avg_faith:.2f}")
        print(f"Answer Relevance (1-5)   : {avg_ar:.2f}")
        
        if unanswerable:
            correctly_refused = sum(1 for r in unanswerable if "không tìm thấy đủ thông tin" in r["answer"].lower())
            print("-" * 50)
            print(f"Metrics cho câu hỏi Unanswerable:")
            print(f"Correctly Refused        : {correctly_refused}/{len(unanswerable)}")

        print("=" * 50)
    else:
        print("\nChưa có câu hỏi nào hoàn thành để kết xuất báo cáo.")

async def run_evaluation():
    if not os.path.exists(TEST_FILE):
        print(f"Không tìm thấy file '{TEST_FILE}'! Vui lòng tạo file trước.")
        return

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Resume nếu có kết quả cũ
    results = []
    if os.path.exists(REPORT_FILE):
        try:
            with open(REPORT_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Đã tải {len(results)} kết quả hoàn thành trước đó.")
        except Exception:
            results = []

    processed_questions = {r["question"] for r in results if r.get("status") in ("success", "fail")}

    # Khởi tạo RAG
    print("\n=== ĐANG KHỞI TẠO LIGHTRAG ===")
    rag = await build_rag()

    # ---- PHA 1: extract entity + generate answer (chỉ dùng qwen3:8b) ----
    print("\n=== PHA 1: EXTRACT + GENERATE (qwen3:8b) ===")
    for idx, item in enumerate(test_data, 1):
        q = item["question"]
        
        # Bỏ qua nếu đã hoàn thành (success/fail)
        if q in processed_questions:
            print(f"[{idx}/{len(test_data)}] => [SKIP] Đã xử lý.")
            continue
            
        # Kiểm tra xem có đang ở pending_judge không (đã qua pha 1)
        existing_result = next((r for r in results if r["question"] == q), None)
        if existing_result and existing_result.get("status") == "pending_judge":
            print(f"[{idx}/{len(test_data)}] => [PENDING_JUDGE] Đã qua pha 1.")
            continue

        print(f"[{idx}/{len(test_data)}] {q}")

        answer = await lightrag_answer(q, rag)
        context = "N/A (LightRAG context is managed internally)"

        results.append({
            "status": "pending_judge",
            "question": q,
            "ground_truth": item["ground_truth"],
            "extracted_entities": [],
            "context": context,
            "answer": answer,
            "metrics": {
                "exact_match": compute_exact_match(answer, item["ground_truth"]),
                "f1_score": compute_f1(answer, item["ground_truth"]),
            }
        })
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # ---- Unload qwen3:8b khỏi VRAM trước khi qua pha judge ----
    print("\n=== UNLOAD qwen3:8b, giải phóng VRAM ===")
    try:
        requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "keep_alive": 0})
    except Exception as e:
        print(f"Lỗi khi unload {OLLAMA_MODEL}: {e}")

    # ---- PHA 2: llm_judge (chỉ dùng gemma3:12b, chạy riêng không tranh GPU) ----
    print(f"\n=== PHA 2: LLM JUDGE ({JUDGE_MODEL}) ===")
    for r in results:
        if r.get("status") != "pending_judge":
            continue
        print(f"Judging: {r['question']}")
        llm_scores = llm_judge(r["question"], r["ground_truth"], r["context"], r["answer"])
        r["metrics"].update(llm_scores)
        r["status"] = "success"
        
        # Bỏ context ra khỏi report cho gọn
        if "context" in r:
            del r["context"]  
            
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # ---- unload gemma3:12b sau khi xong ----
    print(f"\n=== UNLOAD {JUDGE_MODEL}, giải phóng VRAM ===")
    try:
        requests.post(OLLAMA_URL, json={"model": JUDGE_MODEL, "keep_alive": 0})
    except Exception as e:
        print(f"Lỗi khi unload {JUDGE_MODEL}: {e}")

    _print_summary(results, len(test_data))

if __name__ == "__main__":
    asyncio.run(run_evaluation())