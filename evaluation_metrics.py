import re
import string
import unicodedata
import collections
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# ============================================================
# 1. TEXT NORMALIZATION (Chuẩn hóa văn bản tiếng Việt & tiếng Anh)
# ============================================================

def remove_vietnamese_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để so sánh ngữ nghĩa từ vựng cốt lõi."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    # Thay thế các ký tự đặc thù như đ, Đ
    text = text.replace("đ", "d").replace("Đ", "D")
    return text

def normalize_for_eval(text: str, remove_articles: bool = True) -> str:
    """
    Chuẩn hóa văn bản:
    - Lowercase
    - Bỏ dấu tiếng Việt
    - Bỏ dấu câu
    - Bỏ mạo từ tiếng Anh (a, an, the)
    - Chuẩn hóa khoảng trắng
    """
    if text is None:
        return ""
    text = str(text).lower()
    text = remove_vietnamese_accents(text)
    # Bỏ dấu câu
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    tokens = text.split()
    if remove_articles:
        articles = {"a", "an", "the"}
        tokens = [t for t in tokens if t not in articles]
    return " ".join(tokens)

# ============================================================
# 2. TOKEN-LEVEL PRECISION, RECALL, F1 (Tính trên short_answer)
# ============================================================

def compute_token_prf(prediction: str, ground_truth: str) -> Dict[str, float]:
    """
    Đo Precision, Recall và F1 ở cấp độ token.
    F1 phạt câu trả lời dài dòng/lan man của model, 
    Precision chỉ ra tỷ lệ từ sinh ra có ích, Recall chỉ ra mức độ bao phủ ý.
    """
    pred_norm = normalize_for_eval(prediction)
    gt_norm = normalize_for_eval(ground_truth)

    pred_tokens = pred_norm.split()
    gt_tokens = gt_norm.split()

    if not pred_tokens or not gt_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }

# ============================================================
# 3. RETRIEVAL METRICS (Recall@k, Hit@k, MRR trên gold_spans)
# ============================================================

try:
    from eval_data_tools import hit_at_k, recall_at_k, norm as norm_span
except ImportError:
    hit_at_k = None
    recall_at_k = None
    norm_span = None

def compute_retrieval_metrics(
    retrieved_contexts: List[str], 
    gold_spans: List[str],
    k_list: List[int] = [1, 3, 5]
) -> Dict[str, float]:
    """
    Đánh giá năng lực truy xuất không cần LLM judge:
    - Hit@k: Có ít nhất 1 gold_span xuất hiện trong top-k context không.
    - Recall@k: Tỷ lệ các gold_spans được bao phủ trong top-k context.
    - MRR: Nghịch đảo thứ hạng của context đầu tiên chứa ít nhất 1 gold_span.
    """
    if not gold_spans:
        return {}

    norm_spans = [normalize_for_eval(s) for s in gold_spans if s]
    norm_contexts = [normalize_for_eval(c) for c in retrieved_contexts if c]

    if not norm_spans or not norm_contexts:
        metrics = {"mrr": 0.0}
        for k in k_list:
            metrics[f"hit@{k}"] = 0.0
            metrics[f"recall@{k}"] = 0.0
        return metrics

    # Tìm thứ hạng đầu tiên xuất hiện gold span
    first_hit_rank = 0
    for rank, ctx in enumerate(norm_contexts, 1):
        if any(span in ctx for span in norm_spans):
            first_hit_rank = rank
            break

    mrr = (1.0 / first_hit_rank) if first_hit_rank > 0 else 0.0

    metrics = {"mrr": round(mrr, 4)}

    # Tính Hit@k và Recall@k
    for k in k_list:
        sub_contexts = norm_contexts[:k]
        combined_topk_text = " ".join(sub_contexts)
        
        # Hit@k
        hit = 1.0 if any(span in combined_topk_text for span in norm_spans) else 0.0
        
        # Recall@k (tỷ lệ gold spans có trong top k)
        spans_found = sum(1 for span in norm_spans if span in combined_topk_text)
        recall_k = spans_found / len(norm_spans)

        metrics[f"hit@{k}"] = round(hit, 4)
        metrics[f"recall@{k}"] = round(recall_k, 4)

    return metrics

# ============================================================
# 4. ENTITY COVERAGE (Tỷ lệ supporting_entities trong Context)
# ============================================================

def compute_entity_coverage(
    retrieved_context: str, 
    supporting_entities: List[str]
) -> float:
    """
    Đo lường mức độ bao phủ thực thể trong context được truy xuất.
    Rất hữu ích cho GraphRAG để kiểm tra graph có tìm được đúng các node chìa khóa.
    """
    if not supporting_entities:
        return 1.0
    
    norm_context = normalize_for_eval(retrieved_context)
    if not norm_context:
        return 0.0

    found_count = 0
    for entity in supporting_entities:
        norm_ent = normalize_for_eval(entity)
        if norm_ent and norm_ent in norm_context:
            found_count += 1

    return round(found_count / len(supporting_entities), 4)

# ============================================================
# 5. REFUSAL & HALLUCINATION CHECK (Unanswerable & Partial)
# ============================================================

REFUSAL_PATTERNS = [
    r"không\s+(tìm\s+thấy|có|nhắc\s+đến|được\s+đề\s+cập|cung\s+cấp|rõ)",
    r"tài\s+liệu\s+không\s+(nói|nêu|ghi|chứa)",
    r"not\s+mentioned",
    r"not\s+(found|provided|stated|specified|available)",
    r"does\s+not\s+(mention|contain|state|provide)",
    r"cannot\s+(find|answer|determine)",
    r"no\s+information",
]

def check_is_refusal(answer: str) -> bool:
    """Kiểm tra xem câu trả lời có chứa cấu trúc từ chối trả lời hay không."""
    if not answer:
        return True
    ans_clean = remove_vietnamese_accents(answer.lower())
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, ans_clean, re.IGNORECASE):
            return True
    return False

def evaluate_answerability(
    answer: str, 
    answerable_type: Any,  # True, False, "true", "false", "partial"
    question_id: Optional[int] = None,
    ground_truth: Optional[str] = None
) -> Dict[str, Any]:
    """
    Đánh giá khả năng từ chối đúng (Correct Refusal), từ chối sai (False Refusal),
    và bịa đặt (Hallucination) trên tập câu unanswerable và partial.
    """
    is_refused = check_is_refusal(answer)
    str_type = str(answerable_type).strip().lower()

    res = {
        "is_refused": is_refused,
        "correctly_refused": None,
        "false_refusal": False,
        "hallucination": False,
        "partial_credit": None
    }

    if str_type in ("false", "unanswerable"):
        # Câu hoàn toàn không có đáp án trong tài liệu
        if is_refused:
            res["correctly_refused"] = 1.0
            res["hallucination"] = False
        else:
            res["correctly_refused"] = 0.0
            res["hallucination"] = True  # Tự bịa câu trả lời khi văn bản không có
    elif str_type in ("true", "answerable"):
        # Câu có đáp án đầy đủ
        if is_refused:
            res["false_refusal"] = True  # Có thông tin nhưng bot lại từ chối trả lời
        else:
            res["false_refusal"] = False
    elif str_type == "partial":
        # Câu có một phần thông tin, một phần không có (ví dụ Q43)
        # Điểm partial credit: Cần vừa nêu được thông tin có sẵn, vừa từ chối/nêu rõ phần không có
        has_refusal = is_refused or any(re.search(p, remove_vietnamese_accents(answer.lower())) for p in [
            r"không\s+nói\s+rõ\s+thời\s+gian", r"không\s+được\s+đề\s+cập", r"does\s+not\s+mention\s+the\s+duration", r"duration.*not.*stated"
        ])
        res["partial_credit"] = 1.0 if has_refusal else 0.5

    return res

# ============================================================
# 6. LLM JUDGE PROMPT (Answer Correctness 3 mức + Key-points + Context)
# ============================================================

def build_llm_judge_prompt(
    question: str,
    answer: str,
    ground_truth: str,
    context: str,
    short_answer: Optional[str] = None,
    key_points: Optional[List[str]] = None,
    answerable: str = "true"
) -> str:
    """
    Prompt Giám khảo LLM tinh gọn:
    - Bỏ answer_relevance (ít thông tin).
    - Giữ context_relevance (1-5), context_sufficiency (1-5), faithfulness (1-5).
    - Thêm answer_correctness: 3 mức [correct / partially_correct / incorrect].
    - Thêm key_points_matched: kiểm tra danh sách key points.
    """
    kp_str = "\n".join([f"  - Key point {i+1}: {kp}" for i, kp in enumerate(key_points or [])])
    if not kp_str:
        kp_str = "  (Không có danh sách key-points cụ thể, đánh giá dựa trên Ground Truth)"

    prompt = f"""Bạn là một giám khảo chuyên gia đánh giá hệ thống Question Answering / RAG.
Hãy đánh giá câu trả lời của mô hình dựa trên dữ liệu đối chiếu dưới đây:

[Câu hỏi]: {question}
[Loại câu hỏi]: answerable = {answerable}
[Ngắn gọn (Short Answer)]: {short_answer or "N/A"}
[Các ý chính (Key Points)]:
{kp_str}
[Đáp án chuẩn (Ground Truth)]: {ground_truth or "Không có đáp án trong tài liệu (Unanswerable)"}
[Ngữ cảnh tìm được (Context)]:
\"\"\"{context}\"\"\"

[Câu trả lời của Bot (Answer)]:
\"\"\"{answer}\"\"\"

--- HƯỚNG DẪN CHẤM ĐIỂM ---
1. context_relevance (1 đến 5): Ngữ cảnh có liên quan đến câu hỏi không?
2. context_sufficiency (1 đến 5): Ngữ cảnh có đủ thông tin để trả lời trọn vẹn câu hỏi không?
3. faithfulness (1 đến 5): Câu trả lời có dựa HOÀN TOÀN vào Ngữ cảnh được cấp không (1 = bịa đặt hoàn toàn, 5 = hoàn toàn trung thực với ngữ cảnh)?
4. answer_correctness (chọn 1 trong 3 mức):
   - "correct": Trả lời đúng, đầy đủ các ý chính của Ground Truth.
   - "partially_correct": Trả lời đúng được một phần ý chính, nhưng thiếu hoặc thừa ý không nghiêm trọng.
   - "incorrect": Trả lời sai bản chất, lạc đề hoặc bịa đặt thông tin.
   (Nếu là câu unanswerable: bot từ chối đúng thì là "correct", bot bịa đáp án thì là "incorrect").
5. key_points_recalled (số nguyên): Số lượng key point trong danh sách trên được câu trả lời đề cập chính xác.

TRẢ VỀ ĐÚNG ĐỊNH DẠNG JSON SAU, TUYỆT ĐỐI KHÔNG VIẾT THÊM LỜI GIẢI THÍCH:
{{
  "context_relevance": 5,
  "context_sufficiency": 5,
  "faithfulness": 5,
  "answer_correctness": "correct",
  "key_points_recalled": {len(key_points) if key_points else 1}
}}
"""
    return prompt

# ============================================================
# 7. BOOTSTRAP CONFIDENCE INTERVAL (Đo ngưỡng nhiễu cho n=60)
# ============================================================

def compute_bootstrap_ci(
    data: List[float], 
    n_bootstrap: int = 1000, 
    ci: float = 0.95
) -> Tuple[float, float, float]:
    """
    Tính khoảng tin cậy Bootstrap CI (Mean, Lower CI, Upper CI).
    Với n=60, độ lệch dưới 5-8% có thể do nhiễu mẫu ngẫu nhiên.
    """
    if not data:
        return 0.0, 0.0, 0.0
    arr = np.array(data)
    mean_val = float(np.mean(arr))
    if len(arr) < 2:
        return mean_val, mean_val, mean_val

    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return round(mean_val, 4), round(lower, 4), round(upper, 4)

# ============================================================
# 8. AGGREGATE EVALUATION REPORT (Thống kê toàn diện theo type)
# ============================================================

def summarize_evaluation_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tổng hợp kết quả benchmark:
    - Bỏ EM
    - Báo cáo Token Precision, Recall, F1 trên short_answer
    - Báo cáo Retrieval Metrics: Hit@k, Recall@k, MRR
    - Báo cáo Entity Coverage
    - Báo cáo Correctness (correct %, partially %, incorrect %) & Key-point Recall
    - Báo cáo False Refusal Rate & Hallucination Rate
    - Phân rã điểm theo question_type
    - Tính Bootstrap CI 95%
    """
    total = len(results)
    if total == 0:
        return {}

    by_type = collections.defaultdict(list)
    for r in results:
        q_type = r.get("question_type", "unknown")
        by_type[q_type].append(r)

    # Tập hợp các chỉ số
    precisions = [r["metrics"].get("precision", 0.0) for r in results if "precision" in r.get("metrics", {})]
    recalls = [r["metrics"].get("recall", 0.0) for r in results if "recall" in r.get("metrics", {})]
    f1s = [r["metrics"].get("f1_score", 0.0) for r in results if "f1_score" in r.get("metrics", {})]

    mrrs = [r["metrics"].get("mrr", 0.0) for r in results if "mrr" in r.get("metrics", {})]
    ent_covs = [r["metrics"].get("entity_coverage", 0.0) for r in results if "entity_coverage" in r.get("metrics", {})]

    cr_scores = [r["metrics"].get("context_relevance", 0.0) for r in results if "context_relevance" in r.get("metrics", {})]
    cs_scores = [r["metrics"].get("context_sufficiency", 0.0) for r in results if "context_sufficiency" in r.get("metrics", {})]
    faith_scores = [r["metrics"].get("faithfulness", 0.0) for r in results if "faithfulness" in r.get("metrics", {})]

    # Correctness counts
    correctness_counts = collections.Counter(
        r["metrics"].get("answer_correctness", "unknown") for r in results
    )

    # Refusal & Hallucination
    unans_results = [r for r in results if str(r.get("answerable")).lower() in ("false", "unanswerable")]
    ans_results = [r for r in results if str(r.get("answerable")).lower() in ("true", "answerable")]

    hallucination_rate = 0.0
    if unans_results:
        hallucination_rate = sum(1 for r in unans_results if r["metrics"].get("hallucination", False)) / len(unans_results)

    false_refusal_rate = 0.0
    if ans_results:
        false_refusal_rate = sum(1 for r in ans_results if r["metrics"].get("false_refusal", False)) / len(ans_results)

    # Summary object
    summary = {
        "total_questions": total,
        "token_metrics": {
            "precision_mean_ci": compute_bootstrap_ci(precisions),
            "recall_mean_ci": compute_bootstrap_ci(recalls),
            "f1_mean_ci": compute_bootstrap_ci(f1s),
        },
        "retrieval_metrics": {
            "mrr_mean_ci": compute_bootstrap_ci(mrrs),
            "entity_coverage_mean_ci": compute_bootstrap_ci(ent_covs),
        },
        "llm_judge_metrics": {
            "context_relevance_ci": compute_bootstrap_ci(cr_scores),
            "context_sufficiency_ci": compute_bootstrap_ci(cs_scores),
            "faithfulness_ci": compute_bootstrap_ci(faith_scores),
        },
        "answer_correctness": {
            "correct_ratio": round(correctness_counts.get("correct", 0) / total, 4),
            "partially_correct_ratio": round(correctness_counts.get("partially_correct", 0) / total, 4),
            "incorrect_ratio": round(correctness_counts.get("incorrect", 0) / total, 4),
        },
        "unanswerable_safety": {
            "false_refusal_rate": round(false_refusal_rate, 4),
            "hallucination_rate": round(hallucination_rate, 4),
        },
        "by_question_type": {}
    }

    # Phân rã theo từng question_type
    for q_type, q_list in by_type.items():
        type_f1s = [r["metrics"].get("f1_score", 0.0) for r in q_list if "f1_score" in r.get("metrics", {})]
        type_cs = [r["metrics"].get("context_sufficiency", 0.0) for r in q_list if "context_sufficiency" in r.get("metrics", {})]
        type_corr = collections.Counter(r["metrics"].get("answer_correctness", "") for r in q_list)

        summary["by_question_type"][q_type] = {
            "count": len(q_list),
            "f1_mean": round(float(np.mean(type_f1s)), 4) if type_f1s else 0.0,
            "context_sufficiency_mean": round(float(np.mean(type_cs)), 4) if type_cs else 0.0,
            "correct_ratio": round(type_corr.get("correct", 0) / len(q_list), 4),
        }

    return summary
