"""eval_scorer.py — Standalone scorer for evaluating raw JSONL outputs against eval_v2.json ground truth.

Features:
- Hit@k / Recall@k (k=1, 3, 5) on gold_spans via eval_data_tools.py
- Token-level Precision, Recall, F1 on short_answer via evaluation_metrics.py
- LLM-as-a-judge using llama3.1:8b (non-qwen model):
  * Answer Correctness (3-tier: 1.0, 0.5, 0.0)
  * Key-Point Recall (tỷ lệ key_points được trả lời)
  * Context Sufficiency (1-5)
  * Faithfulness (1-5)
  * False Refusal & Hallucination
  * Entity Coverage (supporting_entities trong context)
- Phân tách theo từng tài liệu (doc) và từng loại câu hỏi (question_type)
- Lưu báo cáo kết quả chi tiết dạng JSON và in bảng tóm tắt
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval_data_tools import hit_at_k, recall_at_k, norm
from evaluation_metrics import compute_token_prf, normalize_for_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvalScorer")

OLLAMA_URL = os.getenv("LAB_OLLAMA_HOST", "http://localhost:11434")
JUDGE_MODEL = os.getenv("LAB_JUDGE_MODEL", "llama3.1:8b")


def call_llm_judge(prompt: str, format_json: bool = True, temperature: float = 0.0) -> str:
    """Gọi LLM judge (llama3.1:8b) thông qua Ollama API."""
    payload = {
        "model": JUDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if format_json:
        payload["format"] = "json"

    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
    r.raise_for_status()
    raw = r.json().get("response", "").strip()
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def judge_answer(
    question: str,
    answer: str,
    ground_truth: str,
    key_points: List[str],
    retrieved_context: str,
    answerable: bool = True,
) -> Dict[str, Any]:
    """Chấm điểm câu trả lời bằng LLM judge."""
    prompt = f"""You are an impartial, strict AI benchmark judge.
Evaluate the candidate answer compared against the reference ground truth and context.

Question: {question}
Ground Truth: {ground_truth}
Key Points required: {json.dumps(key_points, ensure_ascii=False)}
Candidate Answer: {answer}
Retrieved Context: {retrieved_context[:3000]}
Is Answerable: {answerable}

Instructions:
1. "answer_correctness": 
   - 1.0 (Fully Correct): captures all key information without major factual error.
   - 0.5 (Partially Correct): captures some key facts but misses details.
   - 0.0 (Incorrect / Refused incorrectly / Hallucinated).
2. "key_points_covered": Number of required key points satisfied by the candidate answer (integer between 0 and {len(key_points)}).
3. "context_sufficiency": Integer 1 to 5 (Does retrieved context contain enough info to answer?).
4. "faithfulness": Integer 1 to 5 (Is the candidate answer grounded entirely in the context without hallucination?).
5. "is_refusal": boolean (true if candidate refused to answer, e.g. "I don't know", "Không có thông tin").

Return ONLY valid JSON matching this exact structure:
{{
  "answer_correctness": 1.0,
  "key_points_covered": 1,
  "context_sufficiency": 5,
  "faithfulness": 5,
  "is_refusal": false,
  "explanation": "Brief explanation"
}}
"""
    try:
        resp = call_llm_judge(prompt, format_json=True)
        data = json.loads(resp)
        return {
            "answer_correctness": float(data.get("answer_correctness", 0.0)),
            "key_points_covered": int(data.get("key_points_covered", 0)),
            "context_sufficiency": int(data.get("context_sufficiency", 1)),
            "faithfulness": int(data.get("faithfulness", 1)),
            "is_refusal": bool(data.get("is_refusal", False)),
            "judge_explanation": data.get("explanation", ""),
        }
    except Exception as e:
        logger.warning(f"Judge error: {e}")
        return {
            "answer_correctness": 0.0,
            "key_points_covered": 0,
            "context_sufficiency": 1,
            "faithfulness": 1,
            "is_refusal": False,
            "judge_explanation": f"Judge error: {e}",
        }


def score_all(
    raw_jsonl_path: Path,
    eval_dataset_path: Path,
    skip_judge: bool = False,
) -> Dict[str, Any]:
    # 1. Đọc ground truth
    gt_items = json.loads(eval_dataset_path.read_text(encoding="utf-8"))
    gt_by_id = {item["id"]: item for item in gt_items}

    # 2. Đọc raw outputs
    raw_lines = []
    with open(raw_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_lines.append(json.loads(line))

    logger.info(f"Đọc {len(raw_lines)} câu trả lời thô từ {raw_jsonl_path}")

    results = []
    for idx, raw in enumerate(raw_lines, 1):
        qid = raw.get("id")
        gt = gt_by_id.get(qid)
        if not gt:
            continue
        if idx % 10 == 0 or idx == len(raw_lines):
            logger.info(f"Scoring [{idx}/{len(raw_lines)}] ID={qid}...")

        chunks = raw.get("retrieved_chunks", [])
        answer = raw.get("answer", "")
        doc = gt.get("doc", raw.get("doc", "unknown"))
        qtype = gt.get("question_type", "unknown")
        is_answerable = gt.get("answerable", True)

        # A. Retrieval metrics (Hit@k / Recall@k)
        h1 = hit_at_k(chunks, gt, k=1)
        h3 = hit_at_k(chunks, gt, k=3)
        h5 = hit_at_k(chunks, gt, k=5)

        r1 = recall_at_k(chunks, gt, k=1)
        r3 = recall_at_k(chunks, gt, k=3)
        r5 = recall_at_k(chunks, gt, k=5)

        # B. Token PRF on short_answer
        short_gt = gt.get("short_answer") or gt.get("ground_truth", "")
        clean_answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        eval_answer = clean_answer if clean_answer else answer
        token_prf = compute_token_prf(eval_answer, short_gt)

        # C. Entity Coverage (supporting_entities trong retrieved chunks)
        entities = gt.get("supporting_entities", [])
        combined_chunks_text = " ".join(chunks).lower()
        if entities:
            ent_cov = sum(norm(e) in combined_chunks_text for e in entities) / len(entities)
        else:
            ent_cov = None

        # D. LLM-as-a-judge
        judge_scores = {}
        if not skip_judge:
            kps = gt.get("key_points", [])
            raw_ctx = " ".join(chunks)
            judge_scores = judge_answer(
                question=gt.get("question", raw.get("question", "")),
                answer=answer,
                ground_truth=gt.get("ground_truth", ""),
                key_points=kps,
                retrieved_context=raw_ctx,
                answerable=is_answerable,
            )
            # Tính key-point recall
            if kps:
                judge_scores["key_point_recall"] = min(
                    1.0, judge_scores["key_points_covered"] / len(kps)
                )
            else:
                judge_scores["key_point_recall"] = None

            # False refusal: answerable mà bị từ chối
            judge_scores["is_false_refusal"] = (
                is_answerable and judge_scores.get("is_refusal", False)
            )

        res_item = {
            "id": qid,
            "doc": doc,
            "question_type": qtype,
            "answerable": is_answerable,
            "latency_sec": raw.get("latency_sec"),
            "tokens": raw.get("total_tokens"),
            "hit@1": h1,
            "hit@3": h3,
            "hit@5": h5,
            "recall@1": r1,
            "recall@3": r3,
            "recall@5": r5,
            "token_f1": token_prf.get("f1_score"),
            "token_precision": token_prf.get("precision"),
            "token_recall": token_prf.get("recall"),
            "entity_coverage": ent_cov,
            **judge_scores,
        }
        results.append(res_item)

    # 3. Tổng hợp metrics & breakdown
    summary = compute_aggregated_metrics(results)
    by_doc = compute_group_metrics(results, group_key="doc")
    by_qtype = compute_group_metrics(results, group_key="question_type")

    return {
        "raw_file": str(raw_jsonl_path),
        "eval_dataset": str(eval_dataset_path),
        "total_scored": len(results),
        "summary": summary,
        "by_doc": by_doc,
        "by_question_type": by_qtype,
        "details": results,
    }


def _avg(vals):
    filtered = [v for v in vals if v is not None]
    return round(sum(filtered) / len(filtered), 4) if filtered else None


def compute_aggregated_metrics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "hit@1": _avg([i.get("hit@1") for i in items]),
        "hit@3": _avg([i.get("hit@3") for i in items]),
        "hit@5": _avg([i.get("hit@5") for i in items]),
        "recall@1": _avg([i.get("recall@1") for i in items]),
        "recall@3": _avg([i.get("recall@3") for i in items]),
        "recall@5": _avg([i.get("recall@5") for i in items]),
        "token_f1": _avg([i.get("token_f1") for i in items]),
        "token_precision": _avg([i.get("token_precision") for i in items]),
        "token_recall": _avg([i.get("token_recall") for i in items]),
        "entity_coverage": _avg([i.get("entity_coverage") for i in items]),
        "answer_correctness": _avg([i.get("answer_correctness") for i in items]),
        "key_point_recall": _avg([i.get("key_point_recall") for i in items]),
        "context_sufficiency": _avg([i.get("context_sufficiency") for i in items]),
        "faithfulness": _avg([i.get("faithfulness") for i in items]),
        "false_refusal_rate": _avg([1.0 if i.get("is_false_refusal") else 0.0 for i in items]),
        "avg_latency_sec": _avg([i.get("latency_sec") for i in items]),
    }


def compute_group_metrics(items: List[Dict[str, Any]], group_key: str) -> Dict[str, Any]:
    groups = defaultdict(list)
    for i in items:
        groups[i.get(group_key, "unknown")].append(i)

    return {g: {"count": len(rows), **compute_aggregated_metrics(rows)} for g, rows in groups.items()}


def print_summary_table(report: Dict[str, Any]):
    s = report["summary"]
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY REPORT")
    print("=" * 60)
    print(f"File evaluated: {report['raw_file']}")
    print(f"Total questions: {report['total_scored']}")
    print("-" * 60)
    print(f"Hit@1: {s.get('hit@1')} | Hit@3: {s.get('hit@3')} | Hit@5: {s.get('hit@5')}")
    print(f"Recall@1: {s.get('recall@1')} | Recall@3: {s.get('recall@3')} | Recall@5: {s.get('recall@5')}")
    print(f"Token F1 (short_ans): {s.get('token_f1')} (P={s.get('token_precision')}, R={s.get('token_recall')})")
    print(f"Entity Coverage: {s.get('entity_coverage')}")
    print(f"Answer Correctness (Judge): {s.get('answer_correctness')}")
    print(f"Key-Point Recall: {s.get('key_point_recall')}")
    print(f"Context Sufficiency (1-5): {s.get('context_sufficiency')}")
    print(f"Faithfulness (1-5): {s.get('faithfulness')}")
    print(f"False Refusal Rate: {s.get('false_refusal_rate')}")
    print(f"Avg Latency: {s.get('avg_latency_sec')}s")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate raw JSONL output")
    parser.add_argument("raw_file", type=Path, help="Duong dan toi file raw jsonl (vi du: eval_outputs/hirag_hi_raw.jsonl)")
    parser.add_argument("--eval-dataset", type=Path, default=ROOT / "data" / "evaluation" / "eval_v2.json")
    parser.add_argument("--output", type=Path, default=None, help="File JSON luu ket qua danh gia")
    parser.add_argument("--skip-judge", action="store_true", help="Bo qua LLM-judge de tinh nhanh metrics truyen thong")

    args = parser.parse_args()

    if not args.raw_file.exists():
        raise FileNotFoundError(f"Khong tim thay {args.raw_file}")
    if not args.eval_dataset.exists():
        fallback = ROOT / "data" / "evaluation" / "merged_questions.json"
        if fallback.exists():
            logger.warning(f"{args.eval_dataset} khong thay, fallback sang {fallback}")
            args.eval_dataset = fallback
        else:
            raise FileNotFoundError(f"Khong tim thay {args.eval_dataset}")

    report = score_all(
        raw_jsonl_path=args.raw_file,
        eval_dataset_path=args.eval_dataset,
        skip_judge=args.skip_judge,
    )

    print_summary_table(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Da luu bao cao vao {args.output}")


if __name__ == "__main__":
    main()
