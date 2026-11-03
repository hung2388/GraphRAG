"""tools/run_all_evaluations.py — Master orchestrator for executing and scoring all 12 benchmark runs
across HiRAG, LightRAG, and MS GraphRAG in both thinking and non-thinking modes, Pre and Post Adapter.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvalOrchestrator")

ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "data" / "evaluation" / "eval_v3_distributed_65.json"

RUNS = [
    # 1. HiRAG Pre-Adapter
    {
        "name": "hirag_hi_pre_nothink",
        "system": "hirag",
        "mode": "hi",
        "think": False,
        "work_dir": "HiRAG/hirag_cache_v2",
        "output": "eval_outputs/v3_hirag_hi_pre_nothink.jsonl",
        "python_env": "HiRAG/.venv/Scripts/python.exe",
    },
    {
        "name": "hirag_hi_pre_thinking",
        "system": "hirag",
        "mode": "hi",
        "think": True,
        "work_dir": "HiRAG/hirag_cache_v2",
        "output": "eval_outputs/v3_hirag_hi_pre_thinking.jsonl",
        "python_env": "HiRAG/.venv/Scripts/python.exe",
    },
    # 2. HiRAG Post-Adapter
    {
        "name": "hirag_hi_post_nothink",
        "system": "hirag",
        "mode": "hi",
        "think": False,
        "work_dir": "HiRAG/hirag_cache_v2_adapted",
        "output": "eval_outputs/v3_hirag_hi_post_nothink.jsonl",
        "python_env": "HiRAG/.venv/Scripts/python.exe",
    },
    {
        "name": "hirag_hi_post_thinking",
        "system": "hirag",
        "mode": "hi",
        "think": True,
        "work_dir": "HiRAG/hirag_cache_v2_adapted",
        "output": "eval_outputs/v3_hirag_hi_post_thinking.jsonl",
        "python_env": "HiRAG/.venv/Scripts/python.exe",
    },

    # 3. LightRAG Pre-Adapter
    {
        "name": "lightrag_hybrid_pre_nothink",
        "system": "lightrag",
        "mode": "hybrid",
        "think": False,
        "work_dir": "light/lightrag_storage_v2",
        "output": "eval_outputs/v3_lightrag_hybrid_pre_nothink.jsonl",
        "python_env": ".venv/Scripts/python.exe",
    },
    {
        "name": "lightrag_hybrid_pre_thinking",
        "system": "lightrag",
        "mode": "hybrid",
        "think": True,
        "work_dir": "light/lightrag_storage_v2",
        "output": "eval_outputs/v3_lightrag_hybrid_pre_thinking.jsonl",
        "python_env": ".venv/Scripts/python.exe",
    },
    # 4. LightRAG Post-Adapter
    {
        "name": "lightrag_hybrid_post_nothink",
        "system": "lightrag",
        "mode": "hybrid",
        "think": False,
        "work_dir": "light/lightrag_storage_v2_adapted",
        "output": "eval_outputs/v3_lightrag_hybrid_post_nothink.jsonl",
        "python_env": ".venv/Scripts/python.exe",
    },
    {
        "name": "lightrag_hybrid_post_thinking",
        "system": "lightrag",
        "mode": "hybrid",
        "think": True,
        "work_dir": "light/lightrag_storage_v2_adapted",
        "output": "eval_outputs/v3_lightrag_hybrid_post_thinking.jsonl",
        "python_env": ".venv/Scripts/python.exe",
    },

    # 5. MS GraphRAG Pre-Adapter
    {
        "name": "ms_graphrag_local_pre_nothink",
        "system": "ms_graphrag",
        "mode": "local",
        "think": False,
        "work_dir": "microsoft/ragtest_ms_v2",
        "output": "eval_outputs/v3_ms_graphrag_local_pre_nothink.jsonl",
        "python_env": ".venv/Scripts/python.exe",
    },
    {
        "name": "ms_graphrag_local_pre_thinking",
        "system": "ms_graphrag",
        "mode": "local",
        "think": True,
        "work_dir": "microsoft/ragtest_ms_v2",
        "output": "eval_outputs/v3_ms_graphrag_local_pre_thinking.jsonl",
        "python_env": ".venv/Scripts/python.exe",
    },
    # 6. MS GraphRAG Post-Adapter
    {
        "name": "ms_graphrag_local_post_nothink",
        "system": "ms_graphrag",
        "mode": "local",
        "think": False,
        "work_dir": "microsoft/ragtest_ms_v2_adapted",
        "output": "eval_outputs/v3_ms_graphrag_local_post_nothink.jsonl",
        "python_env": ".venv/Scripts/python.exe",
    },
    {
        "name": "ms_graphrag_local_post_thinking",
        "system": "ms_graphrag",
        "mode": "local",
        "think": True,
        "work_dir": "microsoft/ragtest_ms_v2_adapted",
        "output": "eval_outputs/v3_ms_graphrag_local_post_thinking.jsonl",
        "python_env": ".venv/Scripts/python.exe",
    },
]


def count_completed_lines(output_path: Path) -> int:
    if not output_path.exists():
        return 0
    cnt = 0
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    d = json.loads(line)
                    if "id" in d and "error" not in d:
                        cnt += 1
                except Exception:
                    pass
    return cnt


def run_single_eval(cfg: dict, eval_file: Path, limit: int = None):
    out_path = ROOT / cfg["output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = count_completed_lines(out_path)
    target_count = limit if limit else 65

    if completed >= target_count:
        logger.info(f"[{cfg['name']}] Da hoan tat day du {completed}/{target_count} cau. Bo qua inference.")
        return

    logger.info(f"=== [START RUN: {cfg['name']}] (Tien do hien tai: {completed}/{target_count}) ===")
    py_exe = ROOT / cfg["python_env"]
    cmd = [
        str(py_exe),
        "eval_harness.py",
        "--system", cfg["system"],
        "--mode", cfg["mode"],
        "--eval-file", str(eval_file),
        "--work-dir", str(ROOT / cfg["work_dir"]),
        "--output", str(out_path),
    ]
    if cfg["think"]:
        cmd.append("--think")
    if limit:
        cmd.extend(["--limit", str(limit)])

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["HIRAG_NUM_CTX"] = "8192"

    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)
    dur = time.perf_counter() - t0
    logger.info(f"=== [FINISH RUN: {cfg['name']}] Hoan tat trong {dur:.2f}s ===")


def score_single_eval(cfg: dict, eval_file: Path):
    out_path = ROOT / cfg["output"]
    score_path = out_path.parent / (out_path.stem + "_score.json")
    if not out_path.exists():
        logger.warning(f"Chua co file output {out_path} de cham diem!")
        return None

    logger.info(f"Dang cham diem cho {cfg['name']}...")
    cmd = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "eval_scorer.py",
        str(out_path),
        "--eval-dataset", str(eval_file),
        "--output", str(score_path),
        "--skip-judge",
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)

    if score_path.exists():
        return json.loads(score_path.read_text(encoding="utf-8"))
    return None


def generate_delta_summary(all_scores: dict):
    logger.info("\n" + "=" * 90)
    logger.info("BANG SO SANH TONG THE PRE-ADAPTER VS POST-ADAPTER (12 CAU HINH)")
    logger.info("=" * 90)

    systems = [
        ("HiRAG (hi)", "hirag_hi_pre_nothink", "hirag_hi_post_nothink", "hirag_hi_pre_thinking", "hirag_hi_post_thinking"),
        ("LightRAG (hybrid)", "lightrag_hybrid_pre_nothink", "lightrag_hybrid_post_nothink", "lightrag_hybrid_pre_thinking", "lightrag_hybrid_post_thinking"),
        ("MS GraphRAG (local)", "ms_graphrag_local_pre_nothink", "ms_graphrag_local_post_nothink", "ms_graphrag_local_pre_thinking", "ms_graphrag_local_post_thinking"),
    ]

    header = f"{'System & Mode':<30} | {'Pre Hit@1':<10} | {'Post Hit@1':<10} | {'Delta H@1':<10} | {'Pre F1':<8} | {'Post F1':<8} | {'Delta F1':<8}"
    logger.info(header)
    logger.info("-" * len(header))

    deltas_report = []

    for sys_label, pre_nt_k, post_nt_k, pre_tk_k, post_tk_k in systems:
        # 1. Non-Thinking
        s_pre_nt = all_scores.get(pre_nt_k, {}).get("summary", {})
        s_post_nt = all_scores.get(post_nt_k, {}).get("summary", {})
        h1_pre_nt = s_pre_nt.get("hit@1", 0.0) or 0.0
        h1_post_nt = s_post_nt.get("hit@1", 0.0) or 0.0
        f1_pre_nt = s_pre_nt.get("token_f1", 0.0) or 0.0
        f1_post_nt = s_post_nt.get("token_f1", 0.0) or 0.0
        d_h1_nt = h1_post_nt - h1_pre_nt
        d_f1_nt = f1_post_nt - f1_pre_nt

        label_nt = f"{sys_label} [No-Think]"
        logger.info(f"{label_nt:<30} | {h1_pre_nt:<10.4f} | {h1_post_nt:<10.4f} | {d_h1_nt:+10.4f} | {f1_pre_nt:<8.4f} | {f1_post_nt:<8.4f} | {d_f1_nt:+8.4f}")

        # 2. Thinking
        s_pre_tk = all_scores.get(pre_tk_k, {}).get("summary", {})
        s_post_tk = all_scores.get(post_tk_k, {}).get("summary", {})
        h1_pre_tk = s_pre_tk.get("hit@1", 0.0) or 0.0
        h1_post_tk = s_post_tk.get("hit@1", 0.0) or 0.0
        f1_pre_tk = s_pre_tk.get("token_f1", 0.0) or 0.0
        f1_post_tk = s_post_tk.get("token_f1", 0.0) or 0.0
        d_h1_tk = h1_post_tk - h1_pre_tk
        d_f1_tk = f1_post_tk - f1_pre_tk

        label_tk = f"{sys_label} [Thinking]"
        logger.info(f"{label_tk:<30} | {h1_pre_tk:<10.4f} | {h1_post_tk:<10.4f} | {d_h1_tk:+10.4f} | {f1_pre_tk:<8.4f} | {f1_post_tk:<8.4f} | {d_f1_tk:+8.4f}")

        deltas_report.append({
            "system": sys_label,
            "no_think": {
                "h1_pre": h1_pre_nt, "h1_post": h1_post_nt, "delta_h1": d_h1_nt,
                "f1_pre": f1_pre_nt, "f1_post": f1_post_nt, "delta_f1": d_f1_nt,
            },
            "thinking": {
                "h1_pre": h1_pre_tk, "h1_post": h1_post_tk, "delta_h1": d_h1_tk,
                "f1_pre": f1_pre_tk, "f1_post": f1_post_tk, "delta_f1": d_f1_tk,
            }
        })

    logger.info("=" * 90 + "\n")
    summary_path = ROOT / "eval_outputs" / "final_benchmark_summary.json"
    summary_path.write_text(json.dumps(deltas_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Da ghi tong hop ket qua vao {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run and score all 12 evaluation configurations")
    parser.add_argument("--limit", type=int, default=None, help="So luong cau hoi can chay moi config (mac dinh toan bo 65 cau)")
    parser.add_argument("--eval-file", type=Path, default=EVAL_FILE)
    parser.add_argument("--score-only", action="store_true", help="Chi tinh diem cho cac file da chay, khong chay inference")
    parser.add_argument("--system-filter", type=str, default=None, help="Loc he thong can chay (hirag, lightrag, ms_graphrag)")
    args = parser.parse_args()

    runs_to_run = RUNS
    if args.system_filter:
        runs_to_run = [r for r in RUNS if args.system_filter in r["system"]]

    logger.info(f"=== BAT DAU TIEN TRINH DANH GIA TONG THE ({len(runs_to_run)} RUNS) ===")
    logger.info(f"Eval File: {args.eval_file}")
    logger.info(f"Limit: {args.limit if args.limit else 'Tat ca 65 cau'}")

    all_scores = {}

    for cfg in runs_to_run:
        if not args.score_only:
            run_single_eval(cfg, args.eval_file, limit=args.limit)
        score_res = score_single_eval(cfg, args.eval_file)
        if score_res:
            all_scores[cfg["name"]] = score_res

    generate_delta_summary(all_scores)
    logger.info("=== TOAN BO TIEN TRINH DANH GIA DA HOAN TAT MY MAN! ===")


if __name__ == "__main__":
    main()
