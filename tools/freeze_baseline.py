"""freeze_baseline.py — Snapshot and freeze baseline runs for GraphRAG benchmark.

Usage:
    python tools/freeze_baseline.py --tag v1_initial --systems hirag,lightrag,ms_graphrag,naive
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_git_commit(cwd: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def freeze_baseline(tag: str, description: str = ""):
    target_dir = ROOT / "baselines" / tag
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = target_dir / "raw_outputs"
    reports_dir = target_dir / "reports"
    raw_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    # 1. Collect config & metadata
    metadata = {
        "tag": tag,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "description": description,
        "git_commits": {
            "root": get_git_commit(ROOT),
            "hirag": get_git_commit(ROOT / "HiRAG"),
        },
        "config": {
            "llm_model": os.getenv("LAB_LLM_MODEL", "qwen3:8b"),
            "embed_model": os.getenv("LAB_EMBED_MODEL", "nomic-embed-text"),
            "judge_model": os.getenv("LAB_JUDGE_MODEL", "llama3.1:8b"),
            "temperature": 0.0,
            "num_ctx": 4096,
        },
    }

    # 2. Copy raw outputs from eval_outputs/
    eval_outputs = ROOT / "eval_outputs"
    copied_raw = 0
    if eval_outputs.exists():
        for f in eval_outputs.glob("*.jsonl"):
            shutil.copy(f, raw_dir / f.name)
            copied_raw += 1

    # 3. Copy reports from each system's reports/ directory
    copied_reports = 0
    for sys_dir in ["HiRAG", "light", "microsoft", "my_rag_microsoft_style"]:
        rep_path = ROOT / sys_dir / "reports"
        if rep_path.exists():
            for rf in rep_path.glob("*.json"):
                shutil.copy(rf, reports_dir / rf.name)
                copied_reports += 1

    meta_file = target_dir / "metadata.json"
    meta_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[*] Da dong bang baseline thanh cong tai: {target_dir}")
    print(f"    - Metadata: {meta_file}")
    print(f"    - Raw outputs chep: {copied_raw} files")
    print(f"    - Reports chep: {copied_reports} files")


def main():
    parser = argparse.ArgumentParser(description="Freeze baseline benchmark results")
    parser.add_argument("--tag", required=True, help="Tag dinh danh cho baseline (vi du: baseline_v1_78q)")
    parser.add_argument("--desc", default="", help="Mo ta them ve dot chay baseline nay")
    args = parser.parse_args()

    freeze_baseline(tag=args.tag, description=args.desc)


if __name__ == "__main__":
    main()
