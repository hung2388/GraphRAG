"""tools/index_ms_graphrag_v2.py — Setup va Index 19 tai lieu eval vao Microsoft GraphRAG (ragtest_ms_v2).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "eval_corpus_19"
SRC_WORK = ROOT / "microsoft" / "ragtest_ms"
DST_WORK = ROOT / "microsoft" / "ragtest_ms_v2"


def find_graphrag_exe():
    venv_ms_exe = ROOT / ".venv-ms" / "Scripts" / ("graphrag.exe" if os.name == "nt" else "graphrag")
    if venv_ms_exe.exists():
        return str(venv_ms_exe)
    curr_exe = Path(sys.executable).parent / ("graphrag.exe" if os.name == "nt" else "graphrag")
    if curr_exe.exists():
        return str(curr_exe)
    return shutil.which("graphrag") or "graphrag"


def run_cmd(cmd: list):
    print(">>", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    subprocess.run(cmd, check=True, env=env)


def setup_workspace():
    DST_WORK.mkdir(parents=True, exist_ok=True)
    dst_input = DST_WORK / "input"
    dst_input.mkdir(parents=True, exist_ok=True)

    # 1. Copy 19 files
    files = sorted([f for f in EVAL_DIR.glob("*.txt") if f.is_file()])
    for f in files:
        shutil.copy(f, dst_input / f.name)
    print(f"[*] Da chep {len(files)} files vao {dst_input}")

    # 2. Copy settings and prompts from SRC_WORK
    src_settings = SRC_WORK / "settings.yaml"
    dst_settings = DST_WORK / "settings.yaml"
    if src_settings.exists():
        cfg = yaml.safe_load(src_settings.read_text(encoding="utf-8"))
        cfg["concurrent_requests"] = 1
        # Ensure max_gleanings = 0 and input dir is input
        if "extract_graph" in cfg and isinstance(cfg["extract_graph"], dict):
            cfg["extract_graph"]["max_gleanings"] = 0
        if "completion_models" in cfg and "default_completion_model" in cfg["completion_models"]:
            comp = cfg["completion_models"]["default_completion_model"]
            comp["model"] = "qwen3:8b-nothink"
            comp.pop("call_args", None)
        dst_settings.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"[*] Da sao chep va settings vao {dst_settings}")

    src_prompts = SRC_WORK / "prompts"
    dst_prompts = DST_WORK / "prompts"
    if src_prompts.exists() and not dst_prompts.exists():
        shutil.copytree(src_prompts, dst_prompts)
        print(f"[*] Da sao chep thu muc prompts sang {dst_prompts}")


def main():
    print("=== BAT DAU SETUP & INDEX MICROSOFT GRAPHRAG V2 ===")
    setup_workspace()

    prog = find_graphrag_exe()
    print(f"[*] Dung graphrag binary: {prog}")
    cmd = [prog, "index", "--root", str(DST_WORK)]
    run_cmd(cmd)
    print("=== HOAN TAT INDEXING MS GRAPHRAG V2! ===")


if __name__ == "__main__":
    main()
