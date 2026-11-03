"""
ms_graphrag.py — Microsoft GraphRAG thuần, chạy với Ollama local.
Độc lập với LightRAG / Neo4j / lazy.

Dùng (trong .venv-ms, từ thư mục gốc Lab/):
    python microsoft/ms_graphrag.py setup
    python microsoft/ms_graphrag.py index
    python microsoft/ms_graphrag.py query "câu hỏi" --method local
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import yaml

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[1]
if load_dotenv:
    load_dotenv(ROOT / ".env")

OLLAMA_HOST = os.getenv("LAB_OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LAB_LLM_MODEL", "qwen3:8b")
EMBED_MODEL = os.getenv("LAB_EMBED_MODEL", "nomic-embed-text")
INPUT_FILES = [
    ROOT / p.strip()
    for p in os.getenv(
        "LAB_INPUT_FILES", "data/input/quang_trung.txt,data/input/src.txt,data/input/love.txt"
    ).split(",")
    if p.strip()
]

WORK = ROOT / "microsoft" / "ragtest_ms"
OLLAMA_BASE = OLLAMA_HOST.rstrip("/") + "/v1"


def graphrag_cmd(*args: str) -> None:
    exe = Path(sys.executable).parent / ("graphrag.exe" if os.name == "nt" else "graphrag")
    prog = str(exe) if exe.exists() else (shutil.which("graphrag") or "graphrag")
    cmd = [prog, *args]
    print(">>", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    subprocess.run(cmd, check=True, env=env)


def patch_settings() -> None:
    path = WORK / "settings.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    for m in (cfg.get("completion_models") or {}).values():
        m.update(
            model_provider="openai",
            model=LLM_MODEL,
            auth_method="api_key",
            api_key="ollama",
            api_base=OLLAMA_BASE,
        )
    for m in (cfg.get("embedding_models") or {}).values():
        m.update(
            model_provider="openai",
            model=EMBED_MODEL,
            auth_method="api_key",
            api_key="ollama",
            api_base=OLLAMA_BASE,
        )

    # model nhỏ hay lỗi khi gleaning nhiều vòng -> tắt cho ổn định
    extract = cfg.get("extract_graph")
    if isinstance(extract, dict):
        extract["max_gleanings"] = 0

    path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Đã vá model trong {path}")


def cmd_setup(_args) -> None:
    (WORK / "input").mkdir(parents=True, exist_ok=True)

    for src in INPUT_FILES:
        if not src.exists():
            raise SystemExit(f"Không tìm thấy file: {src}")
        shutil.copy(src, WORK / "input" / src.name)
        print(f"Đã chép: {src.name}")

    if not (WORK / "settings.yaml").exists():
        graphrag_cmd("init", "--root", str(WORK))

    patch_settings()
    print("Setup xong.")


def cmd_index(_args) -> None:
    graphrag_cmd("index", "--root", str(WORK))


def cmd_query(args) -> None:
    root_dir = args.root or str(WORK)
    graphrag_cmd(
        "query", args.question,
        "--method", args.method,
        "--root", str(root_dir),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup").set_defaults(fn=cmd_setup)
    sub.add_parser("index").set_defaults(fn=cmd_index)

    q = sub.add_parser("query")
    q.add_argument("question")
    q.add_argument("--method", default="local", choices=["local", "global", "drift", "basic"])
    q.add_argument("--root", default=None, help="Root folder chua du lieu graphrag da index")
    q.set_defaults(fn=cmd_query)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
