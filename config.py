import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

OLLAMA_HOST = os.getenv("LAB_OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LAB_LLM_MODEL", "qwen3:8b")
EMBED_MODEL = os.getenv("LAB_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.getenv("LAB_EMBED_DIM", "768"))
LIGHT_DIR = ROOT / "light" / os.getenv("LAB_LIGHT_DIR", "lightrag_storage")

_default_inputs = "data/input/quang_trung.txt,data/input/src.txt,data/input/love.txt"
INPUT_FILES = [
    ROOT / p.strip()
    for p in os.getenv("LAB_INPUT_FILES", _default_inputs).split(",")
    if p.strip()
]
