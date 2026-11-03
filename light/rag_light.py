import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete
from lightrag.utils import EmbeddingFunc

import config
from common.ollama_utils import custom_ollama_embed

try:
    from lightrag.kg.shared_storage import initialize_pipeline_status
except ImportError:
    initialize_pipeline_status = None

config.LIGHT_DIR.mkdir(parents=True, exist_ok=True)


async def build_rag(working_dir=None, think: bool = True):
    if working_dir:
        target_dir = str(working_dir)
    elif think:
        target_dir = str(config.LIGHT_DIR.parent / "lightrag_storage_thinking")
    else:
        target_dir = str(config.LIGHT_DIR)
    options = {
        "num_ctx": 8192,
        "temperature": 0.0,
        "think": think,
        "num_predict": 4096 if think else 3072,
    }
    rag = LightRAG(
        working_dir=target_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=config.LLM_MODEL,
        llm_model_kwargs={
            "host": config.OLLAMA_HOST,
            "options": options,
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=config.EMBED_DIM,
            max_token_size=4096,
            func=custom_ollama_embed,
        ),
        entity_extract_max_gleaning=0,
        entity_extract_max_records=30,
        entity_extract_max_entities=15,
        llm_model_max_async=1,
        default_llm_timeout=300,
    )
    await rag.initialize_storages()
    if initialize_pipeline_status:
        await initialize_pipeline_status()
    return rag


async def lightrag_answer(question, rag, mode="hybrid"):
    return await rag.aquery(question, param=QueryParam(mode=mode))


async def main():
    if not config.INPUT_FILES:
        raise SystemExit("INPUT_FILES rỗng: kiểm tra LAB_INPUT_FILES trong .env")

    rag = await build_rag()

    for path in config.INPUT_FILES:
        if not path.exists():
            raise SystemExit(f"Không tìm thấy file: {path}")
        await rag.ainsert(path.read_text(encoding="utf-8"))
        print(f"Đã nạp: {path}")

    print("Xây KG xong.")

    question = "Quang Trung sinh năm nào và tên khai sinh là gì?"
    for mode in ["naive", "local", "global", "hybrid"]:
        print(f"\n=== {mode} ===")
        print(await lightrag_answer(question, rag, mode))


if __name__ == "__main__":
    asyncio.run(main())