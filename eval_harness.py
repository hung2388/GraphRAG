"""eval_harness.py — Unified evaluation harness runner for GraphRAG systems.

Supports:
- HiRAG (modes: hi, naive, hi_local, hi_global, hi_bridge)
- LightRAG (modes: hybrid, local, global, naive)
- MS GraphRAG (methods: local, global, drift, basic)
- Naive RAG

Dumps raw JSONL per question:
{
  "id": 1,
  "doc": "hong_bang",
  "question": "...",
  "system": "hirag",
  "mode": "hi",
  "answer": "...",
  "retrieved_chunks": [...],
  "latency_sec": 2.45,
  "prompt_tokens": 1200,
  "completion_tokens": 65,
  "total_tokens": 1265,
  "timestamp": "2026-10-20T...",
  "config": {"model": "qwen3:8b", "temperature": 0.0, "num_ctx": 4096}
}
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("EvalHarness")

DEFAULT_EVAL_FILE = ROOT / "data" / "evaluation" / "eval_v2.json"
FALLBACK_EVAL_FILE = ROOT / "data" / "evaluation" / "merged_questions.json"


def load_eval_questions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        if path == DEFAULT_EVAL_FILE and FALLBACK_EVAL_FILE.exists():
            logger.warning(f"File {path} chua ton tai, fallback sang {FALLBACK_EVAL_FILE}")
            path = FALLBACK_EVAL_FILE
        else:
            raise FileNotFoundError(f"Khong tim thay file eval: {path}")

    logger.info(f"Doc bo cau hoi eval tu: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    logger.info(f"Tong so cau hoi: {len(data)}")
    return data


def load_existing_done_ids(output_file: Path) -> set:
    done_ids = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        if "id" in record and "error" not in record:
                            done_ids.add(record["id"])
                    except Exception:
                        pass
    return done_ids


# ============================================================
# 1. HiRAG RUNNER
# ============================================================
class HiRAGRunner:
    def __init__(self, working_dir: Path, num_ctx: int = 4096, temperature: float = 0.0):
        self.working_dir = working_dir
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.rag = None

    async def init(self):
        hirag_root = ROOT / "HiRAG"
        if str(hirag_root) not in sys.path:
            sys.path.insert(0, str(hirag_root))

        from hirag import HiRAG
        from hi_Search_ollama import (
            OLLAMA_embedding,
            OLLAMA_model_if_cache,
            OLLAMA_CHAT_MODEL,
        )

        self.model_name = OLLAMA_CHAT_MODEL
        logger.info(f"Khoi tao HiRAG tai {self.working_dir} (num_ctx={self.num_ctx}, temp={self.temperature})...")
        self.rag = HiRAG(
            working_dir=str(self.working_dir),
            enable_llm_cache=True,
            embedding_func=OLLAMA_embedding,
            best_model_func=OLLAMA_model_if_cache,
            cheap_model_func=OLLAMA_model_if_cache,
            enable_hierachical_mode=True,
            enable_naive_rag=True,
            best_model_max_async=1,
            cheap_model_max_async=1,
            embedding_batch_num=4,
            embedding_func_max_async=2,
        )

    async def query(self, question: str, mode: str = "hi") -> Dict[str, Any]:
        from hirag import QueryParam

        start_time = time.perf_counter()

        hirag_params = {
            "mode": mode,
            "top_k": 10,
            "top_m": 5,
            "max_token_for_text_unit": 3500,
            "max_token_for_local_context": 3500,
            "max_token_for_bridge_knowledge": 1500,
            "max_token_for_community_report": 1500,
        }
        # 1. Retrieve raw context chunks
        context_str = ""
        try:
            context_param = QueryParam(**hirag_params, only_need_context=True)
            context_str = await self.rag.aquery(question, param=context_param)
        except Exception as e:
            logger.warning(f"HiRAG context retrieval loi: {e}")

        # Extract chunks from context string
        retrieved_chunks = self._parse_hirag_chunks(context_str, mode)

        # 2. Generate answer
        answer_param = QueryParam(**hirag_params, only_need_context=False)
        answer = await self.rag.aquery(question, param=answer_param)

        latency = time.perf_counter() - start_time

        # Estimate tokens
        prompt_tokens = len(question.split()) + len(context_str.split())
        completion_tokens = len(answer.split())

        return {
            "answer": answer.strip(),
            "retrieved_chunks": retrieved_chunks,
            "raw_context": context_str,
            "latency_sec": round(latency, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _parse_hirag_chunks(self, context_str: str, mode: str) -> List[str]:
        if not context_str:
            return []
        chunks = []
        if "--New Chunk--" in context_str:
            parts = context_str.split("--New Chunk--")
            chunks = [p.strip() for p in parts if p.strip()]
        elif "-----Sources-----" in context_str:
            parts = context_str.split("-----Sources-----")
            if len(parts) > 1:
                src_block = parts[1]
                chunks = [s.strip() for s in re.split(r"\n\s*\n", src_block) if s.strip()]
        if not chunks:
            # Fallback: split by double newlines or whole text
            lines = [l.strip() for l in context_str.split("\n\n") if len(l.strip()) > 30]
            chunks = lines if lines else [context_str.strip()]
        # Filter out background headers, community tables, etc. to retain pure chunks
        filtered = [c for c in chunks if not c.startswith("-----")]
        return filtered if filtered else chunks


# ============================================================
# 2. LightRAG RUNNER
# ============================================================
class LightRAGRunner:
    def __init__(self, num_ctx: int = 8192, temperature: float = 0.0, think: bool = False, working_dir: Optional[Path] = None):
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.think = think
        self.working_dir = working_dir
        self.rag = None

    async def init(self):
        from light.rag_light import build_rag
        logger.info(f"Khoi tao LightRAG (num_ctx={self.num_ctx}, think={self.think}, working_dir={self.working_dir})...")
        self.rag = await build_rag(working_dir=self.working_dir, think=self.think)

    async def query(self, question: str, mode: str = "hybrid") -> Dict[str, Any]:
        from lightrag import QueryParam

        start_time = time.perf_counter()

        # 1. Retrieve raw context
        context_str = ""
        try:
            context_param = QueryParam(
                mode=mode,
                only_need_context=True,
                top_k=10,
                chunk_top_k=5,
                max_entity_tokens=1000,
                max_relation_tokens=1000,
                max_total_tokens=7000,
            )
            context_str = await self.rag.aquery(question, param=context_param)
        except Exception as e:
            logger.warning(f"LightRAG context retrieval loi: {e}")

        # Extract pure document chunks from Document Chunks section
        chunks = []
        if "Document Chunks" in context_str:
            sec = context_str.split("Document Chunks", 1)[1]
            if "```json" in sec:
                json_part = sec.split("```json", 1)[1].split("```", 1)[0]
                for item_str in json_part.strip().splitlines():
                    try:
                        item = json.loads(item_str.strip())
                        if "content" in item and item["content"].strip():
                            chunks.append(item["content"].strip())
                    except Exception:
                        pass
        if not chunks:
            chunks = [c.strip() for c in context_str.split("\n\n") if len(c.strip()) > 30 and not c.strip().startswith("-----")]
        retrieved_chunks = chunks if chunks else ([context_str.strip()] if context_str else [])

        # 2. Answer
        answer_param = QueryParam(
            mode=mode,
            only_need_context=False,
            top_k=10,
            chunk_top_k=5,
            max_entity_tokens=1000,
            max_relation_tokens=1000,
            max_total_tokens=7000,
        )
        answer = await self.rag.aquery(question, param=answer_param)

        latency = time.perf_counter() - start_time
        prompt_tokens = len(question.split()) + len(context_str.split())
        completion_tokens = len(answer.split())

        return {
            "answer": answer.strip(),
            "retrieved_chunks": retrieved_chunks,
            "raw_context": context_str,
            "latency_sec": round(latency, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }


# ============================================================
# 3. MS GraphRAG RUNNER (CLI or Python wrapper)
# ============================================================
class MSGraphRAGRunner:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.src_map = {}
        # Load text_units mapping if available
        tu_json = self.root_dir / "output" / "text_units.json"
        tu_file = self.root_dir / "output" / "text_units.parquet"
        if tu_json.exists():
            try:
                data = json.loads(tu_json.read_text(encoding="utf-8"))
                self.src_map = {int(k): v for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Khong the load text_units.json: {e}")
        elif tu_file.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(tu_file)
                if "human_readable_id" in df.columns and "text" in df.columns:
                    self.src_map = dict(zip(df["human_readable_id"], df["text"]))
            except Exception as e:
                logger.warning(f"Khong the load text_units.parquet: {e}")

    async def query(self, question: str, mode: str = "local") -> Dict[str, Any]:
        start_time = time.perf_counter()
        python_exe = ROOT / ".venv-ms" / "Scripts" / "python.exe"
        if not python_exe.exists():
            python_exe = Path(sys.executable)

        # Run query via ms_graphrag.py CLI or graphrag.exe
        cmd = [
            str(python_exe),
            str(ROOT / "microsoft" / "ms_graphrag.py"),
            "query",
            question,
            "--method",
            mode,
            "--root",
            str(self.root_dir),
        ]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(ROOT),
            env=env,
        )
        stdout, stderr = await proc.communicate()
        latency = time.perf_counter() - start_time

        ans = stdout.decode("utf-8", errors="replace").strip()
        ans_clean_lines = [l for l in ans.splitlines() if not l.strip().startswith(">>")]
        answer = "\n".join(ans_clean_lines).strip()

        if "SUCCESS: Local Search Response:" in answer:
            answer = answer.split("SUCCESS: Local Search Response:")[-1].strip()
        elif "SUCCESS: Global Search Response:" in answer:
            answer = answer.split("SUCCESS: Global Search Response:")[-1].strip()

        # Parse cited sources from response, e.g. [Data: Sources (4, 5)], [Data: Input Text (1)]
        retrieved_chunks = []
        if self.src_map:
            try:
                matches = re.findall(
                    r"\[Data:[^\]]*?\(([\d\s,;+more]+)\)\]|(?:Sources|Text|Input Text)\s*\(([\d\s,]+)\)",
                    ans,
                )
                all_ids = []
                for m in matches:
                    content = m[0] or m[1]
                    nums = re.findall(r"\b\d+\b", content)
                    all_ids.extend([int(x) for x in nums])
                # preserve order and deduplicate
                seen = set()
                dedup_ids = [x for x in all_ids if not (x in seen or seen.add(x))]
                retrieved_chunks = [self.src_map[i] for i in dedup_ids if i in self.src_map]
            except Exception:
                pass

        return {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "raw_context": "\n\n".join(retrieved_chunks),
            "latency_sec": round(latency, 3),
            "prompt_tokens": 0,
            "completion_tokens": len(answer.split()),
            "total_tokens": len(answer.split()),
        }


# ============================================================
# MAIN HARNESS LOOP
# ============================================================
async def run_harness(
    system_name: str,
    mode: str,
    eval_file: Path,
    output_file: Path,
    limit: Optional[int] = None,
    num_ctx: int = 4096,
    temperature: float = 0.0,
    think: bool = False,
    work_dir: Optional[Path] = None,
):
    questions = load_eval_questions(eval_file)
    if limit:
        questions = questions[:limit]
        logger.info(f"Gioi han chay {limit} cau hoi dau tien.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_existing_done_ids(output_file)
    if done_ids:
        logger.info(f"Phat hien {len(done_ids)} cau da lam trong {output_file}. Se bo qua.")

    # Khoi tao runner
    runner = None
    hirag_dir = work_dir or (
        ROOT / "HiRAG" / "hirag_cache"
        if (ROOT / "HiRAG" / "hirag_cache").exists()
        else ROOT / "HiRAG" / "test_hirag_cache"
    )

    is_adapted = ("adapted" in str(work_dir).lower())
    if is_adapted:
        logger.info(f"==> PHAT HIEN CACHE ADAPTED ({work_dir}) -> Dung embedding server tai http://127.0.0.1:11435")
        os.environ["HIRAG_EMBED_URL"] = "http://127.0.0.1:11435"
        os.environ["HIRAG_EMBED_MODEL"] = "nomic-embed-text-adapted"
        os.environ["LIGHTRAG_EMBED_HOST"] = "http://127.0.0.1:11435"
        os.environ["LIGHTRAG_EMBED_MODEL"] = "nomic-embed-text-adapted"
    else:
        logger.info(f"==> DUNG BASELINE EMBEDDING ({work_dir}) -> Dung Ollama tai http://localhost:11434")
        os.environ.pop("HIRAG_EMBED_URL", None)
        os.environ.pop("HIRAG_EMBED_MODEL", None)
        os.environ.pop("LIGHTRAG_EMBED_HOST", None)
        os.environ.pop("LIGHTRAG_EMBED_MODEL", None)

    if think:
        os.environ["LAB_LLM_MODEL"] = "qwen3:8b"
        os.environ["HIRAG_THINK"] = "true"
    else:
        os.environ["LAB_LLM_MODEL"] = "qwen3:8b-nothink"
        os.environ["HIRAG_THINK"] = "false"

    if system_name == "hirag":
        runner = HiRAGRunner(
            working_dir=hirag_dir,
            num_ctx=num_ctx,
            temperature=temperature,
        )
        await runner.init()
    elif system_name == "lightrag":
        runner = LightRAGRunner(num_ctx=num_ctx, temperature=temperature, think=think, working_dir=work_dir)
        await runner.init()
    elif system_name == "ms_graphrag":
        ms_dir = work_dir or (ROOT / "microsoft" / "ragtest_ms")
        settings_file = ms_dir / "settings.yaml"
        if settings_file.exists():
            c = settings_file.read_text(encoding="utf-8")
            if think:
                c = re.sub(r"model:\s*qwen3:8b-nothink", "model: qwen3:8b", c)
            else:
                c = re.sub(r"model:\s*qwen3:8b(?!-)", "model: qwen3:8b-nothink", c)
            settings_file.write_text(c, encoding="utf-8")
        runner = MSGraphRAGRunner(root_dir=ms_dir)
    elif system_name == "naive":
        runner = HiRAGRunner(
            working_dir=hirag_dir,
            num_ctx=num_ctx,
            temperature=temperature,
        )
        await runner.init()
        mode = "naive"
    else:
        raise ValueError(f"Unknown system: {system_name}")

    total = len(questions)
    completed = 0

    with open(output_file, "a", encoding="utf-8") as out_f:
        for idx, item in enumerate(questions, 1):
            q_id = item.get("id")
            if q_id in done_ids:
                continue

            q_text = item.get("question", "")
            q_doc = item.get("doc", "")
            logger.info(f"[{idx}/{total}] ID={q_id} (doc={q_doc}): {q_text[:70]}...")

            try:
                res = await runner.query(q_text, mode=mode)
                record = {
                    "id": q_id,
                    "doc": q_doc,
                    "question": q_text,
                    "system": system_name,
                    "mode": mode,
                    "answer": res["answer"],
                    "retrieved_chunks": res["retrieved_chunks"],
                    "latency_sec": res["latency_sec"],
                    "prompt_tokens": res["prompt_tokens"],
                    "completion_tokens": res["completion_tokens"],
                    "total_tokens": res["total_tokens"],
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "config": {
                        "model": "qwen3:8b",
                        "temperature": temperature,
                        "num_ctx": num_ctx,
                        "think": think,
                    },
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                completed += 1
                logger.info(f"  --> Xong trong {res['latency_sec']}s ({res['total_tokens']} tokens)")
            except Exception as e:
                logger.error(f"  [LOI] ID={q_id} that bai: {e}")
                error_record = {
                    "id": q_id,
                    "doc": q_doc,
                    "question": q_text,
                    "system": system_name,
                    "mode": mode,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                out_f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                out_f.flush()

    logger.info(f"Hoan tat! Da ghi them {completed} records vao {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Unified Eval Harness Runner")
    parser.add_argument(
        "--system",
        required=True,
        choices=["hirag", "lightrag", "ms_graphrag", "naive"],
        help="System can danh gia",
    )
    parser.add_argument(
        "--mode",
        default="hi",
        help="Mode truy van (hirag: hi/naive/hi_local/hi_global; lightrag: hybrid/local/global; ms_graphrag: local/global)",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=DEFAULT_EVAL_FILE,
        help="Duong dan toi file eval JSON (mac dinh: data/evaluation/eval_v2.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Duong dan file JSONL dau ra thô",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="So luong cau hoi toi da de test",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=4096,
        help="Context length cho model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature sinh cau tra loi (mac dinh 0.0)",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Bat thinking mode cho model (Qwen3)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Duong dan thu muc storage/cache cua he thong can danh gia",
    )

    args = parser.parse_args()

    if args.output is None:
        suffix = "_thinking" if args.think else ""
        out_name = f"{args.system}_{args.mode}{suffix}_raw.jsonl"
        args.output = ROOT / "eval_outputs" / out_name

    asyncio.run(
        run_harness(
            system_name=args.system,
            mode=args.mode,
            eval_file=args.eval_file,
            output_file=args.output,
            limit=args.limit,
            num_ctx=args.num_ctx,
            temperature=args.temperature,
            think=args.think,
            work_dir=args.work_dir,
        )
    )


if __name__ == "__main__":
    main()
