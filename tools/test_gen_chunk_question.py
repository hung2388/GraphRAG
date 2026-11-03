import json
import re
import sys
import requests
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


chunks = [json.loads(line) for line in open("data/eval_corpus_19/chunks_eval_62.jsonl", encoding="utf-8")]
c = chunks[8]

prompt = f"""You are an expert reading comprehension teacher creating a rigorous evaluation benchmark.
Given the following text chunk from article "{c['doc']}", write ONE high-quality, factual question that can be answered specifically and exclusively using this chunk.

Requirements:
- The question must be specific, natural, and require factual information from this chunk.
- Identify 1 or 2 exact quotes (gold_spans) directly copied from the text that contain the answer.
- Provide a concise short_answer and a complete ground_truth.
- List key supporting_entities.

Text:
{c['text'][:1500]}

Respond ONLY in valid JSON matching this schema:
{{
  "question": "...",
  "question_type": "factoid",
  "difficulty": "easy",
  "gold_spans": ["exact verbatim quote 1", "exact verbatim quote 2"],
  "short_answer": "...",
  "ground_truth": "...",
  "supporting_entities": ["..."]
}}"""

res = requests.post("http://localhost:11434/api/generate", json={
    "model": "qwen3:8b",
    "prompt": prompt,
    "stream": False,
    "format": "json",
    "options": {"temperature": 0.0, "num_predict": 500}
})

resp = res.json().get("response", "")
resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
print("Response:", resp)
data = json.loads(resp)
for span in data.get("gold_spans", []):
    in_chunk = span.lower() in c['text'].lower()
    print(f"Span: '{span}' in chunk: {in_chunk}")
