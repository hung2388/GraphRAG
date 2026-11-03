import os
import sys

os.environ["HF_HOME"] = "D:/Lab/.cache/huggingface"

import sentence_transformers
print("ST version:", sentence_transformers.__version__)
from sentence_transformers import SentenceTransformerTrainingArguments

args = SentenceTransformerTrainingArguments(output_dir="tmp_test")
print("Has peft_config in args?", hasattr(args, "peft_config") or "peft_config" in dir(args))
