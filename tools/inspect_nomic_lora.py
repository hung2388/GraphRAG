import os
import sys

os.environ["HF_HOME"] = "D:/Lab/.cache/huggingface"

import torch
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cpu")
print("Transformer module:", type(model[0].auto_model))

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["Wqkv", "out_proj"],
    lora_dropout=0.05,
    bias="none",
)
peft_model = get_peft_model(model[0].auto_model, config)
peft_model.print_trainable_parameters()
print("LoRA successfully configured!")

