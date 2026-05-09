import os
import time
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from prometheus_client import (
    start_http_server,
    Gauge,
)

# ====================================
# Prometheus Metrics
# ====================================

gpu_util = Gauge(
    "aws_gpu_utilization_percent",
    "GPU utilization percent"
)

gpu_mem = Gauge(
    "aws_gpu_memory_used_mb",
    "GPU memory used"
)

loss_metric = Gauge(
    "training_loss",
    "Training loss"
)

step_metric = Gauge(
    "global_step",
    "Training step"
)

start_http_server(8000)

# ====================================
# Dataset
# ====================================

dataset = load_dataset(
    "databricks/databricks-dolly-15k"
)["train"]

def format_example(example):
    return {
        "text":
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['response']}"
    }

dataset = dataset.map(format_example)

# ====================================
# Model
# ====================================

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens

dataset = dataset.map(
    tokenize,
    remove_columns=dataset.column_names,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
)

# ====================================
# LoRA
# ====================================

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

# ====================================
# Training Config
# ====================================

args = SFTConfig(
    output_dir="./output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=1,
    bf16=True,
    fp16=False,
    report_to="none",
)

# ====================================
# Trainer
# ====================================

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=args,
)

print("Starting training...")

trainer.train()

trainer.save_model("./output/final")

print("Training complete")