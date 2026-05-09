import os
import time
import torch
import subprocess
import threading

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from prometheus_client import start_http_server, Gauge, Counter

# --------------------
# Metrics
# --------------------
PLATFORM = os.getenv("PLATFORM", "aws")

loss_gauge = Gauge("training_loss", "Training loss", ["platform"])
step_gauge = Gauge("global_step", "Training step", ["platform"])
samples_counter = Counter("samples_processed_total", "Samples processed", ["platform"])

gpu_util = Gauge("aws_gpu_utilization_percent", "AWS GPU utilization percent", ["platform"])
gpu_mem_used = Gauge("aws_gpu_memory_used_mb", "AWS GPU memory used MB", ["platform"])
gpu_mem_total = Gauge("aws_gpu_memory_total_mb", "AWS GPU memory total MB", ["platform"])
gpu_temp = Gauge("aws_gpu_temperature_celsius", "AWS GPU temperature Celsius", ["platform"])

start_http_server(8000)

def collect_gpu_metrics():
    while True:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            util, mem_used, mem_total, temp = [
                float(x.strip()) for x in output.strip().splitlines()[0].split(",")
            ]

            gpu_util.labels(PLATFORM).set(util)
            gpu_mem_used.labels(PLATFORM).set(mem_used)
            gpu_mem_total.labels(PLATFORM).set(mem_total)
            gpu_temp.labels(PLATFORM).set(temp)

        except Exception as e:
            print(f"GPU metrics error: {e}", flush=True)

        time.sleep(5)

threading.Thread(target=collect_gpu_metrics, daemon=True).start()

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# --------------------
# Model / Tokenizer
# --------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# --------------------
# Dataset
# --------------------
dataset = load_dataset("databricks/databricks-dolly-15k")["train"]

def format_example(example):
    return {
        "text":
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Context:\n{example.get('context','')}\n\n"
        f"### Response:\n{example['response']}"
    }

dataset = dataset.map(format_example)

def tokenize_and_truncate(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding=False,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(
    tokenize_and_truncate,
    remove_columns=dataset.column_names,
)

# --------------------
# Model
# --------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)

# --------------------
# LoRA
# --------------------
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# --------------------
# Training Args
# --------------------
args = SFTConfig(
    output_dir="/home/ubuntu/training-output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    # Same long-running benchmark mode as Akash
    max_steps=55000,

    logging_steps=1,
    save_strategy="steps",
    save_steps=500,

    learning_rate=2e-4,
    bf16=True,
    fp16=False,
    report_to="none",
    gradient_checkpointing=True,
)

# --------------------
# Trainer
# --------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=args,
)

old_log = trainer.callback_handler.on_log

def patched_log(args, state, control, logs=None, **kwargs):
    if logs and "loss" in logs:
        loss_gauge.labels(PLATFORM).set(logs["loss"])

    step_gauge.labels(PLATFORM).set(state.global_step)

    return old_log(args, state, control, logs, **kwargs)

trainer.callback_handler.on_log = patched_log

# --------------------
# Train
# --------------------
print("Starting AWS training...")
t0 = time.time()

trainer.train()

t1 = time.time()

print(f"Training done in {t1 - t0}s")

trainer.save_model("/home/ubuntu/training-output/final")

print("Training complete")