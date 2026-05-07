import os
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from prometheus_client import start_http_server, Gauge, Counter

# --------------------
# Metrics
# --------------------
PLATFORM = os.getenv("PLATFORM", "akash")

loss_gauge = Gauge("training_loss", "Training loss", ["platform"])
step_gauge = Gauge("global_step", "Training step", ["platform"])
samples_counter = Counter("samples_processed_total", "Samples processed", ["platform"])

start_http_server(8000)

print("Prometheus metrics exposed on :8000")

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

# --------------------
# Model
# --------------------
model_name = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True
)

# --------------------
# LoRA
# --------------------
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

# --------------------
# Training Args
# --------------------
args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=1,
    save_steps=200,
    learning_rate=2e-4,
    fp16=True,
    report_to="none"
)

# --------------------
# Trainer
# --------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    peft_config=peft_config,
    args=args
)

# --------------------
# Training loop hook (FIXED)
# --------------------
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
print("Starting training...")
t0 = time.time()

trainer.train()

t1 = time.time()

print(f"Training done in {t1 - t0}s")

trainer.save_model("./output/final")