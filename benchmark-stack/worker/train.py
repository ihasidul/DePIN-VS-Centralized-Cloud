import os
import time
import requests
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from prometheus_client import start_http_server, Gauge, Histogram, Counter

PLATFORM = os.getenv("PLATFORM", "unknown")
PUSHGATEWAY = os.getenv("PUSHGATEWAY")

# Metrics
tokens_sec = Gauge("tokens_per_second", "Tokens/sec", ["platform"])
loss_gauge = Gauge("training_loss", "Loss", ["platform"])
global_step = Gauge("global_step", "Step", ["platform"])
step_hist = Histogram("step_duration_seconds", "Step Duration", ["platform"])
samples = Counter("samples_processed_total", "Samples", ["platform"])

start_http_server(8000)

submit_time = time.time()

def push_metric(name, value):
    requests.post(
        f"{PUSHGATEWAY}/metrics/job/{PLATFORM}",
        data=f"{name} {value}\n"
    )

# Startup timestamps
container_ready = time.time()
push_metric("container_ready_time", container_ready)

# Load dataset
dataset = load_dataset("databricks/databricks-dolly-15k")["train"]

# Load model
model_name = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

train_start = time.time()
push_metric("training_start_time", train_start)

# Format prompts
def fmt(x):
    return {
        "text":
        f"### Instruction:\n{x['instruction']}\n\n"
        f"### Input:\n{x['context']}\n\n"
        f"### Response:\n{x['response']}"
    }

dataset = dataset.map(fmt)

# LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=1,
    save_steps=500,
    learning_rate=2e-4,
    fp16=True
)

class Callback:
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                loss_gauge.labels(PLATFORM).set(logs["loss"])

            global_step.labels(PLATFORM).set(state.global_step)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    peft_config=peft_config,
    args=args
)

t0 = time.time()
trainer.train()
t1 = time.time()

push_metric("training_finished_time", t1)

trainer.save_model("./output/final")

