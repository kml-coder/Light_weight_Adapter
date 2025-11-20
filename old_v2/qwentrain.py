import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "Qwen/Qwen2-1.5B"

# 1. Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    load_in_8bit=True
)

# 2. Prepare LoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 3. Load dataset (jsonl)
dataset = load_dataset("json", data_files={"train": "train.jsonl", "eval": "eval.jsonl"})

def format_example(example):
    user = example["messages"][0]["content"]
    assistant = example["messages"][1]["content"]
    text = f"<|user|>\n{user}\n<|assistant|>\n{assistant}"
    return {"text": text}

dataset = dataset.map(format_example)
dataset = dataset.map(lambda x: tokenizer(x["text"]), batched=True)

# 4. Training arguments
training_args = TrainingArguments(
    output_dir="./qwen-lora-output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=50,
    num_train_epochs=3,
    fp16=True,
    logging_steps=20,
    save_steps=500,
    evaluation_strategy="epoch",
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    data_collator=data_collator,
)

# 6. Train
trainer.train()

# 7. Save LoRA model
trainer.save_model("./qwen-lora-final")
