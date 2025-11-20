from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def load_prompts(path="data/test_prompts.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

MODEL_NAME = "Qwen/Qwen2-1.5B"
LORA_DIR = "./qwen-lora-final"   # 네가 학습해서 export한 LoRA 경로

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
model = PeftModel.from_pretrained(model, LORA_DIR)

prompts = load_prompts()

for p in prompts:
    print("=== INPUT ===")
    print(p)

    qwen_prompt = f"<|user|>\n{p}\n<|assistant|>\n"

    inputs = tokenizer(qwen_prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
    )

    print("=== OUTPUT ===")
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print()
