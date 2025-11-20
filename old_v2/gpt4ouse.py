from openai import OpenAI

def load_prompts(path="data/test_prompts.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

MODEL_ID = "ft:gpt-4o-mini:YOUR_ID_HERE"

client = OpenAI()

prompts = load_prompts()

for p in prompts:
    print("=== INPUT ===")
    print(p)

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": p}]
    )

    print("=== OUTPUT ===")
    print(response.choices[0].message["content"])
    print()
