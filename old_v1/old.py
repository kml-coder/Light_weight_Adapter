import openai
import tiktoken
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tabulate import tabulate
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# -----------------------------
# 1. CONFIG
# -----------------------------

model_name = "gpt-4o"
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

word = "심마"

gold_definition = """
A disturbance of the mind and spirit that obstructs the path toward enlightenment.
"""

# -----------------------------
# 2. PROMPTS
# -----------------------------
before_prompt = f"""
Translate the Korean word '{word}' into English and explain its meaning,
but respond with exactly ONE SINGLE ENGLISH SENTENCE ONLY.
No Korean, no bullet points, no examples, and no multiple sentences.
"""


after_prompt = f"""
Apply the following cultural-meaning framework to the word '{word}':
1) Identify where the word originates.
2) Reconstruct the cultural or narrative scenario behind its usage.
3) Derive the culturally-grounded meaning.
Then translate that culturally-grounded meaning into English,
but respond with exactly ONE SINGLE ENGLISH SENTENCE ONLY.
No Korean, no bullet points, no multiple sentences.
"""


# -----------------------------
# 3. Token Counting Function
# -----------------------------
def count_tokens(prompt, model=model_name):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")  # fallback

    return len(encoding.encode(prompt))

# -----------------------------
# 4. LLM Call Function
# -----------------------------
def ask_llm(prompt):
    response = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# -----------------------------
# 5. Run LLM for both cases
# -----------------------------
print("Running baseline (before framework)…")
before_output = ask_llm(before_prompt)

print("\nRunning reconstructed scenario (after framework)…")
after_output = ask_llm(after_prompt)

# -----------------------------
# 6. Compute Embedding Similarity
# -----------------------------
gold_emb = embedder.encode(gold_definition, convert_to_tensor=True)
emb_before = embedder.encode(before_output, convert_to_tensor=True)
emb_after = embedder.encode(after_output, convert_to_tensor=True)

sim_before_gold = util.cos_sim(emb_before, gold_emb).item()
sim_after_gold = util.cos_sim(emb_after, gold_emb).item()
similarity = util.cos_sim(emb_before, emb_after).item()

sim_before_gold_percent = round(sim_before_gold*100, 2)
sim_after_gold_percent = round(sim_after_gold*100, 2)
similarity_percent = round(similarity * 100, 2)

# -----------------------------
# 7. Token Counts
# -----------------------------
before_tokens_prompt = count_tokens(before_prompt)
after_tokens_prompt = count_tokens(after_prompt)

before_tokens_output = count_tokens(before_output)
after_tokens_output = count_tokens(after_output)

# Total tokens = prompt + output
before_total_tokens = before_tokens_prompt + before_tokens_output
after_total_tokens = after_tokens_prompt + after_tokens_output

# -----------------------------
# 8. Result Table
# -----------------------------
df = pd.DataFrame([
    ["Before: prompt tokens", before_tokens_prompt],
    ["Before: output tokens", before_tokens_output],
    ["Before: total tokens", before_total_tokens],
    ["After: prompt tokens", after_tokens_prompt],
    ["After: output tokens", after_tokens_output],
    ["After: total tokens", after_total_tokens],
    ["Cosine similarity between before and after (%)", similarity_percent],
    ["Cosine similarity between before and gold (%)", sim_before_gold_percent],
    ["Cosine similarity between after and gold (%)", sim_after_gold_percent],
], columns=["Type", "Value"])

print("\n===== LLM EXPERIMENT RESULTS =====\n")
print(tabulate(df, headers="keys", tablefmt="github"))

print("\n=== BEFORE OUTPUT ===\n")
print(before_output)

print("\n=== AFTER OUTPUT ===\n")
print(after_output)
