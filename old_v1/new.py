import os
import openai
import google.generativeai as genai
import anthropic
import tiktoken
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tabulate import tabulate
import torch
import re
import math
from collections import Counter

from sklearn.metrics import silhouette_score
import numpy as np

# =========================================================
# 0. INIT
# =========================================================
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))


embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

culture_words = ["체면","눈치","한(恨)","정(情)", "애교"]
one_one_words = ["책","컴퓨터","사과","바나나","학교"]
culture_word = ""

# =========================================================
# EXTRACT BRACKET TERMS
# =========================================================
def extract_bracket_terms(text):
    return re.findall(r"\[(.*?)\]", text)


# =========================================================
# ★★  METRIC FUNCTIONS 추가
# =========================================================

def entropy(items): # 안 쓸것, 이거는 뭐냐면 llm이 토큰 후보군 만들고, llm이 준 확률들을 써서, 그 확률들이 비슷하면 엔드로피 낮은, 다 다르면 엔트로피 높음 이렇게
    counts = Counter(items)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log(p + 1e-12) for p in probs)

def type_token_ratio(items): # 이거는 쓸만할지도 모름, 그런데 각 번역된 단어군을 비교하는게 아니라 전체 모아서 비교해야될거 같아 우선 후보에서 제외
    if len(items) == 0:
        return 0
    return len(set(items)) / len(items)

def distinct_1(items): # 단어 다양성이나 반복이라, 살짝 애매할 수 있다, 중의적인 단어도 비슷하게 할것이라서
    tokens = " ".join(items).split()
    if len(tokens) == 0:
        return 0
    return len(set(tokens)) / len(tokens)

def distinct_2(items):
    tokens = " ".join(items).split()
    if len(tokens) < 2:
        return 0
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    return len(set(bigrams)) / len(bigrams)

def embedding_variance(sent_list):
    if len(sent_list) == 0:
        return 0
    embs = embedder.encode(sent_list)
    return float(np.var(embs))

def mean_cosine_distance(sent_list): # 
    if len(sent_list) < 2:
        return 0
    embs = embedder.encode(sent_list)
    cos = util.cos_sim(embs, embs)
    # 상삼각행렬 평균
    dists = []
    n = len(sent_list)
    for i in range(n):
        for j in range(i+1, n):
            dists.append(1 - float(cos[i][j]))
    return float(np.mean(dists))

def silhouette(sent_list): # 의미, 군집이 얼마나 다양하게 있냐를 찾음
    if len(sent_list) < 3:
        return 0
    embs = embedder.encode(sent_list)
    labels = np.zeros(len(sent_list))  # 1 cluster
    return silhouette_score(embs, labels)

# =========================================================
# STORAGE
# =========================================================
normal_results = {}
plain_results = {}
translation_results = {}

sim_results = {}
scenario_results = {}

# ★★ 추가: bracket terms 저장 dict
bracket_terms_store = {}         # { model_name: {culture_word: [terms]} }
metrics_store = {}               # { model_name: {culture_word: metric_dict} }

# ★★ 추가: 시나리오와 번역 저장 dict
scenario_store = {}        # { model_name: {culture_word: scenario_text} }
translation_store = {}     # { model_name: {culture_word: translation_text} }


# =========================================================
# 3. API MODEL HELPERS
# =========================================================
def ask_gpt(prompt):
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()


def ask_claude(prompt):
    resp = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text.strip()


def ask_gemini(prompt):
    model = genai.GenerativeModel("gemini-2.5-flash")
    resp = model.generate_content(prompt)
    return resp.text.strip()


# =========================================================
# 11. PROCESS: API MODELS FIRST
# =========================================================
api_models = ["GPT-4o"] # , "Claude", "Gemini"

print("\n=== PROCESSING API MODELS ===\n")

for m in api_models:
    ask = {"GPT-4o": ask_gpt}[m]
    bracket_terms_store[m] = {}
    scenario_store[m] = {}
    translation_store[m] = {}

    for culture_word in culture_words:

        # 1) normal translation
        normal = ask(f"Translate the Korean word '{culture_word}' into English.")
        normal_results[m] = normal

        # 2) scenario generation
        scenario_korean = ask(f"""
        Create 10 Korean scenarios, each written as exactly one sentence, where the '{culture_word}' is included in the sentence.

        Requirements:
        - Write only in Korean.
        - Each scenario must include the exact word '{culture_word}'.
        - Each scenario must be exactly one sentence long.
        - Number each scenario from 1 to 10.
        """)
        scenario_results[m] = scenario_korean

        # ★★ 저장
        scenario_store[m][culture_word] = scenario_korean

        # 3) translation with brackets
        translation = ask(f"""
        Translate the following Korean scenario into natural and fluent English.

        Rules:
        - You MUST translate every occurrence of '{culture_word}' into English.
        - Whenever the word '{culture_word}' appears, wrap ONLY its translated part in brackets '[ ]'.
        - Do NOT explain the meaning.
        - Translate every other part faithfully.

        Scenario:
        {scenario_korean}
        """)

        translation_results[m] = translation
        # ★★ 저장
        translation_store[m][culture_word] = translation

        # =========================================================
        # ★★ 1. Extract bracket terms & STORE
        # =========================================================
        bracket_list = extract_bracket_terms(translation)
        bracket_terms_store[m][culture_word] = bracket_list

        print(f"{m} / {culture_word} complete. Extracted: {len(bracket_list)} terms\n")


# =========================================================
# ★★ 2. Compute METRICS for each model × culture_word
# =========================================================
for m in bracket_terms_store:
    metrics_store[m] = {}

    for cw in bracket_terms_store[m]:
        items = bracket_terms_store[m][cw]

        metrics_store[m][cw] = {
            "entropy": entropy(items),
            "type_token_ratio": type_token_ratio(items),
            "distinct_1": distinct_1(items),
            "distinct_2": distinct_2(items),
            "embedding_variance": embedding_variance(items),
            "mean_cosine_distance": mean_cosine_distance(items),
            "silhouette_score": silhouette(items),
        }


# =========================================================
# ★★  PRINT STORED SCENARIOS + TRANSLATIONS
# =========================================================
print("\n==================== STORED SCENARIOS ====================")
for m in scenario_store:
    for cw in scenario_store[m]:
        print(f"\n[{m}] — Korean Scenarios for '{cw}':\n")
        print(scenario_store[m][cw])

print("\n=================== STORED TRANSLATIONS ===================")
for m in translation_store:
    for cw in translation_store[m]:
        print(f"\n[{m}] — Translated English for '{cw}':\n")
        print(translation_store[m][cw])


# =========================================================
# PRINT RESULT (optional)
# =========================================================
print("\n=== BRACKET TERMS STORED ===")
print(bracket_terms_store)

print("\n=== METRICS STORED ===")
print(metrics_store)
