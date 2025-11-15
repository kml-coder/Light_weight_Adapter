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
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MarianMTModel,
    MarianTokenizer,
    M2M100Tokenizer,
    M2M100ForConditionalGeneration,
    AutoModelForCausalLM
)

# =========================================================
# 0. INIT
# =========================================================
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

model_name = "gpt-4o"
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

culture_word = "체면"

# =========================================================
# 1. Scenario Prompt
# =========================================================
kor_scenario_generation_prompt = f"""
Create 15 Korean scenarios, each written as exactly one sentence, where the '{culture_word}' is included in the sentence.

Requirements:
- Write only in Korean.
- Each scenario must include the exact word '{culture_word}'.
- Each scenario must be exactly one sentence long.
- Number each scenario from 1 to 15.
"""

translation_prompt = f"""
Translate the following Korean scenario into natural and fluent English.

Rules:
- Whenever the word '{culture_word}' appears, replace ONLY its translated content with '[MASK]'.
- Do NOT explain the meaning of the concept.
- Do NOT translate the word '{culture_word}'.
- Do NOT paraphrase, summarize, or alter the structure.
- Translate every other part faithfully.

Scenario:
{{scenario_korean}}
"""



gold_definition = """
A culturally conditioned sensitivity to unspoken social cues and shared situational expectations.
"""

# =========================================================
# 2. Token Count
# =========================================================
def count_tokens(text, model):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

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
# 5. OPEN-SOURCE MODEL LOADERS
# =========================================================
def load_opus():
    name = "Helsinki-NLP/opus-mt-ko-en"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    return tok, model

def load_marian():
    name = "Helsinki-NLP/opus-mt-ko-en"
    tok = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name)
    return tok, model

def load_m2m():
    name = "facebook/m2m100_418M"
    tok = M2M100Tokenizer.from_pretrained(name)
    model = M2M100ForConditionalGeneration.from_pretrained(name)
    return tok, model

def load_nllb():
    name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    return tok, model

def load_llama():
    name = "meta-llama/Llama-3.1-8B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto")
    return tok, model

def load_qwen():
    name = "Qwen/Qwen2.5-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto")
    return tok, model

# =========================================================
# 6. GENERIC TRANSLATION FUNCTIONS
# =========================================================
def translate_opus(text, tok, model):
    enc = tok(text, return_tensors="pt")
    gen = model.generate(**enc)
    return tok.batch_decode(gen, skip_special_tokens=True)[0]

def translate_marian(text, tok, model):
    enc = tok([text], return_tensors="pt", padding=True)
    gen = model.generate(**enc)
    return tok.batch_decode(gen, skip_special_tokens=True)[0]

def translate_m2m(text, tok, model):
    tok.src_lang = "ko"
    enc = tok(text, return_tensors="pt")
    gen = model.generate(**enc, forced_bos_token_id=tok.get_lang_id("en"))
    return tok.batch_decode(gen, skip_special_tokens=True)[0]

def translate_nllb(text, tok, model):
    enc = tok(text, return_tensors="pt")
    gen = model.generate(**enc, forced_bos_token_id=tok.lang_code_to_id["eng_Latn"])
    return tok.batch_decode(gen, skip_special_tokens=True)[0]

def translate_llama(text, tok, model):
    prompt = f"Translate to English:\n{text}"
    inp = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=200)
    return tok.decode(out[0], skip_special_tokens=True)

def translate_qwen(text, tok, model):
    prompt = f"Translate to English:\n{text}"
    inp = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=200)
    return tok.decode(out[0], skip_special_tokens=True)

# =========================================================
# 7. LOAD ALL LOCAL MODELS
# =========================================================
opus_tok, opus_model = load_opus()
marian_tok, marian_model = load_marian()
m2m_tok, m2m_model = load_m2m()
nllb_tok, nllb_model = load_nllb()
llama_tok, llama_model = load_llama()
qwen_tok, qwen_model = load_qwen()

models = {
    "OpusMT": (translate_opus, opus_tok, opus_model),
    "MarianMT": (translate_marian, marian_tok, marian_model),
    "M2M100": (translate_m2m, m2m_tok, m2m_model),
    "NLLB-200": (translate_nllb, nllb_tok, nllb_model),
    "Llama3": (translate_llama, llama_tok, llama_model),
    "Qwen2.5": (translate_qwen, qwen_tok, qwen_model)
}

# =========================================================
# 8. API MODELS AS SAME INTERFACE
# =========================================================
def api_normal(m):
    return {
        "GPT-4o": ask_gpt,
        "Claude": ask_claude,
        "Gemini": ask_gemini
    }[m]

# =========================================================
# 9. METAPHOR PROMPT
# =========================================================
metaphor_template = """
Generate 15 metaphorical English expressions (3–6 words) that can replace [MASK]
in the following situation:

[SITUATION]

Expressions must:
- sound idiomatic and natural
- evoke atmosphere, tension, pressure, balance, or subtle social force
- use physical metaphors (air, heat, weight, shadow, posture, light, space)
- avoid abstract nouns (identity, ego, dignity, pride, context)
- must NOT reference the hidden cultural concept
- must NOT directly describe emotions
- No explanations. Only expressions.
"""

# =========================================================
# 10. STORAGE
# =========================================================
normal_results = {}
plain_results = {}
framework_results = {}
metaphor_results = {}
sim_results = {}
scenario_results = {}

# =========================================================
# 11. PROCESS: API MODELS FIRST (GPT, Claude, Gemini)
# =========================================================
api_models = ["GPT-4o", "Claude", "Gemini"]

print("\n=== PROCESSING API MODELS ===\n")

for m in api_models:
    ask = api_normal(m)

    # 1) normal translation (단어만)
    normal = ask(f"Translate the Korean word '{culture_word}' into English.")
    normal_results[m] = normal

    # 2) scenario generation (한국어, culture_word 포함)
    scenario_korean = ask(kor_scenario_generation_prompt)
    scenario_results[m] = scenario_korean


    # 3) framework masked translation (LLM이 직접 mask)
    masked = ask(f"""
    Translate the following Korean scenario into English.
    Replace ONLY the translated portion of '{culture_word}' with '[MASK]'.
    Do not explain, paraphrase, or summarize.

    Scenario:
    {scenario_korean}
    """)
    framework_results[m] = masked

    # 4) metaphor generation
    metaphor_prompt = metaphor_template.replace("[SITUATION]", masked)
    metaphors = ask(metaphor_prompt)
    metaphor_results[m] = metaphors

    # # 5) similarity
    # out_emb = embedder.encode(metaphors, convert_to_tensor=True)
    # sim_results[m] = float(util.cos_sim(out_emb, gold_emb))

    print(f"{m} complete.\n")


# =========================================================
# OPEN-SOURCE MODELS (LOCAL)
# =========================================================

for name, (func, tok, model) in models.items():

    # 1) normal translation
    normal = func(f"'{culture_word}'를 영어로 번역해줘.", tok, model)
    normal_results[name] = normal

    # 2) scenario generation (GPT 사용 — local models are weak at Korean generation)
    scenario_korean = ask_gpt(kor_scenario_generation_prompt)
    scenario_results[name] = scenario_korean

    # 3) framework masked translation (GPT로 수행)
    # 4) masked translation (framework)
    masked = ask(f"""
    Translate the following Korean scenario into English.
    Replace ONLY the translated portion of '{culture_word}' with '[MASK]'.
    Do not explain or paraphrase.

    Scenario:
    {scenario_korean}
    """)
    framework_results[name] = masked

    # 4) metaphor generation (GPT)
    metaphor_prompt = metaphor_template.replace("[SITUATION]", masked)
    metaphors = ask_gpt(metaphor_prompt)
    metaphor_results[name] = metaphors

    # # 5) similarity
    # out_emb = embedder.encode(metaphors, convert_to_tensor=True)
    # sim_results[name] = float(util.cos_sim(out_emb, gold_emb))

    print(f"{name} complete.\n")


# =========================================================
# 13. TABLE OUTPUT
# =========================================================
rows = []
for m in normal_results:
    rows.append([
        m,
        normal_results[m],
        scenario_results[m],
        plain_results[m],
        framework_results[m],
        round(sim_results[m] * 100, 2)
    ])

df = pd.DataFrame(rows, columns=[
    "Model",
    "Normal Translation",
    "Plain Translation",
    "Framework Translation",
    "Metaphor Similarity (%)"
])

print("\n===== FINAL MULTI-MODEL FULL COMPARISON =====\n")
print(tabulate(df, headers="keys", tablefmt="github"))

