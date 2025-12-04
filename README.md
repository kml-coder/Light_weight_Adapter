# Lightweight English–Korean Sentence Alignment

This repository implements a complete pipeline for aligning English and Korean sentence embeddings.  
The system performs:

- Text loading and segmentation  
- Paragraph-level and sentence-level DP alignment  
- Contextual embedding extraction with BERT and KoBERT  
- Procrustes alignment and trainable mapping models  
- Qualitative evaluation with visualization  
- GPT-based paraphrase perturbation testing  

---

## 1. Environment Setup

### 1.1 Clone repository
```bash
git clone https://github.com/kml-coder/light_weight_adapter.git
cd light_weight_adapter
```

### 1.2 Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate           # macOS / Linux
venv\Scripts\activate            # Windows
```

### 1.3 Install required packages

Install HuggingFace KoBERT:
```bash
pip install "git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf"
```
Install requirements.txt
```bash
pip install -r requirements.txt
```


---

## 2. OpenAI API Key Configuration

If you run the Day3 evaluation (GPT paraphrase tests), you must provide an API key.

### 2.1 Create a `.env` file and put your API Key in there like this:
```
OPENAI_API_KEY=sk-xxxxxx
```

---

## 3. Required Folder Structure

```
text_source/
    alice_eng_short.txt
    alice_kor_short.txt
    alice_eng_long.txt
    alice_kor_long.txt

text_results/
    (generated files)

days_results/
    (generated npz files)
```

---

## 4. Running the Pipeline

### 4.0 Run the function code first for to run all the code

### 4.1 Text Processing and Embedding Extraction
Run the Converter in 1. Converting and result Check
* You can change the ENG_FILE_PATH and KOR_FILE_PATH to alice_long_eng.txt and alice_long_kor.txt to create long .npz data

After you run it, it will:
Generates:
```
days_results/day1_paragraph_data.npz
days_results/day1_sentence_data.npz
days_results/day1_sentence_data_without.npz
days_results/day1_sentence_data_long.npz
```
### 4.2 Check the sentences and .npz data
Use .npz information checker and .npz NaN Checker
---

### 4.3 Train Alignment Models
Run the Adapter Trainer in 2. Training Adapter

Normally it looks like this
run_day2(SENTENCE_INPUT_FILE, SENTENCE_OUTPUT_FILE)

But you can edit it like this
run_day2("days_results/day1_sentence_data_without.npz", "days_results/day2_sentence_results_without.npz")
to train with the no ctx data

Outputs:
```
days_results/day2_sentence_results.npz
days_results/day2_sentence_results_best_model.pt
```

---

### 4.4 Evaluation and GPT Paraphrase Test

Run the code in 3. Test the trained Adapter

It will show Quality Test Data, t-SNE visualizations, and paraphrase evaluation.


## License
Research and academic use only.