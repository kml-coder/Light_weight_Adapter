from openai import OpenAI

client = OpenAI()

# 1. 파일 업로드
train_file = client.files.create(
    file=open("train.jsonl", "rb"),
    purpose="fine-tune"
)

eval_file = client.files.create(
    file=open("eval.jsonl", "rb"),
    purpose="fine-tune"
)

# 2. 파인튜닝 시작
job = client.fine_tuning.jobs.create(
    model="gpt-4o-mini",
    training_file=train_file.id,
    validation_file=eval_file.id,
    hyperparameters={
        "n_epochs": 3,
        "learning_rate_multiplier": 1.1
    }
)

print("Fine-tuning job started:", job.id)
