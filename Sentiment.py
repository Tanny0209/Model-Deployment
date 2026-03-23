import os
import re
import torch
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ===================== PERFORMANCE =====================
torch.set_num_threads(8)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ===================== CONFIG =====================
MONGO_URI = (
    "mongodb+srv://tanmaypawar183_db_user:Tanmay012345@"
    "internal-eye.htwospq.mongodb.net/internal_eye"
    "?retryWrites=true&w=majority&appName=Internal-Eye"
)

DEPARTMENT_DATABASES = ["Finance", "HR", "IT", "Sales"]
COLLECTION_NAME = "Users"

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

MAX_LEN = 96
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== LOAD MODEL =====================
print("[+] Loading sentiment model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

# ===================== CLEANING FUNCTION =====================
def clean_email_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===================== SENTIMENT PREDICTION =====================
@torch.no_grad()
def predict_sentiment_batch(texts):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(DEVICE)

    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    preds = torch.argmax(probs, dim=1).cpu().numpy()

    sentiments = []
    for p in preds:
        if p <= 1:
            sentiments.append("negative")
        elif p == 2:
            sentiments.append("neutral")
        else:
            sentiments.append("positive")

    return sentiments

# ===================== CONNECT TO MONGODB =====================
print("[+] Connecting to MongoDB Atlas...")
client = MongoClient(MONGO_URI)

# ===================== PROCESS DEPARTMENTS =====================
for dept in DEPARTMENT_DATABASES:
    print(f"\n[+] Processing department: {dept}")
    db = client[dept]
    collection = db[COLLECTION_NAME]

    batch_texts = []
    batch_paths = []

    for doc in tqdm(collection.find({})):
        users = doc.get("users", {})

        for user_id, user_data in users.items():
            emails = user_data.get("emails", {})

            for email_id, email_data in emails.items():
                subject = email_data.get("subject", "")
                body = email_data.get("body", "")
                text = email_data.get("text", "")

                full_text = clean_email_text(
                    f"{subject} {body} {text}".strip()
                )

                if len(full_text) < 5:
                    continue

                batch_texts.append(full_text)
                batch_paths.append(
                    (doc["_id"], user_id, email_id)
                )

                # ===== RUN BATCH =====
                if len(batch_texts) >= BATCH_SIZE:
                    sentiments = predict_sentiment_batch(batch_texts)

                    for (doc_id, u_id, e_id), sentiment in zip(batch_paths, sentiments):
                        collection.update_one(
                            {"_id": doc_id},
                            {
                                "$set": {
                                    f"users.{u_id}.emails.{e_id}.sentiment": sentiment
                                }
                            },
                        )

                    batch_texts.clear()
                    batch_paths.clear()

    # ===== FLUSH REMAINING (CRITICAL) =====
    if batch_texts:
        sentiments = predict_sentiment_batch(batch_texts)

        for (doc_id, u_id, e_id), sentiment in zip(batch_paths, sentiments):
            collection.update_one(
                {"_id": doc_id},
                {
                    "$set": {
                        f"users.{u_id}.emails.{e_id}.sentiment": sentiment
                    }
                },
            )

print("\n[✓] Sentiment analysis completed for ALL emails.")
