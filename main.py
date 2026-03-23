from fastapi import FastAPI
from pymongo import MongoClient
import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------ INIT ------------------
app = FastAPI()

# ------------------ ENV ------------------
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

# ------------------ CONFIG ------------------
DEPARTMENTS = ["Finance", "HR", "IT", "Sales"]
COLLECTION_NAME = "Users"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ LOAD INTENT MODEL ------------------
tokenizer_intent = AutoTokenizer.from_pretrained("intent_model/")
model_intent = AutoModelForSequenceClassification.from_pretrained("intent_model/")
model_intent.to(DEVICE)
model_intent.eval()

ID2LABEL = {
    0: "confidential",
    1: "warning",
    2: "casual",
    3: "neutral",
}

# ------------------ LOAD SENTIMENT MODEL ------------------
from Sentiment import predict_sentiment  # your existing file

# ------------------ CLEANING ------------------
def clean_email_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ------------------ INTENT ------------------
@torch.no_grad()
def predict_intent(text):
    text = clean_email_text(text)

    inputs = tokenizer_intent(
        text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    ).to(DEVICE)

    outputs = model_intent(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()

    return ID2LABEL[pred]

# ------------------ RISK FUNCTIONS ------------------

def intent_risk(intent):
    return {
        "confidential": 1.0,
        "warning": 0.7,
        "neutral": 0.2,
        "casual": 0.1,
    }.get(intent, 0.1)

def sentiment_risk(sentiment):
    return {
        "negative": 0.7,
        "neutral": 0.3,
        "positive": 0.1,
    }.get(sentiment, 0.2)

def rule_boost(email):
    boost = 0.0
    if email.get("is_external"):
        boost += 0.05
    if email.get("has_attachment"):
        boost += 0.05
    if email.get("contains_sensitive_keywords"):
        boost += 0.1
    return min(boost, 0.2)

def risk_label(score):
    if score >= 0.75:
        return "HIGH"
    elif score >= 0.45:
        return "MEDIUM"
    else:
        return "LOW"

# ------------------ ROUTES ------------------

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/email")
def process_email(data: dict):
    try:
        department = data.get("department")
        user_id = data.get("user_id")
        email_id = data.get("email_id")

        if department not in DEPARTMENTS:
            return {"error": "Invalid department"}

        db = client[department]
        collection = db[COLLECTION_NAME]

        subject = data.get("subject", "")
        body = data.get("body", "")
        text = data.get("text", "")

        full_text = f"{subject} {body} {text}".strip()

        # -------- RUN MODELS --------
        intent = predict_intent(full_text)
        sentiment = predict_sentiment(full_text)

        # -------- FETCH USER (for GNN score) --------
        doc = collection.find_one({"users." + user_id: {"$exists": True}})

        if not doc:
            return {"error": "User not found"}

        user_data = doc["users"][user_id]
        graph_score = float(user_data.get("cached_user_graph_score", 0.0))

        # -------- RISK FUSION --------
        intent_score = intent_risk(intent)
        sentiment_score = sentiment_risk(sentiment)
        rule_score = rule_boost(data)

        final_score = (
            0.4 * intent_score +
            0.2 * sentiment_score +
            0.4 * graph_score +
            rule_score
        )

        final_score = max(0.0, min(final_score, 1.0))
        final_level = risk_label(final_score)

        # -------- BUILD EMAIL OBJECT --------
        email_obj = {
            "email_id": email_id,
            "subject": subject,
            "body": body,
            "text": text,
            "intent": intent,
            "sentiment": sentiment,
            "risk_score": round(final_score, 4),
            "risk_level": final_level,
            "created_at": data.get("created_at"),
        }

        # -------- SAVE TO MONGODB --------
        collection.update_one(
            {"_id": doc["_id"]},
            {
                "$set": {
                    f"users.{user_id}.emails.{email_id}": email_obj
                }
            }
        )

        return {
            "intent": intent,
            "sentiment": sentiment,
            "risk_score": final_score,
            "risk_level": final_level
        }

    except Exception as e:
        return {"error": str(e)}