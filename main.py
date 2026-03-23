from fastapi import FastAPI
from pymongo import MongoClient
import os
import re
import requests

# ------------------ INIT ------------------
app = FastAPI()

# ------------------ ENV ------------------
MONGO_URI = os.getenv("MONGO_URI")
HF_TOKEN = os.getenv("HF_TOKEN")


client = MongoClient(MONGO_URI)

# ------------------ CONFIG ------------------
DEPARTMENTS = ["Finance", "HR", "IT", "Sales"]
COLLECTION_NAME = "Users"

# ------------------ HF API ------------------
INTENT_API = "https://api-inference.huggingface.co/models/tanmay0209/intent-model"
SENTIMENT_API = "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

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
def predict_intent(text):
    try:
        text = clean_email_text(text)

        response = requests.post(
            INTENT_API,
            headers=headers,
            json={"inputs": text}
        )

        result = response.json()
        print("Intent HF:", result)

        if isinstance(result, list):
            return result[0]["label"]

        return "neutral"

    except Exception as e:
        print("Intent error:", e)
        return "neutral"

# ------------------ SENTIMENT ------------------
def predict_sentiment(text):
    try:
        text = clean_email_text(text)

        response = requests.post(
            SENTIMENT_API,
            headers=headers,
            json={"inputs": text}
        )

        result = response.json()
        print("Sentiment HF:", result)

        if isinstance(result, list):
            label = result[0]["label"]  # e.g. "5 stars"
            stars = int(label.split()[0])

            if stars <= 2:
                return "negative"
            elif stars == 3:
                return "neutral"
            else:
                return "positive"

        return "neutral"

    except Exception as e:
        print("Sentiment error:", e)
        return "neutral"

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

        # -------- MODEL OUTPUT --------
        intent = predict_intent(full_text)
        sentiment = predict_sentiment(full_text)

        # -------- FETCH USER --------
        doc = collection.find_one({"users." + user_id: {"$exists": True}})
        if not doc:
            return {"error": "User not found"}

        graph_score = float(doc["users"][user_id].get("cached_user_graph_score", 0.0))

        # -------- RISK FUSION --------
        final_score = (
            0.4 * intent_risk(intent) +
            0.2 * sentiment_risk(sentiment) +
            0.4 * graph_score +
            rule_boost(data)
        )

        final_score = max(0.0, min(final_score, 1.0))
        final_level = risk_label(final_score)

        # -------- EMAIL OBJECT --------
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

        # -------- SAVE --------
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