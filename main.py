from fastapi import FastAPI
from pymongo import MongoClient
import os

from Sentiment import predict_sentiment
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# ---------------- MongoDB ----------------
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["HR"]  # or dynamic later
collection = db["Users"]

# ---------------- Intent Model ----------------
tokenizer = AutoTokenizer.from_pretrained("intent_model/")
model = AutoModelForSequenceClassification.from_pretrained("intent_model/")
model.eval()

ID2LABEL = {
    0: "confidential",
    1: "warning",
    2: "casual",
    3: "neutral",
}

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return ID2LABEL[pred]

# ---------------- API ----------------
@app.get("/")
def home():
    return {"status": "running"}

@app.post("/email")
def process_email(data: dict):
    text = data.get("text")

    if not text:
        return {"error": "text required"}

    intent = predict_intent(text)
    sentiment = predict_sentiment(text)

    result = {
        "text": text,
        "intent": intent,
        "sentiment": sentiment
    }

    # store in MongoDB
    collection.insert_one(result)

    return result