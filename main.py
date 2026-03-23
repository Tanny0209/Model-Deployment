from fastapi import FastAPI
from pymongo import MongoClient
import torch
from transformers import AutoTokenizer, AutoModel

app = FastAPI()

# MongoDB
client = MongoClient("YOUR_MONGO_URI")
db = client["insider_threat"]
collection = db["emails"]

# Load models ONCE
gnn_model = torch.load("gnn/final_gnn_model.pt", map_location="cpu")

tokenizer = AutoTokenizer.from_pretrained("intent_model/")
intent_model = AutoModel.from_pretrained("intent_model/")

from sentiment_model import predict_sentiment

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/analyze")
def analyze(data: dict):
    text = data["email"]

    # Intent
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    intent_output = intent_model(**inputs)

    # Sentiment
    sentiment = predict_sentiment(text)

    # GNN (dummy example)
    gnn_result = "run your gnn logic"

    result = {
        "email": text,
        "intent": str(intent_output),
        "sentiment": sentiment,
        "gnn": gnn_result
    }

    collection.insert_one(result)

    return result