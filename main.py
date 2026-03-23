from fastapi import FastAPI
from pymongo import MongoClient
import torch
from transformers import AutoTokenizer, AutoModel
import os
from dotenv import load_dotenv

# ------------------ LOAD ENV ------------------
load_dotenv()

# ------------------ INIT APP ------------------
app = FastAPI()

# ------------------ MONGODB ------------------
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI not set")

client = MongoClient(MONGO_URI)
db = client["insider_threat"]
collection = db["emails"]

# ------------------ LOAD MODELS ------------------

# GNN model
gnn_model = torch.load("gnn/final_gnn_model.pt", map_location="cpu")
gnn_model.eval()

# Intent model
tokenizer = AutoTokenizer.from_pretrained("intent_model/")
intent_model = AutoModel.from_pretrained("intent_model/")
intent_model.eval()

# Sentiment model
from Sentiment import predict_sentiment

# ------------------ HELPERS ------------------

def normalize_department(dept: str):
    if not dept:
        return None
    return dept.strip().upper()

def run_intent(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = intent_model(**inputs)

    # Simple representation (replace with classifier later)
    return float(outputs.last_hidden_state.mean())

def run_gnn(text: str):
    # TODO: Replace with real GNN logic
    with torch.no_grad():
        return "gnn_output_placeholder"

# ------------------ ROUTES ------------------

@app.get("/")
def home():
    return {"status": "API is running"}

@app.post("/analyze")
def analyze(data: dict):
    try:
        text = data.get("email")
        department = normalize_department(data.get("department"))

        if not text:
            return {"error": "Email text is required"}

        # -------- RUN MODELS --------
        intent = run_intent(text)
        sentiment = predict_sentiment(text)
        gnn_result = run_gnn(text)

        # -------- RESULT --------
        result = {
            "email": text,
            "department": department,
            "intent": intent,
            "sentiment": sentiment,
            "gnn": gnn_result
        }

        # -------- SAVE --------
        collection.insert_one(result)

        return result

    except Exception as e:
        return {"error": str(e)}