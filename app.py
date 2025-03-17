from fastapi import FastAPI, Form # type: ignore
from fastapi.responses import HTMLResponse, JSONResponse # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from transformers import BertTokenizer, BertForSequenceClassification # type: ignore
import torch

# Initialize FastAPI
app = FastAPI()

# Label mapping dictionary
label_mapping = {0: "Classification", 1: "Segmentation", 2: "Both"}

# Mount the static directory for serving HTML, CSS, and JS files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model and tokenizer
model_path = "/app/checkpoint-1400"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Serve the HTML page
@app.get("/", response_class=HTMLResponse)
def serve_html():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# API endpoint to handle user query
@app.post("/get_response")
def get_response(query: str = Form(...)):
    # Tokenize input text
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted class
    predicted_class = torch.argmax(outputs.logits, dim=-1).item()

    # Map class index to label name
    predicted_label = label_mapping.get(predicted_class, "Unknown")

    # Return response in JSON format
    return JSONResponse(content={"answer": f"Predicted class: {predicted_label}"})


# Run FastAPI with: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
