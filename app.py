from fastapi import FastAPI, Form # type: ignore
from fastapi.responses import HTMLResponse, JSONResponse # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
import gdown  # For Google Drive downloads
import zipfile

# Initialize FastAPI
app = FastAPI()

# Label mapping dictionary
label_mapping = {0: "Classification", 1: "Segmentation", 2: "Both"}

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define model path
model_path = "/app/checkpoint-1400"

def download_from_google_drive():
    file_id = "1X8e9H6E5jNwFSXV5Av0bDPAqxYpC3zjI"  # 🔹 Replace with your actual file ID
    zip_path = f"{model_path}.zip"

    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_path, quiet=False)

        # Extract the ZIP file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("/app/")
        
        print("Model downloaded and extracted.")

# Choose the download method
if not os.path.exists(model_path):
    try:
        download_from_google_drive()
    except:
        print("Google Drive download failed, trying Hugging Face...")

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Serve the HTML page
@app.get("/", response_class=HTMLResponse)
def serve_html():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# API endpoint for classification
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

# To run: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
