"""
FastAPI server for Dogs-vs-Cats classifier (Keras 3.x + TensorFlow 2.18).
Supports:
- GET /              (health check)
- POST /predict-file   (multipart image upload)
- POST /predict-base64 (base64 JSON image)

Configurable via environment variables:
- MODEL_PATH   (default: "my_model.keras")
- CLASS_NAMES  (default: "cat,dog")
"""

import os
import io
import base64
from typing import Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf
import keras

# Config
MODEL_PATH = os.environ.get("MODEL_PATH", "my_model.keras")
CLASS_NAMES = [n.strip() for n in os.environ.get("CLASS_NAMES", "cat,dog").split(",")]
SIGMOID_THRESHOLD = float(os.environ.get("SIGMOID_THRESHOLD", "0.5"))

# Load model (Keras 3.x format preferred)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

print(f"Loading model from {MODEL_PATH} ...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")
print("Input shape:", model.input_shape, "Output shape:", model.output_shape)

# Infer target image size from input shape
def _get_target_size() -> Tuple[int, int]:
    shape = model.input_shape
    if len(shape) == 4:
        _, h, w, c = shape
        if h and w:
            return int(h), int(w)
    return 224, 224  # fallback
TARGET_SIZE = _get_target_size()
print("Target image size:", TARGET_SIZE)

# Preprocessing
def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot open image: {e}")
    img = img.resize(TARGET_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch
    return arr

# Postprocessing
def decode_prediction(pred: np.ndarray):
    pred = np.array(pred)
    if pred.ndim == 2 and pred.shape[1] == 1:
        prob = float(pred[0,0])
        idx = 1 if prob >= SIGMOID_THRESHOLD else 0
        label = CLASS_NAMES[idx]
        confidence = prob if idx == 1 else 1.0 - prob
        return label, float(confidence)
    if pred.ndim == 2 and pred.shape[1] >= 2:
        vec = pred[0]
        idx = int(np.argmax(vec))
        confidence = float(vec[idx])
        label = CLASS_NAMES[idx]
        return label, confidence
    return "unknown", 0.0

# FastAPI app
app = FastAPI(title="Dogs vs Cats API (Keras 3.x)", version="1.0")

# Allow all origins (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Base64Image(BaseModel):
    image: str

@app.get("/")
def root():
    return {"status": "ok", "message": "Keras Dogs-vs-Cats API running!"}

@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        x = preprocess_image_bytes(contents)
        preds = model.predict(x)
        label, conf = decode_prediction(preds)
        return {"label": label, "confidence": conf}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

@app.post("/predict-base64")
def predict_base64(payload: Base64Image):
    raw = payload.image
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(raw)
        x = preprocess_image_bytes(image_bytes)
        preds = model.predict(x)
        label, conf = decode_prediction(preds)
        return {"label": label, "confidence": conf}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
