from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib

app = FastAPI(title="TruthLensAI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load text ML model
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
deepfake_model = load_model("models/deepfake_model.h5")


class NewsInput(BaseModel):
    text: str

# Dummy image logic (learning purpose)
def simple_deepfake_detector(filename: str) -> str:
    if "fake" in filename.lower():
        return "DEEPFAKE ❌"
    return "REAL ✅"

@app.get("/")
def home():
    return {"message": "TruthLensAI backend is running 🚀"}

@app.post("/predict-text")
def predict_text(data: NewsInput):
    vec = vectorizer.transform([data.text])
    pred = model.predict(vec)[0]
    return {
        "prediction": "FAKE ❌" if pred == 1 else "REAL ✅"
    }

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()

    # Convert bytes → image
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = deepfake_model.predict(img)[0][0]

    return {
        "prediction": "DEEPFAKE ❌" if pred > 0.5 else "REAL ✅",
        "confidence": float(pred)
    }



