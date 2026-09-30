import os
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

# FIX: Ab database ki jagah direct local folder se model load hoga
MODEL_URI = f"{BASE_DIR}/house_model"
FEATURES = ["sqft", "bedrooms", "bathrooms", "age_years", "garage", "location_score"]

# Tracking URI set karne ki line hata di gayi hai
model = mlflow.sklearn.load_model(MODEL_URI)

app = FastAPI(title="House Price Predictor")

class HouseFeatures(BaseModel):
    sqft: float = Field(..., gt=0, le=20000)
    bedrooms: int = Field(..., gt=0, le=20)
    bathrooms: int = Field(..., gt=0, le=200)
    age_years: int = Field(..., gt=0, le=150)
    garage: int = Field(..., ge=0, le=10)
    location_score: int = Field(..., ge=1, le=10)
    
@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_URI}

@app.post("/predict")
def predict(features: HouseFeatures):
    input_df = pd.DataFrame([features.model_dump()], columns=FEATURES)
    prediction = model.predict(input_df)[0]
    return {"predicted_price": round(float(prediction), 2)}
    
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"))

@app.get("/")
def frontend():
    return FileResponse(BASE_DIR / "static" / "index.html")