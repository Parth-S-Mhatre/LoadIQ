from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------
# Paths & Artifact Loading
# ---------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "DATA_preprocessing" / "Modelling"

model = joblib.load(MODEL_DIR / "xgb_load_model.pkl")
X_scaler = joblib.load(MODEL_DIR / "X_scaler.pkl")
trained_features = joblib.load(MODEL_DIR / "trained_features.pkl")
# Pre-compute feature mapping for speed
feature_map = {name: i for i, name in enumerate(trained_features)}
# Pre-compute feature mapping for speed
feature_map = {name: i for i, name in enumerate(trained_features)}

app = FastAPI(title="Power Load Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Input schema - Allowing extra fields for 43+ features
# ---------------------------------------------------
class PredictionInput(BaseModel):
    hour: int
    day_of_week: int
    month: int
    
    class Config:
        extra = "allow" # This allows the 43 regional features to be passed in

@app.post("/predict")
def predict(data: PredictionInput):
    # Initialize a dataframe with zeros for all trained features
    # Create input array directly (faster than DataFrame)
    # trained_features is expected to be a list or index of feature names
    
    # Create valid feature map if not exists (inefficient to do every request, but better than DF)
    # Ideally this should be computed once at startup.
    # We will assume feature_map is computed globally after loading.
    
    X_arr = np.zeros((1, len(trained_features)), dtype=np.float32)
    input_dict = data.dict()
    
    # Use the pre-computed map for O(1) lookups
    for key, value in input_dict.items():
        if key in feature_map:
            idx = feature_map[key]
            X_arr[0, idx] = value

    # Scaler typically expects array-like
    X_scaled = X_scaler.transform(X_arr)
    prediction = model.predict(X_scaled)[0]

    return {
        "predicted_load": float(prediction)
    }
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8001)
