# =========================================================
# Electricity Load Forecast Backend (REWRITTEN FINAL VERSION)
# =========================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Optional
from pathlib import Path
import numpy as np
import tensorflow as tf
import joblib
import json
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import traceback
from datetime import datetime

# =========================================================
# APP SETUP
# =========================================================
app = FastAPI(title="Electricity Load Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# FIREBASE & LOGGING SETUP
# =========================================================
print("🔥 BACKEND RUNNING FROM:", __file__)

# Initialize Firebase
try:
    if not firebase_admin._apps:
        # Check for environment variable first (for Render/Cloud)
        firebase_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if firebase_json:
            # Parse the JSON string from env var
            cred_dict = json.loads(firebase_json)
            cred = credentials.Certificate(cred_dict)
        else:
            # Fallback to local file
            cred = credentials.Certificate("serviceAccountKey.json")
            
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase initialized successfully")
except Exception as e:
    print(f"⚠️ Firebase initialization failed: {e}")
    print("⚠️ Logging to Firestore will be disabled.")
    db = None

def log_error_to_db(error_message: str, context: str = "general"):
    """
    Logs error details to Firestore 'backend_logs' collection.
    """
    if db is None:
        return
    
    try:
        doc_ref = db.collection("backend_logs").document()
        doc_ref.set({
            "timestamp": datetime.utcnow(),
            "error_message": error_message,
            "context": context,
            "traceback": traceback.format_exc()
        })
        print(f"📝 Error logged to Firestore: {doc_ref.id}")
    except Exception as log_err:
        print(f"❌ Failed to log error to Firestore: {log_err}")

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "lstm_load_model.keras"
SCALER_X_PATH = BASE_DIR / "scaler_X.save"
SCALER_Y_PATH = BASE_DIR / "scaler_y.save"

# =========================================================
# LOAD MODEL & SCALERS
# =========================================================
print("\n🚀 Loading model & scalers...")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    print("✅ Model loaded")
    print("✅ scaler_X features:", scaler_X.n_features_in_)
    print("✅ scaler_X scale shape:", scaler_X.scale_.shape)
    print("✅ scaler_y scale shape:", scaler_y.scale_.shape)

except Exception as e:
    print("❌ Failed to load model/scalers:", e)
    model = None
    scaler_X = None
    scaler_y = None

# =========================================================
# REQUEST SCHEMA
# =========================================================
class LoadInput(BaseModel):
    last_24_hours: List[float]

    @field_validator("last_24_hours", mode="before")
    @classmethod
    def validate_loads(cls, v):
        if isinstance(v, str):
            v = json.loads(v)
        if not isinstance(v, list):
            raise ValueError("last_24_hours must be a list")
        if len(v) != 24:
            raise ValueError("Exactly 24 values are required")
        values = []
        for i, x in enumerate(v):
            try:
                val = float(x)
                if val < 0:
                    raise ValueError
                values.append(val)
            except Exception:
                raise ValueError(f"Invalid value at index {i}: {x}")
        return values

    class Config:
        json_schema_extra = {
            "example": {
                "last_24_hours": [2500] * 24
            }
        }

# =========================================================
# FEATURE ENGINEERING (24 → 45)
# =========================================================
def engineer_features(loads: List[float]) -> np.ndarray:
    loads = np.asarray(loads, dtype=np.float32)
    if loads.shape != (24,):
        raise ValueError("Expected exactly 24 load values")
    
    # Vectorized implementation
    # Create batch of 1 for simpler vectorization logic reuse or just 2D array
    # Shape: (24, 45)
    X = np.zeros((24, 45), dtype=np.float32)

    # 0 -> original load
    X[:, 0] = loads

    # 1–23 -> hour one-hot using advanced indexing
    # Hours 0-23 map to indices 1-23 (with 23 mapping to 23? Original logic: h+1 if h<23 else 23)
    # Hour 22 -> Index 23. Hour 23 -> Index 23.
    hour_indices = np.arange(24)
    feature_indices = np.where(hour_indices < 23, hour_indices + 1, 23)
    # Applying to diagonal: row h gets 1.0 at col feature_indices[h]
    X[np.arange(24), feature_indices] = 1.0

    # 24 -> day of week (Monday example)
    X[:, 24] = 1.0

    # 31 -> month (January example)
    X[:, 31] = 1.0

    # 43 -> delta
    # Vectorized diff
    X[1:, 43] = loads[1:] - loads[:-1]
    X[0, 43] = 0.0

    # 44 -> rolling mean (3)
    # Vectorized rolling mean is harder without pandas/scipy, but loop for 24 items is negligible.
    # We can keep the loop for rolling mean or use convolution.
    # Given N=24, loop is totally fine and readable.
    for i in range(24):
        X[i, 44] = loads[max(0, i - 2):i + 1].mean()

    return X

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_next_load(loads: List[float]) -> dict:
    if model is None:
        raise RuntimeError("Model not loaded")

    features = engineer_features(loads)  # (24, 45)

    if features.shape[1] != scaler_X.n_features_in_:
        raise RuntimeError(
            f"Feature mismatch: {features.shape[1]} vs {scaler_X.n_features_in_}"
        )

    X_scaled = scaler_X.transform(features)   # (24, 45)
    X_lstm = X_scaled.reshape(1, 24, 45)
    y_scaled = model.predict(X_lstm, verbose=0)
    y_scaled = np.asarray(y_scaled).reshape(-1, 1)
    y_original = scaler_y.inverse_transform(y_scaled)

    result_scaled = float(y_scaled[0, 0])
    result_original = float(y_original[0, 0])

    return {
        "predicted_load": result_original,           # for backward compatibility
        "predicted_load_scaled": result_scaled,      # optional
        "predicted_load_original": result_original   # optional
    }


# =========================================================
# API ROUTES
# =========================================================

# Serve Static Files (Make sure 'build' folder exists in production)
# We mount it AFTER API routes to ensure specific API paths take precedence, 
# BUT FastAPI matching order matters. Mounting "/" usually catches everything.
# So we add a specific route for "/" to serve index.html and then mount generic static files.

@app.post("/predict")
def predict(data: LoadInput):
    try:
        prediction = predict_next_load(data.last_24_hours)
        return prediction
    except Exception as e:
        log_error_to_db(str(e), context="predict_endpoint")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
def predict_batch(payload: dict):
    loads_list = payload.get("loads")
    if not isinstance(loads_list, list):
        raise HTTPException(400, "'loads' must be a list of 24-value sequences")

    results = []
    for i, seq in enumerate(loads_list):
        try:
            results.append(predict_next_load(seq))
        except Exception as e:
            log_error_to_db(str(e), context=f"predict_batch_item_{i}")
            raise HTTPException(400, f"Item {i}: {e}")

    return {"predictions": results}

@app.get("/api/health_check")
def health_check():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "expected_input_shape": "(1, 24, 45)"
    }

@app.get("/health")
def health():
    return {
        "model_loaded": model is not None,
        "scaler_X_loaded": scaler_X is not None,
        "scaler_Y_loaded": scaler_y is not None,
        "scaler_X_features": scaler_X.n_features_in_
    }

@app.post("/debug_predict")
def debug_predict(data: LoadInput):
    try:
        features = engineer_features(data.last_24_hours)
        X_scaled = scaler_X.transform(features)
        X_lstm = X_scaled.reshape(1, 24, 45)
        y_scaled = model.predict(X_lstm, verbose=0)
        y_scaled = np.asarray(y_scaled).reshape(-1, 1)
        y_original = scaler_y.inverse_transform(y_scaled)

        return {
            "features_shape": features.shape,
            "X_scaled_sample": X_scaled[0].tolist(),
            "y_scaled": float(y_scaled[0, 0]),
            "predicted_load_original": float(y_original[0, 0])
        }

    except Exception as e:
        log_error_to_db(str(e), context="debug_predict_endpoint")
        raise HTTPException(400, detail=str(e))

# =========================================================
# STATIC FILES SERVING (Catch-all for React)
# =========================================================

# 1. Mount the 'static' folder inside build (css/js)
# Check if build directory exists (for local dev resilience)
BUILD_DIR = BASE_DIR / "build"

if BUILD_DIR.exists():
    app.mount("/static", StaticFiles(directory=BUILD_DIR / "static"), name="static")
    
    # 2. Serve index.html for root and any other non-API routes (Client-side routing)
    @app.get("/")
    async def serve_root():
        index_path = BUILD_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"error": "Frontend build not found."}

    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        # Allow API routes to pass through if not caught above (though they should be)
        if full_path.startswith("api/") or full_path.startswith("docs") or full_path.startswith("openapi.json"):
             raise HTTPException(status_code=404, detail="Not Found")
        
        # Check if the requested file exists in the build root (e.g. manifest.json, favicon.ico)
        file_path = BUILD_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)

        # Otherwise, serve index.html (client-side routing)
        index_path = BUILD_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
            
        return {"error": "Frontend build not found. Please run 'npm run build' in energy-analytics."}

else:
    print("⚠️ 'build' directory not found. Static file serving is disabled.")
