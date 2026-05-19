"""
model1.py — DE/LU Load Prediction API
======================================
Upgraded to use the new models from energy_model_upgraded.ipynb:
  - LightGBM  (best performer, default)
  - XGBoost
  - Ridge Regression  (needs scaling)
  - Stacking Ensemble (LGB 60% + XGB 40%)

Artifacts required in Backend/Model_assets:
  lgb_load_model.pkl
  xgb_load_model.pkl
  ridge_load_model.pkl
  X_scaler.pkl
  feature_names.pkl
  train_medians.pkl
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Optional, Literal
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------
# Paths & Artifact Loading
# ---------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
MODEL_ASSETS_DIR = BACKEND_DIR / "Model_assets"
ARTIFACT_DIRS = [
    MODEL_ASSETS_DIR,
    BACKEND_DIR,
    PROJECT_ROOT / "DATA_preprocessing" / "Modelling",
    PROJECT_ROOT / "DATA_preprocessing" / "Modeling",
]


def _artifact_search_paths(name: str) -> list[Path]:
    return [directory / name for directory in ARTIFACT_DIRS]


def _find_artifact(name: str) -> Path:
    for path in _artifact_search_paths(name):
        if path.exists():
            return path
    checked = "\n".join(f"  - {path}" for path in _artifact_search_paths(name))
    raise FileNotFoundError(
        f"Artifact not found: {name}\n"
        f"Checked:\n{checked}\n"
        "Run energy_model_upgraded.ipynb to generate the missing .pkl files."
    )


def _load(name: str):
    return joblib.load(_find_artifact(name))


def _checked_artifact_dirs() -> list[str]:
    return [str(path) for path in ARTIFACT_DIRS]


print("🚀 Loading model artifacts…")
try:
    lgb_model = _load("lgb_load_model.pkl")
    xgb_model = _load("xgb_load_model.pkl")
    ridge_model = _load("ridge_load_model.pkl")
    scaler = _load("X_scaler.pkl")              # used ONLY for Ridge
    FEATURE_NAMES = _load("feature_names.pkl")  # ordered list of all feature names
    TRAIN_MEDIANS = _load("train_medians.pkl")  # pd.Series — safe fill for missing inputs
    ARTIFACTS_OK = True
    print(f"✅ Artifacts loaded  |  {len(FEATURE_NAMES)} features")
except Exception as e:
    print(f"❌ Artifact load failed: {e}")
    lgb_model = xgb_model = ridge_model = None
    scaler = FEATURE_NAMES = TRAIN_MEDIANS = None
    ARTIFACTS_OK = False
    ARTIFACT_ERROR = str(e)
else:
    ARTIFACT_ERROR = None

# ---------------------------------------------------
# App
# ---------------------------------------------------
app = FastAPI(
    title="Power Load Prediction API — DE/LU",
    description=(
        "Predicts Germany + Luxembourg electricity demand (MW) "
        "from real-time grid readings. "
        "Supports LightGBM, XGBoost, Ridge, and Ensemble models."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Input Schema
# ---------------------------------------------------
class PredictionInput(BaseModel):
    """
    Required fields: hour, day_of_week, month.
    All other features (regional load, solar, wind, lag values, etc.)
    are optional — missing ones are filled with training-set medians.

    Lag features (from the upgraded notebook):
      load_lag_1h, load_lag_2h, load_lag_3h, load_lag_6h,
      load_lag_12h, load_lag_24h, load_lag_48h, load_lag_168h,
      load_rolling_mean_24h, load_rolling_std_24h,
      load_rolling_mean_168h, load_rolling_std_168h

    If you supply lag features, predictions will be significantly more accurate.
    """
    hour:        int
    day_of_week: int
    month:       int

    # Which model to use (default = ensemble — best accuracy)
    model: Literal["ensemble", "lightgbm", "xgboost", "ridge"] = "ensemble"

    @field_validator("hour")
    @classmethod
    def check_hour(cls, v):
        if not 0 <= v <= 23:
            raise ValueError("hour must be 0–23")
        return v

    @field_validator("day_of_week")
    @classmethod
    def check_dow(cls, v):
        if not 0 <= v <= 6:
            raise ValueError("day_of_week must be 0 (Mon) – 6 (Sun)")
        return v

    @field_validator("month")
    @classmethod
    def check_month(cls, v):
        if not 1 <= v <= 12:
            raise ValueError("month must be 1–12")
        return v

    model_config = {
        "extra": "allow",   # accept all 50+ optional feature fields
        "json_schema_extra": {
            "example": {
                "hour": 10,
                "day_of_week": 1,
                "month": 1,
                "model": "ensemble",
                # Optional — supply as many as you have:
                "DE_load_forecast_entsoe_transparency": 69000,
                "DE_LU_load_forecast_entsoe_transparency": 68500,
                "DE_LU_price_day_ahead": 95,
                "DE_solar_generation_actual": 4200,
                "DE_wind_generation_actual": 18000,
                "load_lag_1h": 67800,
                "load_lag_24h": 66500,
                "load_lag_168h": 65900,
            }
        }
    }


# ---------------------------------------------------
# Core prediction logic
# ---------------------------------------------------
def _build_feature_row(input_dict: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build a (1, n_features) array for the tree models and a scaled copy for Ridge.

    Unknown keys in input_dict are ignored with a warning.
    Missing features are filled with training-set medians (not zeros).
    Returns (X_raw, X_scaled).
    """
    # Start from training medians — safe neutral baseline
    row = TRAIN_MEDIANS.copy()

    # Override with whatever the caller supplied (excluding the 'model' key)
    skipped = []
    for key, value in input_dict.items():
        if key == "model":
            continue
        if key in row.index:
            row[key] = float(value)
        else:
            skipped.append(key)

    if skipped:
        print(f"  ⚠️  Ignored unknown field(s): {skipped}")

    # Keep feature names attached so sklearn/lightgbm do not warn at predict time.
    X_raw = pd.DataFrame(
        [[float(row[name]) for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )
    X_scaled = scaler.transform(X_raw)
    return X_raw, X_scaled


def _require_artifacts():
    if not ARTIFACTS_OK:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Model artifacts are not available yet.",
                "error": ARTIFACT_ERROR,
                "checked_directories": _checked_artifact_dirs(),
            },
        )


def _predict(input_dict: dict) -> dict:
    model_choice = input_dict.get("model", "ensemble")
    X_raw, X_scaled = _build_feature_row(input_dict)

    lgb_pred  = float(lgb_model.predict(X_raw)[0])
    xgb_pred  = float(xgb_model.predict(X_raw)[0])

    if model_choice == "lightgbm":
        final = lgb_pred
    elif model_choice == "xgboost":
        final = xgb_pred
    elif model_choice == "ridge":
        final = float(ridge_model.predict(X_scaled)[0])
    else:  # ensemble (default)
        final = 0.6 * lgb_pred + 0.4 * xgb_pred

    # Count how many real values were supplied (vs median-filled)
    n_supplied = sum(
        1 for k in input_dict
        if k not in ("model",) and k in FEATURE_NAMES
    )
    lag_supplied = any(
        k.startswith("load_lag") or k.startswith("load_rolling")
        for k in input_dict
    )

    return {
        "predicted_load_mw":    round(final, 2),
        "model_used":           model_choice,
        "features_supplied":    n_supplied,
        "features_total":       len(FEATURE_NAMES),
        "lag_features_present": lag_supplied,
        "accuracy_note": (
            "High accuracy — lag features provided."
            if lag_supplied else
            "Moderate accuracy — supply load_lag_1h/24h/168h for best results."
        ),
        # Individual model outputs for transparency
        "model_breakdown": {
            "lightgbm": round(lgb_pred, 2),
            "xgboost":  round(xgb_pred, 2),
            "ensemble": round(0.6 * lgb_pred + 0.4 * xgb_pred, 2),
        }
    }


# ---------------------------------------------------
# Routes
# ---------------------------------------------------
@app.get("/")
def root():
    return JSONResponse({
        "message": "Power Load Prediction API is running.",
        "docs": "/docs",
        "predict": "POST /predict",
        "features": "/features",
        "health": "/health",
    })


@app.post("/predict")
async def predict(data: PredictionInput):
    """
    Predict DE+LU electricity load for a given hour.

    Supply as many optional features as you have — the more context,
    the better the prediction. Lag features have the biggest impact.
    """
    _require_artifacts()
    try:
        input_dict = data.model_dump()
        return await run_in_threadpool(_predict, input_dict)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/features")
def list_features():
    """Return all feature names the model was trained on, with their training medians."""
    _require_artifacts()
    return {
        "feature_count": len(FEATURE_NAMES),
        "features": {
            name: float(TRAIN_MEDIANS[name])
            for name in FEATURE_NAMES
        },
        "lag_features": [f for f in FEATURE_NAMES if "lag" in f or "rolling" in f],
        "note": "Missing features are auto-filled with training medians."
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if ARTIFACTS_OK else "degraded",
        "artifacts_loaded": ARTIFACTS_OK,
        "models_loaded": ["lightgbm", "xgboost", "ridge", "ensemble"] if ARTIFACTS_OK else [],
        "feature_count": len(FEATURE_NAMES) if ARTIFACTS_OK else 0,
        "default_model": "ensemble",
        "checked_directories": _checked_artifact_dirs(),
        "artifact_error": ARTIFACT_ERROR,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
