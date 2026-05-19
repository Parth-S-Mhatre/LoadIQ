"""
model2.py — GB Load Forecast API
==================================
Upgraded to align with energy_model_upgraded.ipynb approach:
  - Replaced LSTM (which required 24-step sequences + its own broken scalers)
    with LightGBM tabular prediction, matching the exact same pattern as model1.py
  - Target: GB_GBN_load_actual_entsoe_transparency (30-min data)
  - Kept: Firebase logging, heuristic fallback, batch/horizon endpoints,
          static file serving — all preserved and cleaned up
  - Fixed: engineer_features() now builds proper lag/rolling features
           instead of the broken hour-one-hot approach
  - Fixed: scaler_X was being refitted on X_train sequences in the old
           notebook — now uses a single consistent joblib-loaded scaler

Artifacts required in Backend/Model_assets:
  lgb_gb_model.pkl
  xgb_gb_model.pkl
  ridge_gb_model.pkl
  X_scaler_gb.pkl
  feature_names_gb.pkl
  train_medians_gb.pkl

NOTE: Run the Model 2 section of your notebook (30min data, GB target)
with the same upgraded pattern to generate these .pkl files.
If GB-specific artifacts are missing, this service can temporarily
fallback to non-GB artifact names in the same directories.
"""

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from typing import List, Optional, Literal, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import json
import os
import traceback
from threading import Lock
from datetime import datetime

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = Path(__file__).resolve().parent
MODEL_ASSETS_DIR = BASE_DIR / "Model_assets"

# ---------------------------------------------------
# App Setup
# ---------------------------------------------------
app = FastAPI(
    title="GB Electricity Load Forecast API",
    description=(
        "Predicts Great Britain electricity load (MW) using LightGBM. "
        "Supports single prediction, batch, and multi-step horizon forecasting."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🔥 BACKEND RUNNING FROM:", __file__)

# ---------------------------------------------------
# Optional Firebase logging (disabled by default for no-NoSQL setup)
# ---------------------------------------------------
ENABLE_FIREBASE_LOGGING = os.getenv("ENABLE_FIREBASE_LOGGING", "false").lower() in ("1", "true", "yes")
db = None

if ENABLE_FIREBASE_LOGGING:
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        def _resolve_service_account_path() -> Path:
            candidate = BASE_DIR / "serviceAccountKey.json"
            if candidate.is_file():
                return candidate
            if candidate.is_dir():
                json_files = sorted(candidate.glob("*.json"))
                if len(json_files) == 1:
                    return json_files[0]
            raise FileNotFoundError(
                f"Firebase service account JSON not found at {candidate}"
            )

        if not firebase_admin._apps:
            firebase_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
            if firebase_json:
                cred = credentials.Certificate(json.loads(firebase_json))
            else:
                cred = credentials.Certificate(_resolve_service_account_path())
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        print("✅ Firebase initialized")
    except Exception as e:
        print(f"⚠️ Firebase unavailable: {e} — logging disabled")
else:
    print("ℹ️ Firebase logging disabled (NoSQL disabled).")


def log_error(message: str, context: str = "general"):
    if db is None:
        return
    try:
        db.collection("backend_logs").document().set({
            "timestamp":     datetime.utcnow(),
            "error_message": message,
            "context":       context,
            "traceback":     traceback.format_exc(),
        })
    except Exception as e:
        print(f"❌ Firestore log failed: {e}")


# ---------------------------------------------------
# Paths & Artifact Loading
# ---------------------------------------------------
ARTIFACT_DIRS = [
    MODEL_ASSETS_DIR,
    BASE_DIR,
    PROJECT_ROOT / "DATA_preprocessing" / "Modelling",
    PROJECT_ROOT / "DATA_preprocessing" / "Modeling",
]


def _artifact_search_paths(name: str) -> List[Path]:
    return [directory / name for directory in ARTIFACT_DIRS]


def _find_artifact(name: str) -> Path:
    for path in _artifact_search_paths(name):
        if path.exists():
            return path
    checked = "\n".join(f"  - {path}" for path in _artifact_search_paths(name))
    raise FileNotFoundError(
        f"Artifact not found: {name}\n"
        f"Checked:\n{checked}\n"
        "Re-run the Model 2 section of energy_model_upgraded.ipynb "
        "with GB target to generate the missing .pkl files."
    )


def _load(name: str):
    return joblib.load(_find_artifact(name))


def _load_first_available(candidates: List[str]):
    last_error = None
    for artifact_name in candidates:
        try:
            return _load(artifact_name), artifact_name
        except FileNotFoundError as e:
            last_error = e
    if last_error is not None:
        raise last_error
    raise FileNotFoundError("No artifact candidates provided.")


def _checked_artifact_dirs() -> List[str]:
    return [str(path) for path in ARTIFACT_DIRS]


print("\n🚀 Loading GB model artifacts…")
print("ℹ️ On-demand mode enabled: artifacts load only when requested.")
ARTIFACT_LOCK = Lock()
MODEL_CACHE = {}
COMMON_CACHE = {}
LOADED_ARTIFACTS = {}
ARTIFACT_ERROR = None
USING_GB_ARTIFACTS = False
UNLOAD_INACTIVE_MODELS = os.getenv("UNLOAD_INACTIVE_MODELS", "true").lower() in ("1", "true", "yes")

MODEL_CANDIDATES = {
    "lightgbm": ["lgb_gb_model.pkl", "lgb_load_model.pkl"],
    "xgboost": ["xgb_gb_model.pkl", "xgb_load_model.pkl"],
    "ridge": ["ridge_gb_model.pkl", "ridge_load_model.pkl"],
}

COMMON_CANDIDATES = {
    "feature_names": ["feature_names_gb.pkl", "feature_names.pkl"],
    "train_medians": ["train_medians_gb.pkl", "train_medians.pkl"],
    "scaler": ["X_scaler_gb.pkl", "X_scaler.pkl"],
}


def _set_artifact_error(message: Optional[str]):
    global ARTIFACT_ERROR
    ARTIFACT_ERROR = message


def _refresh_artifact_mode():
    global USING_GB_ARTIFACTS
    loaded_names = list(LOADED_ARTIFACTS.values())
    USING_GB_ARTIFACTS = bool(loaded_names) and all(name.endswith("_gb.pkl") for name in loaded_names)


def _ensure_common(key: str):
    with ARTIFACT_LOCK:
        if key in COMMON_CACHE:
            return COMMON_CACHE[key]
        artifact, used_name = _load_first_available(COMMON_CANDIDATES[key])
        COMMON_CACHE[key] = artifact
        LOADED_ARTIFACTS[key] = used_name
        _refresh_artifact_mode()
        return artifact


def _ensure_model(model_key: str):
    with ARTIFACT_LOCK:
        if model_key in MODEL_CACHE:
            return MODEL_CACHE[model_key]
        artifact, used_name = _load_first_available(MODEL_CANDIDATES[model_key])
        MODEL_CACHE[model_key] = artifact
        LOADED_ARTIFACTS[model_key] = used_name
        _refresh_artifact_mode()
        return artifact


def _unload_inactive_models(active_models: set[str]):
    if not UNLOAD_INACTIVE_MODELS:
        return
    with ARTIFACT_LOCK:
        inactive = [k for k in MODEL_CACHE if k not in active_models]
        for model_key in inactive:
            MODEL_CACHE.pop(model_key, None)
            LOADED_ARTIFACTS.pop(model_key, None)
        _refresh_artifact_mode()


def _ensure_runtime(model_choice: str):
    """
    Demand-loading runtime bundle:
    - common artifacts are loaded once and reused
    - only required model(s) are kept active
    """
    try:
        feature_names = _ensure_common("feature_names")
        train_medians = _ensure_common("train_medians")
        scaler_obj = _ensure_common("scaler") if model_choice == "ridge" else None

        lgb_obj = xgb_obj = ridge_obj = None
        active = set()
        if model_choice in ("lightgbm", "ensemble"):
            lgb_obj = _ensure_model("lightgbm")
            active.add("lightgbm")
        if model_choice in ("xgboost", "ensemble"):
            xgb_obj = _ensure_model("xgboost")
            active.add("xgboost")
        if model_choice == "ridge":
            ridge_obj = _ensure_model("ridge")
            active.add("ridge")

        _unload_inactive_models(active)
        _set_artifact_error(None)
        return feature_names, train_medians, scaler_obj, lgb_obj, xgb_obj, ridge_obj
    except Exception as e:
        _set_artifact_error(str(e))
        raise


# ---------------------------------------------------
# Static files helper (unchanged from original)
# ---------------------------------------------------
def resolve_build_dir() -> Optional[Path]:
    env_dir = os.getenv("FRONTEND_BUILD_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.extend([
        BASE_DIR / "build",
        BASE_DIR.parent / "build",
        BASE_DIR.parent / "energy-analytics" / "build",
    ])
    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve()
    return None


# ---------------------------------------------------
# Input Schema
# ---------------------------------------------------
class LoadInput(BaseModel):
    """
    Supply hour, day_of_week, month at minimum.
    All 50+ grid features and lag features are optional — missing ones
    are filled with training-set medians.

    For multi-step horizon forecasting via /predict_batch, provide
    last_24_hours (list of 24 recent load values in MW) so the API
    can auto-populate lag features for each iterative step.
    """
    hour:        int
    day_of_week: int
    month:       int

    # Optional: list of last 24 actual load readings for auto-lag population
    last_24_hours: Optional[List[float]] = None

    # Model choice
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
            raise ValueError("day_of_week must be 0–6")
        return v

    @field_validator("month")
    @classmethod
    def check_month(cls, v):
        if not 1 <= v <= 12:
            raise ValueError("month must be 1–12")
        return v

    @field_validator("last_24_hours")
    @classmethod
    def check_history(cls, v):
        if v is None:
            return v
        if len(v) != 24:
            raise ValueError("last_24_hours must contain exactly 24 values")
        if any(x < 0 for x in v):
            raise ValueError("Load values cannot be negative")
        return [float(x) for x in v]

    model_config = {
        "extra": "allow",
        "json_schema_extra": {
            "example": {
                "hour": 14,
                "day_of_week": 2,
                "month": 6,
                "model": "ensemble",
                "last_24_hours": [28000] * 24,
                "load_lag_1h":   27800,
                "load_lag_24h":  27500,
                "load_lag_168h": 27000,
            }
        }
    }


class HorizonInput(BaseModel):
    """For multi-step forecasting: provide last 24 hours + how many steps ahead."""
    last_24_hours: List[float]
    horizon:       int = 1
    model:         Literal["ensemble", "lightgbm", "xgboost", "ridge"] = "ensemble"

    @field_validator("last_24_hours")
    @classmethod
    def check_history(cls, v):
        if len(v) != 24:
            raise ValueError("last_24_hours must contain exactly 24 values")
        return [float(x) for x in v]

    @field_validator("horizon")
    @classmethod
    def check_horizon(cls, v):
        if v < 1 or v > 168:
            raise ValueError("horizon must be 1–168 (up to 1 week ahead)")
        return v


class BatchInput(BaseModel):
    """
    Flexible batch input:
    - Iterative mode: provide last_24_hours + horizon
    - Batch mode: provide loads (or scenarios) as a list of 24-hour sequences
    """
    model: Literal["ensemble", "lightgbm", "xgboost", "ridge"] = "ensemble"
    last_24_hours: Optional[List[float]] = None
    horizon: int = 1
    loads: Optional[List[List[float]]] = None
    scenarios: Optional[List[List[float]]] = None

    @field_validator("last_24_hours")
    @classmethod
    def check_last24(cls, v):
        if v is None:
            return v
        if len(v) != 24:
            raise ValueError("last_24_hours must contain exactly 24 values")
        return [float(x) for x in v]

    @field_validator("horizon")
    @classmethod
    def check_horizon(cls, v):
        if v < 1 or v > 168:
            raise ValueError("horizon must be 1–168 (up to 1 week ahead)")
        return v

    @field_validator("loads", "scenarios")
    @classmethod
    def check_sequences(cls, v):
        if v is None:
            return v
        cleaned = []
        for seq in v:
            if len(seq) != 24:
                raise ValueError("Each load sequence must contain exactly 24 values")
            cleaned.append([float(x) for x in seq])
        return cleaned


# ---------------------------------------------------
# Feature Construction
# ---------------------------------------------------
def _build_feature_row(
    input_dict: dict,
    feature_names: List[str],
    train_medians,
    scaler_obj,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Build (X_raw, X_scaled) from a flat input dictionary.
    Missing features → filled with training medians.
    """
    row = train_medians.copy()

    skipped = []
    for key, value in input_dict.items():
        if key in ("model", "last_24_hours"):
            continue
        if key in row.index:
            row[key] = float(value)
        else:
            skipped.append(key)

    if skipped:
        print(f"  ⚠️  Ignored unknown keys: {skipped}")

    X_raw = pd.DataFrame(
        [[float(row[name]) for name in feature_names]],
        columns=feature_names,
    )
    X_scaled = scaler_obj.transform(X_raw) if scaler_obj is not None else None
    return X_raw, X_scaled


def _populate_lags_from_history(input_dict: dict, history: List[float], feature_names: List[str]) -> dict:
    """
    Auto-populate lag features from a 24-value history list.
    history[-1] = most recent load reading.
    Lag mapping: load_lag_1h = history[-1], load_lag_2h = history[-2], etc.
    """
    history = list(history)
    lag_map = {1: -1, 2: -2, 3: -3, 6: -6, 12: -12, 24: -24}
    for lag, idx in lag_map.items():
        key = f"load_lag_{lag}h"
        if key in feature_names and abs(idx) <= len(history):
            input_dict[key] = history[idx]

    # Rolling mean/std from history
    for key in feature_names:
        if key.startswith("load_rolling_mean") or key.startswith("load_rolling_std"):
            window = int(key.split("_")[-1].replace("h", ""))
            window = min(window, len(history))
            vals = history[-window:]
            if "mean" in key:
                input_dict[key] = float(np.mean(vals))
            else:
                input_dict[key] = float(np.std(vals)) if len(vals) > 1 else 0.0

    return input_dict


# ---------------------------------------------------
# Prediction Logic
# ---------------------------------------------------
def _run_prediction(input_dict: dict) -> dict:
    model_choice = input_dict.get("model", "ensemble")
    if model_choice not in ("ensemble", "lightgbm", "xgboost", "ridge"):
        raise RuntimeError(f"Unsupported model choice: {model_choice}")

    feature_names, train_medians, scaler_obj, lgb_model, xgb_model, ridge_model = _ensure_runtime(model_choice)

    # Auto-populate lags from last_24_hours if provided
    history = input_dict.get("last_24_hours")
    if history:
        input_dict = _populate_lags_from_history(dict(input_dict), history, feature_names)

    X_raw, X_scaled = _build_feature_row(input_dict, feature_names, train_medians, scaler_obj)

    lgb_pred = float(lgb_model.predict(X_raw)[0]) if lgb_model is not None else None
    xgb_pred = float(xgb_model.predict(X_raw)[0]) if xgb_model is not None else None

    if model_choice == "lightgbm":
        final = float(lgb_pred)
    elif model_choice == "xgboost":
        final = float(xgb_pred)
    elif model_choice == "ridge":
        final = float(ridge_model.predict(X_scaled)[0])
    else:
        final = 0.6 * float(lgb_pred) + 0.4 * float(xgb_pred)

    lag_present = any(
        k.startswith("load_lag") or k.startswith("load_rolling")
        for k in input_dict
        if k in feature_names
    )

    model_breakdown = {"ensemble": round(final, 2)}
    if lgb_pred is not None:
        model_breakdown["lightgbm"] = round(lgb_pred, 2)
    if xgb_pred is not None:
        model_breakdown["xgboost"] = round(xgb_pred, 2)
    if model_choice == "ridge":
        model_breakdown["ridge"] = round(final, 2)

    return {
        "predicted_load_mw": round(final, 2),
        "model_used":        model_choice,
        "lag_features_used": lag_present,
        "accuracy_note": (
            "High accuracy — lag features present."
            if lag_present else
            "Moderate accuracy — provide last_24_hours or lag features for best results."
        ),
        "model_breakdown": model_breakdown,
        "prediction_source": "ml_model",
        "fallback_used":     False,
    }


def _heuristic_fallback(history: List[float]) -> float:
    """
    Simple weighted heuristic used when ML model fails.
    Preserved from original model2.py.
    """
    h = np.asarray(history[-24:], dtype=np.float32)
    recent_mean  = float(h[-3:].mean())
    daily_anchor = float(h[0])
    momentum     = float(np.diff(h[-6:]).mean()) if len(h) >= 6 else 0.0
    volatility   = float(np.std(h[-6:]))         if len(h) >= 6 else 0.0

    predicted = (
        daily_anchor * 0.45
        + recent_mean * 0.45
        + float(h[-1]) * 0.10
        + momentum * 0.75
    )
    lo = max(0.0, recent_mean - max(volatility * 2.5, 1500.0))
    hi = recent_mean + max(volatility * 2.5, 1500.0)
    return float(np.clip(predicted, lo, hi))


def _predict_horizon(history: List[float], horizon: int, model_choice: str) -> List[dict]:
    """
    Iterative multi-step forecast.
    Each step: predict next load → append to history → repeat.
    """
    history = list(history)
    results = []

    for step in range(horizon):
        input_dict = {
            "hour":        (step % 24),
            "day_of_week": 0,               # simplified; caller can pass a schedule if needed
            "month":       1,
            "model":       model_choice,
            "last_24_hours": history[-24:],
        }
        try:
            result = _run_prediction(input_dict)
            pred   = result["predicted_load_mw"]
            result["step"] = step + 1
            results.append(result)
            history.append(pred)
        except Exception as e:
            fallback = _heuristic_fallback(history[-24:])
            log_error(str(e), context=f"horizon_step_{step+1}")
            results.append({
                "step":              step + 1,
                "predicted_load_mw": round(fallback, 2),
                "model_used":        "heuristic_fallback",
                "fallback_used":     True,
                "fallback_reason":   str(e),
            })
            history.append(fallback)

    return results


def _predict_batch_sequences(sequences: List[List[float]], model_choice: str) -> dict:
    """
    Predict one next-step value for each provided 24-hour sequence.
    This matches frontend analytics payloads that send `loads`.
    """
    predictions = []
    for idx, history in enumerate(sequences):
        input_dict = {
            "hour": 0,
            "day_of_week": 0,
            "month": 1,
            "model": model_choice,
            "last_24_hours": history,
        }
        try:
            result = _run_prediction(input_dict)
            predictions.append(result)
        except Exception as e:
            fallback = _heuristic_fallback(history)
            log_error(str(e), context=f"batch_sequence_{idx+1}")
            predictions.append({
                "predicted_load_mw": round(fallback, 2),
                "model_used": "heuristic_fallback",
                "fallback_used": True,
                "fallback_reason": str(e),
                "prediction_source": "heuristic_fallback",
            })

    sources = {p.get("model_used", "ensemble") for p in predictions}
    return {
        "predictions": predictions,
        "mode": "batch_sequences",
        "prediction_source": sources.pop() if len(sources) == 1 else "mixed",
        "fallback_used": any(p.get("fallback_used", False) for p in predictions),
    }


# ---------------------------------------------------
# API Routes
# ---------------------------------------------------
@app.post("/predict")
async def predict(data: LoadInput):
    """
    Predict GB electricity load for one time step.

    Tip: supply last_24_hours to auto-populate lag features
    and significantly improve accuracy.
    """
    try:
        return await run_in_threadpool(_run_prediction, data.model_dump())
    except HTTPException:
        raise
    except Exception as e:
        log_error(str(e), context="predict_endpoint")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(data: BatchInput):
    """
    Multi-step iterative forecast.
    Provide last_24_hours + horizon (1–168 steps ahead).
    Each predicted value feeds into the next step's lag features.
    """
    try:
        if data.last_24_hours is not None:
            results = await run_in_threadpool(
                _predict_horizon,
                data.last_24_hours,
                data.horizon,
                data.model,
            )
            sources = {r.get("model_used", "ensemble") for r in results}
            return {
                "horizon":           data.horizon,
                "predictions":       results,
                "mode":              "iterative_forecast",
                "prediction_source": sources.pop() if len(sources) == 1 else "mixed",
                "fallback_used":     any(r.get("fallback_used", False) for r in results),
            }

        sequences = data.loads or data.scenarios
        if sequences:
            return await run_in_threadpool(_predict_batch_sequences, sequences, data.model)

        raise HTTPException(
            status_code=422,
            detail="Provide either `last_24_hours` (+ optional `horizon`) or `loads`/`scenarios`.",
        )
    except HTTPException:
        raise
    except Exception as e:
        log_error(str(e), context="predict_batch_endpoint")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/features")
def list_features():
    """Return all features the model was trained on, with their training medians."""
    try:
        feature_names, train_medians, _, _, _, _ = _ensure_runtime("ensemble")
    except Exception:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Model artifacts are not available yet.",
                "error": ARTIFACT_ERROR,
                "checked_directories": _checked_artifact_dirs(),
            },
        )
    return {
        "feature_count": len(feature_names),
        "features": {
            name: float(train_medians[name])
            for name in feature_names
        },
        "lag_features": [f for f in feature_names if "lag" in f or "rolling" in f],
    }


@app.get("/health")
def health():
    feature_count = len(COMMON_CACHE["feature_names"]) if "feature_names" in COMMON_CACHE else 0
    return {
        "status":        "ok" if ARTIFACT_ERROR is None else "degraded",
        "models_loaded": sorted(list(MODEL_CACHE.keys())),
        "feature_count": feature_count,
        "default_model": "ensemble",
        "artifact_mode": "gb" if USING_GB_ARTIFACTS else "fallback_non_gb",
        "loaded_artifacts": LOADED_ARTIFACTS,
        "fallback_available": True,
        "on_demand_loading": True,
        "unload_inactive_models": UNLOAD_INACTIVE_MODELS,
        "checked_directories": _checked_artifact_dirs(),
        "artifact_error": ARTIFACT_ERROR,
    }


@app.get("/api/health_check")
def health_check():
    """Legacy health check endpoint — preserved for frontend compatibility."""
    return {
        "status":       "running",
        "model_loaded": len(MODEL_CACHE) > 0,
        "models":       ["lightgbm", "xgboost", "ridge", "ensemble"],
    }


# ---------------------------------------------------
# Static Files (React frontend — unchanged from original)
# ---------------------------------------------------
BUILD_DIR = resolve_build_dir()

if BUILD_DIR:
    print(f"✅ Frontend build: {BUILD_DIR}")
    app.mount("/static", StaticFiles(directory=BUILD_DIR / "static"), name="static")

    @app.get("/")
    async def serve_root():
        index = BUILD_DIR / "index.html"
        return FileResponse(index) if index.exists() else {"error": "Frontend build not found"}

    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        if any(full_path.startswith(p) for p in ("api/", "docs", "openapi.json")):
            raise HTTPException(404, "Not Found")
        file_path = BUILD_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        index = BUILD_DIR / "index.html"
        return FileResponse(index) if index.exists() else {"error": "Frontend build not found"}
else:
    print("⚠️ Frontend build not found — static serving disabled")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
