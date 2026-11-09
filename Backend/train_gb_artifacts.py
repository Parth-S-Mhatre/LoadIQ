#!/usr/bin/env python3
"""Train the GB-specific Model2 artifacts.

The live Model2 service is supposed to forecast GB GBN demand, but the
repository only shipped the DE/LU artifacts.  Model2 silently fell back to
those files, which made the GB endpoint return DE-scale predictions.  This
script builds the six GB artifacts expected by ``Model2.py``:

    lgb_gb_model.pkl
    xgb_gb_model.pkl
    ridge_gb_model.pkl
    X_scaler_gb.pkl
    feature_names_gb.pkl
    train_medians_gb.pkl

Run from the repository root (or pass --data):

    python Backend/train_gb_artifacts.py

The input file is the 30-minute OPSD extract in ``DATA/``.  It is resampled to
hourly values so the API's ``last_24_hours`` contract (one value per hour) is
consistent with the feature engineering.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

TARGET = "GB_GBN_load_actual_entsoe_transparency"

RAW_FEATURES = [
    "GB_GBN_load_forecast_entsoe_transparency",
    "GB_GBN_solar_capacity",
    "GB_GBN_solar_generation_actual",
    "GB_GBN_solar_profile",
    "GB_GBN_wind_capacity",
    "GB_GBN_wind_generation_actual",
    "GB_GBN_wind_profile",
    "GB_GBN_wind_offshore_capacity",
    "GB_GBN_wind_offshore_generation_actual",
    "GB_GBN_wind_offshore_profile",
    "GB_GBN_wind_onshore_capacity",
    "GB_GBN_wind_onshore_generation_actual",
    "GB_GBN_wind_onshore_profile",
]

LAGS = [1, 2, 3, 6, 12, 24]

FEATURE_NAMES = RAW_FEATURES + [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
] + [f"load_lag_{lag}h" for lag in LAGS] + [
    "load_rolling_mean_24h",
    "load_rolling_std_24h",
]


def build_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Build the exact feature names Model2 can populate at inference time."""
    features = pd.DataFrame(index=hourly.index)

    for column in RAW_FEATURES:
        if column not in hourly.columns:
            features[column] = np.nan
        else:
            features[column] = pd.to_numeric(hourly[column], errors="coerce")

    features["hour"] = hourly.index.hour
    features["day_of_week"] = hourly.index.dayofweek
    features["month"] = hourly.index.month
    features["is_weekend"] = (hourly.index.dayofweek >= 5).astype(int)

    target = pd.to_numeric(hourly[TARGET], errors="coerce")
    for lag in LAGS:
        features[f"load_lag_{lag}h"] = target.shift(lag)

    shifted = target.shift(1)
    features["load_rolling_mean_24h"] = shifted.rolling(24, min_periods=24).mean()
    # Match Model2._populate_lags_from_history(), which uses np.std(ddof=0).
    features["load_rolling_std_24h"] = shifted.rolling(24, min_periods=24).std(ddof=0)

    return features


def load_hourly(path: Path) -> pd.DataFrame:
    usecols = lambda column: column == "utc_timestamp" or column.startswith("GB_GBN_")
    frame = pd.read_csv(path, usecols=usecols, parse_dates=["utc_timestamp"])
    frame = frame.set_index("utc_timestamp").sort_index()
    hourly = frame.resample("1h").mean(numeric_only=True)
    hourly = hourly[hourly[TARGET].between(10_000, 60_000)]
    return hourly


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "mae_mw": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse_mw": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "mape_pct": round(float(mean_absolute_percentage_error(y_true, y_pred) * 100), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "DATA" / "energy_30min_processed.csv",
        help="Path to the 30-minute processed CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "Model_assets",
        help="Directory for the generated artifacts.",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Data file not found: {args.data}")

    print(f"Loading {args.data} …")
    hourly = load_hourly(args.data)
    print(f"Hourly rows: {len(hourly):,}  range: {hourly.index.min()} → {hourly.index.max()}")
    print(f"Target range: {hourly[TARGET].min():.0f} – {hourly[TARGET].max():.0f} MW")

    features = build_features(hourly)
    dataset = features.join(hourly[[TARGET]]).dropna(
        subset=["load_lag_24h", "load_rolling_mean_24h", TARGET]
    )
    if len(dataset) < 2_000:
        raise SystemExit(f"Not enough training rows after feature engineering: {len(dataset)}")

    split = int(len(dataset) * 0.8)
    train = dataset.iloc[:split]
    test = dataset.iloc[split:]

    medians = train[FEATURE_NAMES].median()
    X_train = train[FEATURE_NAMES].fillna(medians)
    X_test = test[FEATURE_NAMES].fillna(medians)
    y_train = train[TARGET]
    y_test = test[TARGET]

    print(f"Train: {len(train):,} rows ({train.index.min()} → {train.index.max()})")
    print(f"Test : {len(test):,} rows ({test.index.min()} → {test.index.max()})")

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training LightGBM …")
    lgb_model = LGBMRegressor(
        n_estimators=900,
        max_depth=7,
        learning_rate=0.04,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )

    print("Training XGBoost …")
    xgb_model = XGBRegressor(
        n_estimators=900,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        objective="reg:squarederror",
        eval_metric="mae",
        early_stopping_rounds=60,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    print("Training Ridge …")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)

    lgb_pred = lgb_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    ridge_pred = ridge_model.predict(X_test_scaled)
    ensemble_pred = 0.6 * lgb_pred + 0.4 * xgb_pred

    report = {
        "target": TARGET,
        "feature_count": len(FEATURE_NAMES),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_start": str(train.index.min()),
        "train_end": str(train.index.max()),
        "test_start": str(test.index.min()),
        "test_end": str(test.index.max()),
        "lightgbm": metrics(y_test, lgb_pred),
        "xgboost": metrics(y_test, xgb_pred),
        "ridge": metrics(y_test, ridge_pred),
        "ensemble": metrics(y_test, ensemble_pred),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    print("\nHoldout metrics (MW):")
    for name in ("lightgbm", "xgboost", "ridge", "ensemble"):
        print(f"  {name:10s} {report[name]}")

    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(lgb_model, args.output / "lgb_gb_model.pkl")
    joblib.dump(xgb_model, args.output / "xgb_gb_model.pkl")
    joblib.dump(ridge_model, args.output / "ridge_gb_model.pkl")
    joblib.dump(scaler, args.output / "X_scaler_gb.pkl")
    joblib.dump(list(FEATURE_NAMES), args.output / "feature_names_gb.pkl")
    # Force a plain object index: pandas 3 defaults string indexes to the new
    # StringDtype, which raises NotImplementedError when unpickled by pandas 2
    # (the version Render installs with numpy==1.26.4).
    medians = medians.copy()
    medians.index = medians.index.astype(object)
    joblib.dump(medians, args.output / "train_medians_gb.pkl")
    (args.output / "model_metadata_gb.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nSaved GB artifacts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
