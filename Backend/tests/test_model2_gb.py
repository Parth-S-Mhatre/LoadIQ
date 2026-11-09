"""Regression tests for the GB-specific Model2 artifacts.

These tests guard the failure mode that motivated ``train_gb_artifacts.py``:
the GB service silently serving DE/LU-scale predictions from the legacy
artifacts.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import Model2
import main

HISTORY = [
    28100.0,
    27950.0,
    27820.0,
    27740.0,
    27690.0,
    27780.0,
    28120.0,
    28600.0,
    29150.0,
    29780.0,
    30220.0,
    30510.0,
    30780.0,
    30920.0,
    30810.0,
    30480.0,
    30020.0,
    29610.0,
    29240.0,
    28950.0,
    28730.0,
    28560.0,
    28410.0,
    28290.0,
]


def test_gb_artifacts_are_present() -> None:
    assert Model2._gb_artifacts_available(), (
        "GB artifacts are missing. Run `python Backend/train_gb_artifacts.py`."
    )


def test_health_reports_gb_mode() -> None:
    client = TestClient(main.app)
    response = client.get("/model2/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["artifact_mode"] == "gb"
    assert payload["gb_artifacts_available"] is True
    assert payload["allow_non_gb_fallback"] is False
    assert payload["models_available"] == ["lightgbm", "xgboost", "ridge", "ensemble"]


def test_non_gb_artifact_is_refused() -> None:
    with pytest.raises(Model2.ArtifactUnavailable):
        Model2._load_first_available(["lgb_load_model.pkl"])


def test_runtime_loads_and_predicts_with_ml_model() -> None:
    """Artifacts must load on Render's older pandas, not just local pandas 3.

    Pickles written with a pandas-3 ``str`` index raise NotImplementedError
    under pandas 2.x, which silently pushed every request into the heuristic
    fallback. Keep the medians index a plain object dtype.
    """
    feature_names, medians, _, lgb_model, xgb_model, _ = Model2._ensure_runtime("ensemble")
    assert len(feature_names) == 25
    assert medians.index.dtype == object
    assert lgb_model is not None and xgb_model is not None

    result = Model2._run_prediction({"model": "ensemble", "last_24_hours": HISTORY})
    assert result["prediction_source"] == "ml_model"
    assert result["fallback_used"] is False


def test_horizon_forecast_returns_plausible_gb_values() -> None:
    client = TestClient(main.app)
    response = client.post(
        "/model2/predict_batch",
        json={
            "model": "ensemble",
            "hour": 14,
            "day_of_week": 2,
            "month": 6,
            "last_24_hours": HISTORY,
            "horizon": 6,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    predictions = [item["predicted_load_mw"] for item in payload["predictions"]]
    assert len(predictions) == 6
    assert all(20_000 <= value <= 60_000 for value in predictions)
    assert payload["artifact_mode"] == "gb"
    assert payload["fallback_used"] is False


def test_calendar_context_is_validated() -> None:
    client = TestClient(main.app)
    response = client.post(
        "/model2/predict_batch",
        json={"last_24_hours": HISTORY, "hour": 24},
    )
    assert response.status_code == 422


def test_horizon_forecast_blends_seasonal_naive_and_keeps_daily_shape() -> None:
    start_hour = 2  # early morning: load should rise through the day
    results = Model2._predict_horizon(
        HISTORY, 24, "ensemble", start_hour, 2, 6
    )

    assert [result["blend_weight"] for result in results[:6]] == [
        0.85,
        0.6,
        0.4,
        0.25,
        0.2,
        0.2,
    ]
    for result in results:
        assert "model_prediction_mw" in result
        assert "seasonal_naive_mw" in result
        weight = result["blend_weight"]
        expected = (
            weight * result["model_prediction_mw"]
            + (1.0 - weight) * result["seasonal_naive_mw"]
        )
        assert result["predicted_load_mw"] == pytest.approx(expected, abs=0.01)

    # The forecast follows the input daily curve instead of flattening.
    values = [result["predicted_load_mw"] for result in results]
    assert max(values[6:18]) > values[0]
    assert max(values[6:18]) > min(values[18:])


def test_horizon_forecast_extrapolates_beyond_one_day() -> None:
    results = Model2._predict_horizon(HISTORY, 48, "ensemble", 2, 2, 6)
    assert len(results) == 48
    assert all(result["step"] == index + 1 for index, result in enumerate(results))
    assert all(20_000 <= result["predicted_load_mw"] <= 60_000 for result in results)
