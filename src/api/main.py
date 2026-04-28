"""
FastAPI application — serves the dashboard and all data/model endpoints.

Endpoints:
  GET  /                      → dashboard.html
  GET  /health                → {status, test_mae, n_features, lstm_available}
  GET  /metrics               → full params.json
  GET  /plots/{name}          → PNG from metrics/
  POST /predict               → {predicted_delta: float}
  POST /predict/curve         → {predictions: [float, ...], n_laps: int}
  POST /predict/lstm          → {predictions: [float, ...], laps_seen: int}
  GET  /data/filters          → {circuits, drivers, years, compounds}
  GET  /data/degradation      → filtered rows from feature_table
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.inference import (
    get_metrics,
    load_lstm,
    load_model,
    lstm_available,
    predict,
    predict_lstm,
)

STATIC_DIR    = Path("src/static")
PLOTS_DIR     = Path("metrics")
FEATURES_PATH = Path("data/features/feature_table.parquet")

EXPLORER_COLS = [
    "stint_lap_number",
    "lap_time_delta_fuel_corrected",
    "Compound",
    "Driver",
    "Team",
    "EventName",
    "Year",
    "RoundNumber",
    "TrackTemp",
    "AirTemp",
    "TyreLife",
    "Stint",
]

_feature_df: pd.DataFrame | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _feature_df
    load_model()
    load_lstm()  # graceful — warns if lstm_model.pt not found
    _feature_df = pd.read_parquet(FEATURES_PATH, columns=EXPLORER_COLS)
    yield


app = FastAPI(title="F1 Tyre Degradation", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/health")
def health():
    m = get_metrics()
    result = {
        "status":         "ok",
        "test_mae":       m["test"]["mae"],
        "test_r2":        m["test"]["r2"],
        "n_features":     m["n_features"],
        "train_rows":     m["train_rows"],
        "lstm_available": lstm_available(),
    }
    if lstm_available() and "lstm" in m:
        result["lstm_test_mae"] = m["lstm"]["test"]["mae"]
    return result


@app.get("/metrics")
def metrics():
    return get_metrics()


@app.get("/plots/{name}")
def plot(name: str):
    allowed = {
        "shap_importance",
        "shap_waterfall_sample",
        "predicted_vs_actual",
        "residuals_by_compound",
    }
    if name not in allowed:
        raise HTTPException(status_code=404, detail="Plot not found")
    path = PLOTS_DIR / f"{name}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Plot file missing — run make train first")
    return FileResponse(path, media_type="image/png")


@app.post("/predict")
def predict_endpoint(body: dict):
    try:
        delta = predict(body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"predicted_delta": round(delta, 4)}


@app.post("/predict/curve")
def predict_curve_endpoint(body: dict):
    """
    Generate a full degradation curve by iteratively predicting laps 1..n_laps.
    Carries forward deg_rate_last_3 and deg_acceleration between steps so the
    LightGBM model receives realistic rolling features at each lap.

    Body: same fields as /predict, plus optional n_laps (default 30, max 60).
    Returns: {"predictions": [delta_lap1, ..., delta_lapN], "n_laps": N}
    """
    n_laps = min(int(body.get("n_laps", 30)), 60)
    base   = {k: v for k, v in body.items() if k != "n_laps"}

    predictions: list[float] = []
    deg_history: list[float] = []

    try:
        for lap in range(n_laps):
            row = dict(base)
            row["stint_lap_number"] = lap

            # Rolling features carried forward from previous predictions
            if deg_history:
                row["deg_rate_last_3"] = float(np.mean(deg_history[-3:]))
                prev_rate = float(np.mean(deg_history[-4:-1])) if len(deg_history) >= 2 else 0.0
                row["deg_acceleration"] = row["deg_rate_last_3"] - prev_rate
            else:
                row["deg_rate_last_3"]  = 0.0
                row["deg_acceleration"] = 0.0

            delta = predict(row)
            predictions.append(round(delta, 4))
            deg_history.append(delta)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {"predictions": predictions, "n_laps": n_laps}


@app.post("/predict/lstm")
def predict_lstm_endpoint(body: dict):
    """
    Predict a degradation curve using the LSTM sequence model.
    Body: {"laps": [{feature dict}, ...]}  — ordered list of past laps in current stint.
    Returns: {"predictions": [delta, ...], "laps_seen": N}
    """
    if not lstm_available():
        raise HTTPException(
            status_code=503,
            detail="LSTM model not trained yet — run: make train-lstm",
        )
    laps = body.get("laps", [])
    if not laps or not isinstance(laps, list):
        raise HTTPException(status_code=422, detail="'laps' must be a non-empty list")
    try:
        preds = predict_lstm(laps)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"predictions": [round(p, 4) for p in preds], "laps_seen": len(laps)}


@app.get("/data/filters")
def filters():
    df = _feature_df
    return {
        "years":     sorted(df["Year"].dropna().unique().tolist()),
        "circuits":  sorted(df["EventName"].dropna().unique().tolist()),
        "drivers":   sorted(df["Driver"].dropna().unique().tolist()),
        "compounds": sorted(df["Compound"].dropna().unique().tolist()),
    }


@app.get("/data/degradation")
def degradation(
    year:     Optional[int] = Query(None),
    circuit:  Optional[str] = Query(None),
    driver:   Optional[str] = Query(None),
    compound: Optional[str] = Query(None),
    limit:    int           = Query(2000, le=5000),
):
    df = _feature_df
    if year:
        df = df[df["Year"] == year]
    if circuit:
        df = df[df["EventName"] == circuit]
    if driver:
        df = df[df["Driver"] == driver]
    if compound:
        df = df[df["Compound"] == compound]

    df = df.dropna(subset=["stint_lap_number", "lap_time_delta_fuel_corrected"])
    df = df.sort_values(["Driver", "Stint", "stint_lap_number"]).head(limit)
    return df.to_dict(orient="records")
