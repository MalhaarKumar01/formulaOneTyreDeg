"""
FastAPI application — serves the dashboard and all data/model endpoints.

Endpoints:
  GET  /                      → dashboard.html
  GET  /health                → {status, test_mae, n_features}
  GET  /metrics               → full params.json
  GET  /plots/{name}          → PNG from metrics/
  POST /predict               → {predicted_delta: float}
  GET  /data/filters          → {circuits, drivers, years, compounds}
  GET  /data/degradation      → filtered rows from feature_table
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.inference import get_metrics, load_model, predict

STATIC_DIR   = Path("src/static")
PLOTS_DIR    = Path("metrics")
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
    return {
        "status": "ok",
        "test_mae": m["test"]["mae"],
        "test_r2":  m["test"]["r2"],
        "n_features": m["n_features"],
        "train_rows": m["train_rows"],
    }


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
