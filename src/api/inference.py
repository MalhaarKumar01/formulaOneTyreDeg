"""
Model inference — loads model.lgb once and exposes predict() + get_metrics().
"""

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

MODEL_PATH  = Path("models/degradation/model.lgb")
PARAMS_PATH = Path("models/degradation/params.json")

# Alphabetical LabelEncoder order used during training
COMPOUND_MAP = {
    "HARD":         0,
    "INTERMEDIATE": 1,
    "MEDIUM":       2,
    "SOFT":         3,
    "WET":          4,
    "UNKNOWN":      2,
}

_model: lgb.Booster | None = None
_feature_names: list[str] = []


def load_model() -> None:
    global _model, _feature_names
    _model = lgb.Booster(model_file=str(MODEL_PATH))
    _feature_names = _model.feature_name()


def predict(inputs: dict) -> float:
    """
    Return predicted lap_time_delta_fuel_corrected (seconds).
    Accepts 'Compound' as a string; all other keys should match feature names.
    Missing features default to 0.0.
    """
    if _model is None:
        load_model()

    row = dict(inputs)
    if "compound_encoded" not in row and "Compound" in row:
        row["compound_encoded"] = COMPOUND_MAP.get(
            str(row.pop("Compound")).upper(), 2
        )
    elif "Compound" in row:
        row.pop("Compound")

    df = pd.DataFrame([{f: row.get(f, 0.0) for f in _feature_names}])
    return float(_model.predict(df)[0])


def get_metrics() -> dict:
    return json.loads(PARAMS_PATH.read_text())
