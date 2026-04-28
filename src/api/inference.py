"""
Model inference — loads model.lgb + lstm_model.pt once and exposes predict helpers.
"""

import json
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODEL_PATH   = Path("models/degradation/model.lgb")
PARAMS_PATH  = Path("models/degradation/params.json")
LSTM_PATH    = Path("models/degradation/lstm_model.pt")
SCALER_PATH  = Path("models/degradation/lstm_scaler.pkl")

# Alphabetical LabelEncoder order used during LightGBM training
COMPOUND_MAP = {
    "HARD":         0,
    "INTERMEDIATE": 1,
    "MEDIUM":       2,
    "SOFT":         3,
    "WET":          4,
    "UNKNOWN":      2,
}

# ─── LightGBM ────────────────────────────────────────────────────────────────

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
        row["compound_encoded"] = COMPOUND_MAP.get(str(row.pop("Compound")).upper(), 2)
    elif "Compound" in row:
        row.pop("Compound")

    df = pd.DataFrame([{f: row.get(f, 0.0) for f in _feature_names}])
    return float(_model.predict(df)[0])


def get_metrics() -> dict:
    return json.loads(PARAMS_PATH.read_text())


# ─── LSTM ─────────────────────────────────────────────────────────────────────

class LSTMPredictor:
    """Loads the LSTM model + StandardScaler and predicts per-lap deltas for a stint."""

    def __init__(self):
        self._model = None
        self._scaler = None
        self._feature_names: list[str] = []
        self.available = False

    def load(self) -> bool:
        if not LSTM_PATH.exists() or not SCALER_PATH.exists():
            return False
        try:
            import torch
            from src.models.lstm_model import LSTMDegradationModel

            checkpoint = torch.load(str(LSTM_PATH), map_location="cpu", weights_only=False)
            cfg = checkpoint["model_config"]
            self._feature_names = checkpoint["feature_names"]

            self._model = LSTMDegradationModel(
                input_size=cfg["input_size"],
                hidden_size=cfg["hidden_size"],
                num_layers=cfg["num_layers"],
                dropout=0.0,  # no dropout at inference
            )
            self._model.load_state_dict(checkpoint["model_state"])
            self._model.eval()

            with open(SCALER_PATH, "rb") as f:
                self._scaler = pickle.load(f)

            self.available = True
            log.info("LSTM model loaded successfully")
            return True
        except Exception as exc:
            log.warning(f"LSTM model load failed: {exc}")
            return False

    def predict_sequence(self, laps: list[dict]) -> list[float]:
        """
        laps: list of feature dicts (one per lap in current stint, chronological).
        Returns: per-lap predicted delta (seconds), same length as laps.
        """
        import torch

        X = np.array(
            [[lap.get(f, 0.0) for f in self._feature_names] for lap in laps],
            dtype=np.float32,
        )
        X = self._scaler.transform(X)
        X_t   = torch.tensor(X).unsqueeze(0)                     # (1, L, F)
        lens  = torch.tensor([len(laps)], dtype=torch.long)
        with torch.no_grad():
            preds = self._model(X_t, lens)                        # (1, L)
        return preds[0].numpy().tolist()


_lstm: LSTMPredictor | None = None


def load_lstm() -> None:
    global _lstm
    _lstm = LSTMPredictor()
    if not _lstm.load():
        log.info("LSTM model not found — /predict/lstm will return 503 until trained")


def predict_lstm(laps: list[dict]) -> list[float]:
    if _lstm is None or not _lstm.available:
        raise RuntimeError("LSTM model not loaded")
    return _lstm.predict_sequence(laps)


def lstm_available() -> bool:
    return _lstm is not None and _lstm.available
