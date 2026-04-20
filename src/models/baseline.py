"""
Linear baseline model: HuberRegressor on [stint_lap_number, compound, track_temp].
This is the bar the LightGBM model must beat by >15% MAE to justify its complexity.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)

BASELINE_FEATURES = ["stint_lap_number", "compound_encoded", "TrackTemp"]


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[HuberRegressor, LabelEncoder, dict]:
    """
    Fit a HuberRegressor on the three baseline features.
    Returns (model, compound_encoder, metrics_dict).
    """
    # Encode compound — low cardinality, label encode is fine
    enc = LabelEncoder()
    X_train = X_train.copy()
    X_val = X_val.copy()

    if "Compound" in X_train.columns:
        X_train["compound_encoded"] = enc.fit_transform(X_train["Compound"].fillna("UNKNOWN"))
        X_val["compound_encoded"] = enc.transform(
            X_val["Compound"].fillna("UNKNOWN").map(
                lambda c: c if c in enc.classes_ else enc.classes_[0]
            )
        )

    # Select only the baseline features that exist
    features = [f for f in BASELINE_FEATURES if f in X_train.columns]
    if not features:
        raise ValueError(f"None of the baseline features {BASELINE_FEATURES} found in data.")

    X_tr = X_train[features].fillna(0)
    X_vl = X_val[features].fillna(0)

    model = HuberRegressor(epsilon=1.35, max_iter=300)
    model.fit(X_tr, y_train)

    preds = model.predict(X_vl)
    metrics = {
        "baseline_mae": float(mean_absolute_error(y_val, preds)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
        "baseline_r2": float(r2_score(y_val, preds)),
        "baseline_features": features,
    }

    log.info(
        f"Baseline — MAE: {metrics['baseline_mae']:.4f}s  "
        f"RMSE: {metrics['baseline_rmse']:.4f}s  R²: {metrics['baseline_r2']:.4f}"
    )
    return model, enc, metrics
