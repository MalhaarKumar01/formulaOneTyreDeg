"""
Train the tire degradation model.

Pipeline:
  1. Load feature table
  2. Temporal split  (train: 2023-24 | val: 2025 R01-10 | test: 2025 R11+)
  3. Encode features (target-encode Driver + Circuit; label-encode Compound)
  4. Baseline HuberRegressor
  5. Optuna 50-trial hyperparameter search (LightGBM, Huber loss)
  6. Retrain final model on train+val with best params
  7. Evaluate on test — print + save metrics
  8. SHAP: global importance, waterfall, predicted vs actual, residuals by compound
  9. Save model.lgb + params.json

Usage:
    python -m src.models.train_degradation
    python -m src.models.train_degradation --test   # tiny run (5 Optuna trials)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
from category_encoders import TargetEncoder
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from src.models.baseline import train_baseline
from src.models.callbacks import EpochMetricsCallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Paths
FEATURES_PATH = Path("data/features/feature_table.parquet")
MODEL_DIR = Path("models/degradation")
METRICS_DIR = Path("metrics")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "lap_time_delta_fuel_corrected"

# All candidate model features
MODEL_FEATURES = [
    "stint_lap_number",
    "Stint",
    "compound_encoded",
    "fuel_load_kg",
    "TrackTemp",
    "AirTemp",
    "avg_throttle_pct",
    "full_throttle_pct",
    "avg_brake",
    "braking_pct",
    "max_speed_kph",
    "drs_active_pct",
    "avg_rpm",
    "track_evolution",
    "deg_rate_last_3",
    "deg_acceleration",
    "sector_1_pct",
    "sector_2_pct",
    "sector_3_pct",
]

# 2025 round that separates val from test
VAL_CUTOFF_ROUND = 10


# ---------------------------------------------------------------------------
# 1. Load + split
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature table not found at {FEATURES_PATH}. Run make features first.")
    df = pd.read_parquet(FEATURES_PATH)
    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} cols")
    return df


def temporal_split(
    df: pd.DataFrame, smoke_test: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict time-based split — no data leakage.
      train : 2023–2024
      val   : 2025 rounds 1–10   (hyperparameter tuning)
      test  : 2025 rounds 11+    (final held-out evaluation)

    smoke_test=True: fall back to a 60/20/20 round-based split on whatever
    data is present — lets the full pipeline run with only a single session.
    """
    if not smoke_test:
        train = df[df["Year"] <= 2024].copy()
        val   = df[(df["Year"] == 2025) & (df["RoundNumber"] <= VAL_CUTOFF_ROUND)].copy()
        test  = df[(df["Year"] == 2025) & (df["RoundNumber"] >  VAL_CUTOFF_ROUND)].copy()
    else:
        # Sort by (Year, RoundNumber) and split by unique rounds
        rounds = sorted(df[["Year", "RoundNumber"]].drop_duplicates().apply(tuple, axis=1).tolist())
        n = len(rounds)
        train_rounds = set(rounds[: int(n * 0.6) or 1])
        val_rounds   = set(rounds[int(n * 0.6) or 1 : int(n * 0.8) or 2])
        test_rounds  = set(rounds[int(n * 0.8) or 2 :]) or {rounds[-1]}

        def _in(row, s):
            return (row["Year"], row["RoundNumber"]) in s

        train = df[df.apply(lambda r: _in(r, train_rounds), axis=1)].copy()
        val   = df[df.apply(lambda r: _in(r, val_rounds),   axis=1)].copy()
        test  = df[df.apply(lambda r: _in(r, test_rounds),  axis=1)].copy()

        # Last resort: if only one round exists, slice by driver count
        if val.empty or test.empty:
            n_rows = len(df)
            train = df.iloc[: int(n_rows * 0.6)].copy()
            val   = df.iloc[int(n_rows * 0.6) : int(n_rows * 0.8)].copy()
            test  = df.iloc[int(n_rows * 0.8) :].copy()
            log.warning("Only one round available — splitting by row index for smoke test")

    log.info(f"Split  train={len(train):,}  val={len(val):,}  test={len(test):,}")

    # Fallback: if test is empty (e.g. 2025 R11+ not yet ingested),
    # carve the last 30% of val off as test so the pipeline can still run.
    if test.empty and not val.empty:
        log.warning(
            "Test set is empty (no 2025 R11+ data). "
            "Splitting last 30% of val into test — re-run after ingesting more 2025 rounds."
        )
        split_idx = int(len(val) * 0.70)
        test = val.iloc[split_idx:].copy()
        val  = val.iloc[:split_idx].copy()
        log.info(f"Adjusted  val={len(val):,}  test={len(test):,}")

    return train, val, test


# ---------------------------------------------------------------------------
# 2. Feature engineering (encoding + derived cols)
# ---------------------------------------------------------------------------

def add_sector_fractions(df: pd.DataFrame) -> pd.DataFrame:
    """Sector time as fraction of full lap — reveals where tires are failing."""
    for i, col in enumerate(["Sector1Time", "Sector2Time", "Sector3Time"], start=1):
        if col in df.columns and "LapTime" in df.columns:
            df[f"sector_{i}_pct"] = df[col] / df["LapTime"].replace(0, np.nan)
    return df


def encode_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit all encoders on train only — no leakage into val/test.
      - Driver, Circuit  → TargetEncoder (high cardinality)
      - Compound         → LabelEncoder  (5 known values)
    """
    # Sector fractions
    for split in (train, val, test):
        add_sector_fractions(split)

    # Compound label encode — always include UNKNOWN so the fallback mapping
    # never produces a label the encoder hasn't seen
    compound_enc = LabelEncoder()
    known_compounds = sorted(
        set(train["Compound"].dropna().unique()) | {"UNKNOWN"}
    )
    compound_enc.fit(known_compounds)

    def _encode_compound(df: pd.DataFrame) -> pd.DataFrame:
        raw = df["Compound"].fillna("UNKNOWN").map(
            lambda c: c if c in compound_enc.classes_ else "UNKNOWN"
        )
        df["compound_encoded"] = compound_enc.transform(raw)
        return df

    train = _encode_compound(train)
    val   = _encode_compound(val)
    test  = _encode_compound(test)

    # Target-encode Driver + Circuit (fit on train target)
    target_enc = TargetEncoder(cols=["Driver", "EventName"], smoothing=10)
    target_enc.fit(train[["Driver", "EventName"]], train[TARGET])

    for split_df in (train, val, test):
        encoded = target_enc.transform(split_df[["Driver", "EventName"]])
        split_df["driver_encoded"]  = encoded["Driver"].values
        split_df["circuit_encoded"] = encoded["EventName"].values

    return train, val, test


def prepare_xy(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    features = [f for f in MODEL_FEATURES if f in df.columns]
    X = df[features].copy()
    # Fill telemetry nulls (sessions where telemetry wasn't available) with column median
    X = X.fillna(X.median(numeric_only=True))
    y = df[TARGET]
    return X, y


# ---------------------------------------------------------------------------
# 3. Optuna objective
# ---------------------------------------------------------------------------

def make_objective(X_train, y_train, X_val, y_val):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "huber",
            "alpha": 0.9,          # Huber robustness parameter
            "n_estimators": 2000,
            "verbose": -1,
            "n_jobs": -1,
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)

    return objective


# ---------------------------------------------------------------------------
# 4. Final evaluation metrics
# ---------------------------------------------------------------------------

def evaluate(model, X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    preds = model.predict(X)
    mae  = float(mean_absolute_error(y, preds))
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2   = float(r2_score(y, preds))
    log.info(f"{label:10s} → MAE: {mae:.4f}s  RMSE: {rmse:.4f}s  R²: {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ---------------------------------------------------------------------------
# 5. SHAP plots
# ---------------------------------------------------------------------------

def generate_shap(model, X_test: pd.DataFrame) -> None:
    log.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    # Use a 2000-row sample to keep memory reasonable
    sample = X_test.sample(min(2000, len(X_test)), random_state=42)
    shap_values = explainer(sample)

    top3 = (
        pd.Series(np.abs(shap_values.values).mean(axis=0), index=sample.columns)
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )
    log.info(f"Top-3 SHAP features: {top3}")

    # Global feature importance
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.bar(shap_values, max_display=15, show=False, ax=ax)
    ax.set_title("Global Feature Importance (SHAP mean |value|)")
    fig.tight_layout()
    fig.savefig(METRICS_DIR / "shap_importance.png", dpi=150)
    plt.close(fig)

    # Waterfall for a single prediction (index 0)
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(shap_values[0], max_display=12, show=False)
    plt.title("SHAP Waterfall — Single Prediction")
    plt.tight_layout()
    plt.savefig(METRICS_DIR / "shap_waterfall_sample.png", dpi=150)
    plt.close()

    log.info(f"SHAP plots saved to {METRICS_DIR}/")


def plot_predicted_vs_actual(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    preds = model.predict(X_test)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter
    axes[0].scatter(y_test, preds, alpha=0.15, s=8, color="steelblue")
    lim = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    axes[0].plot(lim, lim, "r--", linewidth=1)
    axes[0].set_xlabel("Actual lap_time_delta (s)")
    axes[0].set_ylabel("Predicted (s)")
    axes[0].set_title("Predicted vs Actual")

    # Residuals
    residuals = preds - y_test.values
    axes[1].hist(residuals, bins=80, color="steelblue", edgecolor="none")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual (s)")
    axes[1].set_title("Residual Distribution")

    fig.tight_layout()
    fig.savefig(METRICS_DIR / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)


def plot_residuals_by_compound(model, X_test: pd.DataFrame, y_test: pd.Series, test_df: pd.DataFrame) -> None:
    preds = model.predict(X_test)
    residuals = preds - y_test.values
    compounds = test_df.loc[X_test.index, "Compound"].fillna("UNKNOWN").values

    compound_order = sorted(set(compounds))
    data = [residuals[compounds == c] for c in compound_order]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(data, tick_labels=compound_order, showfliers=False)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_ylabel("Residual (s)")
    ax.set_title("Residuals by Compound")
    fig.tight_layout()
    fig.savefig(METRICS_DIR / "residuals_by_compound.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_trials: int = 50, smoke_test: bool = False) -> None:
    # Auto-build feature table when running in smoke-test mode
    if smoke_test and not FEATURES_PATH.exists():
        log.info("Feature table missing — running feature pipeline first...")
        import subprocess, sys
        subprocess.run(
            [sys.executable, "-m", "src.features.build_features", "--test"],
            check=True,
        )

    df = load_data()
    train_df, val_df, test_df = temporal_split(df, smoke_test=smoke_test)
    train_df, val_df, test_df = encode_features(train_df, val_df, test_df)

    X_train, y_train = prepare_xy(train_df)
    X_val,   y_val   = prepare_xy(val_df)
    X_test,  y_test  = prepare_xy(test_df)

    log.info(f"Feature matrix: {X_train.shape[1]} features — {list(X_train.columns)}")

    # --- Baseline ---
    _, _, baseline_metrics = train_baseline(X_train, y_train, X_val, y_val)

    # --- Optuna search ---
    log.info(f"Starting Optuna ({n_trials} trials)...")
    study = optuna.create_study(direction="minimize", study_name="lgbm_degradation")
    study.optimize(
        make_objective(X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_val_mae = study.best_value
    log.info(f"Best val MAE: {best_val_mae:.4f}s  params: {best_params}")

    # --- Final model: retrain on train + val ---
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    final_params = {
        "objective": "huber",
        "alpha": 0.9,
        "n_estimators": 2000,
        "verbose": -1,
        "n_jobs": -1,
        **best_params,
    }
    n_estimators = final_params.get("n_estimators", 2000)
    metrics_cb = EpochMetricsCallback(
        X_train, y_train, X_val, y_val,
        log_every=50 if smoke_test else 100,
        total_rounds=n_estimators,
    )

    final_model = lgb.LGBMRegressor(**final_params)
    final_model.fit(
        X_trainval, y_trainval,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
            metrics_cb,
        ],
    )

    # --- Evaluate ---
    val_metrics  = evaluate(final_model, X_val,  y_val,  "val")
    test_metrics = evaluate(final_model, X_test, y_test, "test")

    improvement = (baseline_metrics["baseline_mae"] - test_metrics["mae"]) / baseline_metrics["baseline_mae"] * 100
    log.info(f"LightGBM improvement over baseline: {improvement:.1f}%")
    if improvement < 15:
        log.warning("Improvement <15% — check feature engineering, not the model.")

    # --- Save ---
    final_model.booster_.save_model(str(MODEL_DIR / "model.lgb"))

    all_metrics = {
        **baseline_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "improvement_over_baseline_pct": round(improvement, 2),
        "best_optuna_params": best_params,
        "n_features": len(X_train.columns),
        "train_rows": len(X_train),
    }
    with open(MODEL_DIR / "params.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    log.info(f"Model saved → {MODEL_DIR}/model.lgb")
    log.info(f"Metrics saved → {MODEL_DIR}/params.json")

    # --- Plots ---
    generate_shap(final_model, X_test)
    plot_predicted_vs_actual(final_model, X_test, y_test)
    plot_residuals_by_compound(final_model, X_test, y_test, test_df)
    log.info("All plots saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Quick run: 5 Optuna trials, smoke-test split")
    args = parser.parse_args()

    main(n_trials=5 if args.test else 50, smoke_test=args.test)
