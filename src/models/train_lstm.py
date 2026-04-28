"""
Train an LSTM model for tyre degradation sequence prediction.

The LSTM models each stint as a time series, with backpropagation through
time (BPTT) — the hidden state carries memory of how the tyre has been driven
across previous laps, something LightGBM cannot capture.

  Input at each timestep: 16 lap-level features (excludes deg_rate/deg_acceleration
  which are derived from the target and would cause leakage in sequence prediction)
  Output at each timestep: predicted lap_time_delta_fuel_corrected

Temporal split: train 2023-2024, val 2025 R01-10, test 2025 R11+

Usage:
    python -m src.models.train_lstm
    python -m src.models.train_lstm --test   # smoke test (10 epochs)
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.features.build_features import build_stint_sequences
from src.models.lstm_model import LSTMDegradationModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FEATURES_PATH = Path("data/features/feature_table.parquet")
MODEL_DIR     = Path("models/degradation")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "lap_time_delta_fuel_corrected"

# 16 LSTM features — deg_rate_last_3 / deg_acceleration excluded because they
# are rolling windows over the target itself (data leakage for future-lap prediction)
LSTM_FEATURES = [
    "stint_lap_number",
    "compound_encoded",
    "fuel_load_kg",
    "TrackTemp",
    "AirTemp",
    "track_evolution",
    "avg_throttle_pct",
    "full_throttle_pct",
    "avg_brake",
    "braking_pct",
    "max_speed_kph",
    "drs_active_pct",
    "avg_rpm",
    "sector_1_pct",
    "sector_2_pct",
    "sector_3_pct",
]

COMPOUND_MAP    = {"HARD": 0, "INTERMEDIATE": 1, "MEDIUM": 2, "SOFT": 3, "WET": 4, "UNKNOWN": 2}
VAL_CUTOFF_ROUND = 10
HIDDEN_SIZE     = 64
NUM_LAYERS      = 2
DROPOUT         = 0.3
BATCH_SIZE      = 32
MAX_EPOCHS      = 200
PATIENCE        = 20


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def temporal_split(df: pd.DataFrame, smoke_test: bool = False):
    if not smoke_test:
        train = df[df["Year"] <= 2024].copy()
        val   = df[(df["Year"] == 2025) & (df["RoundNumber"] <= VAL_CUTOFF_ROUND)].copy()
        test  = df[(df["Year"] == 2025) & (df["RoundNumber"] >  VAL_CUTOFF_ROUND)].copy()
    else:
        rounds = sorted(df[["Year", "RoundNumber"]].drop_duplicates().apply(tuple, axis=1).tolist())
        n = len(rounds)
        train_set = set(rounds[:max(1, int(n * 0.6))])
        val_set   = set(rounds[max(1, int(n * 0.6)):max(2, int(n * 0.8))])
        test_set  = set(rounds[max(2, int(n * 0.8)):]) or {rounds[-1]}
        def _in(r, s): return (r["Year"], r["RoundNumber"]) in s
        train = df[df.apply(lambda r: _in(r, train_set), axis=1)].copy()
        val   = df[df.apply(lambda r: _in(r, val_set),   axis=1)].copy()
        test  = df[df.apply(lambda r: _in(r, test_set),  axis=1)].copy()
        if val.empty or test.empty:
            n = len(df)
            train = df.iloc[:int(n * 0.6)].copy()
            val   = df.iloc[int(n * 0.6):int(n * 0.8)].copy()
            test  = df.iloc[int(n * 0.8):].copy()

    if test.empty and not val.empty:
        split = int(len(val) * 0.7)
        test, val = val.iloc[split:].copy(), val.iloc[:split].copy()

    log.info(f"Split  train={len(train):,}  val={len(val):,}  test={len(test):,}")
    return train, val, test


def add_lstm_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["compound_encoded"] = (
        df["Compound"].fillna("UNKNOWN")
        .map(lambda c: COMPOUND_MAP.get(str(c).upper(), 2))
        .astype(np.float32)
    )
    for i, col in enumerate(["Sector1Time", "Sector2Time", "Sector3Time"], start=1):
        if col in df.columns and "LapTime" in df.columns:
            df[f"sector_{i}_pct"] = df[col] / df["LapTime"].replace(0, np.nan)
    return df


# ---------------------------------------------------------------------------
# PyTorch dataset + collation
# ---------------------------------------------------------------------------

class StintDataset(torch.utils.data.Dataset):
    def __init__(self, sequences: list[dict]):
        self.seqs = sequences

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int):
        s = self.seqs[idx]
        return (
            torch.tensor(s["X"], dtype=torch.float32),
            torch.tensor(s["y"], dtype=torch.float32),
        )


def collate_fn(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    xs_pad  = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys_pad  = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0.0)
    return xs_pad, ys_pad, lengths


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def masked_huber_loss(
    preds: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    loss_fn = nn.HuberLoss(delta=1.0, reduction="none")
    losses  = loss_fn(preds, targets)
    mask = torch.zeros_like(losses, dtype=torch.bool)
    for i, L in enumerate(lengths):
        mask[i, :L] = True
    return losses[mask].mean()


def collect_predictions(
    model: LSTMDegradationModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y, lengths in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X, lengths)
            for i, L in enumerate(lengths):
                all_preds.extend(preds[i, :L].cpu().tolist())
                all_targets.extend(y[i, :L].cpu().tolist())
    return np.array(all_preds), np.array(all_targets)


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    return {
        "mae":  float(mean_absolute_error(targets, preds)),
        "rmse": float(np.sqrt(mean_squared_error(targets, preds))),
        "r2":   float(r2_score(targets, preds)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(smoke_test: bool = False) -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature table not found at {FEATURES_PATH}. Run make features first."
        )

    df = pd.read_parquet(FEATURES_PATH)
    log.info(f"Loaded {len(df):,} rows")

    train_df, val_df, test_df = temporal_split(df, smoke_test=smoke_test)
    train_df = add_lstm_features(train_df)
    val_df   = add_lstm_features(val_df)
    test_df  = add_lstm_features(test_df)

    # Fill NaNs with train medians (covers sessions without telemetry)
    medians = train_df[LSTM_FEATURES].median()
    for split in (train_df, val_df, test_df):
        split[LSTM_FEATURES] = split[LSTM_FEATURES].fillna(medians)

    # StandardScaler — critical for LSTM convergence (LightGBM doesn't need this)
    scaler = StandardScaler()
    train_df[LSTM_FEATURES] = scaler.fit_transform(train_df[LSTM_FEATURES])
    val_df[LSTM_FEATURES]   = scaler.transform(val_df[LSTM_FEATURES])
    test_df[LSTM_FEATURES]  = scaler.transform(test_df[LSTM_FEATURES])

    log.info("Building stint sequences...")
    train_seqs = build_stint_sequences(train_df, LSTM_FEATURES, augment=True)
    val_seqs   = build_stint_sequences(val_df,   LSTM_FEATURES, augment=False)
    test_seqs  = build_stint_sequences(test_df,  LSTM_FEATURES, augment=False)
    log.info(f"  Sequences — train: {len(train_seqs)}  val: {len(val_seqs)}  test: {len(test_seqs)}")

    if not train_seqs or not val_seqs:
        raise RuntimeError("Not enough stint sequences to train. Check your feature table.")

    def make_loader(seqs, shuffle: bool) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            StintDataset(seqs), batch_size=BATCH_SIZE,
            shuffle=shuffle, collate_fn=collate_fn,
        )

    train_loader = make_loader(train_seqs, shuffle=True)
    val_loader   = make_loader(val_seqs,   shuffle=False)
    test_loader  = make_loader(test_seqs,  shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on {device}")

    model = LSTMDegradationModel(
        input_size=len(LSTM_FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    best_val_mae   = float("inf")
    best_state     = None
    patience_count = 0
    max_epochs     = 10 if smoke_test else MAX_EPOCHS

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for X, y, lengths in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X, lengths)
            loss  = masked_huber_loss(preds, y, lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        val_preds, val_targets = collect_predictions(model, val_loader, device)
        val_mae = float(mean_absolute_error(val_targets, val_preds))
        scheduler.step(val_mae)

        if epoch % 10 == 0 or epoch <= 5:
            log.info(
                f"Epoch {epoch:3d}  "
                f"train_loss={train_loss / len(train_loader):.4f}  "
                f"val_mae={val_mae:.4f}s"
            )

        if val_mae < best_val_mae:
            best_val_mae   = val_mae
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                log.info(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.to(device)

    val_preds,  val_targets  = collect_predictions(model, val_loader,  device)
    test_preds, test_targets = collect_predictions(model, test_loader, device)
    val_m  = compute_metrics(val_preds,  val_targets)
    test_m = compute_metrics(test_preds, test_targets)

    log.info(
        f"LSTM  val  MAE={val_m['mae']:.4f}s  R²={val_m['r2']:.4f}\n"
        f"      test MAE={test_m['mae']:.4f}s  R²={test_m['r2']:.4f}"
    )

    # Compare to LightGBM and update params.json
    params_path = MODEL_DIR / "params.json"
    if params_path.exists():
        existing = json.loads(params_path.read_text())
        lgbm_mae = existing.get("test", {}).get("mae")
        if lgbm_mae:
            winner = "LSTM" if test_m["mae"] < lgbm_mae else "LightGBM"
            log.info(f"LightGBM test MAE: {lgbm_mae:.4f}s  →  Winner: {winner}")
        existing["lstm"] = {
            "val":              val_m,
            "test":             test_m,
            "n_features":       len(LSTM_FEATURES),
            "n_train_sequences": len(train_seqs),
        }
        with open(params_path, "w") as f:
            json.dump(existing, f, indent=2)
        log.info(f"LSTM metrics appended → {params_path}")
    else:
        log.warning("params.json not found — run make train first for LightGBM baseline")

    # Save model checkpoint and scaler
    torch.save(
        {
            "model_state":   model.state_dict(),
            "feature_names": LSTM_FEATURES,
            "model_config": {
                "input_size":  len(LSTM_FEATURES),
                "hidden_size": HIDDEN_SIZE,
                "num_layers":  NUM_LAYERS,
                "dropout":     DROPOUT,
            },
        },
        str(MODEL_DIR / "lstm_model.pt"),
    )
    with open(MODEL_DIR / "lstm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    log.info(f"LSTM model → {MODEL_DIR}/lstm_model.pt")
    log.info(f"Scaler     → {MODEL_DIR}/lstm_scaler.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Smoke test: 10 epochs only")
    args = parser.parse_args()
    main(smoke_test=args.test)
