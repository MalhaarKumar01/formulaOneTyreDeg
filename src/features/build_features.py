"""
Feature engineering pipeline.

Steps:
  1. Load all FastF1 lap parquets + telemetry parquets
  2. Clean laps (drop SC, wet, pit laps, etc.)
  3. Join telemetry aggregates — done in Polars to handle memory
  4. Fuel correction → corrected lap time
  5. Compute target: lap_time_delta_fuel_corrected (delta from stint lap 1)
  6. Rolling features: deg_rate_last_3, deg_acceleration
  7. Session-level track_evolution
  8. Save to data/features/feature_table.parquet

Usage:
    python -m src.features.build_features
    python -m src.features.build_features --test   # 2023 R01 only
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

from src.features.clean import clean_laps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FASTF1_DIR = Path("data/raw/fastf1")
TELEMETRY_DIR = Path("data/raw/telemetry")
FEATURES_DIR = Path("data/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = FEATURES_DIR / "feature_table.parquet"

# Fuel constants (widely-accepted F1 estimates)
FUEL_BURN_RATE_KG_PER_LAP = 1.88
FUEL_TIME_EFFECT_S_PER_KG = 0.035


# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------

def load_fastf1(pattern: str = "*.parquet") -> pd.DataFrame:
    files = sorted(FASTF1_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No FastF1 parquets found in {FASTF1_DIR}")
    log.info(f"Loading {len(files)} FastF1 session files...")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    log.info(f"  FastF1 laps loaded: {len(df):,}")
    return df


def load_telemetry(pattern: str = "*.parquet") -> pd.DataFrame:
    files = sorted(TELEMETRY_DIR.glob(pattern))
    if not files:
        log.warning(f"No telemetry parquets in {TELEMETRY_DIR} — proceeding without telemetry")
        return pd.DataFrame()
    log.info(f"Loading {len(files)} telemetry files...")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    log.info(f"  Telemetry rows loaded: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# 2. Join telemetry → lap data  (Polars for memory / speed)
# ---------------------------------------------------------------------------

TELEMETRY_COLS = [
    "avg_throttle_pct",
    "full_throttle_pct",
    "avg_brake",
    "braking_pct",
    "max_speed_kph",
    "avg_speed_kph",
    "drs_active_pct",
    "avg_rpm",
]

JOIN_KEYS = ["Year", "RoundNumber", "DriverNumber", "LapNumber"]


def join_telemetry(laps: pd.DataFrame, telemetry: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join lap data with aggregated telemetry on (Year, RoundNumber, DriverNumber, LapNumber).
    Uses Polars when available (faster, less memory on 500K+ rows), else falls back to Pandas.
    """
    if telemetry.empty:
        log.warning("No telemetry data — skipping join")
        return laps

    # Normalise key dtypes before joining
    for df in (laps, telemetry):
        for col in ["Year", "RoundNumber", "LapNumber"]:
            if col in df.columns:
                df[col] = df[col].astype("int32")
        if "DriverNumber" in df.columns:
            df["DriverNumber"] = df["DriverNumber"].astype(str)

    tel_cols = JOIN_KEYS + [c for c in TELEMETRY_COLS if c in telemetry.columns]
    tel = telemetry[tel_cols].drop_duplicates(subset=JOIN_KEYS)

    if _POLARS_AVAILABLE:
        log.info("Joining telemetry via Polars...")
        joined = pl.from_pandas(laps).join(pl.from_pandas(tel), on=JOIN_KEYS, how="left").to_pandas()
    else:
        log.info("Joining telemetry via Pandas (install polars for faster joins)...")
        joined = laps.merge(tel, on=JOIN_KEYS, how="left")

    fill_col = next((c for c in TELEMETRY_COLS if c in joined.columns), None)
    if fill_col:
        log.info(f"  After join: {len(joined):,} rows | telemetry fill rate: {joined[fill_col].notna().mean():.1%}")
    return joined


# ---------------------------------------------------------------------------
# 3. Fuel correction + target variable
# ---------------------------------------------------------------------------

def add_fuel_correction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fuel-corrected lap time and the target variable.

    fuel_load_kg        = start_fuel - (race_lap_number × 1.88)
    fuel_time_effect    = fuel_load_kg × 0.035   [seconds]
    corrected_lap_time  = raw_lap_time + fuel_time_effect
                          (normalises to a "zero fuel" reference)

    target = corrected_lap_time_N - corrected_lap_time_stint_lap_1
             (positive = degradation; negative = unlikely but possible)
    """
    log.info("Applying fuel correction...")

    # Estimate start fuel from actual race lap count (race_total_laps × burn_rate)
    # Groups: (Year, RoundNumber) — one race each
    race_laps = (
        df.groupby(["Year", "RoundNumber"])["LapNumber"]
        .max()
        .rename("race_total_laps")
        .reset_index()
    )
    df = df.merge(race_laps, on=["Year", "RoundNumber"], how="left")
    df["start_fuel_kg"] = df["race_total_laps"] * FUEL_BURN_RATE_KG_PER_LAP

    # Lap-level fuel load
    df["fuel_load_kg"] = (df["start_fuel_kg"] - df["LapNumber"] * FUEL_BURN_RATE_KG_PER_LAP).clip(lower=0)

    # Corrected lap time (seconds)
    df["fuel_time_effect_s"] = df["fuel_load_kg"] * FUEL_TIME_EFFECT_S_PER_KG
    df["lap_time_corrected"] = df["LapTime"] + df["fuel_time_effect_s"]

    # Target: delta from first clean lap in this stint
    df = df.sort_values(["Year", "RoundNumber", "Driver", "Stint", "LapNumber"])
    stint_first = (
        df.groupby(["Year", "RoundNumber", "Driver", "Stint"])["lap_time_corrected"]
        .first()
        .rename("stint_lap1_corrected")
        .reset_index()
    )
    df = df.merge(stint_first, on=["Year", "RoundNumber", "Driver", "Stint"], how="left")
    df["lap_time_delta_fuel_corrected"] = df["lap_time_corrected"] - df["stint_lap1_corrected"]

    # Stint lap counter (position within stint, starting at 0)
    df["stint_lap_number"] = df.groupby(["Year", "RoundNumber", "Driver", "Stint"]).cumcount()

    log.info(
        f"  Target stats — mean: {df['lap_time_delta_fuel_corrected'].mean():.3f}s  "
        f"std: {df['lap_time_delta_fuel_corrected'].std():.3f}s  "
        f"max: {df['lap_time_delta_fuel_corrected'].max():.3f}s"
    )
    return df


# ---------------------------------------------------------------------------
# 4. Rolling degradation features
# ---------------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (driver, stint), compute:
      deg_rate_last_3   — rolling mean of lap_time_delta over the last 3 laps
      deg_acceleration  — first diff of deg_rate_last_3 (is degradation speeding up?)
    """
    log.info("Computing rolling degradation features...")
    df = df.sort_values(["Year", "RoundNumber", "Driver", "Stint", "LapNumber"])

    grp = df.groupby(["Year", "RoundNumber", "Driver", "Stint"])["lap_time_delta_fuel_corrected"]
    df["deg_rate_last_3"] = grp.transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    df["deg_acceleration"] = grp.transform(
        lambda s: s.rolling(window=3, min_periods=1).mean().diff()
    )
    return df


# ---------------------------------------------------------------------------
# 5. Track evolution
# ---------------------------------------------------------------------------

def add_track_evolution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cumulative laps completed in the session across all cars up to this lap.
    More rubber on track = more grip = faster lap times (separate from tire deg).
    """
    log.info("Computing track evolution...")
    df = df.sort_values(["Year", "RoundNumber", "LapNumber", "Driver"])
    df["track_evolution"] = df.groupby(["Year", "RoundNumber"]).cumcount() + 1
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build(fastf1_pattern: str = "*.parquet", telemetry_pattern: str = "*.parquet") -> pd.DataFrame:
    laps = load_fastf1(fastf1_pattern)
    laps = clean_laps(laps)

    telemetry = load_telemetry(telemetry_pattern)
    laps = join_telemetry(laps, telemetry)

    laps = add_fuel_correction(laps)
    laps = add_rolling_features(laps)
    laps = add_track_evolution(laps)

    # Drop helper columns not needed downstream
    laps = laps.drop(columns=["race_total_laps", "start_fuel_kg", "stint_lap1_corrected",
                               "fuel_time_effect_s"], errors="ignore")

    log.info(f"Feature table: {len(laps):,} rows × {len(laps.columns)} columns")
    log.info(f"Null rates:\n{laps.isnull().mean().sort_values(ascending=False).head(10)}")
    return laps


# ---------------------------------------------------------------------------
# LSTM sequence builder
# ---------------------------------------------------------------------------

def build_stint_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "lap_time_delta_fuel_corrected",
    augment: bool = False,
) -> list[dict]:
    """
    Reshape a lap-level feature table into per-stint sequences for LSTM training.

    Returns list of {"X": np.ndarray(L, F), "y": np.ndarray(L,), "meta": dict}.

    augment=True creates subsequences [lap 1..2], [lap 1..3], ..., [lap 1..L]
    per stint — multiplies training samples ~10× without data leakage.
    augment=False returns one full sequence per stint (for val/test).
    """
    available = [c for c in feature_cols if c in df.columns]
    sequences: list[dict] = []
    group_cols = ["Year", "RoundNumber", "Driver", "Stint"]

    for key, grp in df.groupby(group_cols, sort=False):
        grp = grp.sort_values("stint_lap_number").dropna(subset=[target_col])
        if len(grp) < 2:
            continue

        X_full = grp[available].to_numpy(dtype=np.float32)
        y_full = grp[target_col].to_numpy(dtype=np.float32)
        meta = dict(zip(group_cols, key))

        if augment:
            for end in range(2, len(grp) + 1):
                sequences.append({"X": X_full[:end], "y": y_full[:end], "meta": meta})
        else:
            sequences.append({"X": X_full, "y": y_full, "meta": meta})

    return sequences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run on 2023 R01 only")
    args = parser.parse_args()

    pattern = "2023_R01.parquet" if args.test else "*.parquet"
    tel_pattern = "2023_R01_telemetry.parquet" if args.test else "*.parquet"

    df = build(pattern, tel_pattern)
    df.to_parquet(OUT_PATH, index=False)
    log.info(f"Saved → {OUT_PATH}")
