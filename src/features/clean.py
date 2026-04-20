"""
Lap filtering — aggressively drops any lap that would corrupt the target variable.

Removal rules (each logged separately so we can audit how much data each cut costs):
  - out_lap      : PitOutTime is not NaT  (first flying lap after a pit stop)
  - in_lap       : PitInTime is not NaT   (lap ending in a pit stop)
  - sc_lap       : TrackStatus contains '4' (Safety Car)
  - vsc_lap      : TrackStatus contains '6' (Virtual Safety Car)
  - yellow_lap   : TrackStatus contains '2' (localised yellow flag)
  - wet_lap      : Rainfall == True
  - inaccurate   : IsAccurate == False    (FastF1 flags these itself)
  - slow_lap     : LapTime > 150s         (formation laps, red-flag crawl laps)
  - null_time    : LapTime is NaN
"""

import logging

import pandas as pd

log = logging.getLogger(__name__)


def _drop(df: pd.DataFrame, mask: pd.Series, label: str) -> pd.DataFrame:
    # Realign mask index to df to avoid pandas reindex warning after prior drops
    mask = mask.reindex(df.index, fill_value=False)
    n = mask.sum()
    if n:
        log.info(f"  Dropping {n:>5} laps — {label}")
    return df[~mask].copy()


def clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all filters in sequence. Returns the cleaned DataFrame and prints
    a removal summary.
    """
    before = len(df)
    log.info(f"clean_laps: starting with {before} laps")

    # Null / unusable lap times first — required before any numeric comparisons
    if "LapTime" in df.columns:
        df = _drop(df, df["LapTime"].isna(), "null LapTime")
        df = _drop(df, df["LapTime"] > 150.0, "slow lap (>150s) — formation / red flag")

    # Pit laps
    if "PitOutTime" in df.columns:
        df = _drop(df, df["PitOutTime"].notna(), "out-lap (PitOutTime set)")
    if "PitInTime" in df.columns:
        df = _drop(df, df["PitInTime"].notna(), "in-lap (PitInTime set)")

    # Track status — FastF1 encodes status as a string that may contain multiple chars
    if "TrackStatus" in df.columns:
        status = df["TrackStatus"].fillna("").astype(str)
        df = _drop(df, status.str.contains("4"), "Safety Car lap")
        df = _drop(df, status.str.contains("6"), "Virtual Safety Car lap")
        df = _drop(df, status.str.contains("2"), "yellow flag lap")

    # Weather
    if "Rainfall" in df.columns:
        df = _drop(df, df["Rainfall"].fillna(False).astype(bool), "wet lap (Rainfall=True)")

    # FastF1 accuracy flag — drops laps with timing anomalies
    if "IsAccurate" in df.columns:
        df = _drop(df, df["IsAccurate"] == False, "IsAccurate=False")  # noqa: E712

    after = len(df)
    removed = before - after
    pct = removed / before * 100 if before else 0
    log.info(f"clean_laps: {after} laps remain ({removed} removed, {pct:.1f}%)")
    return df
