"""
FastF1 ingestion — lap times, tire compounds, stint numbers, and weather.
Uses get_event_schedule for all circuit/session metadata (no Ergast needed).

Usage:
    python -m src.ingestion.fastf1_ingest            # full 2023-2025 run
    python -m src.ingestion.fastf1_ingest --test     # one race only
"""

import argparse
import logging
from pathlib import Path

import fastf1
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/fastf1")
CACHE_DIR = Path("cache/fastf1")
RAW_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))

SEASONS = [2023, 2024, 2025]

# Columns we actually need — drop FastF1's many internal/redundant fields
LAP_COLS = [
    "Driver",
    "DriverNumber",
    "Team",
    "LapNumber",
    "LapTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "Compound",
    "TyreLife",
    "Stint",
    "TrackStatus",
    "LapStartTime",
    "PitInTime",
    "PitOutTime",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
    "IsAccurate",
]


def _timedelta_to_seconds(series: pd.Series) -> pd.Series:
    """Convert a timedelta Series to float seconds. Non-timedelta values become NaN."""
    try:
        return series.dt.total_seconds()
    except AttributeError:
        return pd.to_numeric(series, errors="coerce")


def load_session(year: int, round_number: int) -> pd.DataFrame | None:
    """Load one race session and return a cleaned lap+weather DataFrame."""
    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load(laps=True, telemetry=False, weather=True, messages=False)
    except Exception as e:
        log.warning(f"  [{year} R{round_number}] Load failed: {e}")
        return None

    laps = session.laps.copy()
    if laps.empty:
        log.warning(f"  [{year} R{round_number}] No laps returned")
        return None

    # Keep only columns that exist in this session
    keep = [c for c in LAP_COLS if c in laps.columns]
    laps = laps[keep].copy()

    # Convert timedelta columns → float seconds for Parquet compatibility
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in laps.columns:
            laps[col] = _timedelta_to_seconds(laps[col])

    # --- Weather join (nearest session-time timestamp) ---
    weather = session.weather_data.copy()
    if not weather.empty and "Time" in weather.columns and "LapStartTime" in laps.columns:
        laps = laps.dropna(subset=["LapStartTime"])
        laps = laps.sort_values("LapStartTime").reset_index(drop=True)
        weather = weather.sort_values("Time").reset_index(drop=True)
        laps = pd.merge_asof(
            laps,
            weather[["Time", "AirTemp", "TrackTemp", "Rainfall", "WindSpeed"]].rename(
                columns={"Time": "_wt"}
            ),
            left_on="LapStartTime",
            right_on="_wt",
            direction="nearest",
        ).drop(columns=["_wt"], errors="ignore")

    # --- Circuit / session metadata from get_event_schedule ---
    event = session.event
    laps["Year"] = year
    laps["RoundNumber"] = round_number
    laps["EventName"] = event.get("EventName", "")
    laps["CircuitShortName"] = event.get("OfficialEventName", event.get("EventName", ""))
    laps["Country"] = event.get("Country", "")
    laps["Location"] = event.get("Location", "")

    return laps


def ingest_season(year: int, rounds: list[int] | None = None) -> None:
    """Ingest all (or specified) race rounds for a season."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    # Filter out testing/pre-season events
    races = schedule[schedule["EventFormat"].isin(["conventional", "sprint", "sprint_qualifying"])]

    if rounds:
        races = races[races["RoundNumber"].isin(rounds)]

    log.info(f"Season {year}: {len(races)} rounds to process")

    for _, event in races.iterrows():
        rnd = int(event["RoundNumber"])
        out_path = RAW_DIR / f"{year}_R{rnd:02d}.parquet"

        if out_path.exists():
            log.info(f"  [{year} R{rnd:02d}] Already ingested — skipping")
            continue

        log.info(f"  [{year} R{rnd:02d}] {event['EventName']} ...")
        df = load_session(year, rnd)
        if df is not None:
            df.to_parquet(out_path, index=False)
            log.info(f"  [{year} R{rnd:02d}] Saved {len(df)} laps → {out_path.name}")
        else:
            log.warning(f"  [{year} R{rnd:02d}] Skipped — no data")


def validate(n: int = 3) -> None:
    """Spot-check saved files: lap count, compound coverage, target column presence."""
    import random

    files = list(RAW_DIR.glob("*.parquet"))
    if not files:
        log.error("No parquet files found — run ingestion first.")
        return

    for f in random.sample(files, min(n, len(files))):
        df = pd.read_parquet(f)
        compounds = df["Compound"].value_counts().to_dict() if "Compound" in df.columns else {}
        stints = df["Stint"].nunique() if "Stint" in df.columns else "N/A"
        log.info(
            f"  {f.name}: {len(df)} laps | stints={stints} | compounds={compounds}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Dry run: ingest one race only")
    args = parser.parse_args()

    if args.test:
        log.info("--- TEST MODE: ingesting 2023 R01 only ---")
        ingest_season(2023, rounds=[1])
    else:
        for year in SEASONS:
            ingest_season(year)

    log.info("Validating output...")
    validate(3)
    log.info("Done.")
