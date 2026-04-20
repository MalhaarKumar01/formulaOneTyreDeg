"""
OpenF1 ingestion: car telemetry at ~3.7Hz (throttle, brake, speed, DRS, RPM).
Fetches all car data per session in one request, then aggregates to lap-level.
Saves one Parquet per session to data/raw/openf1/.
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/openf1")
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.openf1.org/v1"
REQUEST_DELAY = 0.5  # seconds between requests

SEASONS = [2022, 2023, 2024, 2025]


def _get(endpoint: str, params: dict) -> list[dict]:
    """GET from OpenF1 API with retry."""
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            log.warning(f"Request failed (attempt {attempt + 1}/3): {e}")
            time.sleep(2**attempt)
    return []


def get_race_sessions(year: int) -> list[dict]:
    data = _get("sessions", {"year": year, "session_name": "Race"})
    time.sleep(REQUEST_DELAY)
    return data


def fetch_laps(session_key: int) -> pd.DataFrame:
    """Lap boundary timestamps for all drivers in a session."""
    data = _get("laps", {"session_key": session_key})
    time.sleep(REQUEST_DELAY)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date_start"] = pd.to_datetime(df["date_start"], utc=True, errors="coerce")
    return df.dropna(subset=["date_start"]).sort_values(["driver_number", "date_start"]).reset_index(drop=True)


def fetch_car_data(session_key: int) -> pd.DataFrame:
    """All telemetry samples for the entire session (one API call)."""
    data = _get("car_data", {"session_key": session_key})
    time.sleep(REQUEST_DELAY)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df.dropna(subset=["date"]).sort_values(["driver_number", "date"]).reset_index(drop=True)


def aggregate_to_laps(car_df: pd.DataFrame, laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sample-level telemetry to one row per (driver, lap).
    Uses lap start timestamps as boundaries; end = next lap's start.
    """
    if car_df.empty or laps_df.empty:
        return pd.DataFrame()

    rows = []
    for driver_number, drv_laps in laps_df.groupby("driver_number"):
        drv_laps = drv_laps.sort_values("date_start").reset_index(drop=True)
        drv_car = car_df[car_df["driver_number"] == driver_number].sort_values("date")

        if drv_car.empty:
            continue

        for pos in range(len(drv_laps)):
            lap_row = drv_laps.iloc[pos]
            lap_start = lap_row["date_start"]
            # End = next lap start; last lap gets a 3-minute ceiling
            lap_end = (
                drv_laps.iloc[pos + 1]["date_start"]
                if pos + 1 < len(drv_laps)
                else lap_start + pd.Timedelta(minutes=3)
            )

            mask = (drv_car["date"] >= lap_start) & (drv_car["date"] < lap_end)
            samples = drv_car[mask]

            if len(samples) < 5:
                continue

            rows.append(
                {
                    "driver_number": driver_number,
                    "lap_number": lap_row.get("lap_number"),
                    "avg_throttle_pct": samples["throttle"].mean() if "throttle" in samples else None,
                    "avg_brake": samples["brake"].mean() if "brake" in samples else None,
                    "max_speed_kph": samples["speed"].max() if "speed" in samples else None,
                    "drs_laps_pct": (samples["drs"] > 0).mean() if "drs" in samples else None,
                    "avg_rpm": samples["rpm"].mean() if "rpm" in samples else None,
                    "n_telemetry_samples": len(samples),
                }
            )

    return pd.DataFrame(rows)


def ingest_session(session: dict) -> None:
    session_key = session["session_key"]
    year = session.get("year")
    circuit = session.get("circuit_short_name", "unknown")

    out_path = RAW_DIR / f"{year}_{session_key}_{circuit}.parquet"
    if out_path.exists():
        log.info(f"  Skip {year} {circuit} (already ingested)")
        return

    log.info(f"  Ingesting {year} {circuit} (session_key={session_key})...")

    laps_df = fetch_laps(session_key)
    if laps_df.empty:
        log.warning(f"  No lap data for session {session_key}")
        return

    car_df = fetch_car_data(session_key)
    if car_df.empty:
        log.warning(f"  No telemetry for session {session_key} — saving lap boundaries only")

    result = aggregate_to_laps(car_df, laps_df)
    if result.empty:
        log.warning(f"  Aggregation produced no rows for session {session_key}")
        return

    result["session_key"] = session_key
    result["year"] = year
    result["circuit"] = circuit
    result.to_parquet(out_path, index=False)
    log.info(f"  Saved {len(result)} lap-level rows → {out_path}")


if __name__ == "__main__":
    for year in SEASONS:
        sessions = get_race_sessions(year)
        log.info(f"Season {year}: {len(sessions)} race sessions")
        for session in sessions:
            ingest_session(session)

    log.info("OpenF1 ingestion complete.")
