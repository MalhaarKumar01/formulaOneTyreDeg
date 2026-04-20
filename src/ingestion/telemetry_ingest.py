"""
High-frequency telemetry ingestion (Speed, Throttle, Brake, DRS) via FastF1.
Aggregates ~3.7Hz samples to lap-level and saves per session to data/raw/telemetry/.

Parallelized with ProcessPoolExecutor — each worker handles one full session.
FastF1 cache is enabled inside each worker to avoid fork-safety issues.

Usage:
    python -m src.ingestion.telemetry_ingest             # full 2023-2025
    python -m src.ingestion.telemetry_ingest --test      # 2023 R01 only
    python -m src.ingestion.telemetry_ingest --workers 6
"""

from __future__ import annotations

import argparse
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fastf1
import pandas as pd

RAW_DIR = Path("data/raw/telemetry")
CACHE_DIR = Path("cache/fastf1")
RAW_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = [2023, 2024, 2025]
DEFAULT_WORKERS = 4

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker — must be a top-level function so multiprocessing can pickle it
# ---------------------------------------------------------------------------

def _process_session(task: tuple[int, int, str]) -> dict:
    """
    Worker: load telemetry for one session, aggregate to lap-level, save Parquet.
    Returns a status dict so the main process can log results cleanly.
    """
    year, round_number, event_name = task
    out_path = RAW_DIR / f"{year}_R{round_number:02d}_telemetry.parquet"

    # Cache must be enabled inside the worker process
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    # Suppress FastF1's own verbose logging inside workers
    logging.getLogger("fastf1").setLevel(logging.WARNING)

    if out_path.exists():
        return {"task": f"{year} R{round_number:02d}", "status": "skipped", "rows": 0}

    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load(laps=True, telemetry=True, weather=False, messages=False)
    except Exception as e:
        return {"task": f"{year} R{round_number:02d}", "status": "error", "error": str(e), "rows": 0}

    laps = session.laps
    if laps.empty:
        return {"task": f"{year} R{round_number:02d}", "status": "no_laps", "rows": 0}

    rows = []
    for _, lap in laps.iterrows():
        try:
            car = lap.get_car_data()
        except Exception:
            continue

        if car is None or len(car) < 5:
            continue

        row: dict = {
            "Year": year,
            "RoundNumber": round_number,
            "EventName": event_name,
            "Driver": lap.get("Driver"),
            "DriverNumber": lap.get("DriverNumber"),
            "LapNumber": lap.get("LapNumber"),
            "Stint": lap.get("Stint"),
            "Compound": lap.get("Compound"),
        }

        if "Speed" in car.columns:
            row["avg_speed_kph"] = car["Speed"].mean()
            row["max_speed_kph"] = car["Speed"].max()

        if "Throttle" in car.columns:
            row["avg_throttle_pct"] = car["Throttle"].mean()
            # Full throttle application fraction (>98% = WOT)
            row["full_throttle_pct"] = (car["Throttle"] >= 98).mean()

        if "Brake" in car.columns:
            row["avg_brake"] = car["Brake"].mean()
            row["braking_pct"] = (car["Brake"] > 0).mean()

        if "DRS" in car.columns:
            # DRS values: 0=closed, 8=open (FastF1 convention)
            row["drs_active_pct"] = (car["DRS"] >= 8).mean()

        if "RPM" in car.columns:
            row["avg_rpm"] = car["RPM"].mean()

        if "nGear" in car.columns:
            row["avg_gear"] = car["nGear"].mean()

        rows.append(row)

    if not rows:
        return {"task": f"{year} R{round_number:02d}", "status": "no_telemetry", "rows": 0}

    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return {"task": f"{year} R{round_number:02d}", "status": "ok", "rows": len(rows)}


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------

def build_task_list(seasons: list[int], rounds: list[int] | None = None) -> list[tuple[int, int, str]]:
    """Return (year, round_number, event_name) tuples for all target sessions."""
    tasks = []
    for year in seasons:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            log.error(f"Could not load schedule for {year}: {e}")
            continue

        races = schedule[schedule["EventFormat"].isin(["conventional", "sprint", "sprint_qualifying"])]
        if rounds:
            races = races[races["RoundNumber"].isin(rounds)]

        for _, event in races.iterrows():
            rnd = int(event["RoundNumber"])
            name = str(event.get("EventName", ""))
            out_path = RAW_DIR / f"{year}_R{rnd:02d}_telemetry.parquet"
            if not out_path.exists():
                tasks.append((year, rnd, name))
            else:
                log.info(f"  [{year} R{rnd:02d}] Already done — skipping")

    return tasks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(tasks: list[tuple[int, int, str]], max_workers: int = DEFAULT_WORKERS) -> None:
    if not tasks:
        log.info("Nothing to do — all sessions already ingested.")
        return

    log.info(f"Processing {len(tasks)} sessions with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_session, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception:
                task = futures[future]
                log.error(f"  [{task[0]} R{task[1]:02d}] Unhandled exception:\n{traceback.format_exc()}")
                continue

            status = result.get("status")
            label = result.get("task")
            if status == "ok":
                log.info(f"  [{label}] OK — {result['rows']} laps saved")
            elif status == "skipped":
                log.info(f"  [{label}] Skipped (already exists)")
            else:
                log.warning(f"  [{label}] {status}: {result.get('error', '')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Ingest 2023 R01 only")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    if args.test:
        log.info("--- TEST MODE: 2023 R01 only ---")
        tasks = build_task_list([2023], rounds=[1])
    else:
        tasks = build_task_list(SEASONS)

    run(tasks, max_workers=args.workers)
    log.info("Telemetry ingestion complete.")
