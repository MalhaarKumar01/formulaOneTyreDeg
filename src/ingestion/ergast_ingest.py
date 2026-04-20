"""
Ergast/Jolpica ingestion: circuit metadata and driver mappings.
Ergast was retired end of 2024; Jolpica (api.jolpi.ca) is the drop-in replacement.
Saves to data/raw/ergast/.
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/ergast")
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.jolpi.ca/ergast/f1"  # Jolpica — Ergast drop-in replacement
REQUEST_DELAY = 0.3
SEASONS = [2022, 2023, 2024, 2025]


def _get_json(url: str, params: dict | None = None) -> dict:
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.json()
        except requests.RequestException as e:
            log.warning(f"Request failed (attempt {attempt + 1}/3): {url} — {e}")
            time.sleep(2**attempt)
    return {}


def _paginate(url: str, table_key: str, item_key: str, limit: int = 100) -> list[dict]:
    """Fetch all pages from a Jolpica endpoint."""
    results = []
    offset = 0
    while True:
        data = _get_json(url, params={"limit": limit, "offset": offset})
        table = data.get("MRData", {}).get(table_key, {})
        items = table.get(item_key, [])
        results.extend(items)
        total = int(data.get("MRData", {}).get("total", 0))
        offset += limit
        if offset >= total:
            break
    return results


def ingest_circuits() -> pd.DataFrame:
    out_path = RAW_DIR / "circuits.parquet"
    if out_path.exists():
        log.info("Circuits already ingested.")
        return pd.read_parquet(out_path)

    log.info("Fetching circuits...")
    circuits = _paginate(f"{BASE_URL}/circuits.json", "CircuitTable", "Circuits")

    rows = [
        {
            "circuit_id": c["circuitId"],
            "circuit_name": c["circuitName"],
            "locality": c.get("Location", {}).get("locality"),
            "country": c.get("Location", {}).get("country"),
            "lat": float(c.get("Location", {}).get("lat", 0)),
            "lon": float(c.get("Location", {}).get("long", 0)),
        }
        for c in circuits
    ]
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    log.info(f"Saved {len(df)} circuits → {out_path}")
    return df


def ingest_drivers(seasons: list[int] = SEASONS) -> pd.DataFrame:
    out_path = RAW_DIR / "drivers.parquet"
    if out_path.exists():
        log.info("Drivers already ingested.")
        return pd.read_parquet(out_path)

    log.info("Fetching drivers...")
    seen: dict[str, dict] = {}
    for year in seasons:
        drivers = _paginate(f"{BASE_URL}/{year}/drivers.json", "DriverTable", "Drivers")
        for d in drivers:
            did = d["driverId"]
            if did not in seen:
                seen[did] = {
                    "driver_id": did,
                    "driver_code": d.get("code"),
                    "given_name": d.get("givenName"),
                    "family_name": d.get("familyName"),
                    "nationality": d.get("nationality"),
                    "permanent_number": d.get("permanentNumber"),
                }

    df = pd.DataFrame(list(seen.values()))
    df.to_parquet(out_path, index=False)
    log.info(f"Saved {len(df)} drivers → {out_path}")
    return df


def ingest_race_results(seasons: list[int] = SEASONS) -> pd.DataFrame:
    out_path = RAW_DIR / "race_results.parquet"
    if out_path.exists():
        log.info("Race results already ingested.")
        return pd.read_parquet(out_path)

    log.info("Fetching race results...")
    rows = []
    for year in seasons:
        races = _paginate(f"{BASE_URL}/{year}/results.json", "RaceTable", "Races", limit=100)
        for race in races:
            circuit_id = race.get("Circuit", {}).get("circuitId")
            round_num = int(race.get("round", 0))
            for result in race.get("Results", []):
                rows.append(
                    {
                        "year": year,
                        "round": round_num,
                        "circuit_id": circuit_id,
                        "driver_id": result.get("Driver", {}).get("driverId"),
                        "constructor_id": result.get("Constructor", {}).get("constructorId"),
                        "grid": int(result.get("grid", 0)),
                        "position": result.get("position"),
                        "points": float(result.get("points", 0)),
                        "status": result.get("status"),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    log.info(f"Saved {len(df)} race result rows → {out_path}")
    return df


if __name__ == "__main__":
    ingest_circuits()
    ingest_drivers()
    ingest_race_results()
    log.info("Ergast/Jolpica ingestion complete.")
