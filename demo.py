"""
demo.py — F1 Tyre Degradation Dashboard demo

Starts the API server, runs a series of illustrative API calls,
prints results, then opens the dashboard in the browser.

Usage:
    .venv/bin/python demo.py
    .venv/bin/python demo.py --no-browser   # skip auto-open
    .venv/bin/python demo.py --port 8001    # custom port
"""

import argparse
import json
import subprocess
import sys
import time
import webbrowser
from typing import Any

import requests

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--port",       type=int, default=8000)
parser.add_argument("--no-browser", action="store_true")
args = parser.parse_args()

BASE = f"http://localhost:{args.port}"

# ── ANSI helpers ─────────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
GREY   = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def banner(text: str) -> None:
    width = 62
    print()
    print(f"{BOLD}{RED}{'─' * width}{RESET}")
    print(f"{BOLD}{RED}  {text.upper()}{RESET}")
    print(f"{BOLD}{RED}{'─' * width}{RESET}")

def step(label: str) -> None:
    print(f"\n{CYAN}▶  {label}{RESET}")

def ok(label: str, value: Any = None) -> None:
    v = f"  {GREY}{value}{RESET}" if value is not None else ""
    print(f"   {GREEN}✓{RESET}  {label}{v}")

def info(label: str, value: Any = "") -> None:
    print(f"   {GREY}·{RESET}  {label}  {YELLOW}{value}{RESET}")

def err(msg: str) -> None:
    print(f"   {RED}✗  {msg}{RESET}")

# ── SERVER STARTUP ────────────────────────────────────────────────────────────
banner("F1 Tyre Degradation — Dashboard Demo")

step("Starting API server")
server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "src.api.main:app",
     "--port", str(args.port), "--log-level", "warning"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
print(f"   {GREY}PID {server.pid} · waiting for readiness...{RESET}", end="", flush=True)

for _ in range(30):
    try:
        requests.get(f"{BASE}/health", timeout=1)
        break
    except Exception:
        time.sleep(0.5)
        print(".", end="", flush=True)
else:
    print()
    err("Server did not start in time. Is port busy? Try --port 8001")
    server.terminate()
    sys.exit(1)

print(f"  {GREEN}ready{RESET}")

# ── DEMO CALLS ────────────────────────────────────────────────────────────────
try:

    # 1. Health check
    step("Health check  →  GET /health")
    h = requests.get(f"{BASE}/health").json()
    ok("Server online")
    info("Test MAE",         f"{h['test_mae']:.4f}s")
    info("Test R²",          f"{h['test_r2']:.4f}")
    info("Features",         h['n_features'])
    info("Training rows",    h['train_rows'])

    # 2. Full metrics
    step("Model metrics  →  GET /metrics")
    m = requests.get(f"{BASE}/metrics").json()
    ok("Metrics loaded")
    info("Val  MAE / R²",  f"{m['val']['mae']:.4f}s  /  {m['val']['r2']:.4f}")
    info("Test MAE / R²",  f"{m['test']['mae']:.4f}s  /  {m['test']['r2']:.4f}")
    info("Baseline MAE",   f"{m['baseline_mae']:.4f}s  →  +{m['improvement_over_baseline_pct']:.1f}% gain")

    # 3. Filters
    step("Data filters  →  GET /data/filters")
    f = requests.get(f"{BASE}/data/filters").json()
    ok("Filters loaded")
    info("Years",     f"{min(f['years'])} – {max(f['years'])}  ({len(f['years'])} seasons)")
    info("Circuits",  f"{len(f['circuits'])} events  (e.g. {f['circuits'][0]})")
    info("Drivers",   f"{len(f['drivers'])} drivers")
    info("Compounds", "  ".join(f['compounds']))

    # 4. Degradation data slice
    step("Data explorer  →  GET /data/degradation?driver=VER&compound=MEDIUM&year=2024")
    rows = requests.get(
        f"{BASE}/data/degradation",
        params={"driver": "VER", "compound": "MEDIUM", "year": 2024},
    ).json()
    ok(f"{len(rows)} laps returned for VER on MEDIUM in 2024")
    if rows:
        deltas = [r["lap_time_delta_fuel_corrected"] for r in rows
                  if r["lap_time_delta_fuel_corrected"] is not None]
        if deltas:
            info("Avg lap delta",  f"+{sum(deltas)/len(deltas):.3f}s")
            info("Max lap delta",  f"+{max(deltas):.3f}s")
            info("Sample events",  ", ".join(sorted({r["EventName"] for r in rows[:8]})))

    # 5. Prediction scenarios
    SCENARIOS = [
        {
            "label":           "Soft — fresh stint, lap 5",
            "Compound":        "SOFT",
            "stint_lap_number": 5,
            "Stint":           2,
            "TrackTemp":       48.0,
            "AirTemp":         30.0,
            "fuel_load_kg":    60.0,
            "avg_throttle_pct": 65.0,
            "full_throttle_pct": 58.0,
            "avg_brake":       0.17,
            "braking_pct":     13.0,
            "max_speed_kph":   318.0,
            "drs_active_pct":  20.0,
            "avg_rpm":         11400.0,
            "track_evolution": 300,
            "deg_rate_last_3": 0.04,
            "deg_acceleration": 0.01,
            "sector_1_pct":    0.28,
            "sector_2_pct":    0.38,
            "sector_3_pct":    0.34,
        },
        {
            "label":           "Medium — mid stint, lap 18",
            "Compound":        "MEDIUM",
            "stint_lap_number": 18,
            "Stint":           2,
            "TrackTemp":       44.0,
            "AirTemp":         28.0,
            "fuel_load_kg":    45.0,
            "avg_throttle_pct": 62.0,
            "full_throttle_pct": 55.0,
            "avg_brake":       0.18,
            "braking_pct":     12.0,
            "max_speed_kph":   315.0,
            "drs_active_pct":  18.0,
            "avg_rpm":         11200.0,
            "track_evolution": 600,
            "deg_rate_last_3": 0.12,
            "deg_acceleration": 0.02,
            "sector_1_pct":    0.28,
            "sector_2_pct":    0.38,
            "sector_3_pct":    0.34,
        },
        {
            "label":           "Soft — cliff zone, lap 28",
            "Compound":        "SOFT",
            "stint_lap_number": 28,
            "Stint":           1,
            "TrackTemp":       52.0,
            "AirTemp":         34.0,
            "fuel_load_kg":    30.0,
            "avg_throttle_pct": 67.0,
            "full_throttle_pct": 60.0,
            "avg_brake":       0.21,
            "braking_pct":     15.0,
            "max_speed_kph":   320.0,
            "drs_active_pct":  22.0,
            "avg_rpm":         11600.0,
            "track_evolution": 900,
            "deg_rate_last_3": 0.38,
            "deg_acceleration": 0.09,
            "sector_1_pct":    0.28,
            "sector_2_pct":    0.38,
            "sector_3_pct":    0.34,
        },
        {
            "label":           "Hard — long stint, lap 35",
            "Compound":        "HARD",
            "stint_lap_number": 35,
            "Stint":           3,
            "TrackTemp":       38.0,
            "AirTemp":         22.0,
            "fuel_load_kg":    20.0,
            "avg_throttle_pct": 60.0,
            "full_throttle_pct": 52.0,
            "avg_brake":       0.15,
            "braking_pct":     10.0,
            "max_speed_kph":   312.0,
            "drs_active_pct":  16.0,
            "avg_rpm":         10900.0,
            "track_evolution": 1200,
            "deg_rate_last_3": 0.07,
            "deg_acceleration": 0.005,
            "sector_1_pct":    0.28,
            "sector_2_pct":    0.38,
            "sector_3_pct":    0.34,
        },
    ]

    step("Live predictions  →  POST /predict")
    for s in SCENARIOS:
        label = s.pop("label")
        r = requests.post(f"{BASE}/predict", json=s).json()
        delta = r["predicted_delta"]
        if delta < 0.1:
            colour, tag = GREEN,  "NOMINAL     "
        elif delta < 0.3:
            colour, tag = YELLOW, "DEGRADING   "
        else:
            colour, tag = RED,    "CLIFF RISK  "
        sign = "+" if delta >= 0 else ""
        print(f"   {colour}{tag}{RESET}  {label:<38}  {colour}{BOLD}{sign}{delta:.3f}s{RESET}")

    # 6. Plot availability
    step("Plot endpoints  →  GET /plots/<name>")
    plots = ["shap_importance", "residuals_by_compound",
             "predicted_vs_actual", "shap_waterfall_sample"]
    for name in plots:
        resp = requests.get(f"{BASE}/plots/{name}")
        size = len(resp.content) // 1024
        ok(f"{name}", f"{size} KB")

    # 7. Open browser
    banner("All checks passed")
    print(f"   {BOLD}Dashboard:{RESET}  {CYAN}{BASE}/{RESET}")
    print(f"   {BOLD}API docs: {RESET}  {CYAN}{BASE}/docs{RESET}")
    print()

    if not args.no_browser:
        step("Opening dashboard in browser")
        time.sleep(0.5)
        webbrowser.open(BASE)
        ok("Browser launched")

    print(f"\n{GREY}   Press Ctrl+C to stop the server.{RESET}\n")
    server.wait()

except KeyboardInterrupt:
    print(f"\n{GREY}   Shutting down...{RESET}")
finally:
    server.terminate()
