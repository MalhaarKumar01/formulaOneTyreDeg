# F1 TireBrain — MVP Checklist

**Goal**: Working degradation model + API in 36 hours.

---

## Phase 1 — Data Ingestion (Hours 0–6)

- [ ] Project setup: `pyproject.toml`, venv, `.gitignore`, directory structure
- [ ] `src/ingestion/fastf1_ingest.py` — lap data + weather for 2022–2025 races → `data/raw/fastf1/`
- [ ] `src/ingestion/openf1_ingest.py` — telemetry (throttle, brake, speed, DRS) → `data/raw/openf1/`
- [ ] `src/ingestion/ergast_ingest.py` — circuit + driver metadata → `data/raw/ergast/`
- [ ] Spot-check 3 sessions: lap counts correct, compounds present, telemetry non-null

> Start ingestion first. It takes ~2 hours to download. Write feature code while it runs.

---

## Phase 2 — Feature Engineering (Hours 6–16)

- [ ] `src/features/clean.py` — remove in-laps, out-laps, SC laps, yellow flag laps, wet laps
- [ ] `src/features/build_features.py`:
  - Join FastF1 laps + weather (nearest timestamp)
  - Aggregate OpenF1 telemetry to lap-level (mean throttle, mean brake, max speed, DRS %)
  - Fuel correction: `fuel_load = start_fuel - (race_lap × 1.88)`
  - Target: `lap_time_delta_fuel_corrected`
  - Rolling features: `deg_rate_last_3`, `deg_acceleration`
  - Target-encode `circuit_id` and `driver_id`
- [ ] Validate: null check, target distribution looks right (normal with right tail)

---

## Phase 3 — Model Training (Hours 16–28)

- [ ] `src/models/baseline.py` — `HuberRegressor` on `[stint_lap, compound, track_temp]`
- [ ] `src/models/train_degradation.py`:
  - Temporal split: train 2022–2024, validate first 10 races 2025, test rest of 2025
  - LightGBM with Huber loss
  - Optuna 50 trials on val MAE → save best params to `models/degradation/params.json`
  - Retrain on train+val, evaluate on test
  - Save: `models/degradation/model.lgb`
- [ ] Output metrics + plots to `metrics/`:
  - `degradation_metrics.json` (MAE, RMSE, R², baseline MAE, improvement %)
  - `shap_importance.png`
  - `predicted_vs_actual.png`
  - `residuals_by_compound.png`

---

## Phase 4 — API + Wrap (Hours 28–36)

- [ ] `src/api/main.py` (FastAPI):
  - `POST /predict/degradation` — inputs: driver, circuit, compound, stint_lap, track_temp, fuel_load → next 5 lap deltas
  - `GET /health` — model version, training date, test MAE
  - Log every prediction to `predictions/log.parquet`
- [ ] `Makefile`: `setup`, `ingest`, `features`, `train`, `serve`, `all`
- [ ] `README.md`: what it is, how to run, metrics table, sample API call
- [ ] Push public repo to GitHub

---

## Done When

- [ ] 40+ races ingested
- [ ] Feature table with all 18 features + fuel correction
- [ ] LightGBM beats linear baseline (target: >15% MAE improvement)
- [ ] SHAP computed and saved
- [ ] API serving predictions with logging
- [ ] Repo is public
