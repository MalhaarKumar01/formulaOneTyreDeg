# F1 TireBrain — Tire Degradation Predictor

## Product Requirements Document — 72-Hour Build

---

## 1. What This Is

A machine learning system that predicts Formula 1 tire degradation in real time — given a driver, compound, circuit, and current stint progress, it forecasts how much lap time they'll lose over their remaining stint and flags when they're approaching a tire cliff.

This is not a dashboard project. This is not a visualization exercise. This is a **data pipeline → feature engineering → model training → serving** system built the way an ML engineer at a team like McLaren Applied or AWS F1 Insights would think about it.

---

## 2. Why This Gets You Hired

Most student ML projects fail the interview in the same way: they download a Kaggle dataset, train a model, report accuracy, done. That tells an interviewer nothing about whether you can do ML *at work*. Here's what this project demonstrates that those don't:

**Data Engineering Competence**: You're not handed a CSV. You ingest from multiple APIs, handle schema mismatches, deal with missing telemetry, join across time-misaligned sources, and build a reproducible feature pipeline. This is 70% of what ML engineers actually do.

**Domain-Informed Feature Engineering**: The fuel correction alone separates you from everyone else. Knowing that raw lap times are contaminated by ~3.5s of fuel effect and building a proper correction shows you understand the problem, not just the tools.

**Rigorous Evaluation**: Walk-forward temporal validation, not random splits. Baseline comparison against a naive linear model. If the ML model doesn't meaningfully beat the baseline, you say so — that honesty is more impressive than inflated metrics.

**Production Thinking**: Model versioning, prediction logging, an API that serves predictions — not just a notebook. You can say "here's the endpoint, here's the latency, here's how I'd monitor drift."

**Anomaly Detection**: The cliff detector is a two-stage unsupervised→supervised pipeline. Changepoint detection with PELT, then a binary classifier on the labels it generates. This is architecturally more interesting than anything a single-model project can show.

---

## 3. Data Sources

| Source | What You Pull | Access |
|--------|---------------|--------|
| **FastF1** (Python library) | Lap times, sector splits, tire compound, stint info, weather (air/track temp, rain), session metadata | `pip install fastf1`, free |
| **OpenF1 API** | Car telemetry: throttle %, brake %, speed, DRS, RPM at sub-second granularity | REST, free, no auth |
| **Ergast API (archived)** | Historical results, circuit metadata, driver/constructor IDs (1950–2024) | REST or static JSON dump |

### Critical Data Relationships

- FastF1 gives you **lap-level** data (one row per driver per lap)
- OpenF1 gives you **sample-level** telemetry (~3.7Hz, hundreds of rows per lap)
- You must aggregate OpenF1 telemetry to lap-level to join with FastF1
- Weather data from FastF1 is timestamped — join to laps by nearest timestamp, not exact match

---

## 4. Architecture

```
 ┌──────────────────────────────────────────────────────┐
 │                   DATA INGESTION                     │
 │                                                      │
 │  fastf1_ingest.py ──┐                                │
 │  openf1_ingest.py ──┼──▶ Raw Parquet files           │
 │  ergast_ingest.py ──┘    (data/raw/{source}/)        │
 └───────────────────────────┬──────────────────────────┘
                             │
                             ▼
 ┌──────────────────────────────────────────────────────┐
 │               FEATURE ENGINEERING                    │
 │                                                      │
 │  Fuel correction ──┐                                 │
 │  Telemetry agg   ──┼──▶ Feature tables (Parquet)     │
 │  Weather join    ──┤    (data/features/)             │
 │  Stint detection ──┘                                 │
 │                                                      │
 │  Output: one row per driver per lap with all features │
 └───────────────────────────┬──────────────────────────┘
                             │
                             ▼
 ┌──────────────────────────────────────────────────────┐
 │                 MODEL TRAINING                       │
 │                                                      │
 │  Degradation Model (LightGBM regressor)              │
 │    - Target: fuel-corrected lap time delta            │
 │    - Temporal split: train 2019-2024, test 2025      │
 │    - Baseline: linear deg = α × stint_lap            │
 │                                                      │
 │  Cliff Detector (two-stage)                          │
 │    - Stage 1: PELT changepoint on stint time series   │
 │    - Stage 2: LightGBM classifier on labeled stints  │
 │                                                      │
 │  Artifacts: models/, metrics/, SHAP plots            │
 └───────────────────────────┬──────────────────────────┘
                             │
                             ▼
 ┌──────────────────────────────────────────────────────┐
 │                  SERVING LAYER                       │
 │                                                      │
 │  FastAPI:                                            │
 │    POST /predict/degradation                         │
 │    POST /predict/cliff-risk                          │
 │    GET  /drivers/{id}/stint-history                  │
 │    GET  /health                                      │
 │                                                      │
 │  + Prediction logging to predictions/ (Parquet)      │
 └──────────────────────────────────────────────────────┘
```

**No Docker. No Kubernetes. No overengineering.** Local Python, virtual environment, Parquet files on disk, FastAPI server. If an interviewer asks why: "I optimized for iteration speed on a 72-hour build. Containerization is a deployment concern — the ML engineering is the hard part."

---

## 5. Feature Specification

This is the most important section of the entire PRD. Features make or break the model.

### 5.1 Target Variable

`lap_time_delta_fuel_corrected`: The difference between a driver's lap N time and their first clean flying lap on that stint, **after subtracting the fuel effect**.

```
fuel_load_kg = start_fuel - (race_lap_number × 1.88)
fuel_time_effect = fuel_load_kg × 0.035  # ~0.035s per kg, widely accepted estimate
corrected_lap_time = raw_lap_time + fuel_time_effect  # Adding because less fuel = faster
lap_time_delta = corrected_lap_time_N - corrected_lap_time_lap1
```

If you skip this correction, your model will learn that "cars get faster over a stint" (because fuel burns off) and attribute it to tire behavior. This is the single most common mistake in F1 ML projects.

### 5.2 Feature Table

| Feature | Type | Source | Engineering Notes |
|---------|------|--------|-------------------|
| `stint_lap_number` | int | FastF1 | Lap count within current stint (resets after pit stop). Primary degradation driver. |
| `compound` | categorical | FastF1 | One-hot encode: SOFT, MEDIUM, HARD, INTERMEDIATE, WET. Each has fundamentally different deg curves. |
| `fuel_load_kg` | float | Derived | `start_fuel - (race_lap × 1.88)`. Even with the correction on the target, include as feature — fuel affects car balance and thus *how* tires wear. |
| `track_temp_c` | float | FastF1 weather | Nearest-timestamp join. High temp → thermal deg on softs. Most important weather feature. |
| `air_temp_c` | float | FastF1 weather | Less impactful than track temp but affects engine cooling and thus power deployment. |
| `rainfall` | boolean | FastF1 weather | Binary. When true, everything changes — flag but don't train the dry model on wet laps. |
| `avg_throttle_pct` | float | OpenF1 telemetry | Mean throttle application over the lap. High throttle = rear tire abuse. Aggregate from ~3.7Hz samples. |
| `avg_brake_pressure` | float | OpenF1 telemetry | Mean brake input. Heavy braking = front tire thermal stress. |
| `max_speed_kph` | float | OpenF1 telemetry | Peak speed per lap — proxy for low-downforce vs high-downforce configuration. |
| `drs_laps_pct` | float | OpenF1 telemetry | Fraction of lap distance with DRS open. DRS laps are artificially fast — must control for this. |
| `circuit_id` | categorical | Ergast | Target-encode or embed. Each circuit has unique surface, abrasiveness, corner profile. |
| `driver_id` | categorical | Ergast | Some drivers are consistently harder on tires (driving style). Target-encode. |
| `stint_number` | int | FastF1 | 1st stint (heavy fuel) vs 3rd stint (light fuel, worn track) behave differently. |
| `track_evolution` | float | Derived | Total number of laps completed in the session so far, across all cars. More rubber on track = more grip. |
| `sector_1_pct` / `sector_2_pct` / `sector_3_pct` | float | FastF1 | Sector time as fraction of total lap. Shifts reveal where the tires are failing (front-limited sectors vs rear-limited). |
| `gap_to_car_ahead_s` | float | OpenF1 positions | Critical: dirty air from car ahead increases tire temps by ~5-10°C. Laps in dirty air degrade tires faster. Must include or the model blames the tires for aero-induced deg. |
| `deg_rate_last_3` | float | Derived | Rolling mean of lap_time_delta over last 3 laps. Trend feature — is degradation accelerating? |
| `deg_acceleration` | float | Derived | Second derivative: `deg_rate_last_3[N] - deg_rate_last_3[N-1]`. Positive = approaching cliff. |

### 5.3 Features You Explicitly Do NOT Include

| Excluded | Why |
|----------|-----|
| Tire surface temperature | Not available in public data. Teams have this; you don't. Don't pretend. |
| Tire pressure | Same — proprietary sensor data. |
| Setup data (wing angles, ride height) | Not public. The `max_speed` and corner-speed proxies approximate this. |
| Safety car periods | Remove these laps from training entirely. SC laps have artificially slow times that corrupt the target variable. |
| Formation laps, in-laps, out-laps | Remove. Non-representative lap times. |
| Laps with yellow flags | Remove. Localized yellows slow sector times unpredictably. |

---

## 6. Model Specifications

### 6.1 Degradation Model

| Aspect | Specification |
|--------|---------------|
| Algorithm | LightGBM Regressor |
| Target | `lap_time_delta_fuel_corrected` |
| Loss | Huber loss (robust to outlier laps — debris, mistakes, blue flags) |
| Validation | Temporal walk-forward: train on seasons 2019–2024, validate on first 10 races of 2025, test on remaining 2025 |
| Baseline | Linear model: `deg = α × stint_lap + β × compound + γ × track_temp`. If LightGBM doesn't beat this by >15% MAE, the complexity isn't justified. |
| Hyperparameter tuning | Optuna, 50 trials, optimizing validation MAE. Key params: `num_leaves`, `learning_rate`, `min_child_samples`, `feature_fraction`. |
| Explainability | SHAP values computed on test set. Global feature importance plot + individual prediction waterfall charts. |

### 6.2 Cliff Detector

| Aspect | Specification |
|--------|---------------|
| Stage 1 | PELT changepoint detection (`ruptures` library) on each stint's lap time series. Penalty tuned to detect 0-1 changepoints per stint. |
| Labeling | A cliff is a changepoint where post-change slope is >2× pre-change slope and magnitude >0.5s/lap jump. ~12-18% of stints have cliffs. |
| Stage 2 Algorithm | LightGBM Binary Classifier |
| Target | `cliff_in_next_5_laps` (boolean) |
| Key features | `deg_acceleration`, `stint_lap_number`, `compound`, `track_temp`, `avg_throttle_pct` |
| Class imbalance | Focal loss (preferred over SMOTE — SMOTE creates synthetic F1 laps that don't physically make sense) |
| Evaluation | Precision-Recall AUC (not ROC-AUC — the dataset is imbalanced, ROC is misleading). Target: PR-AUC > 0.45. Also report precision@recall=0.5 — if you catch 50% of cliffs, how often are you right? |
| False positive cost | A false cliff alert means pitting too early (lose ~2-3s of unnecessary pit delta). A missed cliff means losing 2-4s/lap for potentially 3-5 laps. The cost is asymmetric — tune the threshold to favor recall. |

---

## 7. MVP — First 36 Hours

This is the hard sprint. At the end of 36 hours you have: raw data ingested, features built, one model trained, one model evaluated, and a notebook showing it works.

### Hour 0–6: Data Ingestion

**Deliverables**: Raw data on disk for 40+ races (2022–2025 era).

- [ ] Set up project: `pyproject.toml`, virtual environment, `.gitignore`, directory structure
- [ ] Write `src/ingestion/fastf1_ingest.py` — loop over 2022–2025 race sessions, pull lap data + weather, save as Parquet per session to `data/raw/fastf1/`
- [ ] Write `src/ingestion/openf1_ingest.py` — pull car telemetry (throttle, brake, speed, DRS) per session, save as Parquet
- [ ] Write `src/ingestion/ergast_ingest.py` — pull circuit metadata, driver/constructor mappings, save as Parquet
- [ ] Handle FastF1 caching (it has built-in caching — use it, saves re-downloading)
- [ ] Validate: spot-check 3 random sessions — are lap counts correct? Do compound labels exist? Is telemetry non-null?

**Watch out for**:
- FastF1 takes 30-60s per session to load. 40+ races × 5 sessions = plan for ~2 hours of download time. Start this FIRST and work on other code while it runs.
- Some sessions have missing telemetry (sprint races pre-2023, rain-cancelled sessions). Log these and skip gracefully.
- OpenF1 rate limits: add 0.5s sleep between requests.

### Hour 6–16: Feature Engineering

**Deliverables**: A single feature table (Parquet) with one row per driver per lap, all features populated.

- [ ] Write `src/features/build_features.py`:
  - Join FastF1 laps with weather data (nearest timestamp join)
  - Aggregate OpenF1 telemetry from sample-level to lap-level (mean throttle, mean brake, max speed, DRS fraction)
  - Join aggregated telemetry to lap table on (session, driver, lap_number)
  - Compute fuel correction: `fuel_load = start_fuel - (race_lap × 1.88)`, apply to lap times
  - Compute target: `lap_time_delta_fuel_corrected`
  - Compute rolling features: `deg_rate_last_3`, `deg_acceleration`
  - Compute `gap_to_car_ahead_s` from position/timing data
  - Compute `track_evolution` (cumulative laps in session)
  - Encode categoricals: target-encode `circuit_id` and `driver_id` (use category_encoders library, fit on train only)

- [ ] Write `src/features/clean.py`:
  - Remove in-laps, out-laps, SC laps, formation laps, yellow flag laps
  - Remove laps with pit stop (FastF1 flags these)
  - Remove wet laps from the dry-condition training set (flag `rainfall == True`)
  - Remove laps with telemetry null rate >30%
  - Log how many laps removed and why — put this in the README later

- [ ] Validate: feature table shape, null check, distribution plots for target variable. Does the target look approximately normal with a right tail? It should.

**This is the hardest phase.** Data joining across sources with different schemas and timestamps is where most projects die. Expect 60% of your debugging time here.

### Hour 16–28: Model Training

**Deliverables**: Trained degradation model, evaluation metrics, SHAP analysis, comparison to baseline.

- [ ] Write `src/models/train_degradation.py`:
  - Load feature table
  - Temporal split: train = 2022–2024 seasons, val = first 10 races 2025, test = remaining 2025
  - Train linear baseline: `sklearn.linear_model.HuberRegressor` on `[stint_lap, compound_encoded, track_temp]`
  - Train LightGBM with Huber loss
  - Run Optuna (50 trials) optimizing val MAE — save best params to `models/degradation/params.json`
  - Retrain on train+val with best params, evaluate on test
  - Save model to `models/degradation/model.lgb`
  - Generate and save:
    - `metrics/degradation_metrics.json` (MAE, RMSE, R², baseline MAE, improvement %)
    - `metrics/shap_importance.png` (global feature importance)
    - `metrics/shap_waterfall_sample.png` (single prediction breakdown)
    - `metrics/predicted_vs_actual.png` (scatter plot)
    - `metrics/residuals_by_compound.png` (are errors consistent across tire types?)

- [ ] Write `src/models/baseline.py`:
  - Clean linear baseline implementation
  - This exists so you can say: "My model beat the linear baseline by X%. Here's why that matters."

- [ ] Validate: Does the model beat the baseline? If MAE improvement <15%, investigate — probably a feature engineering issue, not a model issue.

### Hour 28–36: MVP Wrap

**Deliverables**: Working API, prediction logging, README that explains everything.

- [ ] Write `src/api/main.py` (FastAPI):
  - `POST /predict/degradation` — accepts driver, circuit, compound, stint_lap, track_temp, fuel_load; returns predicted delta for next 5 laps
  - `GET /health` — returns model version, training date, test MAE
  - Prediction logging: every prediction saved to `predictions/log.parquet` with timestamp, inputs, outputs
  - Error handling: return sensible errors for unknown circuits/drivers

- [ ] Write `README.md`:
  - What this is (2 sentences)
  - Architecture diagram (copy from this PRD)
  - How to run (`make setup && make ingest && make train && make serve`)
  - Model performance: table of metrics, SHAP plot embedded
  - Sample API call + response
  - What you'd do next (links to post-MVP section)

- [ ] Write `Makefile`:
  - `make setup` — create venv, install deps
  - `make ingest` — run all ingestion scripts
  - `make features` — run feature pipeline
  - `make train` — train model
  - `make serve` — start FastAPI
  - `make all` — full pipeline end to end

- [ ] First commit to GitHub. Repo must be public.

---

## 8. Post-MVP — Hours 36–72

You have a working degradation model. Now you add the things that elevate this from "good student project" to "this person thinks like an ML engineer."

### 8.1 Cliff Detection Model (Hours 36–48)

- [ ] Write `src/models/cliff_labeler.py`:
  - For each stint in the dataset, extract the lap time series
  - Run PELT changepoint detection with `ruptures` (penalty=3, model="rbf")
  - Label stints: `cliff_detected`, `cliff_lap`, `cliff_magnitude`
  - Save labeled dataset to `data/features/cliff_labels.parquet`
  - Generate distribution plot: what % of stints have cliffs, by compound, by circuit?

- [ ] Write `src/models/train_cliff.py`:
  - Build cliff feature set: `deg_rate_last_3`, `deg_acceleration`, `stint_lap`, `compound`, `track_temp`, `avg_throttle_pct`, `gap_to_car_ahead`
  - Train LightGBM classifier with focal loss
  - Evaluate: PR-AUC, precision@recall=0.5, confusion matrix
  - SHAP analysis on cliff predictions
  - Save model to `models/cliff/model.lgb`

- [ ] Add `POST /predict/cliff-risk` endpoint to API
  - Input: current stint data
  - Output: cliff probability, risk level (LOW/ELEVATED/CRITICAL), recommended action

- [ ] Write `notebooks/cliff_analysis.ipynb`:
  - Show example stints with detected cliffs (annotated time series plot)
  - Show SHAP waterfall for a correct cliff prediction
  - Show a false positive and explain why the model was wrong

### 8.2 Prediction Monitoring & Drift Detection (Hours 48–56)

This is what separates you from every other student project. Nobody does this, and every ML job asks about it.

- [ ] Write `src/monitoring/drift.py`:
  - Load prediction logs from `predictions/log.parquet`
  - Compare feature distributions at training time vs. prediction time using PSI (Population Stability Index)
  - Flag features where PSI > 0.2 (significant drift)
  - Generate drift report: `metrics/drift_report.json`

- [ ] Write `src/monitoring/performance.py`:
  - For predictions where ground truth is now available (race has finished), compute actual vs predicted
  - Track rolling MAE over time — is the model degrading?
  - Generate `metrics/rolling_performance.png`

- [ ] Add `GET /monitoring/drift` endpoint — returns current drift status
- [ ] Add `GET /monitoring/performance` endpoint — returns rolling accuracy metrics

### 8.3 Backtesting Framework (Hours 56–64)

Prove the model works across different conditions, not just on a single test set.

- [ ] Write `src/evaluation/backtest.py`:
  - Walk-forward validation: for each race weekend in 2025, train on all prior data, predict that weekend, record metrics
  - Output: per-race MAE table, overall MAE, variance across races
  - Identify failure modes: which circuits/compounds/conditions does the model struggle with?
  - Save `metrics/backtest_results.csv` and `metrics/backtest_by_circuit.png`

- [ ] Write `notebooks/backtest_analysis.ipynb`:
  - Visualize: actual vs predicted degradation curves for 3 interesting races (one where model nailed it, one where it struggled, one wet race)
  - Discuss why the model failed where it did — this is interview gold

### 8.4 Data Validation & Pipeline Robustness (Hours 64–72)

- [ ] Write `src/validation/data_checks.py` using `great_expectations` or manual assertions:
  - Lap times between 60s and 180s
  - Stint lap numbers are sequential with no gaps
  - Compound values are in allowed set
  - Telemetry values physically plausible (throttle 0-100, speed 0-370)
  - No duplicate (session, driver, lap) combinations
  - Fuel load never negative

- [ ] Write `tests/`:
  - `test_features.py` — unit tests for fuel correction math, rolling feature computation
  - `test_api.py` — integration tests for prediction endpoints
  - `test_model.py` — does the model load? Does it predict in <100ms? Is output shape correct?

- [ ] Add GitHub Actions CI:
  - Run tests on every push
  - Lint with ruff
  - Type check with mypy on critical modules

- [ ] Final README polish:
  - Add backtest results table
  - Add drift monitoring explanation
  - Add "Lessons Learned" section — what surprised you, what would you change
  - Add architecture diagram (proper one, not ASCII)

---

## 9. Repo Structure

```
f1-tirebrain/
├── README.md
├── PRD.md
├── Makefile
├── pyproject.toml
├── .gitignore
│
├── src/
│   ├── ingestion/
│   │   ├── fastf1_ingest.py
│   │   ├── openf1_ingest.py
│   │   └── ergast_ingest.py
│   │
│   ├── features/
│   │   ├── build_features.py
│   │   └── clean.py
│   │
│   ├── models/
│   │   ├── baseline.py
│   │   ├── train_degradation.py
│   │   ├── train_cliff.py
│   │   └── cliff_labeler.py
│   │
│   ├── evaluation/
│   │   └── backtest.py
│   │
│   ├── monitoring/
│   │   ├── drift.py
│   │   └── performance.py
│   │
│   ├── validation/
│   │   └── data_checks.py
│   │
│   └── api/
│       ├── main.py
│       ├── routes.py
│       └── schemas.py
│
├── models/
│   ├── degradation/
│   │   ├── model.lgb
│   │   └── params.json
│   └── cliff/
│       ├── model.lgb
│       └── params.json
│
├── data/
│   ├── raw/
│   │   ├── fastf1/
│   │   ├── openf1/
│   │   └── ergast/
│   └── features/
│       ├── feature_table.parquet
│       └── cliff_labels.parquet
│
├── metrics/
│   ├── degradation_metrics.json
│   ├── shap_importance.png
│   ├── predicted_vs_actual.png
│   ├── residuals_by_compound.png
│   ├── backtest_results.csv
│   ├── backtest_by_circuit.png
│   ├── drift_report.json
│   └── rolling_performance.png
│
├── predictions/
│   └── log.parquet
│
├── notebooks/
│   ├── eda.ipynb
│   ├── cliff_analysis.ipynb
│   └── backtest_analysis.ipynb
│
├── tests/
│   ├── test_features.py
│   ├── test_api.py
│   └── test_model.py
│
└── .github/
    └── workflows/
        └── ci.yml
```

---

## 10. Tech Stack

| Component | Tool | Why This and Not Something Else |
|-----------|------|---------------------------------|
| Language | Python 3.11+ | Obviously |
| Data format | Parquet (via pyarrow) | Columnar, compressed, typed. Not CSV — CSVs lose type info and are slow. If an interviewer asks: "Parquet preserves schema, supports predicate pushdown, and compresses 5-10× vs CSV." |
| Feature engineering | Pandas + NumPy | Sufficient at this data scale (~500K laps). No need for Spark/Polars. |
| ML training | LightGBM | Faster than XGBoost on this data size, native categorical support, Huber loss built in. |
| Hyperparameter tuning | Optuna | Better than GridSearch (Bayesian), better than manual (reproducible). |
| Changepoint detection | ruptures | Clean PELT implementation, no heavy dependencies. |
| Explainability | SHAP | Industry standard. If you can't explain why the model predicted something, you can't ship it. |
| Categorical encoding | category_encoders | Target encoding for high-cardinality categoricals (circuit, driver). |
| API | FastAPI | Auto-docs, async, type-validated requests. |
| Data validation | great_expectations or pandera | Schema + value assertions on every pipeline run. |
| Testing | pytest | Standard. |
| CI | GitHub Actions | Free, runs on push. |
| Experiment tracking | MLflow (local) or just JSON + Parquet | MLflow if you want the UI. Flat files if you want speed. Either is fine — what matters is that experiments are tracked, not the tool. |

---

## 11. What You Say in Interviews

This section exists because the project only matters if you can talk about it. Prepare for these questions:

**"Walk me through the project."**
"I built an ML system that predicts F1 tire degradation. The hardest part wasn't the model — it was the feature engineering. F1 lap times are contaminated by fuel burn, dirty air effects, and DRS, so I had to build corrections for all three before the model could learn actual tire behavior. The model beats a linear baseline by [X]% on fuel-corrected MAE."

**"Why LightGBM and not deep learning?"**
"The dataset is ~500K rows of tabular features with mixed types. Tree-based models dominate tabular data at this scale — there's well-established benchmarking literature on this. A transformer would need orders of magnitude more data and training time with no expected accuracy gain."

**"How do you handle model drift?"**
"I log every prediction with its input features. I compute PSI between the training distribution and recent predictions to detect feature drift. I also track rolling MAE against ground truth as races complete. If drift exceeds thresholds, it triggers a retrain. I can show you the monitoring dashboard."

**"What would you do differently with more time?"**
"Three things: first, incorporate position-level data to model undercut/overcut strategy outcomes, not just tire deg. Second, build a proper feature store with Feast so features are versioned and reusable. Third, deploy to a cloud endpoint with A/B testing between model versions."

**"What's the cliff detector and why is it separate?"**
"Degradation and cliffs are fundamentally different phenomena. Degradation is smooth and predictable — a regression problem. Cliffs are sudden phase transitions — an anomaly detection problem. I used PELT changepoint detection to label cliffs in historical data, then trained a classifier to predict them before they happen. The key feature is the second derivative of lap time — if the rate of degradation is accelerating, a cliff is imminent."

---

## 12. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| FastF1 download takes too long | Delays everything | Start ingestion immediately. Work on feature code while data downloads. Cache aggressively. |
| OpenF1 telemetry missing for some sessions | Feature nulls | Make telemetry features optional. Train a model variant without them to measure their marginal value. |
| Fuel correction constant (1.88 kg/lap) is approximate | Target noise | It's the widely accepted estimate. Sensitivity analysis: try 1.8 and 1.95, see if model metrics change significantly. If they don't, the model is robust. |
| Model doesn't beat linear baseline | Ego | This is actually a valid finding. Report it honestly. "The relationship is predominantly linear" is a conclusion, not a failure. Investigate which *subsets* benefit from ML (e.g., high track temp, degradation on softs). |
| Cliff detector has too many false positives | Reduced trust | Tune threshold toward precision at the cost of recall. A system that cries wolf is worse than no system. |
| Scope creep (dashboard, frontend, more models) | Time waste | This PRD is your scope contract. If it's not in here, it doesn't get built in 72 hours. |

---

## 13. Definition of Done

### MVP (36 hours)
- [ ] 40+ races ingested (2022–2025)
- [ ] Feature table built with fuel correction, telemetry aggregation, all 18 features
- [ ] LightGBM degradation model trained, evaluated, SHAP computed
- [ ] Linear baseline comparison documented
- [ ] FastAPI serving predictions
- [ ] Prediction logging active
- [ ] README explains the project, shows metrics, includes sample API call
- [ ] Repo is public on GitHub

### Complete (72 hours)
- [ ] Cliff detection model trained and serving
- [ ] Cliff analysis notebook with annotated examples
- [ ] Drift monitoring endpoints live
- [ ] Walk-forward backtest completed with per-race results
- [ ] Data validation checks in pipeline
- [ ] Unit and integration tests passing
- [ ] GitHub Actions CI running
- [ ] README is portfolio-ready: architecture diagram, metrics tables, lessons learned

---

## 14. Time Budget Reality Check

| Phase | Estimated | Actual Risk |
|-------|-----------|-------------|
| Data ingestion | 6 hours | Could be 8+ if APIs are slow. Start first. |
| Feature engineering | 10 hours | This is where bugs live. Budget extra. |
| Model training | 8 hours | Fast if features are clean. Optuna takes ~30min. |
| API + MVP wrap | 6 hours | Straightforward if model works. |
| Cliff detector | 8 hours | PELT tuning is fiddly. |
| Monitoring + drift | 6 hours | Mostly boilerplate once you have prediction logs. |
| Backtesting | 6 hours | Computationally slow (retrains per race). Let it run overnight. |
| Validation + tests + CI | 6 hours | Tedious but essential. Don't skip. |
| README + polish | 4 hours | Non-negotiable. A project without docs doesn't exist. |

**Total: ~60 working hours across 72 calendar hours.** This assumes you sleep. You should sleep. Debugging on no sleep produces negative productivity after hour 20.