.PHONY: setup ingest ingest-test telemetry telemetry-test features features-test train train-test serve all

setup:
	python -m venv .venv
	.venv/bin/pip install -e .

# --- Ingestion ---

ingest:
	.venv/bin/python -m src.ingestion.fastf1_ingest

ingest-test:
	.venv/bin/python -m src.ingestion.fastf1_ingest --test

telemetry:
	.venv/bin/python -m src.ingestion.telemetry_ingest --workers 4

telemetry-test:
	.venv/bin/python -m src.ingestion.telemetry_ingest --test --workers 1

# --- Features ---

features:
	.venv/bin/python -m src.features.build_features

features-test:
	.venv/bin/python -m src.features.build_features --test

# --- Model ---

train:
	.venv/bin/python -m src.models.train_degradation

train-test:
	.venv/bin/python -m src.models.train_degradation --test

# --- API ---

serve:
	.venv/bin/uvicorn src.api.main:app --reload --port 8000

# --- Full pipeline ---

all: ingest telemetry features train
