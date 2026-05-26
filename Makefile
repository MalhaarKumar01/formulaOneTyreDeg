.PHONY: setup ingest ingest-test telemetry telemetry-test features features-test train train-test train-lstm train-lstm-test serve all embedded bench mqtt-bridge

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

train-lstm:
	.venv/bin/python -m src.models.train_lstm

train-lstm-test:
	.venv/bin/python -m src.models.train_lstm --test

# --- API ---

serve:
	.venv/bin/uvicorn src.api.main:app --reload --port 8000

# --- Embedded C++ layer (CAN bus / RTOS / MQTT) ---

embedded:                          ## Build C++ embedded core
	cmake -S embedded -B embedded/build -DCMAKE_BUILD_TYPE=Release -Wno-dev
	cmake --build embedded/build --parallel

bench: embedded                    ## WCET benchmark (baseline vs. ring-buffer-decoupled)
	./embedded/build/benchmark

mqtt-bridge:                       ## Python MQTT→FastAPI bridge (needs broker + serve)
	.venv/bin/python embedded/scripts/mqtt_consumer.py

# --- Full pipeline ---

all: ingest telemetry features train
