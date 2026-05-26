#!/usr/bin/env python3
"""
MQTT → FastAPI bridge.

Subscribes to f1/telemetry/lap from the C++ rtos_scheduler,
maps the lap telemetry JSON to the LightGBM feature dict,
POSTs to /predict, and prints the predicted degradation delta.

Usage:
    # Terminal 1: start FastAPI
    make serve

    # Terminal 2: start C++ telemetry core
    ./embedded/build/rtos_scheduler

    # Terminal 3: start this bridge
    python embedded/scripts/mqtt_consumer.py
"""

import json
import sys
import time
import requests
import paho.mqtt.client as mqtt

API_URL    = "http://localhost:8000/predict"
BROKER     = "localhost"
BROKER_PORT = 1883
TOPIC      = "f1/telemetry/lap"

COMPOUND_MAP = {"HARD": 0, "INTERMEDIATE": 1, "MEDIUM": 2, "SOFT": 3, "WET": 4}


def lap_to_features(msg: dict) -> dict:
    """Map MQTT lap payload → LightGBM feature dict."""
    compound_str = msg.get("compound", "SOFT")
    compound_enc = COMPOUND_MAP.get(compound_str, 2)
    stint_lap    = int(msg.get("stint_lap", 1))
    fuel_kg      = float(msg.get("fuel_load_kg", 80.0))

    return {
        "stint_lap_number":   stint_lap,
        "Stint":              int(msg.get("lap", 1)),
        "compound_encoded":   compound_enc,
        "fuel_load_kg":       fuel_kg,
        "TrackTemp":          float(msg.get("track_temp", 38.0)),
        "AirTemp":            float(msg.get("air_temp", 26.0)),
        "avg_throttle_pct":   float(msg.get("avg_throttle_pct", 70.0)),
        "full_throttle_pct":  float(msg.get("full_throttle_pct", 0.55)),
        "avg_brake":          float(msg.get("avg_brake", 0.1)),
        "braking_pct":        float(msg.get("braking_pct", 0.12)),
        "max_speed_kph":      float(msg.get("max_speed_kph", 310.0)),
        "drs_active_pct":     float(msg.get("drs_active_pct", 0.15)),
        "avg_rpm":            float(msg.get("avg_rpm", 11500.0)),
        "track_evolution":    stint_lap * 20,  # proxy: laps × drivers
        "deg_rate_last_3":    0.0,
        "deg_acceleration":   0.0,
        "sector_1_pct":       0.33,
        "sector_2_pct":       0.35,
        "sector_3_pct":       0.32,
    }


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
    except json.JSONDecodeError as e:
        print(f"[BRIDGE] JSON decode error: {e}", file=sys.stderr)
        return

    features = lap_to_features(payload)

    try:
        resp = requests.post(API_URL, json=features, timeout=2.0)
        resp.raise_for_status()
        delta = resp.json().get("predicted_delta", float("nan"))
        lap   = payload.get("lap", "?")
        cmp   = payload.get("compound", "?")
        temp  = payload.get("avg_tyre_temp", 0.0)
        print(f"[LAP {lap:>2} | {cmp:<12}] tyre={temp:.1f}°C → Δlap = {delta:+.3f}s")
    except requests.RequestException as e:
        print(f"[BRIDGE] API error: {e}", file=sys.stderr)


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[BRIDGE] Connected to MQTT broker at {BROKER}:{BROKER_PORT}")
        client.subscribe(TOPIC)
        print(f"[BRIDGE] Subscribed to {TOPIC}")
    else:
        print(f"[BRIDGE] Connection failed: rc={rc}", file=sys.stderr)


def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(BROKER, BROKER_PORT, 60)
    except ConnectionRefusedError:
        print(f"[BRIDGE] Cannot connect to MQTT broker at {BROKER}:{BROKER_PORT}", file=sys.stderr)
        print("[BRIDGE] Start broker with: brew services start mosquitto", file=sys.stderr)
        sys.exit(1)

    print("[BRIDGE] Waiting for lap telemetry... (Ctrl+C to stop)")
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[BRIDGE] Stopped.")


if __name__ == "__main__":
    main()
