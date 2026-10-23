"""Load the trained model and predict cold-storage risk."""

from pathlib import Path
import time

import joblib
import pandas as pd


MODEL_PATH = Path("models/cold_storage_model.pkl")
FEATURES = ["temperature_c", "humidity_pct", "excursion_minutes"]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run: python src/train_model.py"
        )
    return joblib.load(MODEL_PATH)


def predict_risk(model, temperature_c, humidity_pct, excursion_minutes):
    values = {
        "temperature_c": float(temperature_c),
        "humidity_pct": float(humidity_pct),
        "excursion_minutes": float(excursion_minutes),
    }

    if not -10 <= values["temperature_c"] <= 25:
        raise ValueError("Temperature must be between -10 and 25 C.")
    if not 0 <= values["humidity_pct"] <= 100:
        raise ValueError("Humidity must be between 0 and 100%.")
    if not 0 <= values["excursion_minutes"] <= 240:
        raise ValueError("Excursion duration must be between 0 and 240 minutes.")

    sample = pd.DataFrame([values], columns=FEATURES)

    start = time.perf_counter()
    prediction = str(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]
    latency_ms = (time.perf_counter() - start) * 1000

    confidence = float(max(probabilities))

    return {
        "risk_label": prediction,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "sensor_reading": values,
    }
