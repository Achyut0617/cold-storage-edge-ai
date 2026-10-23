"""Generate a synthetic cold-storage sensor dataset for PBA-2."""

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n_samples = 6000

    # Simulated temperature, humidity, and excursion duration.
    temperature = rng.normal(4.0, 3.5, n_samples)
    humidity = rng.normal(75.0, 10.0, n_samples)
    excursion_minutes = rng.exponential(12.0, n_samples)

    # Keep generated values within plausible demo ranges.
    temperature = np.clip(temperature, -10.0, 25.0)
    humidity = np.clip(humidity, 20.0, 100.0)
    excursion_minutes = np.clip(excursion_minutes, 0.0, 240.0)

    # Illustrative rules for synthetic demo labels only.
    risk_labels = []
    for temp, hum, minutes in zip(
        temperature, humidity, excursion_minutes
    ):
        if temp >= 10 or minutes >= 90:
            risk_labels.append("High Risk")
        elif temp >= 6 or minutes >= 30 or hum >= 90:
            risk_labels.append("Warning")
        else:
            risk_labels.append("Normal")

    dataset = pd.DataFrame({
        "temperature_c": np.round(temperature, 2),
        "humidity_pct": np.round(humidity, 2),
        "excursion_minutes": np.round(excursion_minutes, 2),
        "risk_label": risk_labels,
    })

    output_file = output_dir / "sensor_dataset.csv"
    dataset.to_csv(output_file, index=False)

    print("\nDataset generated successfully.")
    print(f"File: {output_file.resolve()}")
    print(f"Rows and columns: {dataset.shape}")
    print("\nRisk category counts:")
    print(dataset["risk_label"].value_counts())
    print("\nMissing values:")
    print(dataset.isnull().sum())
    print("\nFirst five rows:")
    print(dataset.head().to_string(index=False))


if __name__ == "__main__":
    main()
