"""Generate illustrative sensor readings for dashboard demonstrations."""

import random


def generate_reading(scenario="Random"):
    scenarios = {
        "Normal": {
            "temperature_c": (2.0, 5.0),
            "humidity_pct": (60.0, 80.0),
            "excursion_minutes": (0.0, 10.0),
        },
        "Warning": {
            "temperature_c": (6.0, 9.0),
            "humidity_pct": (75.0, 95.0),
            "excursion_minutes": (30.0, 60.0),
        },
        "High Risk": {
            "temperature_c": (10.0, 15.0),
            "humidity_pct": (80.0, 100.0),
            "excursion_minutes": (90.0, 180.0),
        },
        "Random": {
            "temperature_c": (-5.0, 15.0),
            "humidity_pct": (40.0, 100.0),
            "excursion_minutes": (0.0, 180.0),
        },
    }

    if scenario not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario}")

    ranges = scenarios[scenario]

    return {
        key: round(random.uniform(low, high), 2)
        for key, (low, high) in ranges.items()
    }
