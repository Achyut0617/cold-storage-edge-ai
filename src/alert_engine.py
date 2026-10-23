"""Translate model risk classes into demonstration alert messages."""


def create_alert(risk_label):
    alerts = {
        "Normal": {
            "level": "NORMAL",
            "message": "No elevated risk detected by the demo model.",
        },
        "Warning": {
            "level": "WARNING",
            "message": "Review sensor conditions and inspect the cold-storage unit.",
        },
        "High Risk": {
            "level": "HIGH RISK",
            "message": "Urgently inspect the unit and verify readings using approved procedures.",
        },
    }

    return alerts.get(
        risk_label,
        {
            "level": "UNKNOWN",
            "message": "Unrecognized risk class. Verify the model output.",
        },
    )
