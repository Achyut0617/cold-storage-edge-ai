import pytest

from src.inference import load_model, predict_risk
from src.alert_engine import create_alert


def test_prediction_returns_valid_class():
    model = load_model()
    result = predict_risk(model, 4.0, 70.0, 5.0)

    assert result["risk_label"] in {"Normal", "Warning", "High Risk"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["latency_ms"] >= 0.0


def test_invalid_humidity_is_rejected():
    model = load_model()

    with pytest.raises(ValueError):
        predict_risk(model, 4.0, 120.0, 5.0)


def test_alert_for_high_risk():
    alert = create_alert("High Risk")

    assert alert["level"] == "HIGH RISK"
    assert alert["message"]
