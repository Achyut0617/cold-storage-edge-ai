"""Live cold-storage monitoring dashboard using simulated sensors."""

import sys
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_model, predict_risk
from src.sensor_simulator import generate_reading
from src.alert_engine import create_alert


st.set_page_config(
    page_title="Cold Storage Live Monitor",
    page_icon="❄️",
    layout="wide",
)

st.title("❄️ Cold Storage Monitoring — Live Simulation")
st.caption("PBA-2 | Simulated sensors | Local ML inference")

if "history" not in st.session_state:
    st.session_state.history = []

if "running" not in st.session_state:
    st.session_state.running = False

if "latest" not in st.session_state:
    st.session_state.latest = None

try:
    model = load_model()
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()

st.sidebar.header("Simulation Controls")

scenario = st.sidebar.selectbox(
    "Sensor scenario",
    ["Random", "Normal", "Warning", "High Risk"],
)

interval = st.sidebar.slider(
    "Update interval (seconds)",
    min_value=1,
    max_value=10,
    value=2,
)

col_start, col_stop = st.sidebar.columns(2)

if col_start.button("Start", type="primary"):
    st.session_state.running = True

if col_stop.button("Stop"):
    st.session_state.running = False

if st.sidebar.button("Generate one reading"):
    st.session_state.running = False

    reading = generate_reading(scenario)
    result = predict_risk(
        model,
        reading["temperature_c"],
        reading["humidity_pct"],
        reading["excursion_minutes"],
    )
    alert = create_alert(result["risk_label"])

    record = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "temperature_c": reading["temperature_c"],
        "humidity_pct": reading["humidity_pct"],
        "excursion_minutes": reading["excursion_minutes"],
        "risk_label": result["risk_label"],
        "confidence_pct": round(result["confidence"] * 100, 2),
        "latency_ms": round(result["latency_ms"], 3),
        "alert": alert["level"],
    }

    st.session_state.history.append(record)
    st.session_state.latest = record

status = "RUNNING" if st.session_state.running else "STOPPED"
st.sidebar.write(f"Simulation status: **{status}**")
st.sidebar.write(f"Recorded readings: {len(st.session_state.history)}")

if st.session_state.running:
    reading = generate_reading(scenario)

    result = predict_risk(
        model,
        reading["temperature_c"],
        reading["humidity_pct"],
        reading["excursion_minutes"],
    )

    alert = create_alert(result["risk_label"])

    record = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "temperature_c": reading["temperature_c"],
        "humidity_pct": reading["humidity_pct"],
        "excursion_minutes": reading["excursion_minutes"],
        "risk_label": result["risk_label"],
        "confidence_pct": round(result["confidence"] * 100, 2),
        "latency_ms": round(result["latency_ms"], 3),
        "alert": alert["level"],
    }

    st.session_state.history.append(record)
    st.session_state.latest = record

    # Retain the most recent 200 readings in dashboard memory.
    st.session_state.history = st.session_state.history[-200:]

if st.session_state.latest is None:
    st.info("Click Start to begin generating live simulated sensor readings.")
else:
    latest = st.session_state.latest

    st.subheader("Latest sensor reading")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature", f"{latest['temperature_c']:.2f} °C")
    c2.metric("Humidity", f"{latest['humidity_pct']:.2f}%")
    c3.metric("Excursion", f"{latest['excursion_minutes']:.2f} min")
    c4.metric("Inference latency", f"{latest['latency_ms']:.3f} ms")

    st.subheader("ML risk prediction")
    st.write(f"**Risk class:** {latest['risk_label']}")
    st.write(f"**Model confidence:** {latest['confidence_pct']:.2f}%")

    alert = create_alert(latest["risk_label"])

    if latest["risk_label"] == "High Risk":
        st.error(alert["message"])
    elif latest["risk_label"] == "Warning":
        st.warning(alert["message"])
    else:
        st.success(alert["message"])

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)

    st.subheader("Live sensor history")
    st.dataframe(df.tail(20), use_container_width=True)

    st.subheader("Temperature trend")
    fig_temp = px.line(
        df,
        x="timestamp",
        y="temperature_c",
        markers=True,
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    st.subheader("Risk-class distribution")
    risk_counts = (
        df["risk_label"]
        .value_counts()
        .rename_axis("Risk class")
        .reset_index(name="Count")
    )
    fig_risk = px.bar(
        risk_counts,
        x="Risk class",
        y="Count",
        title="Predictions during this dashboard session",
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    st.download_button(
        "Download sensor history CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cold_storage_sensor_history.csv",
        mime="text/csv",
    )

if st.session_state.running:
    time.sleep(interval)
    st.rerun()
