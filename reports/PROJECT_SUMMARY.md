# Cold Storage Monitoring and Risk Prediction — PBA-2

## Project scope
A local Python simulation that generates synthetic sensor data, trains a
classification model, predicts risk, and presents results in a dashboard.

## Sensor features
- Temperature (degrees Celsius)
- Humidity (percent)
- Excursion duration (minutes)

## Risk classes
- Normal
- Warning
- High Risk

## Project artifacts
- data/sensor_dataset.csv
- models/cold_storage_model.pkl
- reports/evaluation_metrics.json
- src/generate_dataset.py
- src/train_model.py
- src/inference.py
- src/alert_engine.py
- src/sensor_simulator.py
- dashboard/app.py
- tests/test_inference.py

## Limitations
The data and risk labels are synthetic. The model has not been validated
for actual cold-storage or food-safety decisions. The dashboard runs locally.
Registering the model in Azure ML does not itself deploy an endpoint.
