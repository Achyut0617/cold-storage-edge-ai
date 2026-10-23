"""Train and evaluate the cold-storage risk classifier."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


def main():
    data_path = Path("data/sensor_dataset.csv")
    model_dir = Path("models")
    reports_dir = Path("reports")

    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Run: python src/generate_dataset.py"
        )

    dataset = pd.read_csv(data_path)
    features = [
        "temperature_c",
        "humidity_pct",
        "excursion_minutes",
    ]
    target = "risk_label"

    X = dataset[features]
    y = dataset[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    labels = list(model.classes_)

    metrics = {
        "dataset_rows": int(len(dataset)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "features": features,
        "classes": labels,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=labels
        ).tolist(),
        "note": (
            "Synthetic demonstration data; results do not establish "
            "real-world food-safety performance."
        ),
    }

    model_path = model_dir / "cold_storage_model.pkl"
    metrics_path = reports_dir / "evaluation_metrics.json"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nMODEL TRAINING COMPLETED")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("\nConfusion matrix (class order:", labels, "):")
    print(confusion_matrix(y_test, predictions, labels=labels))
    print(f"\nSaved model: {model_path.resolve()}")
    print(f"Saved metrics: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
