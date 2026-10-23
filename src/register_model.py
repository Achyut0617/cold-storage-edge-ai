from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "mlflow.db"
MODEL_NAME = "ColdStorageRiskModel"

mlflow.set_tracking_uri(f"sqlite:///{DB.as_posix()}")
client = MlflowClient()

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
if not versions:
    raise RuntimeError(
        "No registered model version found. Run python src/train_model.py first."
    )

latest = max(versions, key=lambda item: int(item.version))
print("Registered model:", latest.name)
print("Version:", latest.version)
print("Run ID:", latest.run_id)
print("Source:", latest.source)
