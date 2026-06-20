import json
import os
import joblib
from datetime import datetime

from config import MODEL_DIR


class DeploymentAgent:

    def save_best_model(self, best_model, best_model_name, metadata=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        safe_model_name = best_model_name.lower().replace(" ", "_")

        model_filename = f"{safe_model_name}_{timestamp}.pkl"
        model_path = os.path.join(MODEL_DIR, model_filename)
        metadata_filename = f"{safe_model_name}_{timestamp}.json"
        metadata_path = os.path.join(MODEL_DIR, metadata_filename)

        joblib.dump(best_model, model_path)

        latest_model_path = os.path.join(MODEL_DIR, "best_model.pkl")
        joblib.dump(best_model, latest_model_path)

        metadata = metadata or {}
        metadata = {
            **metadata,
            "model_filename": model_filename,
            "saved_model_path": model_path,
            "created_at": timestamp
        }

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

        latest_metadata_path = os.path.join(MODEL_DIR, "best_model_metadata.json")

        with open(latest_metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

        return {
            "saved_model_path": model_path,
            "latest_model_path": latest_model_path,
            "model_filename": model_filename,
            "metadata_path": metadata_path,
            "latest_metadata_path": latest_metadata_path
        }
