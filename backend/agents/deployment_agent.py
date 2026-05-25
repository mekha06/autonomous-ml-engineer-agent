import os
import joblib
from datetime import datetime

from config import MODEL_DIR


class DeploymentAgent:

    def save_best_model(self, best_model, best_model_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        safe_model_name = best_model_name.lower().replace(" ", "_")

        model_filename = f"{safe_model_name}_{timestamp}.pkl"
        model_path = os.path.join(MODEL_DIR, model_filename)

        joblib.dump(best_model, model_path)

        latest_model_path = os.path.join(MODEL_DIR, "best_model.pkl")
        joblib.dump(best_model, latest_model_path)

        return {
            "saved_model_path": model_path,
            "latest_model_path": latest_model_path,
            "model_filename": model_filename
        }