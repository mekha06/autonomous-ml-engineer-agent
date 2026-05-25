import os
import pandas as pd
import matplotlib.pyplot as plt

from config import REPORT_DIR


class FeatureImportanceAgent:

    def get_feature_names(self, trained_pipeline):
        preprocessor = trained_pipeline.named_steps["preprocessor"]

        feature_names = []

        try:
            feature_names = preprocessor.get_feature_names_out().tolist()
        except Exception:
            feature_names = []

        return feature_names

    def extract_importance(self, best_model_pipeline):
        model = best_model_pipeline.named_steps["model"]
        feature_names = self.get_feature_names(best_model_pipeline)

        importance_values = None
        importance_type = None

        if hasattr(model, "feature_importances_"):
            importance_values = model.feature_importances_
            importance_type = "feature_importances"

        elif hasattr(model, "coef_"):
            importance_values = model.coef_

            if len(importance_values.shape) > 1:
                importance_values = importance_values[0]

            importance_values = abs(importance_values)
            importance_type = "coefficients"

        else:
            return {
                "available": False,
                "message": "Feature importance is not available for this model.",
                "importance_type": None,
                "top_features": [],
                "plot_path": None
            }

        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(len(importance_values))]

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance_values
        })

        importance_df = importance_df.sort_values(
            by="importance",
            ascending=False
        )

        top_features_df = importance_df.head(10)

        plots_dir = os.path.join(REPORT_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        plot_path = os.path.join(plots_dir, "feature_importance.png")

        plt.figure(figsize=(10, 6))
        plt.barh(
            top_features_df["feature"][::-1],
            top_features_df["importance"][::-1]
        )
        plt.title("Top Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        top_features = top_features_df.to_dict(orient="records")

        return {
            "available": True,
            "importance_type": importance_type,
            "top_features": top_features,
            "plot_path": plot_path
        }