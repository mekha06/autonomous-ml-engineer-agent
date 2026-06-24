import os
import shap
import matplotlib.pyplot as plt

from config import REPORT_DIR


class SHAPAgent:

    def explain_model(self, best_model_pipeline, preprocessing_result):
        try:
            X_test = preprocessing_result["X_test"]

            preprocessor = best_model_pipeline.named_steps["preprocessor"]
            model = best_model_pipeline.named_steps["model"]

            X_test_transformed = preprocessor.transform(X_test)

            feature_names = preprocessor.get_feature_names_out()

            sample_size = min(100, len(X_test_transformed))

            try:
                explainer = shap.Explainer(model)

                shap_values = explainer(
                    X_test_transformed[:sample_size],
                    check_additivity=False
                )

            except Exception:
                return {
                    "available": False,
                    "message": "SHAP explanation could not be generated for the selected model.",
                    "plot_path": None
                }

            plots_dir = os.path.join(REPORT_DIR, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            shap_plot_path = os.path.join(
                plots_dir,
                "shap_summary.png"
            )

            plt.figure()

            shap.summary_plot(
                shap_values,
                X_test_transformed[:sample_size],
                feature_names=feature_names,
                show=False
            )

            plt.tight_layout()
            plt.savefig(
                shap_plot_path,
                bbox_inches="tight"
            )
            plt.close()

            return {
                "available": True,
                "message": "SHAP explanation generated successfully.",
                "plot_path": shap_plot_path
            }

        except Exception as e:
            return {
                "available": False,
                "message": str(e),
                "plot_path": None
            }