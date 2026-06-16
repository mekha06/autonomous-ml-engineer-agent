class RecommendationAgent:

    def generate_recommendation(
        self,
        evaluation_report,
        tuning_report,
        dataset_report
    ):
        best_model_name = evaluation_report.get("best_model_name", "N/A")
        best_score = evaluation_report.get("best_score", None)
        task_type = dataset_report.get("task_type", "N/A")

        reasons = []

        reasons.append(
            f"{best_model_name} achieved the best score among all trained models."
        )

        if task_type == "regression":
            reasons.append(
                "For regression tasks, the system selected the model with the highest R2 score."
            )
        else:
            reasons.append(
                "For classification tasks, the system selected the model with the strongest classification performance."
            )

        if tuning_report.get("tuned"):
            reasons.append(
                "Hyperparameter tuning was applied to improve the selected model."
            )
        else:
            reasons.append(
                "No tuning configuration was available for the selected model, so the baseline model was used."
            )

        if "XGBoost" in best_model_name:
            reasons.append(
                "XGBoost is often strong on tabular datasets because it captures non-linear patterns effectively."
            )

        if "Linear Regression" in best_model_name:
            reasons.append(
                "Linear Regression performing best suggests that the dataset has a strong linear relationship."
            )

        if "Random Forest" in best_model_name:
            reasons.append(
                "Random Forest is robust because it combines many decision trees to reduce overfitting."
            )

        return {
            "recommended_model": best_model_name,
            "task_type": task_type,
            "best_score": best_score,
            "reasons": reasons
        }