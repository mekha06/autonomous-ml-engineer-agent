import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class EvaluationAgent:

    def evaluate_models(self, trained_models, preprocessing_result, task_type):
        X_test = preprocessing_result["X_test"]
        y_test = preprocessing_result["y_test"]

        results = {}
        best_model_name = None
        best_score = -float("inf")

        for model_name, model in trained_models.items():
            y_pred = model.predict(X_test)

            if task_type == "regression":
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                results[model_name] = {
                    "MAE": float(mae),
                    "MSE": float(mse),
                    "RMSE": float(rmse),
                    "R2_Score": float(r2)
                }

                score = r2

            else:
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                roc_auc = None
                try:
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X_test)

                        if len(set(y_test)) == 2:
                            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                        else:
                            roc_auc = roc_auc_score(
                                y_test,
                                y_proba,
                                multi_class="ovr",
                                average="weighted"
                            )
                except Exception:
                    roc_auc = None

                results[model_name] = {
                    "Accuracy": float(accuracy),
                    "Precision": float(precision),
                    "Recall": float(recall),
                    "F1_Score": float(f1),
                    "ROC_AUC": float(roc_auc) if roc_auc is not None else None
                }

                score = f1

            if score > best_score:
                best_score = score
                best_model_name = model_name

        return {
            "model_results": results,
            "best_model_name": best_model_name,
            "best_score": float(best_score),
            "best_model": trained_models[best_model_name]
        }