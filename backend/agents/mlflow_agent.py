import mlflow
import mlflow.sklearn


class MLflowAgent:

    def __init__(self):
        mlflow.set_experiment("Autonomous ML Engineer Agent")

    def log_experiment(
        self,
        dataset_report,
        evaluation_report,
        tuning_report,
        deployment_report
    ):

        with mlflow.start_run():

            mlflow.log_param("target_column", dataset_report["target_column"])
            mlflow.log_param("task_type", dataset_report["task_type"])
            mlflow.log_param("rows", dataset_report["rows"])
            mlflow.log_param("columns", dataset_report["columns"])
            mlflow.log_param("best_model", evaluation_report["best_model_name"])
            mlflow.log_param("tuned", tuning_report["tuned"])

            if tuning_report["best_params"]:
                for param_name, param_value in tuning_report["best_params"].items():
                    mlflow.log_param(param_name, param_value)

            mlflow.log_metric("best_score", evaluation_report["best_score"])

            if tuning_report["best_cv_score"] is not None:
                mlflow.log_metric("best_cv_score", tuning_report["best_cv_score"])

            for model_name, metrics in evaluation_report["model_results"].items():
                for metric_name, metric_value in metrics.items():
                    if metric_value is not None:
                        safe_model_name = model_name.replace(" ", "_")
                        mlflow.log_metric(
                            f"{safe_model_name}_{metric_name}",
                            metric_value
                        )

            mlflow.log_artifact(deployment_report["latest_model_path"])

            return {
                "mlflow_logged": True,
                "experiment_name": "Autonomous ML Engineer Agent"
            }