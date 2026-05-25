import os
from datetime import datetime

from config import REPORT_DIR


class ReportAgent:

    def generate_report(
        self,
        dataset_report,
        eda_report,
        preprocessing_report,
        model_training_report,
        evaluation_report,
        deployment_report,
        feature_importance_report
    ):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_filename = f"automl_report_{timestamp}.html"

        report_path = os.path.join(REPORT_DIR, report_filename)

        target_distribution_plot = eda_report["plot_paths"].get(
            "target_distribution",
            ""
        )

        correlation_plot = eda_report["plot_paths"].get(
            "correlation_matrix",
            ""
        )

        feature_importance_plot = feature_importance_report.get(
            "plot_path",
            ""
        )

        html = f"""
        <html>
        <head>
            <title>AutoML Report</title>
        </head>

        <body style="font-family: Arial; padding: 20px;">

            <h1>Autonomous ML Engineer Agent Report</h1>

            <hr>

            <h2>Dataset Summary</h2>

            <p><b>Rows:</b> {dataset_report["rows"]}</p>
            <p><b>Columns:</b> {dataset_report["columns"]}</p>
            <p><b>Target Column:</b> {dataset_report["target_column"]}</p>
            <p><b>Task Type:</b> {dataset_report["task_type"]}</p>
            <p><b>Duplicate Rows:</b> {dataset_report["duplicate_rows"]}</p>

            <hr>

            <h2>EDA Summary</h2>

            <h3>Outlier Report</h3>
            <pre>{eda_report["outlier_report"]}</pre>

            <h3>Target Distribution</h3>
            <img src="{target_distribution_plot}" width="600">

            <h3>Correlation Matrix</h3>
            <img src="{correlation_plot}" width="700">

            <hr>

            <h2>Preprocessing Summary</h2>

            <p><b>Numerical Features:</b> {preprocessing_report["numerical_features"]}</p>

            <p><b>Categorical Features:</b> {preprocessing_report["categorical_features"]}</p>

            <p><b>Train Rows:</b> {preprocessing_report["train_rows"]}</p>

            <p><b>Test Rows:</b> {preprocessing_report["test_rows"]}</p>

            <hr>

            <h2>Model Training Summary</h2>

            <p><b>Models Trained:</b> {model_training_report["models_trained"]}</p>

            <p><b>Total Models:</b> {model_training_report["total_models"]}</p>

            <hr>

            <h2>Evaluation Summary</h2>

            <p><b>Best Model:</b> {evaluation_report["best_model_name"]}</p>

            <p><b>Best Score:</b> {evaluation_report["best_score"]}</p>

            <pre>{evaluation_report["model_results"]}</pre>

            <hr>

            <h2>Deployment Summary</h2>

            <p><b>Saved Model:</b> {deployment_report["model_filename"]}</p>

            <p><b>Latest Model Path:</b> {deployment_report["latest_model_path"]}</p>

            <hr>

            <h2>Feature Importance</h2>

            <p><b>Available:</b> {feature_importance_report["available"]}</p>

            <p><b>Type:</b> {feature_importance_report["importance_type"]}</p>

            <pre>{feature_importance_report["top_features"]}</pre>

            <img src="{feature_importance_plot}" width="700">

            <hr>

            <h2>Future Enhancements</h2>

            <ul>
                <li>Optuna Hyperparameter Tuning</li>
                <li>MLflow Experiment Tracking</li>
                <li>SHAP Explainability</li>
                <li>Docker Deployment</li>
                <li>LangGraph Multi-Agent Orchestration</li>
                <li>Frontend Dashboard</li>
            </ul>

        </body>
        </html>
        """

        with open(report_path, "w", encoding="utf-8") as file:
            file.write(html)

        return {
            "report_filename": report_filename,
            "report_path": report_path
        }