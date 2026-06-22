import os
import textwrap
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
        pdf_report_filename = f"automl_report_{timestamp}.pdf"

        report_path = os.path.join(REPORT_DIR, report_filename)
        pdf_report_path = os.path.join(REPORT_DIR, pdf_report_filename)

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

            <p><b>Excluded Columns:</b> {preprocessing_report.get("excluded_columns", [])}</p>

            <p><b>Stratified Split:</b> {preprocessing_report.get("stratified_split", False)}</p>

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

        self._generate_pdf_report(
            pdf_report_path=pdf_report_path,
            dataset_report=dataset_report,
            eda_report=eda_report,
            preprocessing_report=preprocessing_report,
            model_training_report=model_training_report,
            evaluation_report=evaluation_report,
            deployment_report=deployment_report,
            feature_importance_report=feature_importance_report
        )

        return {
            "report_filename": report_filename,
            "report_path": report_path,
            "pdf_report_filename": pdf_report_filename,
            "pdf_report_path": pdf_report_path
        }

    def _generate_pdf_report(
        self,
        pdf_report_path,
        dataset_report,
        eda_report,
        preprocessing_report,
        model_training_report,
        evaluation_report,
        deployment_report,
        feature_importance_report
    ):
        lines = [
            "Autonomous ML Engineer Agent Report",
            "",
            "Dataset Summary",
            f"Rows: {dataset_report['rows']}",
            f"Columns: {dataset_report['columns']}",
            f"Target Column: {dataset_report['target_column']}",
            f"Task Type: {dataset_report['task_type']}",
            f"Duplicate Rows: {dataset_report['duplicate_rows']}",
            "",
            "Preprocessing Summary",
            f"Numerical Features: {preprocessing_report['numerical_features']}",
            f"Categorical Features: {preprocessing_report['categorical_features']}",
            f"Train Rows: {preprocessing_report['train_rows']}",
            f"Test Rows: {preprocessing_report['test_rows']}",
            f"Excluded Columns: {preprocessing_report.get('excluded_columns', [])}",
            f"Stratified Split: {preprocessing_report.get('stratified_split', False)}",
            "",
            "Model Training Summary",
            f"Models Trained: {model_training_report['models_trained']}",
            f"Total Models: {model_training_report['total_models']}",
            "",
            "Evaluation Summary",
            f"Best Model: {evaluation_report['best_model_name']}",
            f"Best Score: {evaluation_report['best_score']}",
            f"Model Results: {evaluation_report['model_results']}",
            "",
            "Deployment Summary",
            f"Saved Model: {deployment_report['model_filename']}",
            f"Latest Model Path: {deployment_report['latest_model_path']}",
            "",
            "EDA Summary",
            f"Outlier Report: {eda_report['outlier_report']}",
            "",
            "Feature Importance",
            f"Available: {feature_importance_report['available']}",
            f"Type: {feature_importance_report['importance_type']}",
            f"Top Features: {feature_importance_report['top_features']}"
        ]
        wrapped_lines = []

        for line in lines:
            if not line:
                wrapped_lines.append("")
                continue

            wrapped_lines.extend(textwrap.wrap(line, width=95))

        with PdfPages(pdf_report_path) as pdf:
            for start in range(0, len(wrapped_lines), 42):
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.text(
                    0.08,
                    0.95,
                    "\n".join(wrapped_lines[start:start + 42]),
                    va="top",
                    ha="left",
                    fontsize=10,
                    family="monospace"
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
