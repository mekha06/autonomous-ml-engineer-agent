import os
import pandas as pd
import matplotlib.pyplot as plt

from config import REPORT_DIR


class EDAAgent:

    def run_eda(self, file_path, target_column, task_type):
        df = pd.read_csv(file_path)

        plots_dir = os.path.join(REPORT_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        summary_statistics = df.describe(include="all").fillna("").to_dict()

        plot_paths = {}

        # Target distribution plot
        target_plot_path = os.path.join(plots_dir, "target_distribution.png")

        plt.figure(figsize=(8, 5))

        if task_type == "classification":
            df[target_column].value_counts().plot(kind="bar")
            plt.title(f"Class Distribution - {target_column}")
            plt.xlabel(target_column)
            plt.ylabel("Count")
        else:
            df[target_column].hist(bins=30)
            plt.title(f"Target Distribution - {target_column}")
            plt.xlabel(target_column)
            plt.ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(target_plot_path)
        plt.close()

        plot_paths["target_distribution"] = target_plot_path

        # Correlation heatmap-style plot for numerical columns
        numerical_df = df.select_dtypes(include=["int64", "float64"])

        correlation_plot_path = None

        if len(numerical_df.columns) > 1:
            correlation = numerical_df.corr()

            correlation_plot_path = os.path.join(plots_dir, "correlation_matrix.png")

            plt.figure(figsize=(8, 6))
            plt.imshow(correlation, aspect="auto")
            plt.colorbar()
            plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
            plt.yticks(range(len(correlation.columns)), correlation.columns)
            plt.title("Correlation Matrix")
            plt.tight_layout()
            plt.savefig(correlation_plot_path)
            plt.close()

            plot_paths["correlation_matrix"] = correlation_plot_path

        # Outlier summary using IQR
        outlier_report = {}

        for col in numerical_df.columns:
            q1 = numerical_df[col].quantile(0.25)
            q3 = numerical_df[col].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = numerical_df[
                (numerical_df[col] < lower_bound) |
                (numerical_df[col] > upper_bound)
            ]

            outlier_report[col] = int(outliers.shape[0])

        return {
            "summary_statistics": summary_statistics,
            "outlier_report": outlier_report,
            "plot_paths": plot_paths
        }