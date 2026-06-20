import os
import json

from sklearn.metrics import classification_report
from config import REPORT_DIR


class ClassificationReportAgent:

    def generate(
        self,
        y_true,
        y_pred,
        task_type
    ):

        if task_type != "classification":
            return {
                "available": False,
                "report_path": None
            }

        report = classification_report(
            y_true,
            y_pred,
            output_dict=True
        )

        report_path = os.path.join(
            REPORT_DIR,
            "classification_report.json"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                report,
                f,
                indent=4
            )

        return {
            "available": True,
            "report": report,
            "report_path": report_path
        }