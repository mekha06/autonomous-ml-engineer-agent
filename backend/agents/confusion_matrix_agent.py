import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

from config import REPORT_DIR


class ConfusionMatrixAgent:

    def generate(
        self,
        y_test,
        y_pred,
        task_type
    ):

        if task_type != "classification":
            return {
                "available": False,
                "plot_path": None
            }

        plot_dir = os.path.join(
            REPORT_DIR,
            "plots"
        )

        os.makedirs(
            plot_dir,
            exist_ok=True
        )

        plot_path = os.path.join(
            plot_dir,
            "confusion_matrix.png"
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        disp.plot(
            ax=ax,
            cmap="Blues"
        )

        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        return {
            "available": True,
            "plot_path": plot_path
        }