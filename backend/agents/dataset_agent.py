import pandas as pd


class DatasetAgent:

    def analyze_dataset(self, file_path, target_column):
        df = pd.read_csv(file_path)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        num_rows, num_columns = df.shape

        missing_values = {
            col: int(value)
            for col, value in df.isnull().sum().items()
        }

        duplicate_rows = int(df.duplicated().sum())

        numerical_columns = df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_columns = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        unique_target_values = int(df[target_column].nunique())

        if df[target_column].dtype == "object" or unique_target_values < 20:
            task_type = "classification"
        else:
            task_type = "regression"

        class_distribution = None

        if task_type == "classification":
            class_distribution = {
                str(key): float(value)
                for key, value in df[target_column]
                .value_counts(normalize=True)
                .items()
            }

        return {
            "rows": int(num_rows),
            "columns": int(num_columns),
            "target_column": target_column,
            "task_type": task_type,
            "missing_values": missing_values,
            "duplicate_rows": duplicate_rows,
            "numerical_columns": numerical_columns,
            "categorical_columns": categorical_columns,
            "unique_target_values": unique_target_values,
            "class_distribution": class_distribution
        }