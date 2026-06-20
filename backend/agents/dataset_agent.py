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

        target_series = df[target_column]
        missing_target_rows = int(target_series.isna().sum())
        unique_target_values = int(target_series.nunique())
        target_dtype = str(target_series.dtype)

        warnings = []
        if missing_target_rows > 0:
          warnings.append(
        f"{missing_target_rows} rows have missing target values and will be removed before training."
    )

        # ID-like column detection
        id_like_columns = []

        for col in df.columns:
            col_lower = col.lower()

            if (
                "id" in col_lower
                or df[col].nunique() == len(df)
            ):
                id_like_columns.append(col)

        # Invalid target checks
        if target_column in id_like_columns:
            warnings.append(
                f"Target column '{target_column}' appears to be an ID-like column. "
                "This may not be suitable for ML prediction."
            )

        if target_dtype == "object" and unique_target_values > 50:
            warnings.append(
                f"Target column '{target_column}' has high cardinality "
                f"({unique_target_values} unique values). "
                "This may not be a good classification target."
            )

        if unique_target_values == num_rows:
            warnings.append(
                f"Target column '{target_column}' has unique values for every row. "
                "This may indicate an identifier or leakage-prone target."
            )

        # Task type detection
        if target_dtype == "object" or unique_target_values <= 20:
            task_type = "classification"
        else:
            task_type = "regression"

        # Class imbalance check
        class_distribution = None
        imbalance_status = None

        if task_type == "classification":
            class_distribution = {
                str(key): float(value)
                for key, value in target_series
                .value_counts(normalize=True)
                .items()
            }

            max_class_ratio = max(class_distribution.values())

            if max_class_ratio >= 0.90:
                imbalance_status = "severe imbalance"
                warnings.append(
                    "Severe class imbalance detected. "
                    "Consider class weights, resampling, or imbalance-aware models."
                )

            elif max_class_ratio >= 0.75:
                imbalance_status = "moderate imbalance"
                warnings.append(
                    "Moderate class imbalance detected. "
                    "Evaluation should prioritize F1-score, recall, and ROC-AUC."
                )

            else:
                imbalance_status = "balanced"

        # High-cardinality categorical features
        high_cardinality_features = []

        for col in categorical_columns:
            if col != target_column:
                unique_count = df[col].nunique()

                if unique_count > 50:
                    high_cardinality_features.append(
                        {
                            "column": col,
                            "unique_values": int(unique_count)
                        }
                    )

        if high_cardinality_features:
            warnings.append(
                "High-cardinality categorical features detected. "
                "One-hot encoding may create many columns."
            )

        # Time-series-like detection
        datetime_like_columns = []

        for col in df.columns:
            if col == target_column:
                continue

            col_lower = col.lower()

            if any(keyword in col_lower for keyword in ["date", "time", "year"]):
                datetime_like_columns.append(col)

        is_time_series_candidate = len(datetime_like_columns) > 0

        if is_time_series_candidate:
            warnings.append(
                "Datetime-like columns detected. "
                "If the goal is forecasting, a time-series pipeline may be needed."
            )

        dataset_report = {
            "rows": int(num_rows),
            "columns": int(num_columns),
            "target_column": target_column,
            "target_dtype": target_dtype,
            "task_type": task_type,
            "missing_values": missing_values,
            "duplicate_rows": duplicate_rows,
            "numerical_columns": numerical_columns,
            "categorical_columns": categorical_columns,
            "unique_target_values": unique_target_values,
            "class_distribution": class_distribution,
            "imbalance_status": imbalance_status,
            "id_like_columns": id_like_columns,
            "high_cardinality_features": high_cardinality_features,
            "datetime_like_columns": datetime_like_columns,
            "is_time_series_candidate": is_time_series_candidate,
            "missing_target_rows": missing_target_rows,
            "warnings": warnings
        }

        return dataset_report