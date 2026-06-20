import pandas as pd


class DatasetAgent:

    def analyze_dataset(self, file_path, target_column):
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError as exc:
            raise ValueError(
                "Could not read CSV file. Please upload a UTF-8 encoded CSV."
            ) from exc
        except pd.errors.EmptyDataError as exc:
            raise ValueError("Uploaded CSV file is empty.") from exc

        if df.empty:
            raise ValueError("Uploaded CSV file has no rows.")

        if len(df.columns) == 0:
            raise ValueError("Uploaded CSV file has no columns.")

        duplicate_columns = df.columns[df.columns.duplicated()].tolist()

        if duplicate_columns:
            raise ValueError(
                f"Duplicate column names found: {duplicate_columns}."
            )

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        num_rows, num_columns = df.shape

        if num_rows < 10:
            raise ValueError(
                "Dataset must contain at least 10 rows for a reliable split."
            )

        if num_columns < 2:
            raise ValueError(
                "Dataset must contain at least one feature column and one target column."
            )

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
        usable_target_series = target_series.dropna()
        unique_target_values = int(usable_target_series.nunique())
        target_dtype = str(target_series.dtype)

        warnings = []
        if missing_target_rows > 0:
            warnings.append(
                f"{missing_target_rows} rows have missing target values and "
                "will be removed before training."
            )

        if len(usable_target_series) < 10:
            raise ValueError(
                "Dataset must contain at least 10 rows with a non-missing target."
            )

        if unique_target_values < 2:
            raise ValueError(
                f"Target column '{target_column}' must have at least two unique values."
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

        if unique_target_values == len(usable_target_series):
            warnings.append(
                f"Target column '{target_column}' has unique values for every row. "
                "This may indicate an identifier or leakage-prone target."
            )

        # Task type detection
        is_categorical_target = (
            target_dtype in {"object", "category", "bool"}
            or unique_target_values <= 10
            or (
                unique_target_values <= 20
                and unique_target_values / len(usable_target_series) <= 0.05
            )
        )

        if is_categorical_target:
            task_type = "classification"
        else:
            task_type = "regression"

        if target_dtype != "object" and task_type == "classification":
            warnings.append(
                "Numeric target treated as classification because it has a "
                "small number of distinct values."
            )

        excluded_feature_columns = [
            col for col in id_like_columns if col != target_column
        ]

        if excluded_feature_columns:
            warnings.append(
                "ID-like feature columns will be excluded from training to "
                "reduce leakage risk."
            )

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
            "excluded_feature_columns": excluded_feature_columns,
            "high_cardinality_features": high_cardinality_features,
            "datetime_like_columns": datetime_like_columns,
            "is_time_series_candidate": is_time_series_candidate,
            "missing_target_rows": missing_target_rows,
            "warnings": warnings
        }

        return dataset_report
