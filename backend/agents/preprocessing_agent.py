import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from config import TEST_SIZE, RANDOM_STATE


class PreprocessingAgent:

    def preprocess(self, file_path, target_column):
        df = pd.read_csv(file_path)

        df = df.dropna(subset=[target_column])

        X = df.drop(columns=[target_column])
        y = df[target_column]
        label_encoder = None

        if y.dtype == "object" or y.nunique() <= 20:
          label_encoder = LabelEncoder()
          y = label_encoder.fit_transform(y)

        numerical_features = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_features = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        categorical_options = {}

        for col in categorical_features:
            categorical_options[col] = [
                str(value)
                for value in sorted(X[col].dropna().unique().tolist())
            ]

        numerical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, numerical_features),
                ("cat", categorical_pipeline, categorical_features)
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "preprocessor": preprocessor,
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "label_encoder": label_encoder,
            "categorical_options": categorical_options
        }