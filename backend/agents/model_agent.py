from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier
)

from xgboost import XGBRegressor, XGBClassifier


class ModelAgent:

    def get_models(self, task_type):
        if task_type == "regression":
            return {
                "Linear Regression": LinearRegression(),

                "Decision Tree Regressor": DecisionTreeRegressor(
                    random_state=42
                ),

                "Random Forest Regressor": RandomForestRegressor(
                    random_state=42,
                    n_jobs=1
                ),

                "Gradient Boosting Regressor": GradientBoostingRegressor(
                    random_state=42
                ),

                "XGBoost Regressor": XGBRegressor(
                   random_state=42,
                   objective="reg:squarederror",
                   n_estimators=50,
                   max_depth=2,
                   learning_rate=0.1,
                   n_jobs=1,
                   verbosity=0
                )
            }

        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                solver="lbfgs"
            ),

            "Decision Tree Classifier": DecisionTreeClassifier(
                random_state=42
            ),

            "Random Forest Classifier": RandomForestClassifier(
                random_state=42,
                n_jobs=1
            ),

            "Gradient Boosting Classifier": GradientBoostingClassifier(
                random_state=42
            ),

            "XGBoost Classifier": XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_estimators=50,
                max_depth=2,
                learning_rate=0.1,
                n_jobs=1,
                verbosity=0
            )
        }

    def train_models(self, preprocessing_result, task_type):
        models = self.get_models(task_type)

        trained_models = {}

        for model_name, model in models.items():
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessing_result["preprocessor"]),
                ("model", model)
            ])

            pipeline.fit(
                preprocessing_result["X_train"],
                preprocessing_result["y_train"]
            )

            trained_models[model_name] = pipeline

        return trained_models