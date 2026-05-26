from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier
)

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier
)


class TuningAgent:

    def get_param_grid(self, model):

        if isinstance(model, RandomForestRegressor) or isinstance(model, RandomForestClassifier):
            return {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10]
            }

        elif isinstance(model, GradientBoostingRegressor) or isinstance(model, GradientBoostingClassifier):
            return {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [3, 5, 7]
            }

        elif isinstance(model, DecisionTreeRegressor) or isinstance(model, DecisionTreeClassifier):
            return {
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10]
            }

        elif isinstance(model, LinearRegression):
            return {
                "model__fit_intercept": [True, False]
            }

        elif isinstance(model, LogisticRegression):
            return {
                "model__C": [0.01, 0.1, 1, 10],
                "model__solver": ["lbfgs", "liblinear"]
            }

        return None

    def tune_best_model(
        self,
        best_model_pipeline,
        preprocessing_result,
        task_type
    ):

        model = best_model_pipeline.named_steps["model"]

        param_grid = self.get_param_grid(model)

        if param_grid is None:
            return {
                "tuned": False,
                "message": "No tuning configuration available.",
                "best_model": best_model_pipeline,
                "best_params": {}
            }

        scoring = "r2" if task_type == "regression" else "f1_weighted"

        n_iter = min(5, len(list(param_grid.values())[0]))

        randomized_search = RandomizedSearchCV(
            estimator=best_model_pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=3,
            scoring=scoring,
            random_state=42,
            n_jobs=-1
        )

        randomized_search.fit(
            preprocessing_result["X_train"],
            preprocessing_result["y_train"]
        )

        return {
            "tuned": True,
            "best_model": randomized_search.best_estimator_,
            "best_params": randomized_search.best_params_,
            "best_cv_score": float(randomized_search.best_score_)
        }