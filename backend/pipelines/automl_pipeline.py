from typing import Any, Dict, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.classification_report_agent import ClassificationReportAgent
from agents.confusion_matrix_agent import ConfusionMatrixAgent
from agents.dataset_agent import DatasetAgent
from agents.deployment_agent import DeploymentAgent
from agents.eda_agent import EDAAgent
from agents.evaluation_agent import EvaluationAgent
from agents.feature_importance_agent import FeatureImportanceAgent
from agents.mlflow_agent import MLflowAgent
from agents.model_agent import ModelAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.recommendation_agent import RecommendationAgent
from agents.report_agent import ReportAgent
from agents.shap_agent import SHAPAgent
from agents.tuning_agent import TuningAgent


class AutoMLState(TypedDict, total=False):
    file_path: str
    target_column: str
    status: str
    graph_path: list[str]
    warnings: list[str]
    dataset_report: Dict[str, Any]
    eda_report: Dict[str, Any]
    eda_summary_report: Dict[str, Any]
    preprocessing_result: Dict[str, Any]
    preprocessing_report: Dict[str, Any]
    trained_models: Dict[str, Any]
    model_training_report: Dict[str, Any]
    evaluation_result: Dict[str, Any]
    evaluation_report: Dict[str, Any]
    tuning_result: Dict[str, Any]
    tuning_report: Dict[str, Any]
    deployment_report: Dict[str, Any]
    feature_importance_report: Dict[str, Any]
    shap_report: Dict[str, Any]
    report_generation: Dict[str, Any]
    mlflow_report: Dict[str, Any]
    recommendation_report: Dict[str, Any]
    confusion_matrix_report: Dict[str, Any]
    classification_report: Dict[str, Any]
    time_series_report: Dict[str, Any]
    unsupported_report: Dict[str, Any]


class AutoMLPipeline:
    """LangGraph orchestration for the ML engineer agents."""

    def __init__(
        self,
        classification_threshold: float = 0.80,
        regression_threshold: float = 0.60
    ):
        self.classification_threshold = classification_threshold
        self.regression_threshold = regression_threshold

        self.dataset_agent = DatasetAgent()
        self.eda_agent = EDAAgent()
        self.preprocessing_agent = PreprocessingAgent()
        self.model_agent = ModelAgent()
        self.evaluation_agent = EvaluationAgent()
        self.tuning_agent = TuningAgent()
        self.deployment_agent = DeploymentAgent()
        self.report_agent = ReportAgent()
        self.feature_importance_agent = FeatureImportanceAgent()
        self.mlflow_agent = MLflowAgent()
        self.recommendation_agent = RecommendationAgent()
        self.shap_agent = SHAPAgent()
        self.confusion_matrix_agent = ConfusionMatrixAgent()
        self.classification_report_agent = ClassificationReportAgent()

        self.graph = self._build_graph()

    def run(self, file_path, target_column):
        initial_state = {
            "file_path": file_path,
            "target_column": target_column,
            "status": "running",
            "graph_path": [],
            "warnings": []
        }

        final_state = self.graph.invoke(initial_state)
        return self._response_from_state(final_state)

    def _build_graph(self):
        graph = StateGraph(AutoMLState)

        graph.add_node("dataset_analysis", self._dataset_analysis)
        graph.add_node("time_series_agent", self._time_series_agent)
        graph.add_node("eda_agent", self._eda_agent)
        graph.add_node("preprocessing_agent", self._preprocessing_agent)
        graph.add_node("model_selection", self._model_selection)
        graph.add_node("classification_agent", self._train_models)
        graph.add_node("regression_agent", self._train_models)
        graph.add_node("clustering_agent", self._unsupported_task)
        graph.add_node("evaluation_agent", self._evaluation_agent)
        graph.add_node("quality_gate", self._quality_gate)
        graph.add_node("hyperparameter_tuning", self._hyperparameter_tuning)
        graph.add_node("shap_agent", self._shap_agent)
        graph.add_node("recommendation_agent", self._recommendation_agent)
        graph.add_node("report_agent", self._report_agent)
        graph.add_node("deploy_agent", self._deploy_agent)
        graph.add_node("mlflow_agent", self._mlflow_agent)

        graph.add_edge(START, "dataset_analysis")
        graph.add_conditional_edges(
            "dataset_analysis",
            self._route_dataset,
            {
                "time_series": "time_series_agent",
                "normal_ml": "eda_agent"
            }
        )
        graph.add_edge("time_series_agent", "eda_agent")
        graph.add_edge("eda_agent", "preprocessing_agent")
        graph.add_edge("preprocessing_agent", "model_selection")
        graph.add_conditional_edges(
            "model_selection",
            self._route_model_task,
            {
                "classification": "classification_agent",
                "regression": "regression_agent",
                "clustering": "clustering_agent"
            }
        )
        graph.add_edge("classification_agent", "evaluation_agent")
        graph.add_edge("regression_agent", "evaluation_agent")
        graph.add_edge("clustering_agent", END)
        graph.add_edge("evaluation_agent", "quality_gate")
        graph.add_conditional_edges(
            "quality_gate",
            self._route_quality,
            {
                "tune": "hyperparameter_tuning",
                "explain": "shap_agent"
            }
        )
        graph.add_edge("hyperparameter_tuning", "shap_agent")
        graph.add_edge("shap_agent", "recommendation_agent")
        graph.add_edge("recommendation_agent", "deploy_agent")
        graph.add_edge("deploy_agent", "report_agent")
        graph.add_edge("report_agent", "mlflow_agent")
        graph.add_edge("mlflow_agent", END)

        return graph.compile()

    def _append_path(self, state: AutoMLState, node_name: str) -> list[str]:
        return [*state.get("graph_path", []), node_name]

    def _dataset_analysis(self, state: AutoMLState) -> AutoMLState:
        dataset_report = self.dataset_agent.analyze_dataset(
            file_path=state["file_path"],
            target_column=state["target_column"]
        )

        return {
            **state,
            "dataset_report": dataset_report,
            "graph_path": self._append_path(state, "dataset_analysis")
        }

    def _route_dataset(
        self,
        state: AutoMLState
    ) -> Literal["time_series", "normal_ml"]:
        if state["dataset_report"].get("is_time_series_candidate"):
            return "time_series"

        return "normal_ml"

    def _time_series_agent(self, state: AutoMLState) -> AutoMLState:
        report = {
            "available": False,
            "message": (
                "Datetime-like columns were detected, but a dedicated "
                "forecasting agent is not implemented yet. Continuing with "
                "the standard tabular ML workflow."
            ),
            "datetime_like_columns": state["dataset_report"].get(
                "datetime_like_columns",
                []
            )
        }

        return {
            **state,
            "time_series_report": report,
            "warnings": [*state.get("warnings", []), report["message"]],
            "graph_path": self._append_path(state, "time_series_agent")
        }

    def _eda_agent(self, state: AutoMLState) -> AutoMLState:
        eda_report = self.eda_agent.run_eda(
            file_path=state["file_path"],
            target_column=state["target_column"],
            task_type=state["dataset_report"]["task_type"]
        )

        eda_summary_report = {
            "outlier_report": eda_report["outlier_report"],
            "plot_paths": eda_report["plot_paths"]
        }

        return {
            **state,
            "eda_report": eda_report,
            "eda_summary_report": eda_summary_report,
            "graph_path": self._append_path(state, "eda_agent")
        }

    def _preprocessing_agent(self, state: AutoMLState) -> AutoMLState:
        preprocessing_result = self.preprocessing_agent.preprocess(
            file_path=state["file_path"],
            target_column=state["target_column"],
            task_type=state["dataset_report"]["task_type"],
            excluded_columns=state["dataset_report"].get(
                "excluded_feature_columns",
                []
            )
        )

        preprocessing_report = {
            "numerical_features": preprocessing_result["numerical_features"],
            "categorical_features": preprocessing_result["categorical_features"],
            "categorical_options": preprocessing_result["categorical_options"],
            "excluded_columns": preprocessing_result["excluded_columns"],
            "feature_columns": preprocessing_result["feature_columns"],
            "stratified_split": preprocessing_result["stratified_split"],
            "train_rows": len(preprocessing_result["X_train"]),
            "test_rows": len(preprocessing_result["X_test"])
        }

        return {
            **state,
            "preprocessing_result": preprocessing_result,
            "preprocessing_report": preprocessing_report,
            "graph_path": self._append_path(state, "preprocessing_agent")
        }

    def _model_selection(self, state: AutoMLState) -> AutoMLState:
        return {
            **state,
            "graph_path": self._append_path(state, "model_selection")
        }

    def _route_model_task(
        self,
        state: AutoMLState
    ) -> Literal["classification", "regression", "clustering"]:
        task_type = state["dataset_report"].get("task_type")

        if task_type in {"classification", "regression"}:
            return task_type

        return "clustering"

    def _train_models(self, state: AutoMLState) -> AutoMLState:
        trained_models = self.model_agent.train_models(
            preprocessing_result=state["preprocessing_result"],
            task_type=state["dataset_report"]["task_type"]
        )

        model_training_report = {
            "models_trained": list(trained_models.keys()),
            "total_models": len(trained_models)
        }

        return {
            **state,
            "trained_models": trained_models,
            "model_training_report": model_training_report,
            "graph_path": self._append_path(
                state,
                f'{state["dataset_report"]["task_type"]}_agent'
            )
        }

    def _unsupported_task(self, state: AutoMLState) -> AutoMLState:
        return {
            **state,
            "status": "unsupported",
            "unsupported_report": {
                "message": "Clustering is not implemented for this pipeline yet."
            },
            "graph_path": self._append_path(state, "clustering_agent")
        }

    def _evaluation_agent(self, state: AutoMLState) -> AutoMLState:
        evaluation_result = self.evaluation_agent.evaluate_models(
            trained_models=state["trained_models"],
            preprocessing_result=state["preprocessing_result"],
            task_type=state["dataset_report"]["task_type"]
        )

        task_type = state["dataset_report"]["task_type"]

        confusion_matrix_report = self.confusion_matrix_agent.generate(
            evaluation_result["y_test"],
            evaluation_result["best_predictions"],
            task_type
        )

        classification_report = self.classification_report_agent.generate(
            evaluation_result["y_test"],
            evaluation_result["best_predictions"],
            task_type
        )

        evaluation_report = {
            "model_results": evaluation_result["model_results"],
            "best_model_name": evaluation_result["best_model_name"],
            "best_score": evaluation_result["best_score"]
        }

        return {
            **state,
            "evaluation_result": evaluation_result,
            "evaluation_report": evaluation_report,
            "confusion_matrix_report": confusion_matrix_report,
            "classification_report": classification_report,
            "graph_path": self._append_path(state, "evaluation_agent")
        }

    def _quality_gate(self, state: AutoMLState) -> AutoMLState:
        return {
            **state,
            "graph_path": self._append_path(state, "quality_gate")
        }

    def _route_quality(self, state: AutoMLState) -> Literal["tune", "explain"]:
        task_type = state["dataset_report"]["task_type"]
        best_score = state["evaluation_report"]["best_score"]

        threshold = (
            self.regression_threshold
            if task_type == "regression"
            else self.classification_threshold
        )

        if best_score < threshold:
            return "tune"

        return "explain"

    def _hyperparameter_tuning(self, state: AutoMLState) -> AutoMLState:
        tuning_result = self.tuning_agent.tune_best_model(
            best_model_pipeline=state["evaluation_result"]["best_model"],
            preprocessing_result=state["preprocessing_result"],
            task_type=state["dataset_report"]["task_type"]
        )

        tuning_report = self._build_tuning_report(tuning_result)

        return {
            **state,
            "tuning_result": tuning_result,
            "tuning_report": tuning_report,
            "graph_path": self._append_path(state, "hyperparameter_tuning")
        }

    def _shap_agent(self, state: AutoMLState) -> AutoMLState:
        tuning_result = state.get("tuning_result")

        if tuning_result is None:
            tuning_result = {
                "tuned": False,
                "message": "Best score met the quality gate; tuning skipped.",
                "best_model": state["evaluation_result"]["best_model"],
                "best_params": {},
                "best_cv_score": None
            }

        tuning_report = self._build_tuning_report(tuning_result)
        best_model = tuning_result["best_model"]

        feature_importance_report = (
            self.feature_importance_agent.extract_importance(
                best_model_pipeline=best_model
            )
        )

        shap_report = self.shap_agent.explain_model(
            best_model_pipeline=best_model,
            preprocessing_result=state["preprocessing_result"]
        )

        return {
            **state,
            "tuning_result": tuning_result,
            "tuning_report": tuning_report,
            "feature_importance_report": feature_importance_report,
            "shap_report": shap_report,
            "graph_path": self._append_path(state, "shap_agent")
        }

    def _recommendation_agent(self, state: AutoMLState) -> AutoMLState:
        recommendation_report = (
            self.recommendation_agent.generate_recommendation(
                evaluation_report=state["evaluation_report"],
                tuning_report=state["tuning_report"],
                dataset_report=state["dataset_report"]
            )
        )

        return {
            **state,
            "recommendation_report": recommendation_report,
            "graph_path": self._append_path(state, "recommendation_agent")
        }

    def _report_agent(self, state: AutoMLState) -> AutoMLState:
        report_generation = self.report_agent.generate_report(
            dataset_report=state["dataset_report"],
            eda_report=state["eda_summary_report"],
            preprocessing_report=state["preprocessing_report"],
            model_training_report=state["model_training_report"],
            evaluation_report=state["evaluation_report"],
            deployment_report=state["deployment_report"],
            feature_importance_report=state["feature_importance_report"]
        )

        return {
            **state,
            "report_generation": report_generation,
            "graph_path": self._append_path(state, "report_agent")
        }

    def _deploy_agent(self, state: AutoMLState) -> AutoMLState:
        deployment_report = self.deployment_agent.save_best_model(
            best_model=state["tuning_result"]["best_model"],
            best_model_name=state["evaluation_report"]["best_model_name"],
            metadata={
                "task_type": state["dataset_report"]["task_type"],
                "target_column": state["dataset_report"]["target_column"],
                "best_model_name": state["evaluation_report"]["best_model_name"],
                "best_score": state["evaluation_report"]["best_score"],
                "feature_columns": state["preprocessing_report"]["feature_columns"],
                "numerical_features": state["preprocessing_report"]["numerical_features"],
                "categorical_features": state["preprocessing_report"]["categorical_features"],
                "excluded_columns": state["preprocessing_report"]["excluded_columns"],
                "label_classes": (
                    state["preprocessing_result"]["label_encoder"]
                    .classes_
                    .tolist()
                    if state["preprocessing_result"]["label_encoder"] is not None
                    else None
                ),
                "graph_path": state.get("graph_path", [])
            }
        )

        return {
            **state,
            "deployment_report": deployment_report,
            "graph_path": self._append_path(state, "deploy_agent")
        }

    def _mlflow_agent(self, state: AutoMLState) -> AutoMLState:
        mlflow_report = self.mlflow_agent.log_experiment(
            dataset_report=state["dataset_report"],
            evaluation_report=state["evaluation_report"],
            tuning_report=state["tuning_report"],
            deployment_report=state["deployment_report"]
        )

        return {
            **state,
            "status": "success",
            "mlflow_report": mlflow_report,
            "graph_path": self._append_path(state, "mlflow_agent")
        }

    def _build_tuning_report(
        self,
        tuning_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        tuning_result = tuning_result or {}

        return {
            "tuned": tuning_result.get("tuned", False),
            "best_params": tuning_result.get("best_params", {}),
            "best_cv_score": tuning_result.get("best_cv_score"),
            "message": tuning_result.get("message")
        }

    def _response_from_state(self, state: AutoMLState) -> Dict[str, Any]:
        if state.get("status") == "unsupported":
            return {
                "status": "unsupported",
                "graph_path": state.get("graph_path", []),
                "dataset_report": state.get("dataset_report", {}),
                "unsupported_report": state.get("unsupported_report", {})
            }

        return {
            "status": state.get("status", "success"),
            "graph_path": state.get("graph_path", []),
            "warnings": state.get("warnings", []),
            "dataset_report": state["dataset_report"],
            "eda_report": state["eda_summary_report"],
            "preprocessing_report": state["preprocessing_report"],
            "model_training_report": state["model_training_report"],
            "evaluation_report": state["evaluation_report"],
            "tuning_report": state["tuning_report"],
            "deployment_report": state["deployment_report"],
            "feature_importance_report": state["feature_importance_report"],
            "shap_report": state["shap_report"],
            "report_generation": state["report_generation"],
            "mlflow_report": state["mlflow_report"],
            "recommendation_report": state["recommendation_report"],
            "confusion_matrix_report": state["confusion_matrix_report"],
            "classification_report": state["classification_report"],
            "time_series_report": state.get("time_series_report")
        }
