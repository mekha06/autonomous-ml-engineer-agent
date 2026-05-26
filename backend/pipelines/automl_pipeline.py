from agents.dataset_agent import DatasetAgent
from agents.eda_agent import EDAAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.model_agent import ModelAgent
from agents.evaluation_agent import EvaluationAgent
from agents.tuning_agent import TuningAgent
from agents.deployment_agent import DeploymentAgent
from agents.report_agent import ReportAgent
from agents.feature_importance_agent import FeatureImportanceAgent
from agents.mlflow_agent import MLflowAgent


class AutoMLPipeline:

    def run(self, file_path, target_column):

        dataset_agent = DatasetAgent()
        eda_agent = EDAAgent()
        preprocessing_agent = PreprocessingAgent()
        model_agent = ModelAgent()
        evaluation_agent = EvaluationAgent()
        tuning_agent = TuningAgent()
        deployment_agent = DeploymentAgent()
        report_agent = ReportAgent()
        feature_importance_agent = FeatureImportanceAgent()
        mlflow_agent = MLflowAgent()

        dataset_report = dataset_agent.analyze_dataset(
            file_path=file_path,
            target_column=target_column
        )

        eda_report = eda_agent.run_eda(
            file_path=file_path,
            target_column=target_column,
            task_type=dataset_report["task_type"]
        )

        preprocessing_result = preprocessing_agent.preprocess(
            file_path=file_path,
            target_column=target_column
        )

        trained_models = model_agent.train_models(
            preprocessing_result=preprocessing_result,
            task_type=dataset_report["task_type"]
        )

        evaluation_result = evaluation_agent.evaluate_models(
            trained_models=trained_models,
            preprocessing_result=preprocessing_result,
            task_type=dataset_report["task_type"]
        )

        tuning_result = tuning_agent.tune_best_model(
            best_model_pipeline=evaluation_result["best_model"],
            preprocessing_result=preprocessing_result,
            task_type=dataset_report["task_type"]
        )

        deployment_result = deployment_agent.save_best_model(
            best_model=tuning_result["best_model"],
            best_model_name=evaluation_result["best_model_name"]
        )

        feature_importance_report = feature_importance_agent.extract_importance(
            best_model_pipeline=tuning_result["best_model"]
        )

        preprocessing_report = {
            "numerical_features": preprocessing_result["numerical_features"],
            "categorical_features": preprocessing_result["categorical_features"],
            "train_rows": len(preprocessing_result["X_train"]),
            "test_rows": len(preprocessing_result["X_test"])
        }

        model_training_report = {
            "models_trained": list(trained_models.keys()),
            "total_models": len(trained_models)
        }

        evaluation_report = {
            "model_results": evaluation_result["model_results"],
            "best_model_name": evaluation_result["best_model_name"],
            "best_score": evaluation_result["best_score"]
        }

        tuning_report = {
            "tuned": tuning_result["tuned"],
            "best_params": tuning_result["best_params"],
            "best_cv_score": tuning_result.get("best_cv_score")
        }

        eda_summary_report = {
            "outlier_report": eda_report["outlier_report"],
            "plot_paths": eda_report["plot_paths"]
        }

        report_result = report_agent.generate_report(
            dataset_report=dataset_report,
            eda_report=eda_summary_report,
            preprocessing_report=preprocessing_report,
            model_training_report=model_training_report,
            evaluation_report=evaluation_report,
            deployment_report=deployment_result,
            feature_importance_report=feature_importance_report
        )

        mlflow_report = mlflow_agent.log_experiment(
            dataset_report=dataset_report,
            evaluation_report=evaluation_report,
            tuning_report=tuning_report,
            deployment_report=deployment_result
        )

        return {
            "status": "success",
            "dataset_report": dataset_report,
            "eda_report": eda_summary_report,
            "preprocessing_report": preprocessing_report,
            "model_training_report": model_training_report,
            "evaluation_report": evaluation_report,
            "tuning_report": tuning_report,
            "deployment_report": deployment_result,
            "feature_importance_report": feature_importance_report,
            "report_generation": report_result,
            "mlflow_report": mlflow_report
        }