import pandas as pd
import requests
import streamlit as st
import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

def show_graph_route(result):
    graph_path = result.get("graph_path", [])

    agent_names = {
        "dataset_analysis": "Dataset Agent",
        "time_series_agent": "Time Series Agent",
        "eda_agent": "EDA Agent",
        "preprocessing_agent": "Preprocessing Agent",
        "model_selection": "Model Selection Agent",
        "classification_agent": "Model Training Agent",
        "regression_agent": "Model Training Agent",
        "evaluation_agent": "Evaluation Agent",
        "quality_gate": "Quality Gate Agent",
        "hyperparameter_tuning": "Hyperparameter Tuning Agent",
        "shap_agent": "SHAP Agent",
        "recommendation_agent": "Recommendation Agent",
        "deploy_agent": "Deployment Agent",
        "report_agent": "Report Agent",
        "mlflow_agent": "MLflow Agent"
    }

    st.subheader("Agent Execution Flow")

    if graph_path:

        progress = len(graph_path)
        total_steps = 11

        st.progress(min(progress / total_steps, 1.0))

        for step in graph_path:
            st.success(
                f" {agent_names.get(step, step)} Completed"
            )

    else:
        st.warning(
            "No agent execution path found. "
            "Run AutoML again."
        )

def show_report_download(report_generation, key):
    report_filename = (
        report_generation.get("pdf_report_filename")
        or report_generation.get("report_filename")
    )

    st.subheader("Report Download")

    if not report_filename:
        st.warning(
            "No report filename found in the backend response. "
            "Run AutoML again after restarting the backend."
        )
        return

    report_response = requests.get(
        f"{API_URL}/download-report/{report_filename}"
    )

    if report_response.status_code != 200:
        st.warning("Report was generated, but download failed.")
        st.code(report_response.text)
        return

    is_pdf = report_filename.lower().endswith(".pdf")

    st.download_button(
        label="Download PDF Report" if is_pdf else "Download HTML Report",
        data=report_response.content,
        file_name=report_filename,
        mime="application/pdf" if is_pdf else "text/html",
        key=key
    )

st.set_page_config(
    page_title="Autonomous ML Engineer Agent",
    layout="wide"
)

st.title("Autonomous ML Engineer Agent")

st.markdown(
    """
Upload a dataset and let the AI agent analyze, train, tune, explain, track, and deploy ML models automatically.
"""
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Training",
    "Visualizations",
    "Prediction",
    "Experiments"
])


with tab1:
    st.header("Training Pipeline")

    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    target_column = st.text_input("Enter Target Column")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

    if st.button("Run AutoML", key="run_automl_button"):

        if uploaded_file is None:
            st.error("Please upload a CSV file.")

        elif not target_column:
            st.error("Please enter target column.")

        else:
            with st.spinner("Running AutoML Pipeline..."):
                uploaded_file.seek(0)

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file,
                        "text/csv"
                    )
                }

                data = {"target_column": target_column}

                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                    data=data
                )

            if response.status_code == 200:
                result = response.json()
                st.session_state["automl_result"] = result
                st.success("AutoML Pipeline Completed Successfully")

            else:
                st.error("AutoML Pipeline Failed")
                st.code(response.text)

    if "automl_result" in st.session_state:
        result = st.session_state["automl_result"]

        dataset_report = result.get("dataset_report", {})
        evaluation_report = result.get("evaluation_report", {})
        tuning_report = result.get("tuning_report", {})
        model_training_report = result.get("model_training_report", {})
        preprocessing_report = result.get("preprocessing_report", {})
        recommendation_report = result.get("recommendation_report", {})
        report_generation = result.get("report_generation", {})
        time_series_report = result.get("time_series_report", {})

        target_name = dataset_report.get("target_column", "Prediction")

        st.divider()

        st.subheader("Pipeline Summary")

        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            st.metric("Target Column", target_name)

        with row1_col2:
            st.metric("Task Type", dataset_report.get("task_type", "N/A"))

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            st.metric(
                "Best Model",
                evaluation_report.get("best_model_name", "N/A")
            )

        with row2_col2:
            score = evaluation_report.get("best_score", 0)
            st.metric("Best Score", round(score, 4))

        show_graph_route(result)
        show_report_download(report_generation, key="training_report_download")

        model_results = evaluation_report.get("model_results", {})

        if model_results:
            st.subheader("Model Leaderboard")
            leaderboard_df = pd.DataFrame(model_results).T
            st.dataframe(leaderboard_df, use_container_width=True)

        st.subheader("Model Recommendation")

        recommended_model = recommendation_report.get(
            "recommended_model",
            evaluation_report.get("best_model_name", "N/A")
        )

        st.success(f"Recommended Model: {recommended_model}")

        for reason in recommendation_report.get("reasons", []):
            st.write(f"- {reason}")

        st.subheader("Dataset Health Check")

        health_col1, health_col2, health_col3 = st.columns(3)

        with health_col1:
            st.metric(
                "Imbalance Status",
                dataset_report.get("imbalance_status", "N/A")
            )

        with health_col2:
            st.metric(
                "Time Series Candidate",
                str(dataset_report.get("is_time_series_candidate", False))
            )

        with health_col3:
            st.metric(
                "ID-like Columns",
                len(dataset_report.get("id_like_columns", []))
            )

        warnings = dataset_report.get("warnings", [])

        if warnings:
            st.warning("Dataset Health Warnings Detected")
            for warning in warnings:
                st.write(f"- {warning}")
        else:
            st.success("No major dataset health warnings detected.")

        if time_series_report:
            st.subheader("Time Series Check")
            st.info(time_series_report.get("message", "No time-series route used."))
            datetime_columns = time_series_report.get(
                "datetime_like_columns",
                []
            )

            if datetime_columns:
                st.write("Datetime-like columns:", datetime_columns)

        excluded_columns = preprocessing_report.get("excluded_columns", [])

        if excluded_columns:
            st.subheader("Leakage Protection")
            st.warning(
                "Excluded ID-like feature columns: "
                + ", ".join(excluded_columns)
            )

        if preprocessing_report.get("stratified_split"):
            st.info("Classification split was stratified.")

        st.subheader("Models Trained")
        st.write(model_training_report.get("models_trained", []))

        st.subheader("Hyperparameter Tuning")
        if tuning_report.get("tuned"):
            st.success("Hyperparameter tuning applied.")
        else:
            st.info(
                tuning_report.get(
                    "message",
                    "Hyperparameter tuning was skipped."
                )
            )

        st.json(tuning_report.get("best_params", {}))


with tab2:
    st.header("Visualizations")

    if "automl_result" not in st.session_state:
        st.info("Run AutoML first to generate visualizations.")

    else:
        result = st.session_state["automl_result"]

        feature_importance_report = result.get(
            "feature_importance_report",
            {}
        )

        shap_report = result.get("shap_report", {})
        classification_report = result.get("classification_report", {})

        st.subheader("Feature Importance")

        top_features = feature_importance_report.get("top_features", [])

        if top_features:
            st.dataframe(top_features)
        else:
            st.info("Feature importance not available.")

        st.subheader("SHAP Explainability")

        if shap_report.get("available"):
            st.success(shap_report.get("message", "SHAP generated."))
        else:
            st.warning(shap_report.get("message", "SHAP unavailable."))

        st.subheader("Generated Plots")

        plot_names = [
            "target_distribution.png",
            "correlation_matrix.png",
            "feature_importance.png",
            "shap_summary.png",
            "confusion_matrix.png"
        ]

        for plot_name in plot_names:
            plot_url = f"{API_URL}/plot/{plot_name}"

            try:
                plot_response = requests.get(plot_url)

                if plot_response.status_code == 200:
                    st.image(
                        plot_response.content,
                        caption=plot_name,
                        use_container_width=True
                    )
                else:
                    st.info(f"{plot_name} not available.")

            except Exception:
                st.warning(f"Could not load {plot_name}.")

        if classification_report.get("available"):
            st.subheader("Classification Report")
            report_df = pd.DataFrame(classification_report.get("report", {})).T
            st.dataframe(report_df, use_container_width=True)


with tab3:
    st.header("Prediction")

    if "automl_result" not in st.session_state:
        st.info("Train a model first using Run AutoML.")

    else:
        result = st.session_state["automl_result"]

        dataset_report = result.get("dataset_report", {})
        preprocessing_report = result.get("preprocessing_report", {})

        target_name = dataset_report.get("target_column", "Prediction")

        st.info(f"Enter feature values to predict: {target_name}")

        numerical_features = preprocessing_report.get("numerical_features", [])
        categorical_features = preprocessing_report.get("categorical_features", [])
        categorical_options = preprocessing_report.get("categorical_options", {})

        prediction_data = {}

        with st.expander("Numerical Features", expanded=True):
            if numerical_features:
                for feature in numerical_features:
                    prediction_data[feature] = st.number_input(
                        label=feature,
                        value=0.0,
                        key=f"num_{feature}"
                    )
            else:
                st.info("No numerical features found.")

        with st.expander("Categorical Features", expanded=True):
            if categorical_features:
                for feature in categorical_features:
                    options = categorical_options.get(feature, [])

                    if options:
                        prediction_data[feature] = st.selectbox(
                            label=feature,
                            options=options,
                            key=f"cat_{feature}"
                        )
                    else:
                        prediction_data[feature] = st.text_input(
                            label=feature,
                            value="",
                            key=f"cat_{feature}"
                        )
            else:
                st.info("No categorical features found.")

        with st.expander("Prediction Payload"):
            st.json(prediction_data)

        if st.button("Predict", key="predict_button"):
            predict_response = requests.post(
                f"{API_URL}/predict",
                json={"data": prediction_data}
            )

            if predict_response.status_code == 200:
                prediction_result = predict_response.json()
                predicted_value = prediction_result["prediction"][0]

                st.success("Prediction completed successfully.")

                st.metric(
                    label=f"Predicted {target_name}",
                    value=predicted_value
                )

            else:
                st.error("Prediction failed.")
                st.code(predict_response.text)


with tab4:
    st.header("Experiments and Raw Outputs")

    if "automl_result" not in st.session_state:
        st.info("Run AutoML first to view experiment details.")

    else:
        result = st.session_state["automl_result"]

        mlflow_report = result.get("mlflow_report", {})
        deployment_report = result.get("deployment_report", {})
        report_generation = result.get("report_generation", {})
        show_graph_route(result)

        st.subheader("MLflow Tracking")
        st.json(mlflow_report)

        st.subheader("Deployment Report")
        st.json(deployment_report)

        st.subheader("Report Generation")
        st.json(report_generation)

        show_report_download(report_generation, key="experiments_report_download")

        with st.expander("View Full Raw Response"):
            st.json(result)
