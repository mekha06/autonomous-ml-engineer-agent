import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Autonomous ML Engineer Agent",
    layout="wide"
)

st.title("Autonomous ML Engineer Agent")

st.markdown(
    """
Upload a dataset and let the AI agent:

- Analyze the dataset
- Perform EDA
- Preprocess features
- Train multiple models
- Tune hyperparameters
- Generate explanations
- Track experiments
- Create deployment artifacts
"""
)

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

target_column = st.text_input(
    "Enter Target Column"
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])


if st.button("Run AutoML"):

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

            data = {
                "target_column": target_column
            }

            response = requests.post(
                f"{API_URL}/upload",
                files=files,
                data=data
            )

        if response.status_code == 200:

            result = response.json()

            st.success("AutoML Pipeline Completed Successfully")

            dataset_report = result.get("dataset_report", {})
            evaluation_report = result.get("evaluation_report", {})
            tuning_report = result.get("tuning_report", {})
            model_training_report = result.get("model_training_report", {})
            feature_importance_report = result.get(
                "feature_importance_report",
                {}
            )
            shap_report = result.get("shap_report", {})
            mlflow_report = result.get("mlflow_report", {})

            st.subheader("Pipeline Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Task Type",
                    dataset_report.get("task_type", "N/A")
                )

            with col2:
                st.metric(
                    "Best Model",
                    evaluation_report.get("best_model_name", "N/A")
                )

            with col3:
                score = evaluation_report.get("best_score", 0)
                st.metric("Best Score", round(score, 4))

            st.divider()

            st.subheader("Models Trained")
            st.write(model_training_report.get("models_trained", []))

            st.subheader("Hyperparameter Tuning")
            st.write("Tuned:", tuning_report.get("tuned"))
            st.json(tuning_report.get("best_params", {}))

            st.subheader("Feature Importance")

            top_features = feature_importance_report.get("top_features", [])

            if top_features:
                st.dataframe(top_features)
            else:
                st.info("Feature importance not available.")

            st.subheader("SHAP Explainability")

            if shap_report.get("available"):
                st.success(
                    shap_report.get("message", "SHAP generated.")
                )
            else:
                st.warning(
                    shap_report.get("message", "SHAP unavailable.")
                )

            st.subheader("Generated Visualizations")

            plot_names = [
                "target_distribution.png",
                "correlation_matrix.png",
                "feature_importance.png",
                "shap_summary.png"
            ]

            for plot_name in plot_names:
                plot_url = f"{API_URL}/plot/{plot_name}"

                try:
                    plot_response = requests.get(plot_url)

                    if plot_response.status_code == 200:
                        st.image(
                            plot_response.content,
                            caption=plot_name
                        )

                except Exception:
                    pass

            st.subheader("MLflow Tracking")
            st.json(mlflow_report)

            with st.expander("View Full Raw Response"):
                st.json(result)

        else:
            st.error("AutoML Pipeline Failed")
            st.code(response.text)