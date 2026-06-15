import pandas as pd
import requests
import streamlit as st

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
    preprocessing_report = result.get("preprocessing_report", {})
    evaluation_report = result.get("evaluation_report", {})
    tuning_report = result.get("tuning_report", {})
    model_training_report = result.get("model_training_report", {})
    feature_importance_report = result.get("feature_importance_report", {})
    shap_report = result.get("shap_report", {})
    mlflow_report = result.get("mlflow_report", {})

    target_name = dataset_report.get("target_column", "Prediction")

    st.subheader("Pipeline Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Target Column", target_name)

    with col2:
        st.metric("Task Type", dataset_report.get("task_type", "N/A"))

    with col3:
        st.metric("Best Model", evaluation_report.get("best_model_name", "N/A"))

    with col4:
        score = evaluation_report.get("best_score", 0)
        st.metric("Best Score", round(score, 4))

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
        st.success(shap_report.get("message", "SHAP generated."))
    else:
        st.warning(shap_report.get("message", "SHAP unavailable."))

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
                st.image(plot_response.content, caption=plot_name)

        except Exception:
            pass

    st.subheader("MLflow Tracking")
    st.json(mlflow_report)

    with st.expander("View Full Raw Response"):
        st.json(result)

    st.divider()

    st.subheader("Make Prediction")

    st.info(f"Enter feature values to predict: {target_name}")

    numerical_features = preprocessing_report.get("numerical_features", [])
    categorical_features = preprocessing_report.get("categorical_features", [])
    categorical_options = preprocessing_report.get("categorical_options", {})

    prediction_data = {}

    for feature in numerical_features:
        prediction_data[feature] = st.number_input(
            label=feature,
            value=0.0,
            key=f"num_{feature}"
        )

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

    if st.button("Predict", key="predict_button"):

        predict_response = requests.post(
            f"{API_URL}/predict",
            json={"data": prediction_data}
        )

        if predict_response.status_code == 200:
            prediction_result = predict_response.json()
            predicted_value = prediction_result["prediction"][0]

            st.success("Prediction completed successfully!")
            st.metric(
                label=f"Predicted {target_name}",
                value=predicted_value
            )

        else:
            st.error("Prediction failed.")
            st.code(predict_response.text)

else:
    st.divider()
    st.subheader("Make Prediction")
    st.info("Train a model first using Run AutoML, then prediction fields will appear.")