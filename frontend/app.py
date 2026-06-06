import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Autonomous ML Engineer Agent",
    layout="wide"
)

st.title("Autonomous ML Engineer Agent")
st.write("Upload a CSV dataset and let the agent train, evaluate, explain, and deploy the best ML model.")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
target_column = st.text_input("Enter Target Column")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

if st.button("Run AutoML"):
    if uploaded_file is None:
        st.error("Please upload a CSV file.")
    elif not target_column:
        st.error("Please enter the target column.")
    else:
        with st.spinner("Running AutoML pipeline..."):
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

            st.success("AutoML pipeline completed successfully!")

            st.subheader("Dataset Report")
            st.json(result.get("dataset_report", {}))

            st.subheader("EDA Report")
            st.json(result.get("eda_report", {}))

            st.subheader("Preprocessing Report")
            st.json(result.get("preprocessing_report", {}))

            st.subheader("Model Training Report")
            st.json(result.get("model_training_report", {}))

            st.subheader("Evaluation Report")
            st.json(result.get("evaluation_report", {}))

            st.subheader("Tuning Report")
            st.json(result.get("tuning_report", {}))

            st.subheader("Feature Importance Report")
            st.json(result.get("feature_importance_report", {}))

            st.subheader("SHAP Report")
            st.json(result.get("shap_report", {}))

            st.subheader("Deployment Report")
            st.json(result.get("deployment_report", {}))

            st.subheader("Report Generation")
            st.json(result.get("report_generation", {}))

            st.subheader("MLflow Report")
            st.json(result.get("mlflow_report", {}))

        else:
            st.error("AutoML pipeline failed.")
            st.write(response.text)