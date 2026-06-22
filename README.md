# ML Engineer Agent

ML Engineer Agent is a multi-agent AutoML platform that automates the machine learning workflow from dataset upload to model deployment.

Simply upload a CSV file and specify the target column. The system automatically analyzes the dataset, performs preprocessing, trains multiple machine learning models, evaluates their performance, selects the best model, generates reports, and exposes a prediction API.

## Features

* Automated dataset analysis
* Classification and regression task detection
* Missing value and duplicate detection
* Automated EDA and visualization generation
* Data preprocessing pipeline
* Training of multiple ML models
* Automatic best model selection
* Feature importance analysis
* Model persistence using Joblib
* FastAPI prediction endpoint
* HTML report generation

## Workflow

Dataset Upload → Data Analysis → EDA → Preprocessing → Model Training → Evaluation → Best Model Selection → Report Generation → Prediction API

## Tech Stack

**Backend:** Python, FastAPI

**Machine Learning:** Scikit-learn, Pandas, NumPy

**Visualization:** Matplotlib

**Models:** Linear Regression, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting

## Why I Built This

While learning Machine Learning, I noticed that a lot of time is spent on repetitive tasks like data analysis, preprocessing, model training, and evaluation. I wanted to explore how an agent-based system could automate these steps and make experimentation faster.

## Future Plans

* XGBoost Integration
* Hyperparameter Tuning with Optuna
* SHAP Explainability
* MLflow Tracking
* Docker Deployment
* LangGraph Workflow Orchestration
* Streamlit / React Dashboard
