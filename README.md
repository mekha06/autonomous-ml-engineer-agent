An intelligent multi-agent AutoML platform that accepts raw CSV datasets, automatically analyzes the data, performs preprocessing, trains multiple machine learning models, evaluates performance, selects the best model, generates explainable reports, and exposes a prediction API using FastAPI.

Features
Current MVP Features
CSV Dataset Upload
Automatic Dataset Understanding
Classification / Regression Task Detection
Missing Value Detection
Duplicate Row Detection
Numerical & Categorical Feature Detection
Automated EDA
Correlation Matrix Generation
Outlier Detection
Automatic Preprocessing Pipeline
Multiple Model Training
Automatic Best Model Selection
Feature Importance Analysis
Model Persistence using Joblib
FastAPI Prediction Endpoint
HTML Report Generation
EDA Plot Generation
Project Architecture
User Uploads CSV
        ↓
Dataset Agent
        ↓
EDA Agent
        ↓
Preprocessing Agent
        ↓
Model Training Agent
        ↓
Evaluation Agent
        ↓
Feature Importance Agent
        ↓
Deployment Agent
        ↓
Report Agent
        ↓
Prediction API
Tech Stack
Backend
Python
FastAPI
Scikit-learn
Pandas
NumPy
Matplotlib
ML Models
Linear Regression
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
Model Persistence
Joblib
Planned Additions
XGBoost
Optuna
MLflow
Docker
LangGraph
SHAP
Streamlit Dashboard
Folder Structure
ml_engineer_agent/
│
├── backend/
│   ├── agents/
│   │   ├── dataset_agent.py
│   │   ├── eda_agent.py
│   │   ├── preprocessing_agent.py
│   │   ├── model_agent.py
│   │   ├── evaluation_agent.py
│   │   ├── feature_importance_agent.py
│   │   ├── deployment_agent.py
│   │   └── report_agent.py
│   │
│   ├── pipelines/
│   │   └── automl_pipeline.py
│   │
│   ├── schemas/
│   │   └── prediction_schema.py
│   │
│   ├── uploads/
│   ├── models/
│   ├── reports/
│   │   └── plots/
│   │
│   ├── config.py
│   ├── main.py
│   └── requirements.txt
│
└── README.md
Installation
Clone Repository
git clone <your-repo-url>
cd ml_engineer_agent/backend
Install Dependencies
pip install -r requirements.txt
Run FastAPI Server
python -m uvicorn main:app --reload
API Documentation

After starting the server:

http://127.0.0.1:8000/docs
API Endpoints
Upload Dataset & Train Models
Endpoint
POST /upload
Parameters
Parameter	Type	Description
file	CSV File	Dataset
target_column	String	Column to predict
Prediction Endpoint
Endpoint
POST /predict
Example Input
{
  "data": {
    "Hours Studied": 7,
    "Previous Scores": 85,
    "Extracurricular Activities": "Yes",
    "Sleep Hours": 7,
    "Sample Question Papers Practiced": 5
  }
}
Example Output
{
  "prediction": [77.39]
}
Generated Outputs
Models
backend/models/

Contains:

best_model.pkl
timestamped trained models
Reports
backend/reports/

Contains:

HTML reports
EDA visualizations
feature importance plots
Current Workflow
CSV Upload
↓
Dataset Analysis
↓
EDA Generation
↓
Preprocessing
↓
Model Training
↓
Evaluation
↓
Best Model Selection
↓
Feature Importance Extraction
↓
Model Saving
↓
HTML Report Generation
↓
Prediction API
Future Improvements
ML Enhancements
Hyperparameter Tuning
XGBoost Integration
Ensemble Models
Auto Feature Selection
Explainability
SHAP Explainability
Model Interpretability Dashboard
MLOps
MLflow Experiment Tracking
Docker Containerization
CI/CD Pipelines
Agentic AI
LangGraph Workflow Orchestration
Autonomous Experiment Retry
AI-driven Model Recommendation
Frontend
Streamlit Dashboard
React Frontend
Drag-and-Drop Dataset Upload
