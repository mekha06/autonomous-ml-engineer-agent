# Autonomous ML Engineer Agent

An agentic AI system that automates the complete machine learning workflow using **LangGraph** and multiple specialized agents.

The project accepts a dataset and task description, analyzes the data, preprocesses it, trains multiple machine learning models, evaluates their performance, explains predictions, generates reports, and logs experiments automatically.

---

# Features

* Dataset Analysis
* Exploratory Data Analysis (EDA)
* Data Preprocessing
* Automatic Task Detection
* Multi-model Training
* Hyperparameter Tuning
* Model Evaluation
* SHAP Explainability
* Automated Report Generation
* MLflow Experiment Tracking
* LangGraph Agent Orchestration
* Dockerized Deployment

---

# Tech Stack

* Python
* FastAPI
* Streamlit
* LangGraph
* Scikit-learn
* XGBoost
* SHAP
* MLflow
* Docker
* Docker Compose

---

# Project Structure

```text
backend/
    agents/
    reports/
    models/
    uploads/
    main.py

frontend/
    app.py

docker-compose.yml
README.md
```

---

# Getting Started

Clone the repository

```bash
git clone <repository-url>
cd autonomous-ml-engineer-agent
```

Build the Docker containers

```bash
docker compose build
```

Start the application

```bash
docker compose up
```

Frontend

```
http://localhost:8501
```

Backend API

```
http://localhost:8000/docs
```

---

# Agent Workflow

```text
Dataset
    │
    ▼
Dataset Agent
    │
    ▼
EDA Agent
    │
    ▼
Preprocessing Agent
    │
    ▼
Model Agent
    │
    ▼
Evaluation Agent
    │
    ▼
SHAP Agent
    │
    ▼
Report Agent
    │
    ▼
MLflow Agent
```

---

# What I Learned

This project became my introduction to building production-style AI systems beyond traditional machine learning.

Through this project I learned:

* Agent orchestration using LangGraph
* Designing multi-agent workflows
* Building REST APIs with FastAPI
* Creating interactive interfaces using Streamlit
* Containerizing applications using Docker
* Managing multi-container applications with Docker Compose
* Environment configuration using environment variables
* Experiment tracking using MLflow
* Model explainability using SHAP
* Structuring scalable ML projects

---

# Future Improvements

* Persistent Agent Memory
* Reflection Agent
* Dynamic Graph Routing
* CI/CD using GitHub Actions
* Cloud Deployment
* Monitoring and Logging
* Production Health Checks

---

# Author

**Mekha S R**
**AI and ML**
**SCTCE**
