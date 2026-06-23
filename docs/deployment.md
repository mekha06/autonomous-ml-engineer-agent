# Deployment Guide

This document explains how the Autonomous ML Engineer Agent is deployed, from local development to production.

---

# Deployment Architecture

The application follows a multi-container architecture.

```text
                 User
                   │
                   ▼
        Streamlit Frontend
             (Port 8501)
                   │
         HTTP API Requests
                   │
                   ▼
          FastAPI Backend
             (Port 8000)
                   │
                   ▼
              LangGraph
                   │
        Multi-Agent Workflow
                   │
                   ▼
         Models • Reports • MLflow
```

The frontend is responsible for user interaction, while the backend handles machine learning workflows and agent orchestration.

---

# Technology Stack

| Component                  | Technology     |
| -------------------------- | -------------- |
| Frontend                   | Streamlit      |
| Backend                    | FastAPI        |
| Agent Orchestration        | LangGraph      |
| Experiment Tracking        | MLflow         |
| Containerization           | Docker         |
| Multi-Container Management | Docker Compose |

---

# Local Development

Clone the repository.

```bash
git clone https://github.com/<username>/autonomous-ml-engineer-agent.git

cd autonomous-ml-engineer-agent
```

---

## Build Docker Images

```bash
docker compose build
```

Docker builds separate images for

* Backend
* Frontend

---

## Start Containers

```bash
docker compose up
```

or

```bash
docker compose up --build
```

The application starts two services.

| Service   | Port |
| --------- | ---- |
| Streamlit | 8501 |
| FastAPI   | 8000 |

---

# Docker Architecture

Each application component runs inside its own container.

```text
┌─────────────────────┐
│ Frontend Container  │
│     Streamlit       │
└──────────┬──────────┘
           │
           │ API_URL
           ▼
┌─────────────────────┐
│ Backend Container   │
│      FastAPI        │
└──────────┬──────────┘
           │
           ▼
      LangGraph
           │
           ▼
     ML Agent Pipeline
```

Containers communicate using Docker's internal network.

The frontend communicates with the backend using

```text
http://backend:8000
```

instead of

```text
http://127.0.0.1:8000
```

because each container has its own isolated network namespace.

---

# Docker Volumes

Persistent directories are mounted as Docker volumes.

```text
backend/uploads
backend/models
backend/reports
backend/mlruns
```

This ensures that generated models, reports, uploaded datasets and MLflow experiments remain available even after restarting containers.

---

# Environment Variables

Application configuration is managed through environment variables.

Example

```env
API_URL=http://backend:8000

UPLOAD_DIR=uploads

MODEL_DIR=models

REPORT_DIR=reports

MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

This allows the same codebase to run in multiple environments without changing the source code.

---

# Health Check

The backend exposes

```text
GET /health
```

This endpoint verifies that the backend is running and available to process requests.

Health checks are commonly used by cloud platforms to determine whether a service should continue receiving traffic or be restarted.

---

# Deployment Workflow

The deployment process follows the standard software engineering workflow.

```text
Development

        │

        ▼

Docker Build

        │

        ▼

Local Testing

        │

        ▼

GitHub Repository

        │

        ▼

GitHub Actions (CI)

        │

        ▼

Cloud Deployment
```

Each stage validates the application before it progresses to production.

---

# Continuous Integration

Every code change is pushed to GitHub.

GitHub Actions can automatically

* Build Docker images
* Validate dependencies
* Detect build failures
* Prepare the application for deployment

This ensures the repository always remains deployable.

---

# Production Deployment

The project is designed to be deployed using Docker-compatible cloud platforms.

Recommended platforms include

* Render
* Azure Container Apps
* Google Cloud Run
* Railway
* AWS ECS

Each platform runs the backend and frontend as separate services while preserving the same architecture used during local development.

---

# Deployment Strategy

The frontend and backend are deployed independently.

```text
GitHub Repository

      │

 ┌────┴────┐

 ▼         ▼

Frontend   Backend

(Streamlit) (FastAPI)

      │

      └────HTTP────►
```

This separation allows each service to scale, update and restart independently.

---

# Lessons Learned

During the deployment process, several production engineering concepts were explored.

* Writing Dockerfiles
* Building Docker images
* Running containers
* Managing multi-container applications using Docker Compose
* Container networking
* Environment variable management
* Persistent Docker volumes
* Health endpoints
* Service isolation
* Preparing applications for cloud deployment

These concepts transformed the project from a locally executed machine learning application into a deployment-ready AI system.

---

# Future Improvements

The deployment pipeline will continue to evolve with

* Automated CI/CD using GitHub Actions
* Automatic cloud deployment
* Production logging
* Monitoring and alerting
* Container health monitoring
* HTTPS support

These enhancements will further improve the reliability, maintainability and scalability of the application.
