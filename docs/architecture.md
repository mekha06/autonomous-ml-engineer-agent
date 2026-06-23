
---

# Architecture

The project follows a multi-agent architecture built using LangGraph.

Each agent has a single responsibility and passes structured outputs to the next stage.

# Agents

The Autonomous ML Engineer Agent follows a modular multi-agent architecture where each agent is responsible for a single stage of the machine learning workflow. This design keeps the pipeline organized, extensible, and easy to maintain.

---

## Dataset Agent

**Purpose:** Understands the uploaded dataset before processing begins.

**Responsibilities**

* Analyze dataset structure
* Detect target column
* Identify task type
* Detect missing values
* Check class imbalance
* Generate dataset insights

**Output:** Dataset summary and recommendations.

---

## EDA Agent

**Purpose:** Performs exploratory data analysis.

**Responsibilities**

* Statistical summary
* Distribution analysis
* Correlation analysis
* Data visualization

**Output:** EDA report and plots.

---

## Preprocessing Agent

**Purpose:** Cleans and prepares the dataset.

**Responsibilities**

* Handle missing values
* Encode categorical features
* Scale numerical features
* Remove unnecessary columns
* Split training and testing data

**Output:** Model-ready dataset.

---

## Model Agent

**Purpose:** Trains multiple machine learning models.

**Responsibilities**

* Train classification or regression models
* Compare candidate algorithms
* Store trained pipelines

**Output:** Trained models.

---

## Evaluation Agent

**Purpose:** Evaluates model performance.

**Responsibilities**

* Calculate evaluation metrics
* Compare model performance
* Select the best-performing model

**Output:** Model leaderboard and best model.

---

## Tuning Agent

**Purpose:** Improves the selected model through hyperparameter optimization.

**Responsibilities**

* Perform hyperparameter search
* Optimize model performance
* Save tuned model

**Output:** Tuned model and optimized parameters.

---

## Recommendation Agent

**Purpose:** Provides intelligent recommendations based on dataset quality and model performance.

**Responsibilities**

* Detect potential issues
* Suggest preprocessing improvements
* Recommend better modelling strategies

**Output:** Dataset and model recommendations.

---

## Feature Importance Agent

**Purpose:** Identifies the most influential features.

**Responsibilities**

* Calculate feature importance
* Rank important features
* Generate feature importance plots

**Output:** Ranked feature importance.

---

## SHAP Agent

**Purpose:** Explains model predictions using SHAP.

**Responsibilities**

* Generate SHAP values
* Create explainability plots
* Interpret feature contributions

**Output:** SHAP visualizations and explanations.

---

## Confusion Matrix Agent

**Purpose:** Visualizes classification performance.

**Responsibilities**

* Generate confusion matrix
* Highlight prediction errors

**Output:** Confusion matrix visualization.

---

## Classification Report Agent

**Purpose:** Generates detailed classification metrics.

**Responsibilities**

* Precision
* Recall
* F1-score
* Support

**Output:** Classification report.

---

## Report Agent

**Purpose:** Creates the final consolidated project report.

**Responsibilities**

* Combine outputs from all agents
* Generate downloadable report
* Summarize the ML workflow

**Output:** PDF/HTML report.

---

## MLflow Agent

**Purpose:** Tracks machine learning experiments.

**Responsibilities**

* Log parameters
* Log metrics
* Save artifacts
* Register trained models

**Output:** Experiment history.

---

## Deployment Agent

**Purpose:** Prepares the best-performing model for deployment.

**Responsibilities**

* Save trained model
* Export deployment artifacts
* Prepare prediction pipeline

**Output:** Deployment-ready model.

---

# Workflow

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
Tuning Agent
   │
   ▼
Recommendation Agent
   │
   ├──────────────┐
   ▼              ▼
Feature       SHAP Agent
Importance        │
Agent             │
   │              │
   └──────┬───────┘
          ▼
Confusion Matrix Agent
          │
          ▼
Classification Report Agent
          │
          ▼
Report Agent
          │
          ▼
MLflow Agent
          │
          ▼
Deployment Agent
```
