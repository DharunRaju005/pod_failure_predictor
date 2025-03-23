# Kubernetes Pod Failure Detection Project Documentation

This document provides an in-depth overview of the Kubernetes Pod Failure Detection project, a machine learning-based solution designed to predict pod failures in a Kubernetes cluster. The project leverages metrics from Prometheus, a pre-trained XGBoost model, and a Flask-based API to classify pod statuses, enhancing cluster reliability and operational efficiency.

## 📌 Project Overview

The Kubernetes Pod Failure Detection project aims to proactively identify and predict pod failures within a Kubernetes cluster using real-time metrics and machine learning. It integrates with Prometheus for metrics collection and employs an XGBoost model to classify pod statuses into seven categories: `ContainerCreating`, `CrashLoopBackOff`, `Error`, `NotFound`, `Pending`, `Running`, and `Unknown`. The system is deployed as a Flask application within a Docker container, providing a scalable and accessible prediction API.

### 🎯 Objectives
- **Predict Pod Failures:** Classify pod statuses using real-time metrics to anticipate failures.
- **Monitor Cluster Health:** Utilize Prometheus to collect and analyze pod and node metrics.
- **Provide Actionable Insights:** Deliver predictions with confidence levels and probability distributions for operational decision-making.

---

## 📁 Directory Structure

The project directory structure is as follows:

```plaintext
D:\ML\kubernetesPodFailureDetection\devxV1\devx
├── .venv/                          # Virtual environment for Python dependencies
├── app/                            # Application source code
│   └── predict.py                  # Main Flask application for prediction
├── model/                          # Directory for model artifacts
│   ├── categorical_cols.txt        # List of categorical columns used in training
│   ├── feature_columns.txt         # List of feature columns used in training
│   ├── label_encoder.pkl           # Pickle file for the label encoder
│   └── model.pkl                   # Trained XGBoost model
├── .dockerignore                   # Docker ignore file
├── Dockerfile                      # Dockerfile for building the Flask application image
└── requirements.txt                # Python dependencies for the application
```

### 🔍 Directory Details
- **`.venv/`**: Contains the virtual environment for Python dependencies, ensuring isolated dependency management.
- **`app/predict.py`**: The core Flask application that handles prediction requests, fetches metrics from Prometheus, and uses the trained model to predict pod statuses.
- **`model/`:**
  - `categorical_cols.txt`: Stores the names of categorical columns used during training (e.g., `Node Name`, `Pod Event Reason`).
  - `feature_columns.txt`: Lists all feature columns used by the model after preprocessing and feature engineering.
  - `label_encoder.pkl`: A serialized `LabelEncoder` object used to encode pod statuses into numerical values.
  - `model.pkl`: The trained XGBoost model saved using `joblib`.
- **`.dockerignore`**: Excludes unnecessary files (e.g., `.venv/`, `__pycache__`) from the Docker build context.
- **`Dockerfile`**: Defines the steps to build the Docker image for the Flask application.
- **`requirements.txt`**: Lists Python dependencies, including `pandas`, `numpy`, `xgboost`, `joblib`, `flask`, `scikit-learn`, and `requests`.

---

## ⚙️ Architecture

The architecture of the Kubernetes Pod Failure Detection system is modular, scalable, and tightly integrated with Kubernetes and Prometheus.

### 🔥 High-Level Architecture Diagram

```plaintext
+-------------------+       +-------------------+       +-------------------+
| Kubernetes Cluster|       | Prometheus        |       | Flask Application |
|                   |       |                   |       | (predict.py)      |
|  +-------------+  |       |  +-------------+  |       |  +-------------+  |
|  | Pods        |  |<----->|  | Metrics     |  |<----->|  | API Endpoints|  |
|  | (Node.js)   |  | scrape |  | (CPU, Memory)|  | query |  | /predict     |  |
|  +-------------+  |       |  +-------------+  |       |  | /test-prometheus|  |
|                   |       |                   |       |  +-------------+  |
|  +-------------+  |       |  +-------------+  |       |  +-------------+  |
|  | Node Exporter|  |<----->|  | Node Metrics|  |       |  | XGBoost Model|  |
|  +-------------+  | scrape |  +-------------+  |       |  +-------------+  |
+-------------------+       +-------------------+       +-------------------+
```

### 🔧 Components

#### 1️⃣ Kubernetes Cluster
- **Purpose:** Hosts all application and monitoring components.
- **Subcomponents:**
  - **Pods (Node.js Application):**
    - Exposes custom metrics via `/metrics` endpoint using `prom-client`.
    - Runs on port `8080`.
  - **Node Exporter:**
    - Deployed as a `DaemonSet`.
    - Collects node-level metrics (CPU, memory) on port `9100`.
  - **Kube-State Metrics:**
    - Provides Kubernetes-specific metrics on port `8080`.

#### 2️⃣ Prometheus
- **Purpose:** Collects and stores time-series metrics from Kubernetes.
- **Configuration:**
  - Deployed via `prometheus-deployment.yaml`.
  - Exposed on port `9090`.
- **Scrape Jobs:**
  - `node-app`: Scrapes Node.js app.
  - `kube-state-metrics`: Scrapes Kubernetes state metrics.

#### 3️⃣ Flask Application (`predict.py`)
- **Purpose:** Serves as the prediction API.
- **API Endpoints:**
  - `/predict`: Predicts pod status with confidence levels.
  - `/test-prometheus`: Tests Prometheus connectivity.

#### 4️⃣ XGBoost Model
- **Purpose:** Multi-class classification of pod statuses.
- **Training:**
  - `n_estimators=300`, `learning_rate=0.05`, `max_depth=6`.
- **Deployment:**
  - Saved as `model.pkl`.

#### 5️⃣ Docker
- **Purpose:** Containerizes the Flask application.
- **Configuration:**
  - Base image: `python:3.9-slim`.
  - Runs on port `6000`.

---

## 🚀 Deployment

### Docker
```bash
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --timeout=900 --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY model/ ./model/
EXPOSE 6000
CMD ["flask", "run", "--host=0.0.0.0", "--port=6000"]
```

### Kubernetes
```bash
kubectl apply -f prometheus-deployment.yaml
kubectl apply -f prometheus-rbac.yaml
kubectl apply -f node-exporter.yaml
```

---

## 🛠️ Usage

### Predicting Pod Status
```bash
curl -X POST -H "Content-Type: application/json" -d '{"pod_name": "my-pod", "namespace": "default"}' http://localhost:6000/predict
```

### Testing Prometheus Connectivity
```bash
curl http://localhost:6000/test-prometheus
```

---

## 📊 Performance Metrics

- **Model Accuracy:** 97.95% on the test set.
- **Top 10 Feature Importances:**
```plaintext
Ready Containers: 0.183854
Pod Event Reason_Killing: 0.175592
Total Containers: 0.174264
```

---

## 🚀 Future Improvements
- Real-Time Event Integration.
- Model Optimization.
- Alerting with Alertmanager.

---

