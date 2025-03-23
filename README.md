markdown

Collapse

Wrap

Copy
# Kubernetes Pod Failure Detection Project Documentation

This document provides an in-depth overview of the Kubernetes Pod Failure Detection project, a machine learning-based solution designed to predict pod failures in a Kubernetes cluster. The project leverages metrics from Prometheus, a pre-trained XGBoost model, and a Flask-based API to classify pod statuses, enhancing cluster reliability and operational efficiency.

## Project Overview

The Kubernetes Pod Failure Detection project aims to proactively identify and predict pod failures within a Kubernetes cluster using real-time metrics and machine learning. It integrates with Prometheus for metrics collection and employs an XGBoost model to classify pod statuses into seven categories: `ContainerCreating`, `CrashLoopBackOff`, `Error`, `NotFound`, `Pending`, `Running`, and `Unknown`. The system is deployed as a Flask application within a Docker container, providing a scalable and accessible prediction API.

### Objectives
- **Predict Pod Failures**: Classify pod statuses using real-time metrics to anticipate failures.
- **Monitor Cluster Health**: Utilize Prometheus to collect and analyze pod and node metrics.
- **Provide Actionable Insights**: Deliver predictions with confidence levels and probability distributions for operational decision-making.

## Directory Structure

The project directory structure is as follows, based on the provided details:
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

text

Collapse

Wrap

Copy

### Directory Details
- **`.venv/`**: Contains the virtual environment for Python dependencies, ensuring isolated dependency management.
- **`app/predict.py`**: The core Flask application that handles prediction requests, fetches metrics from Prometheus, and uses the trained model to predict pod statuses.
- **`model/`**:
  - `categorical_cols.txt`: Stores the names of categorical columns used during training (e.g., `Node Name`, `Pod Event Reason`).
  - `feature_columns.txt`: Lists all feature columns used by the model after preprocessing and feature engineering.
  - `label_encoder.pkl`: A serialized `LabelEncoder` object used to encode pod statuses into numerical values.
  - `model.pkl`: The trained XGBoost model saved using `joblib`.
- **`.dockerignore`**: Excludes unnecessary files (e.g., `.venv/`, `__pycache__`) from the Docker build context.
- **`Dockerfile`**: Defines the steps to build the Docker image for the Flask application.
- **`requirements.txt`**: Lists Python dependencies, including `pandas`, `numpy`, `xgboost`, `joblib`, `flask`, `scikit-learn`, and `requests`.

## Architecture

The architecture of the Kubernetes Pod Failure Detection system is modular, scalable, and tightly integrated with Kubernetes and Prometheus. It comprises several components that work together to collect metrics, process data, and deliver predictions. Below is a detailed breakdown of the architecture, including a textual representation of the high-level design.

### High-Level Architecture Diagram
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
|                   |       |                   |       |                   |
|  +-------------+  |       |  +-------------+  |       |  +-------------+  |
|  | Kube-State   |  |<----->|  | Kube Metrics|  |       |  | Preprocessing|  |
|  | Metrics     |  | scrape |  +-------------+  |       |  | & Feature Eng|  |
|  +-------------+  |       |                   |       |  +-------------+  |
+-------------------+       +-------------------+       +-------------------+

text

Collapse

Wrap

Copy

### Components

#### 1. Kubernetes Cluster
- **Purpose**: Hosts all application and monitoring components.
- **Subcomponents**:
  - **Pods (Node.js Application)**:
    - A lightweight Express.js application (`server.js`) exposes custom metrics (e.g., `http_requests_total`) via a `/metrics` endpoint using the `prom-client` library.
    - Runs on port `8080`.
    - Deployed using a `Dockerfile` based on `node:20-alpine`.
  - **Node Exporter**:
    - Deployed as a `DaemonSet` via `node-exporter.yaml`.
    - Collects node-level metrics (e.g., CPU, memory, network packet drops) on port `9100`.
    - Uses `hostNetwork: true` to access the host’s network stack.
  - **Kube-State Metrics**:
    - Provides Kubernetes-specific metrics (e.g., pod statuses, resource limits) on port `8080`.
    - Scraped by Prometheus for cluster state monitoring.

#### 2. Prometheus
- **Purpose**: Collects and stores time-series metrics from the Kubernetes cluster and its components.
- **Configuration**:
  - Deployed via `prometheus-deployment.yaml` as a `Deployment` with 1 replica in the `default` namespace.
  - Uses a `ConfigMap` (`prometheus-config`) to define `prometheus.yml`.
  - Exposed on port `9090` via a `Service`.
  - Scrape interval set to 5 seconds (`scrape_interval: 5s`).
- **Scrape Jobs** (from `prometheus.yml`):
  - **`node-app`**: Scrapes the Node.js app at `node-app-service.default.svc.cluster.local:5000`.
  - **`kubernetes-cadvisor`**: Scrapes cAdvisor metrics (e.g., CPU, memory usage) from nodes at port `10250` using HTTPS and `insecure_skip_verify`.
  - **`kube-state-metrics`**: Scrapes Kubernetes state metrics at `kube-state-metrics.kube-system.svc.cluster.local:8080`.
  - **`local-app`**: Scrapes the Node.js app at `host.docker.internal:8080`.
  - **`node-exporter`**: Scrapes node-level metrics at `host.docker.internal:9100`.
- **RBAC**:
  - Configured via `prometheus-rbac.yaml` with a `ClusterRole` and `ClusterRoleBinding`.
  - Grants access to resources like `nodes`, `pods`, and `services` for metrics collection.

#### 3. Flask Application (`predict.py`)
- **Purpose**: Serves as the prediction API, fetching metrics from Prometheus and using the XGBoost model to predict pod statuses.
- **Configuration**:
  - Runs on port `6000`.
  - Containerized using a `Dockerfile` based on `python:3.9-slim`.
  - Dependencies listed in `requirements.txt`.
- **API Endpoints**:
  - **`/predict`**:
    - **Method**: POST
    - **Input**: JSON with `pod_name` and optional `namespace`.
    - **Functionality**: Fetches metrics from Prometheus, preprocesses them, and returns predicted pod status with confidence levels.
    - **Example Response**:
      ```json
      {
        "predictions": [
          {
            "Pod Name": "my-pod",
            "Predicted Status": "Running",
            "Confidence Level": 0.95,
            "Probability Distribution": [0.02, 0.03, 0.95]
          }
        ],
        "metrics_fetched": {
          "Node Name": "docker-desktop",
          "CPU Usage (%)": 10.5,
          "Memory Usage (%)": 20.3,
          "Network Receive Packets Dropped (p/s)": 0.0
        },
        "pod_activity": "No significant activity detected"
      }
/test-prometheus:
Method: GET
Functionality: Tests connectivity to Prometheus with a simple up query.
Example Response:
json

Collapse

Wrap

Copy
{
  "status": "success",
  "message": "Prometheus query successful",
  "response": {...}
}
Model Integration:
Loads model.pkl, label_encoder.pkl, feature_columns.txt, and categorical_cols.txt from the model/ directory.
Preprocesses incoming metrics (e.g., one-hot encoding, feature engineering) to match the training data format.
Feature Engineering:
Adds binary flags: CPU_High_Usage (>90%), Memory_High_Usage (>90%), Network_Drop_High (>0).
Uses raw values for CPU_Usage_Rolling_Mean and Memory_Usage_Rolling_Mean (simplified for single-row predictions).
4. XGBoost Model
Purpose: Performs multi-class classification of pod statuses.
Training:
Trained on a synthetic dataset (dataSynthetic_cleaned.csv) with features like CPU usage, memory usage, and network metrics.
Classes: ContainerCreating, CrashLoopBackOff, Error, NotFound, Pending, Running, Unknown.
Parameters: n_estimators=300, learning_rate=0.05, max_depth=6, eval_metric='mlogloss'.
Uses class weights to handle imbalance: {0: 14.39, 1: 2.25, 2: 20.34, 3: 12.42, 4: 2.93, 5: 0.18, 6: 1.75}.
Deployment:
Saved as model.pkl and loaded by the Flask app.
5. Docker
Purpose: Containerizes the Flask application for consistent deployment.
Configuration:
Base image: python:3.9-slim.
Installs dependencies from requirements.txt.
Copies app/ and model/ directories.
Runs on port 6000.
Implementation Details
Data Preprocessing and Feature Engineering
Dataset: Synthetic dataset (dataSynthetic_cleaned.csv) from Google Drive.
Preprocessing:
Excluded columns: Pod Name, Timestamp, Pod Event Message, Event Message, Pod Status.
Filled missing values with 0.
One-hot encoded categorical columns: Node Name, Pod Event Reason, Pod Event Age, Pod Event Source, Event Age, Event Source.
Feature Engineering:
Added binary flags: CPU_High_Usage, Memory_High_Usage, Network_Drop_High.
Computed rolling means: CPU_Usage_Rolling_Mean and Memory_Usage_Rolling_Mean (window=3, simplified in Flask app).
Model Training
Algorithm: XGBoost (XGBClassifier).
Parameters:
n_estimators=300
learning_rate=0.05
max_depth=6
eval_metric='mlogloss'
Process:
Split data: 80% training, 20% testing (stratified).
Applied class weights to address imbalance.
Trained with validation monitoring (no explicit early stopping due to fit usage).
Output:
Saved to /content/drive/MyDrive/Project_ML/KubernetesPodfailureDetection/Output/pod_failure_model_xgb.pkl.
Flask Application (predict.py)
Key Functions:
fetch_prometheus_metrics: Queries Prometheus for CPU, memory, and network metrics; mocks event data.
preprocess_new_data: Aligns fetched metrics with training features.
predict_pod_status: Uses the XGBoost model to predict statuses and returns results.
Deployment
Docker
dockerfile

Collapse

Wrap

Copy
FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --timeout=900 --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY model/ ./model/
ENV FLASK_APP=app/predict.py
ENV FLASK_DEBUG=0
EXPOSE 6000
CMD ["flask", "run", "--host=0.0.0.0", "--port=6000"]
Kubernetes
Prometheus: Deployed with prometheus-deployment.yaml and prometheus-rbac.yaml.
Node Exporter: Deployed with node-exporter.yaml.
Setup and Installation
Prerequisites
Docker
Kubernetes cluster (e.g., Docker Desktop)
Python 3.9+
Node.js 20+
Prometheus
Steps
Clone the Repository:
bash

Collapse

Wrap

Copy
git clone <repository-url>
cd kubernetesPodFailureDetection/devxV1/devx
Set Up Virtual Environment:
bash

Collapse

Wrap

Copy
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
Install Dependencies:
bash

Collapse

Wrap

Copy
pip install -r requirements.txt
Build Docker Image:
bash

Collapse

Wrap

Copy
docker build -t kubernetes-pod-failure-detection .
Deploy Prometheus and Node Exporter:
bash

Collapse

Wrap

Copy
kubectl apply -f prometheus-deployment.yaml
kubectl apply -f prometheus-rbac.yaml
kubectl apply -f node-exporter.yaml
Run the Flask Application:
Locally:
bash

Collapse

Wrap

Copy
export FLASK_APP=app/predict.py
export FLASK_DEBUG=0
flask run --host=0.0.0.0 --port=6000
Via Docker:
bash

Collapse

Wrap

Copy
docker run -p 6000:6000 kubernetes-pod-failure-detection
Usage
Predicting Pod Status
Send a POST request to /predict:

bash

Collapse

Wrap

Copy
curl -X POST -H "Content-Type: application/json" -d '{"pod_name": "my-pod", "namespace": "default"}' http://localhost:6000/predict
Testing Prometheus Connectivity
Send a GET request to /test-prometheus:

bash

Collapse

Wrap

Copy
curl http://localhost:6000/test-prometheus
Performance Metrics
Model Accuracy: 97.95% on the test set.
Classification Report:
text

Collapse

Wrap

Copy
                  precision    recall  f1-score   support
ContainerCreating   0.93      0.85      0.89       198
CrashLoopBackOff    0.94      0.96      0.95      1271
Error               0.93      0.71      0.81       141
NotFound            1.00      0.96      0.98       230
Pending             0.96      0.96      0.96       974
Running             0.99      0.99      0.99     15552
Unknown             0.93      0.91      0.92      1634
accuracy            0.98      0.98      0.98         -
macro avg           0.96      0.91      0.93     20000
weighted avg        0.98      0.98      0.98     20000
Top 10 Feature Importances:
text

Collapse

Wrap

Copy
Feature                     Importance
Ready Containers            0.183854
Pod Event Reason_Killing    0.175592
Total Containers            0.174264
Pod Restarts                0.085117
Pod Event Reason_Pulling    0.048317
Pod Event Type_No recent events 0.048211
Memory_Usage_Rolling_Mean   0.019616
Pod Event Age_0:03:41       0.014722
Memory Usage (%)            0.013096
Pod Event Age_0:00:55       0.011510
Future Improvements
Real-Time Event Integration: Fetch Kubernetes events directly instead of mocking them.
Model Optimization: Tune hyperparameters or explore other algorithms.
Alerting: Integrate with Alertmanager for failure notifications.
Scalability: Deploy the Flask app with multiple replicas and a load balancer.
Conclusion
The Kubernetes Pod Failure Detection project demonstrates an effective approach to predicting pod failures using machine learning, achieving a test accuracy of 97.95%. Its modular architecture ensures seamless integration with Kubernetes and Prometheus, providing a scalable and reliable solution for cluster monitoring.

For further details or contributions, refer to the project repository or contact the maintainers.

text

Collapse

Wrap

Copy

---

### How to Use This File
1. Copy the content above into a file named `README.md` or any other `.md` file.
2. Place it in your project directory (e.g., `D:\ML\kubernetesPodFailureDetection\devxV1\devx\`).
3. View it in a Markdown viewer (e.g., GitHub, VS Code, or any Markdown renderer) to see the formatted documentation with code blocks.

Let me know if you need further assistance!
