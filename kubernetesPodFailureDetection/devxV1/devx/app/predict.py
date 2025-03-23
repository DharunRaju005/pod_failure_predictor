import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame
from prometheus_api_client.utils import parse_datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
import logging
from kubernetes import client, config
from datetime import datetime, timezone

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_PATH = os.path.join(os.path.dirname(__file__), "../output/")
MODEL_PATH = f"{BASE_PATH}pod_failure_model_xgb.pkl"
ENCODER_PATHS = {
    'le_node': f"{BASE_PATH}le_node.joblib",
    'le_reason': f"{BASE_PATH}le_reason.joblib",
    'le_pod_source': f"{BASE_PATH}le_pod_source.joblib",
    'le_event_source': f"{BASE_PATH}le_event_source.joblib",
    'le_target': f"{BASE_PATH}le_target.joblib"
}

# Load model and encoders
xgb_model = joblib.load(MODEL_PATH)
scaler = MinMaxScaler(feature_range=(0, 1))  # Define scaler in-line
le_node = joblib.load(ENCODER_PATHS['le_node'])
le_reason = joblib.load(ENCODER_PATHS['le_reason'])
le_pod_source = joblib.load(ENCODER_PATHS['le_pod_source'])
le_event_source = joblib.load(ENCODER_PATHS['le_event_source'])
le_target = joblib.load(ENCODER_PATHS['le_target'])

# Ensure 'unknown' is in encoder classes
for encoder in [le_node, le_reason, le_pod_source, le_event_source]:
    if 'unknown' not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, 'unknown')

logger.info("Model and encoders loaded successfully")

# Prometheus configuration
PROMETHEUS_URL = os.getenv('PROMETHEUS_URL', 'http://host.docker.internal:9090/')
prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)

# Feature columns expected by the model
FEATURE_COLUMNS = [
    'CPU Usage (%)',
    'Memory Usage (%)',
    'Network Receive Packets Dropped (p/s)',
    'Node Name',
    'Pod Event Reason',
    'Pod Event Source',
    'Pod Event Age',
    'Event Age',
    'Event Source'
]

CATEGORICAL_COLUMNS = [
    'Node Name',
    'Pod Event Reason',
    'Pod Event Source',
    'Event Source'
]

NUMERICAL_COLUMNS = [
    'CPU Usage (%)',
    'Memory Usage (%)',
    'Network Receive Packets Dropped (p/s)',
    'Pod Event Age',
    'Event Age'
]

# Define alert thresholds
ALERT_THRESHOLDS = {
    'CPU_USAGE_HIGH': 80.0,  # Percentage
    'MEMORY_USAGE_HIGH': 80.0,  # Percentage
    'NETWORK_DROPS_HIGH': 1.0,  # Packets/second
    'PROBABILITY_CONFIDENCE': 0.7,  # Minimum probability for confident prediction
    'CRASH_PROBABILITY': 0.3  # Threshold for crash-related alert
}

# Load Kubernetes config (detect if running in-cluster)
kubernetes_available = False
try:
    config.load_incluster_config()  # If running inside a Kubernetes cluster
    logger.info("Loaded in-cluster Kubernetes config.")
    kubernetes_available = True
except config.ConfigException:
    try:
        config.load_kube_config()  # If running locally
        logger.info("Loaded local kube config.")
        kubernetes_available = True
    except Exception as e:
        logger.warning(f"Could not load Kubernetes config: {e}. Proceeding without Kubernetes.")
        kubernetes_available = False

def get_pod_event_age_from_k8s(pod_name, namespace='default'):
    """Fetch Pod Event Age and Event Reason from Kubernetes API if missing in Prometheus."""
    if not kubernetes_available:
        return None, None  # Return defaults if Kubernetes is unavailable
    try:
        v1 = client.CoreV1Api()
        events = v1.list_namespaced_event(namespace, field_selector=f"involvedObject.name={pod_name}")

        if not events.items:
            return None, None  # No events found

        latest_event = max(events.items, key=lambda e: e.last_timestamp or e.event_time or e.metadata.creation_timestamp)
        
        event_time = latest_event.last_timestamp or latest_event.event_time or latest_event.metadata.creation_timestamp
        event_age = (datetime.now(timezone.utc) - event_time).total_seconds() if event_time else 0

        return event_age, latest_event.reason if latest_event.reason else 'unknown'
    except Exception as e:
        logger.error(f"Error fetching event age from Kubernetes API: {e}")
        return None, None

def fetch_prometheus_metrics(pod_name, namespace='default'):
    """Fetch relevant metrics from Prometheus and Kubernetes for a given pod."""
    try:
        start_time = parse_datetime("5m")
        end_time = parse_datetime("now")

        queries = {
            'CPU Usage (%)': f'rate(container_cpu_usage_seconds_total{{pod="{pod_name}", namespace="{namespace}"}}[5m]) * 100',
            'Memory Usage (%)': f'container_memory_usage_bytes{{pod="{pod_name}", namespace="{namespace}"}} / container_spec_memory_limit_bytes{{pod="{pod_name}", namespace="{namespace}"}} * 100',
            'Network Receive Packets Dropped (p/s)': f'rate(container_network_receive_packets_dropped_total{{pod="{pod_name}", namespace="{namespace}"}}[5m])'
        }

        metrics = {}
        for metric_name, query in queries.items():
            data = prom.custom_query_range(query=query, start_time=start_time, end_time=end_time, step='1m')
            metrics[metric_name] = MetricRangeDataFrame(data)['value'].mean() if data else 0.0

        # Fetch pod events dynamically from Prometheus
        event_query = f'kube_event_message{{involved_object_name="{pod_name}", namespace="{namespace}"}}'
        event_data = prom.custom_query(event_query)
        if event_data:
            event_info = event_data[0]['metric']
            metrics['Pod Event Reason'] = event_info.get('reason', 'unknown')
            metrics['Pod Event Source'] = event_info.get('source_component', 'unknown')
            metrics['Event Source'] = event_info.get('reporting_component', 'unknown')
        else:
            metrics.update({'Pod Event Reason': 'unknown', 'Pod Event Source': 'unknown', 'Event Source': 'unknown'})

        # Fetch Node Name and Event Age from Kubernetes API if available
        if kubernetes_available:
            v1 = client.CoreV1Api()
            pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
            metrics['Node Name'] = pod.spec.node_name if pod.spec.node_name else 'unknown'
            
            pod_event_age, event_reason = get_pod_event_age_from_k8s(pod_name, namespace)
            metrics['Pod Event Age'] = pod_event_age if pod_event_age is not None else 0
            metrics['Event Age'] = pod_event_age if pod_event_age is not None else 0
            if event_reason and metrics['Pod Event Reason'] == 'unknown':
                metrics['Pod Event Reason'] = event_reason
        else:
            metrics['Node Name'] = 'unknown'
            metrics['Pod Event Age'] = 0
            metrics['Event Age'] = 0

        logger.info(f"Retrieved Metrics for {pod_name}: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Error fetching Prometheus metrics: {e}")
        raise

def preprocess_metrics(metrics):
    """Preprocess fetched metrics into a DataFrame for prediction with normalization."""
    df = pd.DataFrame([metrics])
    
    # Ensure all feature columns are present
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0 if col in NUMERICAL_COLUMNS else 'unknown'
    df = df[FEATURE_COLUMNS]  # Reorder to match model expectations
    
    # Map column names to their respective encoders
    encoder_map = {
        'Node Name': le_node,
        'Pod Event Reason': le_reason,
        'Pod Event Source': le_pod_source,
        'Event Source': le_event_source
    }
    
    # Encode categorical columns
    for col in CATEGORICAL_COLUMNS:
        encoder = encoder_map[col]
        df[col] = df[col].astype(str).map(
            lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
        )
    
    # Scale numerical columns
    df[NUMERICAL_COLUMNS] = scaler.fit_transform(df[NUMERICAL_COLUMNS])
    
    return df

def generate_alerts(prediction_results, metrics):
    """Generate alerts based on predictions and metrics."""
    alerts = []
    
    predicted_status = prediction_results['predicted_status']
    probabilities = prediction_results['probabilities']
    max_probability = max(probabilities.values())
    
    if metrics['CPU Usage (%)'] > ALERT_THRESHOLDS['CPU_USAGE_HIGH']:
        alerts.append({
            'type': 'HIGH_CPU',
            'message': f'CPU usage exceeded threshold: {metrics["CPU Usage (%)"]:.2f}%',
            'severity': 'warning'
        })
    
    if metrics['Memory Usage (%)'] > ALERT_THRESHOLDS['MEMORY_USAGE_HIGH']:
        alerts.append({
            'type': 'HIGH_MEMORY',
            'message': f'Memory usage exceeded threshold: {metrics["Memory Usage (%)"]:.2f}%',
            'severity': 'warning'
        })
    
    if metrics['Network Receive Packets Dropped (p/s)'] > ALERT_THRESHOLDS['NETWORK_DROPS_HIGH']:
        alerts.append({
            'type': 'NETWORK_ISSUES',
            'message': f'High network packet drops: {metrics["Network Receive Packets Dropped (p/s)"]:.2f} p/s',
            'severity': 'warning'
        })
    
    if max_probability < ALERT_THRESHOLDS['PROBABILITY_CONFIDENCE']:
        alerts.append({
            'type': 'LOW_CONFIDENCE',
            'message': f'Prediction confidence low: {max_probability:.2%}',
            'severity': 'info'
        })
    
    if predicted_status in ['CrashLoopBackOff', 'Error'] or probabilities.get('CrashLoopBackOff', 0) > ALERT_THRESHOLDS['CRASH_PROBABILITY']:
        alerts.append({
            'type': 'POTENTIAL_CRASH',
            'message': f'High crash risk detected (Status: {predicted_status}, Crash probability: {probabilities.get("CrashLoopBackOff", 0):.2%})',
            'severity': 'critical'
        })
    
    if predicted_status in ['unknown', 'NotFound']:
        alerts.append({
            'type': 'UNKNOWN_STATE',
            'message': f'Pod in uncertain state: {predicted_status}',
            'severity': 'warning'
        })
    
    return alerts if alerts else None

@app.route('/predict', methods=['POST'])
def predict():
    """Predict pod failure based on metrics with alerts."""
    try:
        data = request.get_json()
        if not data or 'pod_name' not in data:
            return jsonify({'error': 'No pod_name provided'}), 400

        pod_name = data['pod_name']
        namespace = data.get('namespace', 'default')

        # Fetch metrics
        metrics = fetch_prometheus_metrics(pod_name, namespace)

        # Preprocess metrics
        X = preprocess_metrics(metrics)

        # Predict
        prediction = xgb_model.predict(X)[0]
        probabilities = xgb_model.predict_proba(X)[0]
        predicted_status = le_target.inverse_transform([prediction])[0]
        prob_dict = dict(zip(le_target.classes_, probabilities))

        # Generate alerts
        prediction_results = {
            'predicted_status': predicted_status,
            'probabilities': {k: float(v) for k, v in prob_dict.items()}
        }
        alerts = generate_alerts(prediction_results, metrics)

        return jsonify({
            'pod_name': pod_name,
            'namespace': namespace,
            'predictions': {
                'predicted_status': predicted_status,
                'probabilities': prediction_results['probabilities'],
                'alert': alerts
            },
            'metrics': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in metrics.items()}
        }), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)