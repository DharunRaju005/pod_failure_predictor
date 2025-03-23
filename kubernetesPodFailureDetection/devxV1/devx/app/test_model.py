import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_PATH = os.path.join(os.path.dirname(__file__), "../Outputofweights/")
MODEL_PATH = f"{BASE_PATH}xgb_model.joblib"
ENCODER_PATHS = {
    'le_node': f"{BASE_PATH}le_node.joblib",
    'le_reason': f"{BASE_PATH}le_reason.joblib",
    'le_pod_source': f"{BASE_PATH}le_pod_source.joblib",
    'le_event_source': f"{BASE_PATH}le_event_source.joblib",
    'le_target': f"{BASE_PATH}le_target.joblib"
}

# Load model and encoders
try:
    xgb_model = joblib.load(MODEL_PATH)
    le_node = joblib.load(ENCODER_PATHS['le_node'])
    le_reason = joblib.load(ENCODER_PATHS['le_reason'])
    le_pod_source = joblib.load(ENCODER_PATHS['le_pod_source'])
    le_event_source = joblib.load(ENCODER_PATHS['le_event_source'])
    le_target = joblib.load(ENCODER_PATHS['le_target'])
    logger.info("Model and encoders loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or encoders: {e}")
    exit(1)

# Sample test data
test_data = {
    'CPU Usage (%)': 0.002564151384,
        'Memory Usage (%)': 0.002021123549,
        'Network Receive Packets Dropped (p/s)': 0.04490201064,
        'Node Name': 'aks-npubuntu-25645003-vmss000004',
        'Pod Event Reason': 'Pulled',
        'Pod Event Source': 'kubelet',
        'Pod Event Age': '0:02:02',
        'Event Age': '0:11:04',
        'Event Source': 'kernel-monitor'
}

def time_to_seconds(time_str):
    """Convert time string to seconds."""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def preprocess_test_data(data):
    """Preprocess sample data for model prediction."""
    df = pd.DataFrame([data])

    # Convert time columns
    df['Pod Event Age'] = df['Pod Event Age'].apply(time_to_seconds)
    df['Event Age'] = df['Event Age'].apply(time_to_seconds)

    # Encode categorical variables
    for col, encoder in [
        ('Node Name', le_node),
        ('Pod Event Reason', le_reason),
        ('Pod Event Source', le_pod_source),
        ('Event Source', le_event_source)
    ]:
        try:
            df[col] = encoder.transform(df[col])
        except ValueError:
            logger.warning(f"Unseen value in {col}. Using default encoding.")
            df[col] = encoder.transform([encoder.classes_[0]])[0]

    return df

# Process and predict
try:
    X_test = preprocess_test_data(test_data)
    prediction = xgb_model.predict(X_test)[0]
    probabilities = xgb_model.predict_proba(X_test)[0]

    predicted_status = le_target.inverse_transform([prediction])[0]
    prob_dict = dict(zip(le_target.classes_, probabilities))

    logger.info(f"Predicted Status: {predicted_status}")
    logger.info(f"Prediction Probabilities: {prob_dict}")
except Exception as e:
    logger.error(f"Error during prediction: {e}")
