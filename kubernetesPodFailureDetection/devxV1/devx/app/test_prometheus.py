import os
import logging
from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame
from prometheus_api_client.utils import parse_datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus configuration
PROMETHEUS_URL = os.getenv('PROMETHEUS_URL', 'http://host.docker.internal:9090/')
prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)

def fetch_prometheus_metrics(pod_name, namespace='default'):
    """Fetch relevant metrics from Prometheus for a given pod."""
    try:
        # Time range for metrics (last 5 minutes)
        start_time = parse_datetime("5m")
        end_time = parse_datetime("now")

        # Prometheus queries
        queries = {
            'CPU Usage (%)': f'rate(container_cpu_usage_seconds_total{{pod="{pod_name}", namespace="{namespace}"}}[5m]) * 100',
            'Memory Usage (%)': f'container_memory_usage_bytes{{pod="{pod_name}", namespace="{namespace}"}} / container_spec_memory_limit_bytes{{pod="{pod_name}", namespace="{namespace}"}} * 100',
            'Network Receive Packets Dropped (p/s)': f'rate(container_network_receive_packets_dropped_total{{pod="{pod_name}", namespace="{namespace}"}}[5m])'
        }

        metrics = {}
        for metric_name, query in queries.items():
            data = prom.custom_query_range(
                query=query,
                start_time=start_time,
                end_time=end_time,
                step='1m'
            )
            if data:
                df = MetricRangeDataFrame(data)
                metrics[metric_name] = df['value'].mean()
            else:
                metrics[metric_name] = 0.0

        # Placeholder values for event-related features
        metrics['Node Name'] = 'unknown_node'
        metrics['Pod Event Reason'] = 'Running'
        metrics['Pod Event Source'] = 'kubelet'
        metrics['Pod Event Age'] = '0:05:00'
        metrics['Event Age'] = '0:05:00'
        metrics['Event Source'] = 'kernel-monitor'

        logger.info(f"Fetched metrics for pod {pod_name}: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Error fetching Prometheus metrics: {e}")
        return None

# Test with a sample pod name
POD_NAME = "test-pod"
NAMESPACE = "default"

metrics = fetch_prometheus_metrics(POD_NAME, NAMESPACE)
if metrics:
    print("Retrieved Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
else:
    print("Failed to retrieve metrics.")
