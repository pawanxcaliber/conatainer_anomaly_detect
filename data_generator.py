import datetime
import random
from faker import Faker
import time
import csv
import numpy as np # Import numpy for numerical data generation

# Initialize Faker
fake = Faker()

# --- Global Configuration ---
# Container and Service Identifiers
CONTAINER_IDS = [f'app-container-{i}' for i in range(1, 6)]
ANOMALOUS_CONTAINER_ID = 'malicious-container-x'
SERVICE_NAMES = ['auth-service', 'user-service', 'product-catalog', 'payment-gateway', 'reporting-tool']

# --- Log Generation Configuration ---
LOG_LEVELS = ['INFO', 'DEBUG', 'INFO', 'WARNING'] # INFO/DEBUG more frequent
ERROR_LOG_LEVELS = ['ERROR', 'CRITICAL'] # For anomalies

# --- Network Metrics Configuration ---
NORMAL_BANDWIDTH_RANGE = (100, 500) # KBps
ANOMALOUS_BANDWIDTH_RANGE = (1000, 5000) # KBps for spike
NORMAL_LATENCY_RANGE = (20, 100) # ms
ANOMALOUS_LATENCY_RANGE = (500, 2000) # ms for high latency
NORMAL_PACKET_DROP_RATE = (0, 0.01) # Percentage
ANOMALOUS_PACKET_DROP_RATE = (0.1, 0.5) # Percentage for high drops
NORMAL_CONNECTIONS_ACTIVE = (10, 50)
ANOMALOUS_CONNECTIONS_ACTIVE = (100, 500) # For connection surge
NORMAL_CONNECTION_FAILURES = (0, 1)
ANOMALOUS_CONNECTION_FAILURES = (5, 20) # For high failures

# --- Infrastructure Telemetry Configuration ---
NORMAL_CPU_RANGE = (10, 60) # Percentage
ANOMALOUS_CPU_RANGE = (85, 99) # Percentage for high CPU
NORMAL_MEMORY_RANGE = (200, 800) # MB
ANOMALOUS_MEMORY_RANGE = (1500, 3000) # MB for high memory
NORMAL_DISK_IO_RANGE = (50, 200) # KBps
ANOMALOUS_DISK_IO_RANGE = (1000, 5000) # KBps for high disk IO
NORMAL_PROCESS_COUNT_RANGE = (5, 15)
ANOMALOUS_PROCESS_COUNT_RANGE = (30, 80) # For process surge

# --- Log Generation Functions ---
def generate_normal_log(current_container_id=None, current_service_name=None):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
    level = random.choice(LOG_LEVELS)

    container = current_container_id if current_container_id else random.choice(CONTAINER_IDS)
    service = current_service_name if current_service_name else random.choice(SERVICE_NAMES)

    if level == 'INFO':
        message = random.choice([
            f"Processed request for user {fake.uuid4()}",
            f"Service {service} started successfully.",
            f"Health check passed for {container}.",
            f"API call to {fake.uri_path()} completed in {random.randint(50, 500)}ms."
        ])
    elif level == 'DEBUG':
        message = random.choice([
            f"Debugging {fake.word()} in {service} for {container}.",
            f"Variable {fake.word()} has value {random.randint(0, 100)}.",
            f"Entering function {fake.word()} in module {fake.word()}."
        ])
    elif level == 'WARNING':
        message = random.choice([
            f"High latency detected for service {service}.",
            f"Resource usage for {container} is approaching threshold.",
            f"Outbound connection to {fake.ipv4_private()} failed, retrying."
        ])

    return [timestamp, level, container, service, message]

def generate_anomalous_log(current_container_id=None, current_service_name=None):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
    level = random.choice(ERROR_LOG_LEVELS)
    container = current_container_id if current_container_id else ANOMALOUS_CONTAINER_ID
    service = current_service_name if current_service_name else random.choice(SERVICE_NAMES)

    message = random.choice([
        f"Unauthorized access attempt detected from {fake.ipv4()} to service {service} on port {random.randint(1024, 65535)}.",
        f"CRITICAL: Out of memory error in {container}. System unstable.",
        f"ERROR: Database connection failed for {service}. Retries exhausted.",
        f"Suspicious file access denied in {container}: {fake.file_path()}.",
        f"Repeated authentication failures for user '{fake.user_name()}' from {fake.ipv4_private()}."
    ])

    return [timestamp, level, container, service, message]

# --- Network Metrics Generation Functions ---
def generate_normal_network_metrics(current_container_id=None, current_service_name=None):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    container = current_container_id if current_container_id else random.choice(CONTAINER_IDS)
    service = current_service_name if current_service_name else random.choice(SERVICE_NAMES)

    bytes_sent = round(np.random.normal(np.mean(NORMAL_BANDWIDTH_RANGE), 50))
    bytes_received = round(np.random.normal(np.mean(NORMAL_BANDWIDTH_RANGE), 40))
    packets_dropped = round(np.random.uniform(*NORMAL_PACKET_DROP_RATE) * 100) # Convert rate to count
    latency_ms = round(np.random.normal(np.mean(NORMAL_LATENCY_RANGE), 10))
    connections_active = random.randint(*NORMAL_CONNECTIONS_ACTIVE)
    connection_failures = random.randint(*NORMAL_CONNECTION_FAILURES)

    return [timestamp, container, service, max(0, bytes_sent), max(0, bytes_received),
            max(0, packets_dropped), max(0, latency_ms), max(0, connections_active), max(0, connection_failures)]

def generate_anomalous_network_metrics(current_container_id=None, current_service_name=None):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    container = current_container_id if current_container_id else ANOMALOUS_CONTAINER_ID
    service = current_service_name if current_service_name else random.choice(SERVICE_NAMES)

    bytes_sent = round(np.random.normal(np.mean(ANOMALOUS_BANDWIDTH_RANGE), 500)) # High traffic
    bytes_received = round(np.random.normal(np.mean(ANOMALOUS_BANDWIDTH_RANGE), 400))
    packets_dropped = round(np.random.uniform(*ANOMALOUS_PACKET_DROP_RATE) * 500) # Many drops
    latency_ms = round(np.random.normal(np.mean(ANOMALOUS_LATENCY_RANGE), 200)) # High latency
    connections_active = random.randint(*ANOMALOUS_CONNECTIONS_ACTIVE) # Many connections
    connection_failures = random.randint(*ANOMALOUS_CONNECTION_FAILURES) # Many failures

    return [timestamp, container, service, max(0, bytes_sent), max(0, bytes_received),
            max(0, packets_dropped), max(0, latency_ms), max(0, connections_active), max(0, connection_failures)]

# --- Infrastructure Telemetry Generation Functions ---
def generate_normal_infra_metrics(current_container_id=None, current_service_name=None):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    container = current_container_id if current_container_id else random.choice(CONTAINER_IDS)
    service = current_service_name if current_service_name else random.choice(SERVICE_NAMES)

    cpu_util = round(np.random.normal(np.mean(NORMAL_CPU_RANGE), 10), 1)
    memory_usage = round(np.random.normal(np.mean(NORMAL_MEMORY_RANGE), 50))
    disk_read = round(np.random.normal(np.mean(NORMAL_DISK_IO_RANGE), 20))
    disk_write = round(np.random.normal(np.mean(NORMAL_DISK_IO_RANGE), 20))
    process_count = random.randint(*NORMAL_PROCESS_COUNT_RANGE)

    return [timestamp, container, service, max(0.0, cpu_util), max(0, memory_usage),
            max(0, disk_read), max(0, disk_write), max(0, process_count)]

def generate_anomalous_infra_metrics(current_container_id=None, current_service_name=None):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    container = current_container_id if current_container_id else ANOMALOUS_CONTAINER_ID
    service = current_service_name if current_service_name else random.choice(SERVICE_NAMES)

    cpu_util = round(np.random.normal(np.mean(ANOMALOUS_CPU_RANGE), 5), 1) # High CPU
    memory_usage = round(np.random.normal(np.mean(ANOMALOUS_MEMORY_RANGE), 100)) # High Memory
    disk_read = round(np.random.normal(np.mean(ANOMALOUS_DISK_IO_RANGE), 500)) # High Disk IO
    disk_write = round(np.random.normal(np.mean(ANOMALOUS_DISK_IO_RANGE), 500))
    process_count = random.randint(*ANOMALOUS_PROCESS_COUNT_RANGE) # Many processes

    return [timestamp, container, service, max(0.0, cpu_util), max(0, memory_usage),
            max(0, disk_read), max(0, disk_write), max(0, process_count)]

# --- Main Generation Function ---
def generate_all_data(
    num_entries,
    log_output_file='synthetic_container_logs.csv',
    net_output_file='synthetic_network_metrics.csv',
    infra_output_file='synthetic_infra_telemetry.csv',
    anomaly_frequency_log=50, # Every N log entries, inject anomaly burst
    anomaly_frequency_metrics=10, # Every N metric intervals, inject anomaly
    error_burst_size_log=5,
    metric_interval_seconds=5 # How often to generate metric data points (e.g., every 5 seconds)
):
    # --- Define CSV Headers ---
    log_headers = ['Timestamp', 'LogLevel', 'ContainerID', 'ServiceName', 'Message']
    net_headers = ['Timestamp', 'ContainerID', 'ServiceName', 'BytesSentKBps', 'BytesReceivedKBps',
                   'PacketsDropped', 'LatencyMs', 'ConnectionsActive', 'ConnectionFailures']
    infra_headers = ['Timestamp', 'ContainerID', 'ServiceName', 'CPUUtilizationPercent', 'MemoryUsageMB',
                     'DiskReadKBps', 'DiskWriteKBps', 'ProcessCount']

    # --- Open all CSV writers ---
    with open(log_output_file, 'w', newline='') as log_f, \
         open(net_output_file, 'w', newline='') as net_f, \
         open(infra_output_file, 'w', newline='') as infra_f:

        log_writer = csv.writer(log_f)
        net_writer = csv.writer(net_f)
        infra_writer = csv.writer(infra_f)

        # Write headers
        log_writer.writerow(log_headers)
        net_writer.writerow(net_headers)
        infra_writer.writerow(infra_headers)

        print(f"Generating data for {num_entries} intervals...")
        print(f"Logs -> '{log_output_file}'")
        print(f"Network Metrics -> '{net_output_file}'")
        print(f"Infra Telemetry -> '{infra_output_file}'")

        # Keep track of when to generate metrics
        next_metric_time = time.time() + metric_interval_seconds

        for i in range(num_entries):
            current_time = time.time()

            # --- Log Generation ---
            log_data_row = []
            if (i + 1) % anomaly_frequency_log == 0:
                # Inject a burst of anomalous logs
                target_container = random.choice(CONTAINER_IDS + [ANOMALOUS_CONTAINER_ID]) # Pick a container for this anomaly context
                target_service = random.choice(SERVICE_NAMES)

                for _ in range(error_burst_size_log):
                    log_data_row = generate_anomalous_log(current_container_id=target_container, current_service_name=target_service)
                    log_writer.writerow(log_data_row)
                    # print(f"LOG ANOMALY: {', '.join(map(str, log_data_row))}")
                    time.sleep(0.01) # Small delay for realism in logs
            else:
                log_data_row = generate_normal_log()
                log_writer.writerow(log_data_row)
                # print(f"LOG NORMAL: {', '.join(map(str, log_data_row))}")


            # --- Metric Generation (Network & Infra) - less frequent than logs ---
            if current_time >= next_metric_time:
                # Decide if this interval should have an anomaly for metrics
                is_metric_anomaly = random.randint(1, anomaly_frequency_metrics) == 1

                target_container_for_metrics = ANOMALOUS_CONTAINER_ID if is_metric_anomaly else random.choice(CONTAINER_IDS)
                target_service_for_metrics = random.choice(SERVICE_NAMES)

                if is_metric_anomaly:
                    net_data_row = generate_anomalous_network_metrics(current_container_id=target_container_for_metrics, current_service_name=target_service_for_metrics)
                    infra_data_row = generate_anomalous_infra_metrics(current_container_id=target_container_for_metrics, current_service_name=target_service_for_metrics)
                    # print(f"METRIC ANOMALY ({target_container_for_metrics}): Net={net_data_row[3]}, CPU={infra_data_row[3]}%")
                else:
                    net_data_row = generate_normal_network_metrics()
                    infra_data_row = generate_normal_infra_metrics()
                    # print(f"METRIC NORMAL: Net={net_data_row[3]}, CPU={infra_data_row[3]}%")

                net_writer.writerow(net_data_row)
                infra_writer.writerow(infra_data_row)

                next_metric_time = current_time + metric_interval_seconds

            # General sleep for the main loop iteration
            time.sleep(0.02) # Adjusted sleep to keep overall time reasonable with more operations

    print(f"\nData generation complete. Check the generated CSV files.")

if __name__ == '__main__':
    # You are using Google Colab. Run `!pip install Faker` in a separate cell first.
    # Setting num_entries to 15000 for approximately 5 minutes (300 seconds) of data generation
    generate_all_data(
        num_entries=15000, # Increased for ~5 minutes of data
        log_output_file='synthetic_container_logs.csv',
        net_output_file='synthetic_network_metrics.csv',
        infra_output_file='synthetic_infra_telemetry.csv',
        anomaly_frequency_log=100,
        anomaly_frequency_metrics=5,
        error_burst_size_log=10,
        metric_interval_seconds=0.1
    )
