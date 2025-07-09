import pandas as pd
import numpy as np

def preprocess_telemetry_data(
    log_file='synthetic_container_logs.csv',
    network_file='synthetic_network_metrics.csv',
    infra_file='synthetic_infra_telemetry.csv',
    resample_interval='1s', # Define the common resampling interval
    output_file=None # New parameter to specify output CSV file
):
    """
    Loads, preprocesses, and merges synthetic container telemetry data.
    Optionally stores the processed data to a CSV file.

    Steps include:
    1. Loading CSV files.
    2. Converting Timestamp columns to datetime objects.
    3. Resampling Network and Infrastructure metrics to a common interval and filling NaNs.
    4. Extracting features from logs (counts of log levels) and filling NaNs.
    5. Merging all processed data into a single DataFrame.
    6. Optionally saving the final processed DataFrame to a CSV file.

    Args:
        log_file (str): Path to the synthetic container logs CSV.
        network_file (str): Path to the synthetic network metrics CSV.
        infra_file (str): Path to the synthetic infrastructure telemetry CSV.
        resample_interval (str): The time interval for resampling metrics and logs (e.g., '1s' for 1 second).
        output_file (str, optional): If provided, the final processed DataFrame will be
                                     saved to this CSV file. Defaults to None (no saving).

    Returns:
        pd.DataFrame: A single, comprehensive DataFrame with all features, or None if an error occurs.
    """
    try:
        # --- 1. Load CSV files into DataFrames ---
        df_logs = pd.read_csv(log_file)
        df_network = pd.read_csv(network_file)
        df_infra = pd.read_csv(infra_file)

        print("CSV files loaded successfully!")

        # --- 2. Convert 'Timestamp' columns to datetime objects ---
        df_logs['Timestamp'] = pd.to_datetime(df_logs['Timestamp'], format='%Y-%m-%d %H:%M:%S,%f')
        df_network['Timestamp'] = pd.to_datetime(df_network['Timestamp'])
        df_infra['Timestamp'] = pd.to_datetime(df_infra['Timestamp'])

        print("\nTimestamp columns converted to datetime objects.")

        # --- 3. Resample Network Metrics to specified intervals and fill NaNs ---
        df_network_indexed = df_network.set_index('Timestamp')
        network_agg = df_network_indexed.groupby(['ContainerID', 'ServiceName']).resample(resample_interval).mean().reset_index()
        network_agg = network_agg.fillna(0)
        print(f"Network metrics resampled to {resample_interval} intervals and NaNs filled.")

        # --- 4. Resample Infrastructure Telemetry to specified intervals and fill NaNs ---
        df_infra_indexed = df_infra.set_index('Timestamp')
        infra_agg = df_infra_indexed.groupby(['ContainerID', 'ServiceName']).resample(resample_interval).mean().reset_index()
        infra_agg = infra_agg.fillna(0)
        print(f"Infrastructure telemetry resampled to {resample_interval} intervals and NaNs filled.")

        # --- 5. Feature Engineering from Logs: Aggregate and Count Log Levels ---
        df_logs_indexed = df_logs.set_index('Timestamp')

        def log_features(group):
            """Helper function to extract features from log groups."""
            total_logs = len(group)
            info_logs = (group['LogLevel'] == 'INFO').sum()
            debug_logs = (group['LogLevel'] == 'DEBUG').sum()
            warning_logs = (group['LogLevel'] == 'WARNING').sum()
            error_logs = (group['LogLevel'] == 'ERROR').sum()
            critical_logs = (group['LogLevel'] == 'CRITICAL').sum()

            return pd.Series({
                'TotalLogs': total_logs,
                'InfoLogs': info_logs,
                'DebugLogs': debug_logs,
                'WarningLogs': warning_logs,
                'ErrorLogs': error_logs,
                'CriticalLogs': critical_logs
            })

        logs_features_df = df_logs_indexed.groupby(['ContainerID', 'ServiceName']).resample(resample_interval).apply(log_features).reset_index()
        logs_features_df = logs_features_df.fillna(0)
        print("Log features extracted and NaNs filled.")

        # --- 6. Merge all DataFrames ---
        # Merge network and infra data first
        combined_metrics_df = pd.merge(network_agg, infra_agg,
                                       on=['Timestamp', 'ContainerID', 'ServiceName'],
                                       how='outer')

        # Then merge with log features
        final_df = pd.merge(combined_metrics_df, logs_features_df,
                            on=['Timestamp', 'ContainerID', 'ServiceName'],
                            how='outer')

        # Fill any remaining NaNs after merging (for time-container-service combinations not present in all original DFs)
        final_df = final_df.fillna(0)

        print("\n--- Final Merged DataFrame created successfully! ---")
        print(final_df.head())
        print(f"\nFinal Merged DataFrame shape: {final_df.shape}")

        # --- 7. Optionally Save the Processed Data ---
        if output_file:
            final_df.to_csv(output_file, index=False)
            print(f"\nProcessed data saved to {output_file}")

        return final_df

    except FileNotFoundError:
        print("One or more CSV files not found. Please ensure the data generation script has been run successfully.")
        return None
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        return None

if __name__ == '__main__':
    # This block will run when the script is executed directly

    # Ensure your synthetic data generation script has been run first,
    # or place the CSV files in the same directory as this script.

    # Example usage:
    processed_data = preprocess_telemetry_data(
        log_file='synthetic_container_logs.csv',
        network_file='synthetic_network_metrics.csv',
        infra_file='synthetic_infra_telemetry.csv',
        resample_interval='1s',
        output_file='processed_telemetry_data.csv' # The processed data will be saved here
    )

    if processed_data is not None:
        print("\nPreprocessing complete. The 'processed_data' DataFrame is ready for ML model development.")
        # You can now proceed with further steps like:
        # - Loading 'processed_telemetry_data.csv' for ML training
        # - Splitting data into training/testing sets
        # - Scaling numerical features
        # - Applying ML anomaly detection models
