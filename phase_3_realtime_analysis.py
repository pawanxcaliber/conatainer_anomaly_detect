import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. Data Preparation Function (from previous steps, consolidated) ---
def prepare_data_for_ml(
    processed_data_file='processed_telemetry_data.csv',
    test_size=0.15, # 15% for test
    val_size=0.15,  # 15% for validation
    random_seed=42,
    scaler_output_path='standard_scaler.joblib' # Path to save the fitted scaler
):
    """
    Loads processed data, splits it into training, validation, and test sets,
    scales the features, and saves the fitted scaler.

    Args:
        processed_data_file (str): Path to the processed telemetry data CSV.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include in the validation split.
        random_seed (int): Seed for random operations for reproducibility.
        scaler_output_path (str): Path to save the fitted StandardScaler object.

    Returns:
        tuple: (X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, df_processed_original)
               The scaled training, validation, test feature sets, and original processed DF (for inspection).
        None: If an error occurs.
    """
    try:
        # --- Load the processed data ---
        df_processed = pd.read_csv(processed_data_file)
        print(f"Processed data loaded successfully from {processed_data_file}")
        print(f"Shape of loaded data: {df_processed.shape}")

        df_processed['Timestamp'] = pd.to_datetime(df_processed['Timestamp'])

        # --- Identify features (X) ---
        columns_to_exclude = ['Timestamp', 'ContainerID', 'ServiceName']
        features_columns = [col for col in df_processed.columns if col not in columns_to_exclude]
        X = df_processed[features_columns]
        print(f"\nFeatures (X) DataFrame shape: {X.shape}")

        # --- Calculate test and validation sizes from total remaining ---
        if (test_size + val_size) >= 1.0:
            raise ValueError("test_size + val_size must be less than 1.0")

        temp_size = test_size + val_size
        X_train, X_temp = train_test_split(X, test_size=temp_size, random_state=random_seed)

        val_size_relative_to_temp = val_size / temp_size
        X_val, X_test = train_test_split(X_temp, test_size=val_size_relative_to_temp, random_state=random_seed)

        print("\nData Split Complete:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"X_test shape: {X_test.shape}")

        # --- Feature Scaling/Normalization ---
        scaler = StandardScaler()
        scaler.fit(X_train) # Fit ONLY on the training data

        X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=features_columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=features_columns, index=X_val.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features_columns, index=X_test.index)

        print("\nFeature Scaling Complete.")

        # --- Save the fitted scaler ---
        joblib.dump(scaler, scaler_output_path)
        print(f"\nFitted StandardScaler saved to {scaler_output_path}")

        # Return original df_processed too, for easier inspection of anomalies later
        return X_train_scaled, X_val_scaled, X_test_scaled, df_processed

    except FileNotFoundError:
        print(f"Error: The file '{processed_data_file}' was not found. Please ensure the preprocessing script was run and the file exists.")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        return None, None, None, None


# --- Define the anomaly detection function for deployment ---
def detect_anomaly_realtime(
    new_data_point: pd.DataFrame,
    scaler_path: str = 'standard_scaler.joblib',
    model_path: str = 'one_class_svm_model.joblib'
) -> tuple:
    """
    Simulates real-time anomaly detection for a single or multiple new data points.

    Args:
        new_data_point (pd.DataFrame): A DataFrame containing the new telemetry data point(s).
                                       Must have the same feature columns as the training data.
        scaler_path (str): Path to the saved StandardScaler object.
        model_path (str): Path to the saved OneClassSVM model object.

    Returns:
        tuple: (prediction, anomaly_score)
               prediction: -1 for anomaly, 1 for normal
               anomaly_score: The decision function score (negative is anomalous)
        None: If an error occurs during loading or prediction.
    """
    try:
        # Load the pre-trained scaler and model
        # In a real system, these would be loaded once when the service starts up
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
        print(f"Loaded scaler from {scaler_path} and model from {model_path}")

        # If new_data_point is a Series (single row), convert it to DataFrame (single row, multiple columns)
        if isinstance(new_data_point, pd.Series):
            new_data_point = new_data_point.to_frame().T
        
        # Preprocess the new data point using the loaded scaler
        scaled_data_point = scaler.transform(new_data_point)
        
        # Convert back to DataFrame to maintain column names for clarity if needed,
        # though scikit-learn models usually work with numpy arrays directly.
        scaled_data_point_df = pd.DataFrame(scaled_data_point, columns=new_data_point.columns, index=new_data_point.index)

        # Get anomaly prediction and score
        prediction = model.predict(scaled_data_point_df)
        anomaly_score = model.decision_function(scaled_data_point_df)

        return prediction, anomaly_score

    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        return None, None
    except Exception as e:
        print(f"An error occurred during real-time detection: {e}")
        return None, None

# --- Main Execution Block ---
if __name__ == '__main__':
    # Make sure you have 'processed_telemetry_data.csv' generated from previous steps
    # and the models/scalers saved as 'standard_scaler.joblib' and 'one_class_svm_model.joblib'.

    # 1. Prepare the data (load, split, scale, save scaler) - Only needs to run once
    print("\n--- Starting Data Preparation (for model training/saving if not done) ---")
    X_train_scaled, X_val_scaled, X_test_scaled, df_processed_original = prepare_data_for_ml(
        processed_data_file='processed_telemetry_data.csv',
        scaler_output_path='standard_scaler.joblib'
    )

    if X_train_scaled is None:
        print("Exiting due to data preparation error. Please check previous logs.")
        exit()

    # --- Train and Save One-Class SVM (if not already saved from Pass 3) ---
    # This block ensures the OCSVM model is trained and saved if you're running this script standalone
    print("\n--- Training/Loading One-Class SVM Model for Deployment ---")
    ocsvm_model_output_path = 'one_class_svm_model.joblib'
    try:
        # Attempt to load the model first
        ocsvm_model = joblib.load(ocsvm_model_output_path)
        print(f"One-Class SVM model loaded from {ocsvm_model_output_path}.")
    except FileNotFoundError:
        # If not found, train and save it
        print("One-Class SVM model not found, training new one.")
        ocsvm_model = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')
        ocsvm_model.fit(X_train_scaled)
        joblib.dump(ocsvm_model, ocsvm_model_output_path)
        print(f"Trained One-Class SVM model saved to {ocsvm_model_output_path}.")

    # --- Simulating Real-time Anomaly Detection (Phase 3) ---
    print("\n--- Simulating Real-time Anomaly Detection (Phase 3) ---")

    # --- Scenario 1: Simulate a known anomalous data point (from malicious-container-x) ---
    # Let's pick an actual anomalous row index from your previous output (e.g., 4149)
    anomalous_sample_index = 4149
    simulated_anomalous_data = df_processed_original[features_columns].loc[[anomalous_sample_index]]
    original_anomalous_info = df_processed_original.loc[[anomalous_sample_index]]

    print(f"\n--- Simulating Anomaly: {original_anomalous_info['ContainerID'].iloc[0]} at {original_anomalous_info['Timestamp'].iloc[0]} ---")
    pred, score = detect_anomaly_realtime(simulated_anomalous_data)
    
    if pred is not None:
        status = "Anomaly Detected!" if pred[0] == -1 else "Normal"
        print(f"Prediction: {status}, Anomaly Score: {score[0]:.4f}")
        if status == "Anomaly Detected!":
            print("  -> This data point indeed looks like an anomaly based on its characteristics (high resource/network usage, errors).")
        print(f"  Original Info: ContainerID={original_anomalous_info['ContainerID'].iloc[0]}, Service={original_anomalous_info['ServiceName'].iloc[0]}")
        
    # --- Scenario 2: Simulate a known normal data point ---
    # CORRECTED LINE: Access the element of the Index object using [0]
    normal_sample_index = df_processed_original[df_processed_original['ContainerID'] != 'malicious-container-x'].sample(1, random_state=1).index[0]
    simulated_normal_data = df_processed_original[features_columns].loc[[normal_sample_index]]
    original_normal_info = df_processed_original.loc[[normal_sample_index]]

    print(f"\n--- Simulating Normal: {original_normal_info['ContainerID'].iloc[0]} at {original_normal_info['Timestamp'].iloc[0]} ---")
    pred, score = detect_anomaly_realtime(simulated_normal_data)

    if pred is not None:
        status = "Anomaly Detected!" if pred[0] == -1 else "Normal"
        print(f"Prediction: {status}, Anomaly Score: {score[0]:.4f}")
        if status == "Normal":
            print("  -> This data point is correctly classified as normal.")
        print(f"  Original Info: ContainerID={original_normal_info['ContainerID'].iloc[0]}, Service={original_normal_info['ServiceName'].iloc[0]}")

    # --- Scenario 3: Simulate a *new*, custom anomalous data point (hypothetical) ---
    print("\n--- Simulating a NEW Custom Anomaly ---")
    custom_anomaly_data = pd.DataFrame([{
        'BytesSentKBps': 5000.0,
        'BytesReceivedKBps': 6000.0,
        'PacketsDropped': 500.0,
        'LatencyMs': 2000.0,
        'ConnectionsActive': 300.0,
        'ConnectionFailures': 50.0,
        'CPUUtilizationPercent': 150.0, # Very high
        'MemoryUsageMB': 5000.0,
        'DiskReadKBps': 5000.0,
        'DiskWriteKBps': 4000.0,
        'ProcessCount': 100.0,
        'TotalLogs': 20.0,
        'InfoLogs': 0.0,
        'DebugLogs': 0.0,
        'WarningLogs': 0.0,
        'ErrorLogs': 10.0,
        'CriticalLogs': 10.0 # Very high critical errors
    }], columns=features_columns) # Ensure columns match your training features

    pred, score = detect_anomaly_realtime(custom_anomaly_data)

    if pred is not None:
        status = "Anomaly Detected!" if pred[0] == -1 else "Normal"
        print(f"Prediction: {status}, Anomaly Score: {score[0]:.4f}")
        if status == "Anomaly Detected!":
            print("  -> The model successfully flagged our custom-designed anomaly!")
        else:
            print("  -> Unexpectedly, the model classified our custom anomaly as normal. Review parameters/features.")
    
    print("\n--- Phase 3: Deployment & Integration Simulation Complete ---")
    print("This demonstrates how your trained One-Class SVM model would operate in a real-time inference scenario.")
