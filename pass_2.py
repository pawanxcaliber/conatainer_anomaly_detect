import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib # For saving/loading scaler and potentially models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy for percentile calculation

# --- 1. Data Preparation Function (from previous steps, consolidated) ---
# (Keeping this function defined as it was, no changes needed here)
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

        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Convert back to DataFrame to keep column names and indices
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features_columns, index=X_train.index)
        X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=features_columns, index=X_val.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features_columns, index=X_test.index)

        print("\nFeature Scaling Complete.")

        # --- Save the fitted scaler ---
        joblib.dump(scaler, scaler_output_path)
        print(f"\nFitted StandardScaler saved to {scaler_output_path}")

        # Return original df_processed too, for easier inspection of anomalies later
        return X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, df_processed

    except FileNotFoundError:
        print(f"Error: The file '{processed_data_file}' was not found. Please ensure the preprocessing script was run and the file exists.")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        return None, None, None, None


# --- Main Execution Block for Pass 2: Hyperparameter Tuning ---
if __name__ == '__main__':
    # 1. Prepare the data (load, split, scale, save scaler) - same as Pass 1
    print("\n--- Starting Data Preparation for Pass 2 ---")
    X_train_scaled, X_val_scaled, X_test_scaled, df_processed_original = prepare_data_for_ml(
        processed_data_file='processed_telemetry_data.csv',
        scaler_output_path='standard_scaler.joblib'
    )

    if X_train_scaled is None:
        print("Exiting due to data preparation error. Please check previous logs.")
        exit()

    # Define contamination values to test
    contamination_values = [0.005, 0.01, 0.02, 0.05]
    results = {}

    print("\n--- Starting Pass 2: Hyperparameter Tuning (Contamination) ---")

    for contamination_rate in contamination_values:
        print(f"\n--- Testing Isolation Forest with contamination={contamination_rate} ---")

        # 2. Train ML Model with current contamination rate
        isolation_forest_model = IsolationForest(
            n_estimators=100, # Keeping n_estimators constant for this tuning round
            contamination=contamination_rate,
            random_state=42,
            n_jobs=-1
        )
        isolation_forest_model.fit(X_train_scaled)
        print(f"Isolation Forest Model Trained with contamination={contamination_rate}.")

        # 3. Generate Anomaly Predictions for Evaluation (using Validation Set)
        y_val_pred_if = isolation_forest_model.predict(X_val_scaled)
        y_val_scores_if = isolation_forest_model.decision_function(X_val_scaled)
        print("Anomaly Predictions Generated for Validation Set.")

        # 4. Evaluate Model Performance for current contamination rate
        # Calculate the decision threshold based on the 'contamination' parameter
        calculated_threshold = np.percentile(y_train_scores_if, contamination_rate * 100) # Use training scores for threshold

        # Visualize Anomaly Scores Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(y_val_scores_if, bins=50, kde=True)
        plt.title(f'Anomaly Scores on Validation Set (Contamination={contamination_rate})')
        plt.xlabel('Anomaly Score (lower = more anomalous)')
        plt.ylabel('Number of Instances')
        plt.axvline(x=calculated_threshold, color='r', linestyle='--', label=f'Decision Threshold ({calculated_threshold:.4f})')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Count Detected Anomalies
        num_val_anomalies = (y_val_pred_if == -1).sum()
        total_val_samples = len(y_val_pred_if)
        percentage_anomalies = (num_val_anomalies / total_val_samples) * 100
        print(f"Validation Set: Detected {num_val_anomalies} anomalies out of {total_val_samples} samples ({percentage_anomalies:.2f}% of total)")

        # Store results
        results[contamination_rate] = {
            'num_anomalies': num_val_anomalies,
            'percentage_anomalies': percentage_anomalies,
            'threshold': calculated_threshold
        }

        # Optional: Inspect Sample Anomalies for this contamination (if desired, can get verbose)
        # For brevity, we'll skip detailed sample inspection for each iteration here,
        # but you would typically do this for your chosen final parameter.

    print("\n--- Pass 2: Hyperparameter Tuning (Contamination) Complete ---")
    print("\nSummary of Results:")
    for cont, res in results.items():
        print(f"Contamination {cont}:")
        print(f"  - Detected Anomalies: {res['num_anomalies']} ({res['percentage_anomalies']:.2f}%)")
        print(f"  - Decision Threshold: {res['threshold']:.4f}")
    print("\nBased on these results, you can choose the 'contamination' value that best fits your operational requirements for anomaly detection.")
