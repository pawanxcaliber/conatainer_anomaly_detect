import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib # For saving/loading scaler and potentially models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy for percentile calculation

# --- 1. Data Preparation Function (from previous steps, now consolidated) ---
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


# --- Main Execution Block for First Pass ML Model Development ---
if __name__ == '__main__':
    # Ensure processed_telemetry_data.csv exists from the previous preprocessing step.

    # 1. Prepare the data (load, split, scale, save scaler)
    print("\n--- Starting Data Preparation ---")
    X_train_scaled, X_val_scaled, X_test_scaled, df_processed_original = prepare_data_for_ml(
        processed_data_file='processed_telemetry_data.csv',
        scaler_output_path='standard_scaler.joblib' # Scaler will be saved here
    )

    if X_train_scaled is None: # Check if data preparation failed
        print("Exiting due to data preparation error. Please check previous logs.")
        exit() # Terminate script if data isn't ready

    # 2. Train ML Model: Isolation Forest (First Pass)
    print("\n--- Training Isolation Forest Model ---")
    # Initialize Isolation Forest Model
    # contamination: Expected proportion of outliers. Set to 1% for this first pass.
    # n_estimators: Number of trees in the forest.
    # n_jobs=-1: Use all available CPU cores.
    isolation_forest_model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
    
    # Fit the model to the scaled training data
    isolation_forest_model.fit(X_train_scaled)

    # Save the trained model (important for future inference)
    model_output_path = 'isolation_forest_model.joblib'
    joblib.dump(isolation_forest_model, model_output_path)
    print(f"Trained Isolation Forest model saved to {model_output_path}")


    # 3. Generate Anomaly Predictions for Evaluation
    print("\n--- Generating Anomaly Predictions ---")
    # Note: decision_function returns raw anomaly scores (lower = more anomalous)
    # predict returns -1 for anomaly, 1 for normal
    y_train_pred_if = isolation_forest_model.predict(X_train_scaled)
    y_train_scores_if = isolation_forest_model.decision_function(X_train_scaled)

    y_val_pred_if = isolation_forest_model.predict(X_val_scaled)
    y_val_scores_if = isolation_forest_model.decision_function(X_val_scaled)

    y_test_pred_if = isolation_forest_model.predict(X_test_scaled)
    y_test_scores_if = isolation_forest_model.decision_function(X_test_scaled)

    print("Anomaly Predictions Generated for Training, Validation, and Test Sets.")


    # 4. Evaluate Model Performance (First Pass)
    print("\n--- Isolation Forest Model Evaluation (First Pass) ---")

    # Calculate the decision threshold based on the 'contamination' parameter
    # The threshold is the score value below which points are classified as anomalies.
    # It corresponds to the (contamination * 100) percentile of the decision_function scores
    # on the training data.
    contamination_rate = isolation_forest_model.contamination # Get contamination from the model
    # Calculate the threshold as the value at the (contamination * 100) percentile of the training scores
    calculated_threshold = np.percentile(y_train_scores_if, contamination_rate * 100)


    # Visualize Anomaly Scores Distribution (Validation Set is good for this)
    plt.figure(figsize=(10, 6))
    sns.histplot(y_val_scores_if, bins=50, kde=True)
    plt.title('Distribution of Anomaly Scores on Validation Set (Isolation Forest)')
    plt.xlabel('Anomaly Score (lower = more anomalous)')
    plt.ylabel('Number of Instances')
    # Use the calculated threshold for plotting the vertical line
    plt.axvline(x=calculated_threshold, color='r', linestyle='--', label=f'Decision Threshold ({calculated_threshold:.4f})')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Count Detected Anomalies
    num_val_anomalies = (y_val_pred_if == -1).sum()
    total_val_samples = len(y_val_pred_if)
    print(f"Validation Set: Detected {num_val_anomalies} anomalies out of {total_val_samples} samples ({num_val_anomalies/total_val_samples:.2%} of total)")

    num_test_anomalies = (y_test_pred_if == -1).sum()
    total_test_samples = len(y_test_pred_if)
    print(f"Test Set: Detected {num_test_anomalies} anomalies out of {total_test_samples} samples ({num_test_anomalies/total_test_samples:.2%} of total)")

    # Inspect Sample Anomalies from Validation Set (using original data for context)
    anomalous_val_indices = X_val_scaled.index[y_val_pred_if == -1] # Get original indices of anomalies
    
    # Ensure there are anomalies to display
    if not anomalous_val_indices.empty:
        # Use .loc to get the rows from the original processed DataFrame
        original_anomalous_val_data = df_processed_original.loc[anomalous_val_indices]
        print("\nSample of 5 Data Points Flagged as Anomalies in Validation Set (Original Data):")
        print(original_anomalous_val_data.head())
    else:
        print("\nNo anomalies detected in the validation set to display in detail.")

    print("\n--- First Pass ML Model Development & Evaluation Complete ---")
    print("You now have a trained Isolation Forest model and initial insights into its performance.")
