import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib # Often preferred over pickle for sklearn objects due to efficiency

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
        tuple: (X_train_scaled_df, X_val_scaled_df, X_test_scaled_df)
               The scaled training, validation, and test feature sets.
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
        # Ensure that test_size + val_size does not exceed 1.0
        if (test_size + val_size) >= 1.0:
            raise ValueError("test_size + val_size must be less than 1.0")

        # First, split into training and a temporary set (validation + test)
        # Remaining for train = 1.0 - (test_size + val_size)
        temp_size = test_size + val_size
        X_train, X_temp = train_test_split(X, test_size=temp_size, random_state=random_seed)

        # Then, split the temporary set into validation and test sets
        # The proportion for val/test from X_temp will be relative to X_temp's size
        # e.g., if temp_size is 0.30, and you want val=0.15, test=0.15,
        # then val_size_relative_to_temp = 0.15 / 0.30 = 0.5
        val_size_relative_to_temp = val_size / temp_size
        X_val, X_test = train_test_split(X_temp, test_size=val_size_relative_to_temp, random_state=random_seed)

        print("\nData Split Complete:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"X_test shape: {X_test.shape}")

        # --- Feature Scaling/Normalization ---
        scaler = StandardScaler()

        # Fit the scaler ONLY on the training data to prevent data leakage
        scaler.fit(X_train)

        # Transform all three sets using the fitted scaler
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Convert back to DataFrame to keep column names and indices
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features_columns, index=X_train.index)
        X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=features_columns, index=X_val.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features_columns, index=X_test.index)

        print("\nFeature Scaling Complete:")
        print(f"X_train_scaled shape: {X_train_scaled_df.shape}")
        print(f"X_val_scaled shape: {X_val_scaled_df.shape}")
        print(f"X_test_scaled shape: {X_test_scaled_df.shape}")

        print("\nSample of Scaled Training Data (first 5 rows, first 5 columns):")
        print(X_train_scaled_df.iloc[:5, :5])

        # --- Save the fitted scaler ---
        joblib.dump(scaler, scaler_output_path)
        print(f"\nFitted StandardScaler saved to {scaler_output_path}")

        return X_train_scaled_df, X_val_scaled_df, X_test_scaled_df

    except FileNotFoundError:
        print(f"Error: The file '{processed_data_file}' was not found. Please ensure the preprocessing script was run and the file exists.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        return None, None, None

if __name__ == '__main__':
    # Ensure processed_telemetry_data.csv exists from the previous preprocessing step.

    # Example usage:
    X_train_scaled, X_val_scaled, X_test_scaled = prepare_data_for_ml(
        processed_data_file='processed_telemetry_data.csv',
        scaler_output_path='standard_scaler.joblib'
    )

    if X_train_scaled is not None:
        print("\nData preparation for ML complete. Scaled datasets are ready for model training.")
        # Now you would proceed to train your Isolation Forest model
        # using X_train_scaled, and evaluate using X_val_scaled and X_test_scaled.
