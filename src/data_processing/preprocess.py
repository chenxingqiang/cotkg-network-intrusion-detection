import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def validate_required_columns(df):
    """Validate if DataFrame contains all required columns"""
    required_columns = {
        'flow_duration',  # 新的列名
        'Header_Length',
        'Protocol Type',
        'Rate', 
        'Srate',
        'Drate',
        'label',
        # 添加其他必需的列...
    }
    
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Warning: Missing required columns: {list(missing_columns)}")
        return False
    return True

def load_and_preprocess_data(file_path, test_mode=False, test_size=0.2, random_state=42):
    """
    Load and preprocess network traffic data.

    Args:
        file_path (str): Path to the CSV data file
        test_mode (bool): If True, only load a small portion of data for testing
        test_size (float): Proportion of the dataset to include in the test split (0.0 to 1.0)
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("Loading and preprocessing data...")

    try:
        # Load the data
        df = pd.read_csv(file_path)
        
        # Validate columns
        if not validate_required_columns(df):
            raise ValueError("Missing required columns in the dataset")
        
        # Store label column before selecting numeric features
        labels = df['label'].copy()
        
        # Ensure numeric features (排除 label 列)
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        X = df[numeric_columns].copy()

        # Convert labels to binary (0 for BENIGN, 1 for attacks)
        y = (labels != 'BENIGN').astype(int)

        # Handle missing values
        X = X.fillna(0)

        # Remove duplicate columns if any exist
        X = X.loc[:, ~X.columns.duplicated()]
        
        print(f"Final X shape: {X.shape}")
        print(f"Final y shape: {y.shape}")

        if test_mode:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            return X_train, X_test, y_train, y_test
        else:
            return X, y

    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    file_path = "data/raw/CICIDS2017.csv"

    try:
        print("\nRunning in test mode...")
        X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path,
            test_mode=True,
            test_size=0.2,
            random_state=42
        )
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Testing labels shape: {y_test.shape}")
    except Exception as e:
        print(f"Failed to load data: {str(e)}")